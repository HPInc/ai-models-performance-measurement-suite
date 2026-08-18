"""Linux-specific GPU performance sampling utilities."""

import csv
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation


class GPUSampler:  # pylint: disable=too-few-public-methods
    """Sample GPU utilization and memory on Linux."""

    def __init__(self):
        self.q = None
        self.adapter_counters = {}
        self.mem_counters = {}
        self.linux_devices = []
        self.use_nvidia_smi = False
        self.ready = False
        self._init_linux()

    @staticmethod
    def _safe_read_text(path):
        try:
            with open(path, "r", encoding="utf-8") as file_handle:
                return file_handle.read().strip()
        except OSError:
            return ""

    @staticmethod
    def _parse_int(text):
        if not text:
            return -1
        token = text.split()[0]
        try:
            return int(token, 0)
        except ValueError:
            return -1

    def _sample_nvidia_smi(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return []

        if result.returncode != 0 or not result.stdout.strip():
            return []

        adapter_stats = []
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) < 4:
                continue
            try:
                idx = row[0].strip()
                name = row[1].strip()
                util = min(max(float(row[2].strip()), 0.0), 100.0)
                mem_gb = max(float(row[3].strip()), 0.0) / 1024.0
            except ValueError:
                continue

            adapter_stats.append({
                "adapter": f"nvidia-{idx}:{name}",
                "utilization_percent": round(util, 2),
                "memory_gb": round(mem_gb, 3),
            })

        return adapter_stats

    def _discover_linux_devices(self, skip_vendors=None):
        drm_root = "/sys/class/drm"
        if not os.path.isdir(drm_root):
            return []

        skip_vendors = {vendor.lower() for vendor in (skip_vendors or set())}
        devices = []
        seen_device_dirs = set()

        for entry in sorted(os.listdir(drm_root)):
            if not entry.startswith("card") or not entry[4:].isdigit():
                continue

            card_dir = os.path.join(drm_root, entry)
            device_dir = os.path.join(card_dir, "device")
            if not os.path.isdir(device_dir):
                continue

            real_device_dir = os.path.realpath(device_dir)
            if real_device_dir in seen_device_dirs:
                continue
            seen_device_dirs.add(real_device_dir)

            vendor = self._safe_read_text(os.path.join(device_dir, "vendor")).lower()
            if vendor in skip_vendors:
                continue

            driver_name = ""
            driver_link = os.path.join(device_dir, "driver")
            try:
                if os.path.islink(driver_link):
                    driver_name = os.path.basename(os.path.realpath(driver_link))
            except OSError:
                driver_name = ""

            util_paths = []
            busy_percent = os.path.join(device_dir, "gpu_busy_percent")
            if os.path.isfile(busy_percent):
                util_paths.append(busy_percent)

            gt_root = os.path.join(card_dir, "gt")
            if os.path.isdir(gt_root):
                for gt_name in sorted(os.listdir(gt_root)):
                    gt_busy_percent = os.path.join(gt_root, gt_name, "busy_percent")
                    if os.path.isfile(gt_busy_percent):
                        util_paths.append(gt_busy_percent)

            mem_paths = []
            for filename in ("mem_info_vram_used", "mem_info_vis_vram_used", "mem_info_gtt_used"):
                candidate = os.path.join(device_dir, filename)
                if os.path.isfile(candidate):
                    mem_paths.append(candidate)

            if not util_paths and not mem_paths:
                continue

            adapter_name = entry if not driver_name else f"{entry}:{driver_name}"
            devices.append({
                "adapter": adapter_name,
                "util_paths": util_paths,
                "mem_paths": mem_paths,
            })

        return devices

    def _init_linux(self):
        nvidia_stats = self._sample_nvidia_smi()
        self.use_nvidia_smi = bool(nvidia_stats)
        skip_vendors = {"0x10de"} if self.use_nvidia_smi else set()
        self.linux_devices = self._discover_linux_devices(skip_vendors=skip_vendors)
        self.ready = self.use_nvidia_smi or bool(self.linux_devices)
        if self.ready:
            logger.debug(
                f"[GPU] Linux backend initialized: nvidia_smi={self.use_nvidia_smi}, "
                f"sysfs_devices={len(self.linux_devices)}"
            )
        else:
            logger.debug("[GPU] No supported Linux GPU telemetry found")

    def sample(self):
        if not self.ready:
            self._init_linux()
            if not self.ready:
                return 0.0, 0.0, []

        try:
            adapter_stats = []
            overall_util = 0.0
            overall_mem_gb = 0.0

            nvidia_stats = self._sample_nvidia_smi() if self.use_nvidia_smi else []
            if self.use_nvidia_smi and not nvidia_stats:
                self.use_nvidia_smi = False

            for stat in nvidia_stats:
                adapter_stats.append(stat)
                overall_util = max(overall_util, float(stat["utilization_percent"]))
                overall_mem_gb += float(stat["memory_gb"])

            for device in self.linux_devices:
                util_values = []
                for path in device["util_paths"]:
                    value = self._parse_int(self._safe_read_text(path))
                    if value >= 0:
                        util_values.append(float(value))

                adapter_util = min(sum(util_values), 100.0) if util_values else 0.0

                mem_bytes = 0
                for path in device["mem_paths"]:
                    value = self._parse_int(self._safe_read_text(path))
                    if value > 0:
                        mem_bytes += value
                mem_gb = mem_bytes / (1024 ** 3) if mem_bytes > 0 else 0.0

                adapter_stats.append({
                    "adapter": device["adapter"],
                    "utilization_percent": round(adapter_util, 2),
                    "memory_gb": round(mem_gb, 3),
                })
                overall_util = max(overall_util, adapter_util)
                overall_mem_gb += mem_gb

            return round(overall_util, 2), round(overall_mem_gb, 3), adapter_stats
        except Exception as e:
            logger.warning(f"[GPU] Linux sample failed: {e}")
            self.ready = False
            return 0.0, 0.0, []
