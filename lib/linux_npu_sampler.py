"""
NPUSampler: samples total NPU memory consumption on Linux.

This is a faithful Python port of the algorithm implemented in the
"Resources" Rust project (https://github.com/nokyan/resources), specifically:

  - src/utils/npu/mod.rs           (device discovery via /sys/class/accel)
  - lib/process_data/src/pci_slot.rs (PCI slot parsing/formatting)
  - lib/process_data/src/lib.rs    (fdinfo scanning, MEM_DRM_FIELDS, dedup/merge)
  - lib/process_data/src/npu_usage.rs (per-driver memory extraction)
  - src/utils/app.rs               (App::npu_mem: sum of per-process mem per NPU)

Two independent data sources are combined per NPU device to be exhaustive
across "all possible" NPU vendors/drivers, exactly mirroring what the
upstream code does for each individual driver:

  1. Per-process accounting via DRM `fdinfo`, for drivers that expose a
     memory field there (this is the "MEM_DRM_FIELDS" table upstream).
     This is summed across all processes holding a handle to that device
     (after de-duplicating fds that point at the same target, and taking
     the max across multiple fds of the *same* process against the *same*
     device, exactly like upstream's `NpuUsageStats::greater`).

  2. Device-level sysfs counters (`.../device/npu_memory_utilization`),
     used by upstream as the fallback/general path for any NPU driver
     that does not (yet) expose per-process memory via fdinfo.

  For a given device, source (1) is used if any process reports memory
  for it via fdinfo; otherwise source (2) is used as a fallback. This
  mirrors the fact that upstream treats these as two independent metrics
  and never double-counts a single driver via both paths in practice
  (only amdxdna currently populates fdinfo memory; all other drivers only
  have the sysfs counter, if any).

Units: everything is normalized to bytes, matching upstream (device sysfs
values are raw bytes with no conversion in upstream code; fdinfo values are
reported in KiB by the kernel and are multiplied by 1024 in upstream code).
"""

from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PciSlot: mirrors lib/process_data/src/pci_slot.rs exactly.
#
# Format: "domain:bus:number.function", all hex, where:
#   domain   : 16 bits (0x0000-0xffff)
#   bus      : 8 bits  (0x00-0xff)
#   number   : 5 bits  (0x00-0x1f)   <- device number
#   function : 3 bits  (0x0-0x7)
#
# Canonical display form (Display impl): "{domain:04x}:{bus:02x}:{number:02x}.{function:x}"
# Parsing (FromStr impl): split on ':' and '.', exactly 4 hex parts required.
# ---------------------------------------------------------------------------


class PciSlotError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class PciSlot:
    domain: int
    bus: int
    number: int
    function: int

    @staticmethod
    def try_new(domain: int, bus: int, number: int, function: int) -> "PciSlot":
        if number > 0x1F:
            raise PciSlotError(f"PCI slot number out of bounds (is {number}, must be < 32)")
        if function > 0x7:
            raise PciSlotError(f"PCI slot function out of bounds (is {function}, must be < 8)")
        return PciSlot(domain & 0xFFFF, bus & 0xFF, number & 0x1F, function & 0x7)

    @staticmethod
    def from_str(s: str) -> "PciSlot":
        parts = re.split(r"[:.]", s)
        if len(parts) != 4:
            raise PciSlotError("invalid PCI slot format: expected 'domain:bus:number.function'")
        try:
            domain = int(parts[0], 16)
            bus = int(parts[1], 16)
            number = int(parts[2], 16)
            function = int(parts[3], 16)
        except ValueError as exc:
            raise PciSlotError(f"invalid hex component in PCI slot string {s!r}") from exc
        return PciSlot.try_new(domain, bus, number, function)

    def __str__(self) -> str:
        return f"{self.domain:04x}:{self.bus:02x}:{self.number:02x}.{self.function:x}"


def _try_parse_pci_slot(s: Optional[str]) -> Optional[PciSlot]:
    if s is None:
        logger.debug("[NPU] _try_parse_pci_slot: no PCI_SLOT_NAME value provided")
        return None
    try:
        parsed = PciSlot.from_str(s.strip())
        logger.debug("[NPU] _try_parse_pci_slot: parsed %r -> %s", s, parsed)
        return parsed
    except PciSlotError as exc:
        logger.debug("[NPU] _try_parse_pci_slot: failed to parse %r: %s", s, exc)
        return None


# ---------------------------------------------------------------------------
# uevent parsing: mirrors `read_uevent` usage in src/utils/npu/mod.rs.
# uevent files are simple "KEY=VALUE" lines.
# ---------------------------------------------------------------------------


def _read_uevent(path: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "=" in line:
                    key, _, value = line.partition("=")
                    result[key.strip()] = value.strip()
    except OSError as exc:
        logger.debug("[NPU] _read_uevent: could not read %s: %s", path, exc)
        return result
    logger.debug("[NPU] _read_uevent: %s -> %r", path, result)
    return result


# ---------------------------------------------------------------------------
# fdinfo parsing: mirrors `parse_fdinfo` in lib/process_data/src/lib.rs.
# fdinfo files are ":"-separated "key: value" lines.
# ---------------------------------------------------------------------------


def _parse_fdinfo(content: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in content.splitlines():
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        result[name.strip()] = value.strip()
    return result


# ---------------------------------------------------------------------------
# MEM_DRM_FIELDS: mirrors the static table in lib/process_data/src/lib.rs.
#
# Maps a DRM driver name (as reported in fdinfo's "drm-driver" field) to the
# list of fdinfo field names whose values (each formatted as "<N> KiB") are
# summed to produce that driver's total memory usage for one fd.
#
# NOTE: amdgpu / i915 / v3d / xe are primarily GPU drivers, but are included
# here verbatim because upstream's fdinfo scanning logic is shared between
# GPUs and NPUs (a process's fdinfo can contain any DRM driver's fields);
# "amdxdna_accel_driver" is the only driver currently used for a dedicated
# NPU/accel device by upstream.
# ---------------------------------------------------------------------------

MEM_DRM_FIELDS: Dict[str, List[str]] = {
    "amdgpu": ["drm-memory-gtt", "drm-memory-vram"],
    "amdxdna_accel_driver": ["drm-total-memory"],
    "i915": ["drm-total-local0", "drm-total-system0"],
    "v3d": ["drm-total-memory"],
    "xe": ["drm-total-gtt", "drm-total-vram0"],
}

_RE_DRM_KIB = re.compile(r"(\d+)\s*KiB")

DRM_DRIVER_KEY = "drm-driver"
DRM_PDEV_KEY = "drm-pdev"


def _parse_drm_kib_fields(fdinfo_content: Dict[str, str], field_names: List[str]) -> int:
    """Mirrors `parse_drm_fields::<u64, _>(fdinfo, names, &RE_DRM_KIB)`: sum of
    parsed KiB values (not yet converted to bytes) across all listed fields
    that are present and match the regex."""
    total = 0
    for name in field_names:
        value = fdinfo_content.get(name)
        if value is None:
            continue
        match = _RE_DRM_KIB.search(value)
        if match:
            total += int(match.group(1))
    return total


def _extract_mem_bytes_from_fdinfo(
    fdinfo_content: Dict[str, str],
) -> Optional[Tuple[PciSlot, int]]:
    """Mirrors `extract_npu_usage_from_fdinfo`, memory-only subset.

    Returns (pci_slot, mem_bytes) if this fdinfo belongs to a DRM device
    whose driver has a known memory field mapping; otherwise None.
    """
    driver = fdinfo_content.get(DRM_DRIVER_KEY)
    if driver is None:
        return None

    field_names = MEM_DRM_FIELDS.get(driver)
    if field_names is None:
        # Driver not recognized for memory extraction (mirrors the `_ => bail!`
        # arm in extract_npu_usage_from_fdinfo, generalized to any driver
        # since we want full vendor coverage rather than amdxdna-only).
        return None

    pci_slot = _try_parse_pci_slot(fdinfo_content.get(DRM_PDEV_KEY)) or PciSlot(0, 0, 0, 0)
    mem_kib = _parse_drm_kib_fields(fdinfo_content, field_names)
    mem_bytes = mem_kib * 1024  # saturating_mul(1024) in upstream; Python ints don't overflow
    return pci_slot, mem_bytes


# ---------------------------------------------------------------------------
# NPU device discovery: mirrors `Npu::get_npus` / `Npu::from_sysfs_path`
# in src/utils/npu/mod.rs.
# ---------------------------------------------------------------------------


@dataclass
class _NpuDevice:
    accel_path: str  # e.g. /sys/class/accel/accel0
    sysfs_device_path: str  # e.g. /sys/class/accel/accel0/device
    pci_slot: Optional[PciSlot]
    driver: Optional[str]


def _discover_npus() -> List[_NpuDevice]:
    devices: List[_NpuDevice] = []

    accel_paths = sorted(glob.glob("/sys/class/accel/accel*"))
    logger.debug("[NPU] _discover_npus: found %d candidate accel path(s): %s", len(accel_paths), accel_paths)

    for accel_path in accel_paths:
        sysfs_device_path = os.path.join(accel_path, "device")
        uevent = _read_uevent(os.path.join(sysfs_device_path, "uevent"))

        driver = uevent.get("DRIVER")

        # Mirrors: "if the driver is simple-framebuffer, it's likely not a GPU"
        if driver == "simple-framebuffer":
            logger.debug("[NPU] _discover_npus: skipping %s (driver=simple-framebuffer)", accel_path)
            continue

        pci_slot = _try_parse_pci_slot(uevent.get("PCI_SLOT_NAME"))

        logger.debug(
            "[NPU] _discover_npus: registering device accel_path=%s driver=%s pci_slot=%s",
            accel_path, driver, pci_slot,
        )

        devices.append(
            _NpuDevice(
                accel_path=accel_path,
                sysfs_device_path=sysfs_device_path,
                pci_slot=pci_slot,
                driver=driver,
            )
        )

    logger.debug("[NPU] _discover_npus: discovered %d device(s) total", len(devices))
    return devices


def _read_parsed_u64(path: str) -> Optional[int]:
    """Mirrors `read_parsed::<u64>(path)`: read file, trim, parse as integer.
    Returns None on any failure (missing file, permission error, parse
    failure), matching upstream's `Result` -> discarded-on-error semantics.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            value = int(f.read().strip())
            logger.debug("[NPU] _read_parsed_u64: %s -> %d", path, value)
            return value
    except (OSError, ValueError) as exc:
        logger.debug("[NPU] _read_parsed_u64: failed to read/parse %s: %s", path, exc)
        return None


def _device_used_memory_bytes(device: _NpuDevice) -> Optional[int]:
    """Mirrors `NpuImpl::drm_used_memory` default impl:
    reads `<sysfs_device_path>/npu_memory_utilization` as raw bytes."""
    path = os.path.join(device.sysfs_device_path, "npu_memory_utilization")
    result = _read_parsed_u64(path)
    logger.debug("[NPU] _device_used_memory_bytes: device=%s path=%s -> %s", device.accel_path, path, result)
    return result


# ---------------------------------------------------------------------------
# Per-process fdinfo scanning: mirrors `Process::retrieve`'s call chain into
# `collect_fdinfos` + `npu_usage_stats` in lib/process_data/src/lib.rs.
#
# Caching, mirroring upstream exactly:
#   - `symlink_cache`: (pid, fd_num) -> resolved /proc/<pid>/fd/<n> target.
#     Avoids a `readlink()` syscall for fds we've already resolved.
#   - `non_relevant_fdinfos`: set of (pid, fd_num) known, from a previous
#     cycle, to not be relevant to memory extraction (i.e. their
#     `drm-driver` isn't in MEM_DRM_FIELDS, or the fdinfo file couldn't be
#     read). Upstream splits this into `non_gpu_fdinfos` / `non_npu_fdinfos`
#     and only skips re-reading a fd when BOTH are true for it; since our
#     single extraction routine already unifies the GPU+NPU driver name
#     table (MEM_DRM_FIELDS), one combined set is the exact equivalent.
#   - Invalidation: if a cached-as-non-relevant fd's resolved target has
#     since changed to point at `/dev/dri/*` or `/dev/accel/*`, the cache
#     entry is dropped and the fd is re-read (fd numbers get reused by the
#     kernel across close()/open() calls, so a fd once irrelevant can later
#     refer to a relevant device).
#
# These caches are stored per-PID on the `NPUSampler` instance and pruned
# for PIDs that have exited between samples, mirroring how upstream's
# caches live on a per-process object that is dropped when the process
# disappears from the tracked process list.
# ---------------------------------------------------------------------------


def _iter_pids() -> List[int]:
    pids = []
    try:
        entries = os.listdir("/proc")
    except OSError as exc:
        logger.debug("[NPU] _iter_pids: failed to list /proc: %s", exc)
        return pids
    for entry in entries:
        if entry.isdigit():
            pids.append(int(entry))
    logger.debug("[NPU] _iter_pids: found %d pid(s)", len(pids))
    return pids


@dataclass
class _PidCache:
    symlink_cache: Dict[int, str] = field(default_factory=dict)
    non_relevant_fds: set = field(default_factory=set)


def _collect_process_npu_mem(pid: int, cache: _PidCache) -> Dict[PciSlot, int]:
    """Returns, for this single process, a mapping of PciSlot -> mem_bytes,
    where multiple fds targeting the same device within this process have
    already been reduced via max() (mirrors `NpuUsageStats::greater`).

    Mutates `cache` in place exactly as upstream's `collect_fdinfos` mutates
    its `symlink_cache`/`non_gpu_fdinfos`/`non_npu_fdinfos` arguments.
    """

    fdinfo_dir = f"/proc/{pid}/fdinfo"
    fd_dir = f"/proc/{pid}/fd"

    try:
        fd_entries = os.listdir(fdinfo_dir)
    except OSError as exc:
        logger.debug("[NPU] _collect_process_npu_mem: pid=%d could not list %s: %s", pid, fdinfo_dir, exc)
        return {}

    seen_targets: Dict[str, int] = {}  # symlink target -> first fd number that had it
    per_device_max: Dict[PciSlot, int] = {}

    for entry in fd_entries:
        try:
            fd_num = int(entry)
        except ValueError:
            continue

        is_cached_non_relevant = fd_num in cache.non_relevant_fds

        cached_target = cache.symlink_cache.get(fd_num)
        if cached_target is not None:
            target: Optional[str] = cached_target
        else:
            fd_symlink = os.path.join(fd_dir, str(fd_num))
            try:
                target = os.readlink(fd_symlink)
            except OSError:
                target = None

        if is_cached_non_relevant:
            if target is not None:
                if "/dev/dri/" not in target and "/dev/accel/" not in target:
                    # Still not relevant: skip without reading/parsing fdinfo.
                    continue
                # Target changed to a GPU/NPU device: invalidate and re-process.
                cache.non_relevant_fds.discard(fd_num)
            # If target is None (couldn't resolve), fall through and
            # re-process anyway, matching upstream's behavior exactly.

        if target is not None:
            first_fd = seen_targets.get(target)
            if first_fd is not None:
                # Duplicate fd pointing at an already-processed target: skip
                # (mirrors the seen_targets de-duplication in collect_fdinfos).
                continue
            seen_targets[target] = fd_num
            cache.symlink_cache[fd_num] = target

        fdinfo_path = os.path.join(fdinfo_dir, entry)
        try:
            with open(fdinfo_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as exc:
            logger.debug("[NPU] _collect_process_npu_mem: pid=%d fd=%s unreadable: %s", pid, entry, exc)
            cache.non_relevant_fds.add(fd_num)
            continue

        parsed = _parse_fdinfo(content)
        extracted = _extract_mem_bytes_from_fdinfo(parsed)
        if extracted is None:
            logger.debug(
                "[NPU] _collect_process_npu_mem: pid=%d fd=%s not relevant (driver=%s)",
                pid, entry, parsed.get(DRM_DRIVER_KEY),
            )
            cache.non_relevant_fds.add(fd_num)
            continue

        pci_slot, mem_bytes = extracted
        logger.debug(
            "[NPU] _collect_process_npu_mem: pid=%d fd=%s pci_slot=%s mem_bytes=%d",
            pid, entry, pci_slot, mem_bytes,
        )
        existing = per_device_max.get(pci_slot)
        if existing is None or mem_bytes > existing:
            per_device_max[pci_slot] = mem_bytes

    logger.debug("[NPU] _collect_process_npu_mem: pid=%d totals=%s", pid, per_device_max)
    return per_device_max


def _collect_all_processes_npu_mem(
    pid_caches: Dict[int, _PidCache],
) -> Dict[PciSlot, int]:
    """Mirrors `App::npu_mem`: sums per-process memory (already max-merged
    per process/device above) across all processes, per PciSlot.

    `pid_caches` is mutated in place: entries for PIDs no longer present are
    pruned, and entries for newly-seen PIDs are created on demand.
    """

    totals: Dict[PciSlot, int] = {}

    current_pids = set(_iter_pids())

    # Prune caches for processes that have exited since the last sample,
    # mirroring upstream dropping a Process's caches when it's no longer
    # tracked.
    for stale_pid in list(pid_caches.keys()):
        if stale_pid not in current_pids:
            logger.debug("[NPU] _collect_all_processes_npu_mem: pruning cache for exited pid=%d", stale_pid)
            del pid_caches[stale_pid]

    for pid in current_pids:
        cache = pid_caches.get(pid)
        if cache is None:
            cache = _PidCache()
            pid_caches[pid] = cache

        try:
            per_device = _collect_process_npu_mem(pid, cache)
        except OSError as exc:
            logger.debug("[NPU] _collect_all_processes_npu_mem: pid=%d raised OSError: %s", pid, exc)
            continue

        for pci_slot, mem_bytes in per_device.items():
            totals[pci_slot] = totals.get(pci_slot, 0) + mem_bytes

    logger.debug("[NPU] _collect_all_processes_npu_mem: fdinfo totals across %d pid(s): %s", len(current_pids), totals)
    return totals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class NPUSampler:
    """Samples total NPU memory consumption (in GiB) across all NPUs
    present in the system, using the same algorithm as the Rust
    "Resources" system monitor.

    Devices are (re-)discovered on every `sample()` call via
    `/sys/class/accel/accel*`, so NPUs that are hot-plugged/unplugged
    between calls are handled transparently. This mirrors upstream's
    behavior of re-running `Npu::get_npus()` on every refresh cycle.

    Attributes
    ----------
    has_npu : bool
        True when at least one NPU device was found. Refreshed on every
        `sample()` call, mirroring the Windows sampler's interface.
    """

    def __init__(self) -> None:
        logger.debug("[NPU] NPUSampler.__init__: starting device discovery")
        # Discover once eagerly so construction fails loudly if /sys/class/accel
        # is inaccessible; sample() re-discovers to stay accurate over time.
        self._devices: List[_NpuDevice] = _discover_npus()

        # Per-PID fdinfo caches (symlink resolutions + "not relevant to memory
        # extraction" markers), persisted across sample() calls and pruned for
        # exited PIDs. Mirrors upstream's per-Process fdinfo caches.
        self._pid_caches: Dict[int, _PidCache] = {}

        # True when at least one NPU device was found. Mirrors the Windows
        # sampler's ``has_npu`` attribute, which callers (e.g. resource_monitor.py)
        # rely on to decide whether to keep polling.
        self.has_npu: bool = bool(self._devices)
        logger.debug(
            "[NPU] NPUSampler.__init__: has_npu=%s devices=%s",
            self.has_npu, [d.accel_path for d in self._devices],
        )

    def sample(self) -> float:
        """
        Collect the latest NPU committed memory statistics.

        Returns
        -------
        float
            Total NPU memory currently consumed, in GiB (rounded to 3
            decimals), summed across every NPU device found on the system.
            Matches the Windows sampler's units.

        For each discovered NPU device, memory is computed as follows,
        exactly mirroring upstream:

          1. Sum, across all processes on the system, the per-process NPU
             memory reported via that process's `/proc/<pid>/fdinfo/*`
             entries whose `drm-pdev` matches this device's PCI slot and
             whose `drm-driver` has a known memory field mapping
             (MEM_DRM_FIELDS). Multiple fds within the same process
             pointing at the same device are combined via max(), not sum
             (matching `NpuUsageStats::greater`), and duplicate fds
             pointing at the same underlying resource are de-duplicated
             before that.

          2. If no process reported any fdinfo-based memory for this
             device (e.g. its driver does not expose the fdinfo memory
             field, such as `intel_vpu`/`ivpu`), fall back to reading
             `<accel_sysfs_path>/device/npu_memory_utilization`, which is
             the device-wide committed memory reported directly by the
             kernel driver, in bytes.
        """
        logger.debug("[NPU] sample: starting sample cycle")
        self._devices = _discover_npus()
        self.has_npu = bool(self._devices)
        logger.debug("[NPU] sample: has_npu=%s device_count=%d", self.has_npu, len(self._devices))

        if not self._devices:
            logger.debug("[NPU] sample: no devices discovered, returning 0.0 GiB")
            return 0.0

        fdinfo_totals = _collect_all_processes_npu_mem(self._pid_caches)

        total_bytes = 0

        for device in self._devices:
            device_bytes: Optional[int] = None
            source = None

            if device.pci_slot is not None and device.pci_slot in fdinfo_totals:
                device_bytes = fdinfo_totals[device.pci_slot]
                source = "fdinfo"

            if device_bytes is None:
                device_bytes = _device_used_memory_bytes(device)
                source = "sysfs" if device_bytes is not None else None

            logger.debug(
                "[NPU] sample: device=%s driver=%s pci_slot=%s source=%s device_bytes=%s",
                device.accel_path, device.driver, device.pci_slot, source, device_bytes,
            )

            if device_bytes is not None:
                total_bytes += device_bytes

        mem_gib = total_bytes / (1024 ** 3) if total_bytes > 0 else 0.0
        mem_gib = round(mem_gib, 3)
        logger.debug("[NPU] sample: total_bytes=%d -> %.3f GiB", total_bytes, mem_gib)
        return mem_gib
