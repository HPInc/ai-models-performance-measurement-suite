#!/usr/bin/env python3
# pylint: disable=duplicate-code,too-many-lines
"""
Monitor a worker process while collecting system (RAM), CPU, GPU, and NPU statistics.

This module provides two public entry points:
    - ResourceMonitor.monitor_process(...): Run a worker function while monitoring.
    - CLI execution: Run this file directly to monitor any shell command string,
      optionally writing JSON results.

High-level behavior
-------------------
- Spawns lightweight monitor processes to sample memory (RAM), CPU utilization,
  GPU utilization/memory, and NPU utilization/memory at configurable intervals
  worker function runs in a separate process.
- Collects time-stamped measurements into a manager-backed namespace so data is
  safely shared across processes.
- Gathers system information via the Windows `systeminfo` command, enriched with
  CPU name, GPU name(s), and Short Product ID.
- Returns both the worker function's result and a dictionary of normalized
  time-series samples whose timestamps are adjusted relative to the worker's
  start time (t=0).
- When run as a CLI tool, can output JSON results with system information.

Key design choices
------------------
- Multiprocessing "spawn" start method for Windows compatibility and isolation.
- Monitor loops implement a responsive sleep (short waits) so the stop signal
  is honored promptly.
- GPU stats are sourced from a sibling `gpu_sampler` module, allowing the
  monitoring orchestration here to remain vendor-neutral and decoupled from
  PDH details.

Expected environment
--------------------
- Requires `psutil` for system metrics.
- Requires `gpu_sampler.GPUSampler` for GPU metrics.
- Requires `npu_sampler.NPUSampler` for NPU metrics.
- Requires `platform_support` for command execution and directory creation.

Imports
-------
- argparse: CLI argument parsing for standalone usage.
- json: serialize results to JSON files.
- logging: structured logging throughout the module.
- multiprocessing: isolate monitors and the worker, and to share state via a
  Manager() namespace and lists.
- os, sys: filesystem and path manipulation.
- platform: detect Windows environment.
- time: timestamps and pacing (sampling intervals and responsive waits).
- typing: type annotations for readability and editor/tooling support.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import platform
import shlex
import sys
import time
from typing import Any, Callable, Dict, Tuple, List

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

# Pylint assumes that multiline comments should only be used for file
# or function doc strings. I profoundly disagree.
# pylint: disable=pointless-string-statement

"""
Third‑party dependency: psutil
- Used for RAM and CPU metrics in a cross-platform manner.
- We fail fast with a clear message so callers learn early about the
  dependency requirement.
"""
try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Please make sure psutil is installed.") from exc

# Compute the directory containing this script to make relative imports robust.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Look in the lib directory for library files.
# Prepend our local ./lib to sys.path so local support modules can be imported
# without requiring site-wide installation.
lib_dir = os.path.join(script_dir, 'lib')
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

try:
    from mass_bench_common import check_python_version
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing mass_bench_common.py library file.") from exc

"""
Library module dependency: gpu_sampler.GPUSampler
- Provides GPU utilization (%) and memory (GiB) sampling.
- Kept separate so the orchestration logic here remains simple and testable.
"""
try:
    from gpu_sampler import GPUSampler
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing gpu_sampler.py library file.") from exc

try:
    from npu_sampler import NPUSampler
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing npu_sampler.py library file.") from exc
    
try:
    from platform_support import create_output_directory_tree, execute_command
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing platform_support.py library file.") from exc

try:
    from resource_monitor_helpers import sample_interval
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing resource_monitor_helpers.py library file.") from exc

try:
    from power_config import get_power_profile, get_power_mode, set_power_mode
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing power_config.py library file.") from exc

class ResourceMonitor:     # pylint: disable=too-few-public-methods
    """
    Orchestrates background sampling of system resources while a worker runs.

    Design notes
    -----------
    - Uses `multiprocessing` with the 'spawn' context (Windows-friendly).
    - Private staticmethods are used as process targets so they are picklable.
    """

    # ---------------------------
    # Private helpers
    # ---------------------------
    @staticmethod
    def _get_cpu_info() -> str:
        """
        Get the CPU name for the current platform.

        Returns
        -------
        str
            The CPU name (e.g., "AMD Ryzen AI 7 PRO 350 w/ Radeon 860M"),
            or an empty string if detection fails.
        """
        if IS_LINUX:
            try:
                with open('/proc/cpuinfo', 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.lower().startswith('model name') and ':' in line:
                            return line.split(':', 1)[1].strip()
            except OSError:
                pass

            stdout, _, returncode = execute_command(['lscpu'])
            if returncode == 0 and stdout:
                for line in stdout.splitlines():
                    if line.startswith('Model name:'):
                        return line.split(':', 1)[1].strip()
            return platform.processor()

        cmd = [
            'powershell', '-NoProfile', '-Command',
            '(Get-CimInstance -ClassName Win32_Processor).Name'
        ]
        stdout, _, returncode = execute_command(cmd)
        if returncode == 0 and stdout:
            return stdout.strip()
        return ""

    @staticmethod
    def _get_gpu_info() -> str:
        """
        Get the GPU name(s) for the current platform.

        Returns
        -------
        str
            The GPU name(s) (e.g., "AMD Radeon (TM) 860M Graphics"),
            or an empty string if detection fails. Multiple GPUs are
            separated by " | ".
        """
        if IS_LINUX:
            stdout, _, returncode = execute_command(['lspci'])
            if returncode == 0 and stdout:
                gpu_names = []
                for line in stdout.splitlines():
                    lowered = line.lower()
                    if any(token in lowered for token in ('vga compatible controller',
                                                          '3d controller',
                                                          'display controller')):
                        gpu_names.append(line.split(': ', 1)[1].strip()
                                         if ': ' in line else line.strip())
                if gpu_names:
                    return ' | '.join(gpu_names)

            drm_root = '/sys/class/drm'
            gpu_names = []
            if os.path.isdir(drm_root):
                for entry in sorted(os.listdir(drm_root)):
                    if not entry.startswith('card') or not entry[4:].isdigit():
                        continue
                    device_dir = os.path.join(drm_root, entry, 'device')
                    driver_link = os.path.join(device_dir, 'driver')
                    try:
                        if os.path.islink(driver_link):
                            gpu_names.append(os.path.basename(os.path.realpath(driver_link)))
                    except OSError:
                        continue
            if gpu_names:
                return ' | '.join(dict.fromkeys(gpu_names))
            return ''

        cmd = [
            'powershell', '-NoProfile', '-Command',
            '(Get-CimInstance -ClassName Win32_VideoController).Name'
        ]
        stdout, _, returncode = execute_command(cmd)
        if returncode == 0 and stdout:
            # Each GPU name is on a separate line
            gpu_names = []
            for line in stdout.strip().split('\n'):
                line = line.strip()
                if line:
                    gpu_names.append(line)
            if gpu_names:
                return ' | '.join(gpu_names)
        return ""

    @staticmethod
    def _get_gpu_driver_version() -> str:
        """
        Get the GPU driver version(s) for the current platform.

        Returns
        -------
        str
            The GPU driver version(s) (e.g., "32.0.11035.1002"),
            or an empty string if detection fails. Multiple driver versions
            are separated by " | ".
        """
        if IS_LINUX:
            return ''

        cmd = [
            'powershell', '-NoProfile', '-Command',
            '(Get-CimInstance -ClassName Win32_VideoController).DriverVersion'
        ]
        stdout, _, returncode = execute_command(cmd)
        if returncode == 0 and stdout:
            # Each driver version is on a separate line
            driver_versions = []
            for line in stdout.strip().split('\n'):
                line = line.strip()
                if line:
                    driver_versions.append(line)
            if driver_versions:
                return ' | '.join(driver_versions)
        return ""

    @staticmethod
    def _get_gpu_driver_date() -> str:
        """
        Get the GPU driver date(s) for the current platform.

        Returns
        -------
        str
            The GPU driver date(s) formatted as YYYY-MM-DD (e.g., "2024-11-15"),
            or an empty string if detection fails. Multiple driver dates
            are separated by " | ".
        """
        if IS_LINUX:
            return ''

        cmd = [
            'powershell', '-NoProfile', '-Command',
            '(Get-CimInstance -ClassName Win32_VideoController).DriverDate | '
            'ForEach-Object { if ($_) { $_.ToString("yyyy-MM-dd") } }'
        ]
        stdout, _, returncode = execute_command(cmd)
        if returncode == 0 and stdout:
            # Each driver date is on a separate line
            driver_dates = []
            for line in stdout.strip().split('\n'):
                line = line.strip()
                if line:
                    driver_dates.append(line)
            if driver_dates:
                return ' | '.join(driver_dates)
        return ""

    @staticmethod
    def _get_shared_vram() -> str:
        """
        Get the Shared System Memory for the GPU when available.

        Uses dxdiag to export diagnostic info and parses the
        "Shared System Memory" field which accurately reports
        the shared VRAM for integrated/hybrid GPUs.

        Returns
        -------
        str
            The shared VRAM (e.g., "8144 MB"),
            or an empty string if detection fails.
        """
        if IS_LINUX:
            return ''

        import tempfile

        # Create a temporary file for dxdiag output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Run dxdiag with /t to export to text file (no UI)
            cmd = ['dxdiag', '/t', tmp_path]
            _, _, returncode = execute_command(cmd)

            if returncode != 0:
                return ""

            # Wait for dxdiag to finish writing (it returns immediately)
            # Poll for the file to have content, with a timeout
            max_wait = 30  # seconds
            poll_interval = 0.5
            waited = 0
            while waited < max_wait:
                time.sleep(poll_interval)
                waited += poll_interval
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    # Give it a bit more time to finish writing
                    time.sleep(0.5)
                    break

            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                return ""

            # Read and parse the output file
            try:
                with open(tmp_path, 'r', encoding='utf-16') as f:
                    content = f.read()
            except (UnicodeError, UnicodeDecodeError):
                # Try UTF-8 as fallback
                with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

            # Look for "Shared System Memory:" line
            for line in content.split('\n'):
                if 'Shared System Memory:' in line:
                    # Extract the value (e.g., "Shared System Memory: 8144 MB")
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        return parts[1].strip()

            return ""
        except Exception:  # pylint: disable=broad-except
            return ""
        finally:
            # Clean up temp file
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _get_short_product_id() -> str:
        """
        Get the short product identifier for the current platform.

        Returns
        -------
        str
            The short product ID from Win32_BaseBoard (e.g., "8E0D"),
            or an empty string if detection fails.
        """
        if IS_LINUX:
            board_name = '/sys/devices/virtual/dmi/id/board_name'
            try:
                with open(board_name, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except OSError:
                return ''

        cmd = [
            'powershell', '-NoProfile', '-Command',
            '(Get-CimInstance -ClassName Win32_BaseBoard).Product'
        ]
        stdout, _, returncode = execute_command(cmd)
        if returncode == 0 and stdout:
            return stdout.strip()
        return ""

    @staticmethod
    def _parse_systeminfo_output(stdout: str) -> Dict[str, str]:
        """
        Parse the raw systeminfo command output into a dictionary.

        Parameters
        ----------
        stdout : str
            The raw stdout from the systeminfo command.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping field names to their values.
        """
        lines = stdout.strip().split('\n')
        system_info: Dict[str, str] = {}
        current_key: str | None = None

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Lines with key-value pairs have a colon
            if ':' in line:
                # Split on the first colon only
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key:
                        system_info[key] = value
                        current_key = key
            elif current_key and current_key in system_info:
                # Continuation line (e.g., for hotfixes) - append to previous value
                system_info[current_key] += ' ' + line.strip()

        return system_info

    @staticmethod
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _insert_product_info(
            system_info: Dict[str, str],
            cpu_info: str,
            gpu_info: str,
            gpu_driver_version: str,
            gpu_driver_date: str,
            shared_vram: str,
            power_profile: str,
            power_mode: str,
            short_product_id: str) -> Dict[str, str]:
        """
        Insert CPU, GPU, GPU Driver Version/Date, Shared VRAM, Power Profile/Mode, and Short Product ID.

        Short Product ID is inserted after the systeminfo "Product ID" field.
        CPU Info, GPU Info, GPU Driver Version, GPU Driver Date, Shared VRAM,
        Power Profile, and Power Mode are inserted after "Processor(s)".

        Parameters
        ----------
        system_info : Dict[str, str]
            The parsed systeminfo dictionary.
        cpu_info : str
            The CPU name string (may be empty).
        gpu_info : str
            The GPU name string (may be empty).
        gpu_driver_version : str
            The GPU driver version string (may be empty).
        gpu_driver_date : str
            The GPU driver date string (may be empty).
        shared_vram : str
            The shared VRAM string (may be empty).
        power_profile : str
            The active Windows power plan name (may be empty).
        power_mode : str
            The active Windows power mode/overlay (may be empty).
        short_product_id : str
            The four-digit hexadecimal short product ID (may be empty).

        Returns
        -------
        Dict[str, str]
            A new dictionary with Short Product ID, CPU Info, GPU Info,
            GPU Driver Version, GPU Driver Date, Shared VRAM, Power Profile,
            and Power Mode inserted.
        """
        new_system_info: Dict[str, str] = {}
        for key, value in system_info.items():
            new_system_info[key] = value
            # Insert Short Product ID after the systeminfo Product ID field
            if key == "Product ID" and short_product_id:
                new_system_info["Short Product ID"] = short_product_id
            # Insert CPU, GPU info, driver version, driver date, shared VRAM,
            # power profile, and power mode after Processor(s)
            if key == "Processor(s)":
                if cpu_info:
                    new_system_info["CPU Info"] = cpu_info
                if gpu_info:
                    new_system_info["GPU Info"] = gpu_info
                if gpu_driver_version:
                    new_system_info["GPU Driver Version"] = gpu_driver_version
                if gpu_driver_date:
                    new_system_info["GPU Driver Date"] = gpu_driver_date
                if shared_vram:
                    new_system_info["Shared VRAM"] = shared_vram
                if power_profile:
                    new_system_info["Power Profile"] = power_profile
                if power_mode:
                    new_system_info["Power Mode"] = power_mode

        # If Product ID wasn't found, add Short Product ID at the end
        if "Short Product ID" not in new_system_info and short_product_id:
            new_system_info["Short Product ID"] = short_product_id
        # If Processor(s) wasn't found, add CPU/GPU/power info at the end
        if "CPU Info" not in new_system_info and cpu_info:
            new_system_info["CPU Info"] = cpu_info
        if "GPU Info" not in new_system_info and gpu_info:
            new_system_info["GPU Info"] = gpu_info
        if "GPU Driver Version" not in new_system_info and gpu_driver_version:
            new_system_info["GPU Driver Version"] = gpu_driver_version
        if "GPU Driver Date" not in new_system_info and gpu_driver_date:
            new_system_info["GPU Driver Date"] = gpu_driver_date
        if "Shared VRAM" not in new_system_info and shared_vram:
            new_system_info["Shared VRAM"] = shared_vram
        if "Power Profile" not in new_system_info and power_profile:
            new_system_info["Power Profile"] = power_profile
        if "Power Mode" not in new_system_info and power_mode:
            new_system_info["Power Mode"] = power_mode

        return new_system_info

    @staticmethod
    def _get_linux_system_info() -> Dict[str, str]:
        """
        Collect a compact Linux system information dictionary.
        """
        uname = platform.uname()
        pretty_name = ''
        os_release = '/etc/os-release'
        if os.path.exists(os_release):
            try:
                with open(os_release, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('PRETTY_NAME='):
                            pretty_name = line.split('=', 1)[1].strip().strip('"')
                            break
            except OSError:
                pass

        cpu_count = psutil.cpu_count(logical=True) or 0
        system_info: Dict[str, str] = {
            'Host Name': uname.node,
            'OS Name': 'Linux',
            'OS Version': pretty_name or uname.release,
            'Kernel': uname.release,
            'Architecture': uname.machine,
            'Processor(s)': str(cpu_count),
        }

        product_name = '/sys/devices/virtual/dmi/id/product_name'
        if os.path.exists(product_name):
            try:
                with open(product_name, 'r', encoding='utf-8') as f:
                    value = f.read().strip()
                    if value:
                        system_info['System Model'] = value
            except OSError:
                pass
                
        return system_info

    @staticmethod
    def _get_system_info() -> Dict[str, str]:
        """
        Run the platform-specific system information collection and return a dictionary.

        Also collects CPU, GPU, GPU Driver Version, GPU Driver Date, Shared VRAM,
        Power Profile, Power Mode, and Short Product ID information via PowerShell
        and powercfg. Short Product ID is inserted after the systeminfo "Product ID"
        field. CPU Info, GPU Info, GPU Driver Version, GPU Driver Date, Shared VRAM,
        Power Profile, and Power Mode are inserted after the "Processor(s)" field.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping field names to their values from systeminfo output,
            plus "Short Product ID", "CPU Info", "GPU Info", "GPU Driver Version",
            "GPU Driver Date", "Shared VRAM", "Power Profile", and "Power Mode" fields.
            Returns an empty dictionary if the command fails or is unavailable.
        """
        if IS_LINUX:
            system_info = ResourceMonitor._get_linux_system_info()
            short_product_id = ResourceMonitor._get_short_product_id()
            cpu_info = ResourceMonitor._get_cpu_info()
            gpu_info = ResourceMonitor._get_gpu_info()
            gpu_driver_version = ResourceMonitor._get_gpu_driver_version()
            gpu_driver_date = ResourceMonitor._get_gpu_driver_date()
            shared_vram = ResourceMonitor._get_shared_vram()
            return ResourceMonitor._insert_product_info(system_info,
                                                        cpu_info,
                                                        gpu_info,
                                                        gpu_driver_version,
                                                        gpu_driver_date,
                                                        shared_vram,
                                                        get_power_profile(),
                                                        get_power_mode(),
                                                        short_product_id)

        stdout, stderr, returncode = execute_command(['systeminfo'])

        if returncode != 0:
            logger.warning("systeminfo command failed with code %d: %s",
                           returncode, stderr)
            return {}

        if not stdout:
            logger.warning("systeminfo command returned no output")
            return {}

        system_info = ResourceMonitor._parse_systeminfo_output(stdout)

        # Get Short Product ID, CPU, GPU info, driver version, driver date,
        # shared VRAM, power profile, and power mode; then insert at appropriate locations
        short_product_id = ResourceMonitor._get_short_product_id()
        cpu_info = ResourceMonitor._get_cpu_info()
        gpu_info = ResourceMonitor._get_gpu_info()
        gpu_driver_version = ResourceMonitor._get_gpu_driver_version()
        gpu_driver_date = ResourceMonitor._get_gpu_driver_date()
        shared_vram = ResourceMonitor._get_shared_vram()
        power_profile = get_power_profile()
        power_mode = get_power_mode()

        return ResourceMonitor._insert_product_info(system_info,
                                                    cpu_info,
                                                    gpu_info,
                                                    gpu_driver_version,
                                                    gpu_driver_date,
                                                    shared_vram,
                                                    power_profile,
                                                    power_mode,
                                                    short_product_id)

    @staticmethod
    def _get_installed_memory_size() -> float:  # pylint: disable=no-method-argument
        """
        Returns the total installed memory size in Gigabytes (GiB).
        """
        memory = psutil.virtual_memory()
        total_memory_bytes = memory.total
        total_memory_gb = total_memory_bytes / (1024 ** 3)
        return round(total_memory_gb, 3)

    @staticmethod
    def _get_used_memory_gb() -> float:
        """
        Return the current used physical memory (RAM) in GiB.

        Uses psutil.virtual_memory().available, converting from bytes (base 2).
        Values are rounded to three decimals for compactness in time-series logs.
        """
        mem_info = psutil.virtual_memory()
        gigs_available = round((mem_info.available / (1024 ** 3)), 3)
        gigs_used = ResourceMonitor._get_installed_memory_size() - gigs_available
        return round(gigs_used, 3)

    @staticmethod
    def _get_average_cpu_utilization(interval: float = 0.1) -> float:
        """
        Return average CPU utilization (%) across all logical cores.

        The measurement is taken over a short sampling interval (default 100 ms),
        computed as the mean of per-core utilizations, and rounded to two decimals.
        """
        by_cores = psutil.cpu_percent(interval=interval, percpu=True)
        return round((sum(by_cores) / len(by_cores)), 2)

    @staticmethod
    def _get_gpu_utilization(gpu_sampler: GPUSampler) -> Tuple[float, float]:
        """
        Return (utilization_percent, memory_gib) from GPUSampler.

        Delegates to the GPUSampler library module and returns the overall
        utilization (%) and total memory (GiB) reported by the sampler.
        """
        gpu_utilization_pct, gpu_memory_gb, _ = gpu_sampler.sample()
        return gpu_utilization_pct, gpu_memory_gb

    @staticmethod
    def _get_npu_memory(npu_sampler: NPUSampler) -> float:
        """
        Return committed memory (GiB) from NPUSampler.

        Delegates to the NPUSampler library module for NPU committed
        memory (GiB).  Returns 0.0 if no NPU is present.
        """
        return npu_sampler.sample()

    # --------------------------------------
    # Private resource monitoring processes.
    # Each process writes timestamped samples into shared lists on stats_ns.
    # --------------------------------------

    @staticmethod
    def _configure_child_logging(log_level: int) -> None:
        """
        Configure logging in spawned child processes.

        Spawned processes do not reliably inherit parent logging configuration,
        so each monitor process configures its own root logger.
        """
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,
        )

    @staticmethod
    def _memory_monitor_process(
        stop_event: "multiprocessing.synchronize.Event",
        stats_ns: "multiprocessing.managers.Namespace",
        interval_sec: float,
        log_level: int = logging.INFO,
    ) -> None:
        """
        Continuously sample used RAM and append [timestamp, value] to stats_ns.
        """
        ResourceMonitor._configure_child_logging(log_level)
        while not stop_event.is_set():
            end_time = time.perf_counter() + interval_sec
            used_gb = ResourceMonitor._get_used_memory_gb()
            stats_ns.used_memory_gb.append([time.perf_counter(), used_gb])

            # Responsive sleep loop
            while time.perf_counter() < end_time:
                if stop_event.wait(timeout=0.01):
                    return

    @staticmethod
    def _cpu_monitor_process(
        stop_event: "multiprocessing.synchronize.Event",
        stats_ns: "multiprocessing.managers.Namespace",
        interval_sec: float,
        log_level: int = logging.INFO,
    ) -> None:
        """
        Continuously sample average CPU utilization and append [timestamp, value].
        """
        ResourceMonitor._configure_child_logging(log_level)
        while not stop_event.is_set():
            end_time = time.perf_counter() + interval_sec
            cpu_utilization_pct = ResourceMonitor._get_average_cpu_utilization()
            stats_ns.cpu_utilization_pct.append([time.perf_counter(), cpu_utilization_pct])

            # Responsive sleep loop
            while time.perf_counter() < end_time:
                if stop_event.wait(timeout=0.01):
                    return

    @staticmethod
    def _gpu_monitor_process(
        stop_event: "multiprocessing.synchronize.Event",
        stats_ns: "multiprocessing.managers.Namespace",
        interval_sec: float,
        log_level: int = logging.INFO,
    ) -> None:
        """
        Continuously sample GPU utilization and memory; append [timestamp, value] pairs.
        """
        ResourceMonitor._configure_child_logging(log_level)
        gpu_sampler = GPUSampler()
        while not stop_event.is_set():
            end_time = time.perf_counter() + interval_sec
            gpu_utilization_pct, gpu_memory_gb = ResourceMonitor._get_gpu_utilization(gpu_sampler)
            timestamp = time.perf_counter()
            stats_ns.gpu_utilization_pct.append([timestamp, gpu_utilization_pct])
            stats_ns.gpu_memory_gb.append([timestamp, gpu_memory_gb])

            # Responsive sleep loop
            while time.perf_counter() < end_time:
                if stop_event.wait(timeout=0.01):
                    return

    @staticmethod
    def _npu_monitor_process(
        stop_event: "multiprocessing.synchronize.Event",
        stats_ns: "multiprocessing.managers.Namespace",
        interval_sec: float = 0.2,
        log_level: int = logging.INFO,
    ) -> None:
        """
        Continuously sample NPU memory; append [timestamp, value] pairs.

        If no NPU is detected, writes a single zero sample and returns immediately
        so the wait loop in monitor_process() does not block indefinitely.
        """
        ResourceMonitor._configure_child_logging(log_level)
        npu_sampler = NPUSampler()
        logger.debug("[NPU] Monitor process initialized sampler: has_npu=%s", npu_sampler.has_npu)
        if not npu_sampler.has_npu:
            logger.debug("[NPU] No NPU detected in monitor process; recording single zero sample")
            timestamp = time.perf_counter()
            stats_ns.npu_memory_gb.append([timestamp, 0.0])
            return

        while not stop_event.is_set():
            end_time = time.perf_counter() + interval_sec
            npu_memory_gb = ResourceMonitor._get_npu_memory(npu_sampler)
            timestamp = time.perf_counter()
            stats_ns.npu_memory_gb.append([timestamp, npu_memory_gb])

            # Responsive sleep loop
            while time.perf_counter() < end_time:
                if stop_event.wait(timeout=0.01):
                    return

    @staticmethod
    def _adjust_timestamps(
        data: List[List[Any]],
        model_start_time: float
    ) -> List[List[Any]]:
        """
        Return a new list with timestamps shifted so model_start_time maps to 0.0.
        """
        adjusted: List[List[Any]] = []
        for timestamp, value in data:
            new_ts = round((timestamp - model_start_time), 3)
            adjusted.append([new_ts, value])
        return adjusted

    @staticmethod
    def _average_stats(data: List[List[Any]]) -> float | None:
        """
        Average of values for samples where timestamp > 0.0. Returns None if no samples.
        """
        total = 0.0
        count = 0
        for timestamp, value in data:
            if timestamp > 0.0:
                total += value
                count += 1
        if count == 0:
            return None
        return round((total / count), 3)

    @staticmethod
    def _max_stats(data: List[List[Any]]) -> float | None:
        maximum: float | None = None
        for timestamp, value in data:
            if timestamp > 0.0:
                if maximum is None:
                    maximum = round(value, 3)
                elif value > maximum:
                    maximum = round(value, 3)
        return maximum

    @staticmethod
    def _min_stats(data: List[List[Any]]) -> float | None:
        minimum: float | None = None
        for timestamp, value in data:
            if timestamp > 0.0:
                if minimum is None:
                    minimum = round(value, 3)
                elif value < minimum:
                    minimum = round(value, 3)
        return minimum

    @staticmethod
    def _prestart_stat(data: List[List[Any]]) -> float | None:
        """
        Return the pre-start value (timestamp < 0), or None if not present.

        Assumes data has been trimmed via _trim_to_prestart(), so the prestart
        sample (if present) is the first element.
        """
        if data and data[0][0] < 0.0:
            return data[0][1]
        return None

    @staticmethod
    def _trim_to_prestart(data: List[List[Any]]) -> List[List[Any]]:
        """
        Remove all samples earlier than the prestart sample (last sample before t=0).

        Uses the same algorithm as _prestart_stat to find the index of the last
        sample where timestamp < 0, then returns data from that index onward.

        Returns
        -------
        List[List[Any]]
            A new list starting from the prestart sample. If no prestart sample
            exists (all timestamps >= 0), returns the original data unchanged.
        """
        prestart_index = None
        for i, (timestamp, _) in enumerate(data):
            if timestamp < 0.0:
                prestart_index = i
                # Don't break - keep iterating to find the last (most recent) pre-start index
            else:
                # Once we hit t >= 0, all remaining values are post-start
                break
        if prestart_index is not None:
            return data[prestart_index:]
        return data

    @staticmethod
    def _normalize_to_prestart(data: List[List[Any]]) -> List[List[Any]]:
        """
        Normalize data values by subtracting the prestart value from each data point.

        Assumes data has been trimmed via _trim_to_prestart(), so the prestart
        sample (if present) is the first element.

        Parameters
        ----------
        data : List[List[Any]]
            Time-series data as [[timestamp, value], ...].

        Returns
        -------
        List[List[Any]]
            A new list with the prestart value subtracted from each data point.
            If no prestart sample exists, returns the original data unchanged.
        """
        if not data or data[0][0] >= 0.0:
            return data

        prestart_value = data[0][1]
        normalized: List[List[Any]] = []
        for timestamp, value in data:
            normalized.append([timestamp, round(value - prestart_value, 3)])
        return normalized

    @staticmethod
    def _start_resource_monitors(
        sample_interval_s: float,
        monitor_log_level: int = logging.INFO,
    ) -> Tuple[List[Any], Any, Any, Any]:
        """
        Start background processes for memory, CPU, and GPU sampling.

        Returns
        -------
        (processes, stats_ns, stop_event, ctx)
        """
        ctx = multiprocessing.get_context("spawn")

        stop_event = ctx.Event()
        manager = ctx.Manager()

        # Shared stats namespace populated by monitors; lists hold [timestamp, value]
        stats_ns = manager.Namespace()
        stats_ns.used_memory_gb = manager.list()
        stats_ns.cpu_utilization_pct = manager.list()
        stats_ns.gpu_utilization_pct = manager.list()
        stats_ns.gpu_memory_gb = manager.list()
        stats_ns.npu_memory_gb = manager.list()
        stats_ns.benchmark_duration_s = float
        stats_ns.sample_interval_s = float

        # Build the memory monitor process.
        mem_proc = ctx.Process(
            target=ResourceMonitor._memory_monitor_process,
            args=(stop_event, stats_ns),
            kwargs={"interval_sec": sample_interval_s, "log_level": monitor_log_level},
            daemon=True,
        )

        # Build the CPU monitor process.
        cpu_proc = ctx.Process(
            target=ResourceMonitor._cpu_monitor_process,
            args=(stop_event, stats_ns),
            kwargs={"interval_sec": sample_interval_s, "log_level": monitor_log_level},
            daemon=True,
        )

        # Build the GPU monitor process.
        gpu_proc = ctx.Process(
            target=ResourceMonitor._gpu_monitor_process,
            args=(stop_event, stats_ns),
            kwargs={"interval_sec": sample_interval_s, "log_level": monitor_log_level},
            daemon=True,
        )

        # Build the NPU monitor process.
        npu_proc = ctx.Process(
            target=ResourceMonitor._npu_monitor_process,
            args=(stop_event, stats_ns),
            kwargs={"interval_sec": sample_interval_s, "log_level": monitor_log_level},
            daemon=True,
        )

        # Start in an order that begins capturing GPU data ASAP; order is not critical.
        gpu_proc.start()
        npu_proc.start()
        cpu_proc.start()
        mem_proc.start()

        return [mem_proc, cpu_proc, gpu_proc, npu_proc], stats_ns, stop_event, ctx

    @staticmethod
    def _build_normalized_stats(
        stats_ns: Any,
        worker_start_time: float,
        normalize_resource_data: bool = False
    ) -> Dict[str, Any]:
        """
        Build normalized stats (timestamps shifted so t=0 at worker start).

        Parameters
        ----------
        stats_ns : Any
            Shared namespace containing raw time-series data.
        worker_start_time : float
            The perf_counter timestamp when the worker started.
        normalize_resource_data : bool
            If True, subtract the prestart value from each data point.
        """
        used_memory_gb = ResourceMonitor._adjust_timestamps(list(stats_ns.used_memory_gb),
                                                            worker_start_time)
        cpu_utilization_pct = ResourceMonitor._adjust_timestamps(list(stats_ns.cpu_utilization_pct),
                                                          worker_start_time)
        gpu_utilization_pct = ResourceMonitor._adjust_timestamps(list(stats_ns.gpu_utilization_pct),
                                                          worker_start_time)
        gpu_memory_gb = ResourceMonitor._adjust_timestamps(list(stats_ns.gpu_memory_gb),
                                                        worker_start_time)
        npu_memory_gb = ResourceMonitor._adjust_timestamps(list(stats_ns.npu_memory_gb),
                                                        worker_start_time)

        # Trim each array to remove samples earlier than the prestart sample
        used_memory_gb = ResourceMonitor._trim_to_prestart(used_memory_gb)
        cpu_utilization_pct = ResourceMonitor._trim_to_prestart(cpu_utilization_pct)
        gpu_utilization_pct = ResourceMonitor._trim_to_prestart(gpu_utilization_pct)
        gpu_memory_gb = ResourceMonitor._trim_to_prestart(gpu_memory_gb)
        npu_memory_gb = ResourceMonitor._trim_to_prestart(npu_memory_gb)

        # Optionally normalize data by subtracting prestart values
        if normalize_resource_data:
            used_memory_gb = ResourceMonitor._normalize_to_prestart(used_memory_gb)
            cpu_utilization_pct = ResourceMonitor._normalize_to_prestart(cpu_utilization_pct)
            gpu_utilization_pct = ResourceMonitor._normalize_to_prestart(gpu_utilization_pct)
            gpu_memory_gb = ResourceMonitor._normalize_to_prestart(gpu_memory_gb)
            npu_memory_gb = ResourceMonitor._normalize_to_prestart(npu_memory_gb)

        return_stats: Dict[str, Any] = {
            "benchmark_duration_s": stats_ns.benchmark_duration_s,
            "sample_interval_s": stats_ns.sample_interval_s,
            "prestart_used_memory_gb": ResourceMonitor._prestart_stat(used_memory_gb),
            "avg_used_memory_gb": ResourceMonitor._average_stats(used_memory_gb),
            "max_used_memory_gb": ResourceMonitor._max_stats(used_memory_gb),
            "min_used_memory_gb": ResourceMonitor._min_stats(used_memory_gb),
            "prestart_cpu_utilization_pct": ResourceMonitor._prestart_stat(cpu_utilization_pct),
            "avg_cpu_utilization_pct": ResourceMonitor._average_stats(cpu_utilization_pct),
            "max_cpu_utilization_pct": ResourceMonitor._max_stats(cpu_utilization_pct),
            "min_cpu_utilization_pct": ResourceMonitor._min_stats(cpu_utilization_pct),
            "prestart_gpu_utilization_pct": ResourceMonitor._prestart_stat(gpu_utilization_pct),
            "avg_gpu_utilization_pct": ResourceMonitor._average_stats(gpu_utilization_pct),
            "max_gpu_utilization_pct": ResourceMonitor._max_stats(gpu_utilization_pct),
            "min_gpu_utilization_pct": ResourceMonitor._min_stats(gpu_utilization_pct),
            "prestart_gpu_memory_gb": ResourceMonitor._prestart_stat(gpu_memory_gb),
            "avg_gpu_memory_gb": ResourceMonitor._average_stats(gpu_memory_gb),
            "max_gpu_memory_gb": ResourceMonitor._max_stats(gpu_memory_gb),
            "min_gpu_memory_gb": ResourceMonitor._min_stats(gpu_memory_gb),
            "prestart_npu_memory_gb": ResourceMonitor._prestart_stat(npu_memory_gb),
            "avg_npu_memory_gb": ResourceMonitor._average_stats(npu_memory_gb),
            "max_npu_memory_gb": ResourceMonitor._max_stats(npu_memory_gb),
            "min_npu_memory_gb": ResourceMonitor._min_stats(npu_memory_gb),
            "used_memory_gb": used_memory_gb,
            "cpu_utilization_pct": cpu_utilization_pct,
            "gpu_utilization_pct": gpu_utilization_pct,
            "gpu_memory_gb": gpu_memory_gb,
            "npu_memory_gb": npu_memory_gb
        }
        return return_stats

    # ---------------------------
    # Public API
    # ---------------------------

    @staticmethod
    def monitor_process(
        worker_fn: Callable[..., None],
        worker_args: Tuple[Any, ...],
        sample_interval_s: float = 0.2,
        normalize_resource_data: bool = False
    ) -> Tuple[Any, Dict[str, Any], Dict[str, str]]:
        """
        Run a worker function in a separate process while collecting resource stats.

        Parameters
        ----------
        worker_fn : Callable[..., None]
            The worker function to run.
        worker_args : Tuple[Any, ...]
            Arguments to pass to the worker function.
        sample_interval_s : float
            Interval between resource samples in seconds.
        normalize_resource_data : bool
            If True, subtract the prestart value from each data point.

        Returns
        -------
        (worker_result, return_stats_dict, system_info_dict)
        """
        # Collect system information from the systeminfo command before we get started.
        system_info = ResourceMonitor._get_system_info()

        # Start the resource monitor
        monitor_log_level = logging.getLogger().getEffectiveLevel()
        proc_list, stats_ns, stop_event, ctx = \
            ResourceMonitor._start_resource_monitors(sample_interval_s, monitor_log_level)

        # Create a queue for the worker to publish its result back to the parent.
        result_queue: "multiprocessing.queues.Queue" = ctx.Queue()

        # Prepare the worker process.
        worker_proc = ctx.Process(
            target=worker_fn,
            args=(*worker_args, result_queue),
            daemon=False,
        )

        # Wait until all stats lists have at least one sample before starting the worker.
        while (len(stats_ns.gpu_utilization_pct) < 1 or
               len(stats_ns.gpu_memory_gb) < 1 or
               len(stats_ns.npu_memory_gb) < 1 or
               len(stats_ns.cpu_utilization_pct) < 1 or
               len(stats_ns.used_memory_gb) < 1):
            time.sleep(0.01)

        # Record the worker's start time (monotonic) for later timestamp normalization.
        worker_start_time = time.perf_counter()

        # Start the worker process.
        worker_proc.start()

        # Block until the worker exits; then compute elapsed wall time.
        worker_proc.join()

        stats_ns.benchmark_duration_s = round((time.perf_counter() - worker_start_time), 3)
        stats_ns.sample_interval_s = sample_interval_s

        # Sleep for a sample interval to catch trailing data.
        time.sleep(sample_interval_s)

        # Signal monitors to stop and join them with a short timeout to avoid hangs.
        stop_event.set()
        for proc in proc_list:
            proc.join(timeout=2.0)

        # Propagate worker failures with a helpful message if the *Python* worker failed.
        if worker_proc.exitcode != 0:
            raise RuntimeError(f"Worker exited with code {worker_proc.exitcode}")

        # Retrieve the worker's result from the queue.
        worker_result = result_queue.get()

        # Normalize the timestamps for the data against the worker start time.
        return_stats = ResourceMonitor._build_normalized_stats(stats_ns,
                                                               worker_start_time,
                                                               normalize_resource_data)

        return worker_result, return_stats, system_info


# ---------------------------------------------------------------------
# Stand-alone CLI: run any shell command under monitoring
# ---------------------------------------------------------------------

def run_shell_command_worker(
    cmd_str: str,
    echo_stderr: bool,
    result_queue: "multiprocessing.queues.Queue"
) -> None:
    """
    Worker function that executes a shell command (string) and returns a dict result.

    The worker itself always exits normally so that monitor_process can complete.
    The actual command's exit code is returned in the result payload.

    Parameters
    ----------
    cmd_str : str
        The command string to execute (will be split on whitespace).
    echo_stderr : bool
        If True, echo stderr to console. If False, capture stderr in the result.
    result_queue : multiprocessing.queues.Queue
        Queue to put the result dictionary into.
    """
    cmd = shlex.split(cmd_str, posix=not IS_WINDOWS)

    # Special case: if we're running waiter.py, always echo stderr in real time.
    # This keeps the console responsive/useful for interactive monitoring runs.
    if cmd and os.path.basename(cmd[0]).lower() == "python" and any(
        os.path.basename(arg).lower() == "waiter.py" for arg in cmd[1:]
    ):
        echo_stderr = True

    stdout, stderr, returncode = execute_command(cmd, echo_stderr=echo_stderr)
    result = {
        "exit_code": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
    result_queue.put(result)


def _parse_cli_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor CPU/RAM/GPU while running a shell command."
    )
    # Sampling interval for resource monitoring; validated by sample_interval().
    parser.add_argument(
        "-s", "--sample-interval",
        type=sample_interval,
        default=0.2,
        help="Number of seconds between resource monitoring samples, "
             "range 0.2 to 1.0 inclusive (default: 0.2)"
    )
    parser.add_argument(
        "--cmd",
        type=str,
        help="Shell command string to execute (alternative to using '--' remainder)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write JSON results."
    )
    parser.add_argument(
        "-e", "--echo-stderr",
        action="store_true",
        default=False,
        help="Echo stderr to console instead of capturing (default: capture stderr)."
    )
    parser.add_argument(
        "-n", "--normalize-resource-data",
        action="store_true",
        default=False,
        help="Normalize resource data by subtracting prestart values."
    )
    parser.add_argument(
        "-p", "--power-mode",
        type=str,
        default=None,
        choices=["performance", "balanced", "power-saver"],
        help="Set Windows power mode before running the command (Windows only). "
             "Choices: performance, balanced, power-saver."
    )
    # Accept remainder after '--' as the command to run.
    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Use '-- <command and args>' form to pass a command without quoting."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose debug logging (including monitor child processes)."
    )
    return parser.parse_args(argv)


def _extract_command_from_args(ns: argparse.Namespace) -> str:
    """
    Determine the command string either from --cmd or from the remainder after '--'.
    """
    cmd: str | None = ns.cmd
    if ns.remainder:
        # If user used the `--` style, argparse includes the literal '--' at index 0.
        # Example: python resource_monitor.py -- ls -la
        parts = ns.remainder
        if parts and parts[0] == "--":
            parts = parts[1:]
        if parts:
            cmd_from_remainder = " ".join(parts)
            # Prefer explicit --cmd if both were provided; otherwise use remainder.
            if cmd is None:
                cmd = cmd_from_remainder
    if not cmd:
        raise SystemExit(
            "ERROR: No command provided. Use --cmd \"your command\" "
            "or the '-- <command and args>' style."
        )
    return cmd


def _print_human_summary(cmd: str, worker_result: Dict[str, Any], stats: Dict[str, Any]) -> None:
    print("\n=== Resource Monitor Summary ===")
    print(f"Command           : {cmd}")
    print(f"Exit code         : {worker_result.get('exit_code')}")
    print(f"Duration (s)      : {stats.get('benchmark_duration_s')}")
    print(f"Sample interval(s): {stats.get('sample_interval_s')}")
    print("\nAverage metrics (post-start):")
    print(f"  CPU Util (%)    : {stats.get('avg_cpu_utilization_pct')}")
    print(f"  GPU Util (%)    : {stats.get('avg_gpu_utilization_pct')}")
    print(f"  GPU Mem (GiB)   : {stats.get('avg_gpu_memory_gb')}")
    print(f"  NPU Mem (GiB)   : {stats.get('avg_npu_memory_gb')}")
    print(f"  Used RAM (GiB)  : {stats.get('avg_used_memory_gb')}")
    print("\nPeaks (post-start):")
    print(f"  CPU Util (%) : {stats.get('max_cpu_utilization_pct')}")
    print(f"  GPU Util (%) : {stats.get('max_gpu_utilization_pct')}")
    print(f"  GPU Mem (GiB): {stats.get('max_gpu_memory_gb')}")
    print(f"  NPU Mem (GiB): {stats.get('max_npu_memory_gb')}")
    print(f"  Used RAM (GiB): {stats.get('max_used_memory_gb')}")
    print("\nPre-start metrics:")
    print(f"  CPU Util (%) : {stats.get('prestart_cpu_utilization_pct')}")
    print(f"  GPU Util (%) : {stats.get('prestart_gpu_utilization_pct')}")
    print(f"  GPU Mem (GiB): {stats.get('prestart_gpu_memory_gb')}")
    print(f"  NPU Mem (GiB): {stats.get('prestart_npu_memory_gb')}")
    print(f"  Used RAM (GiB): {stats.get('prestart_used_memory_gb')}\n")


def main(argv: List[str]) -> int:
    """
    Perform resource monitoring on an arbitrary command.
    """
    check_python_version()

    ns = _parse_cli_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if ns.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    cmd = _extract_command_from_args(ns)

    # Save the initial power mode so we can restore it when done
    initial_power_mode = get_power_mode()
    power_mode_changed = False

    # Set power mode if requested
    if ns.power_mode:
        mode_map = {
            "performance": "Performance",
            "balanced": "Balanced",
            "power-saver": "Power Saver",
        }
        target_mode = mode_map.get(ns.power_mode)
        if target_mode:
            print(f"Setting power mode to: {target_mode}")
            if set_power_mode(target_mode):
                power_mode_changed = True
            else:
                print("WARNING: Failed to set power mode. Continuing anyway.")

    try:
        # Run the shell command under monitoring
        worker_result, stats, system_info = ResourceMonitor.monitor_process(
            worker_fn=run_shell_command_worker,
            worker_args=(cmd, ns.echo_stderr),
            sample_interval_s=ns.sample_interval,
            normalize_resource_data=ns.normalize_resource_data
        )

        # Print a concise summary
        _print_human_summary(cmd, worker_result, stats)

        # Optionally write JSON to the specified directory
        if ns.output_dir:
            if not create_output_directory_tree(ns.output_dir):
                logger.error('Unable to create output directory %s', ns.output_dir)
                sys.exit(1)

            # Write JSON containing the worker result, stats, and system info
            filename_base = "resource_monitor"
            json_path = os.path.join(ns.output_dir, filename_base + ".json")
            payload = {
                "command": cmd,
                "result": worker_result,
                "stats": stats,
                "system_info": system_info,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote JSON results to: {json_path}")

        # Also surface command failure to the caller via this script's exit code,
        # while still letting the monitoring finish cleanly.
        return int(worker_result.get("exit_code", 0))

    finally:
        # Restore the initial power mode if we changed it
        if power_mode_changed and initial_power_mode:
            print(f"Restoring power mode to: {initial_power_mode}")
            if not set_power_mode(initial_power_mode):
                print("WARNING: Failed to restore power mode.")


def cli_main() -> int:
    """Console entry point for setuptools scripts."""
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(cli_main())
