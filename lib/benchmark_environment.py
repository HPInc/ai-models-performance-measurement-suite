import gc
import os
import time
import ctypes
import logging
import platform

logger = logging.getLogger(__name__)
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

def reset_benchmark_environment(
    memory_pressure_gb: float = 8.0,
    cpu_stress_seconds: int = 5,
) -> None:
    """
    Reset the local system state to approximate a cold-start environment for benchmarking.

    This function attempts to reduce the effects of:
        - OS file/page cache (via memory pressure)
        - CPU cache and branch predictor warm-up
        - allocator and working set reuse

    NOTE:
        This function uses best-effort methods only. Some operations are platform-
        specific and may require elevated privileges to have full effect.

    Args:
        memory_pressure_gb:
            Amount of RAM (in GB) to allocate temporarily to force cache eviction.

        cpu_stress_seconds:
            Duration of CPU-intensive work used to pollute CPU caches.

    Raises:
        RuntimeError:
            If any critical operation fails.
    """

    def _apply_memory_pressure(gb: float) -> None:
        """
        Allocate and release memory to evict OS file cache.
        """
        try:
            size = int(gb * (1024 ** 3))
            chunk_size = 256 * 1024 * 1024  # 256 MB chunks
            buffers = []

            allocated = 0
            while allocated < size:
                buffers.append(bytearray(chunk_size))
                allocated += chunk_size

            # Touch memory to force actual allocation
            for buf in buffers:
                buf[0] = 1

            # Release memory
            del buffers

        except MemoryError:
            # Expected if we hit system limits — still useful
            pass
        except Exception as exc:
            raise RuntimeError(f"Memory pressure step failed: {exc}") from exc

    def _pollute_cpu_cache(duration: int) -> None:
        """
        Perform CPU-intensive work to disrupt cache locality and branch predictors.
        """
        end_time = time.time() + duration
        x = 0
        while time.time() < end_time:
            x = (x * 1664525 + 1013904223) & 0xFFFFFFFF

    def _trim_working_set() -> None:
        """
        Ask Windows to trim the working set of the current process.
        """
        if not IS_WINDOWS:
            return

        try:
            kernel32 = ctypes.windll.kernel32
            h_process = kernel32.GetCurrentProcess()
            kernel32.SetProcessWorkingSetSize(h_process, -1, -1)
        except Exception:
            # Non-critical; ignore failures
            pass

    def _trim_linux_allocator() -> None:
        """
        Ask glibc to release free heap pages back to the OS when available.
        """
        if not IS_LINUX:
            return

        try:
            libc = ctypes.CDLL(None)
            malloc_trim = getattr(libc, "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim.argtypes = [ctypes.c_size_t]
                malloc_trim.restype = ctypes.c_int
                malloc_trim(0)
        except Exception:
            pass

    def _drop_linux_caches() -> None:
        """
        Request Linux page-cache reclamation when the kernel interface is writable.
        """
        if not IS_LINUX:
            return

        try:
            if hasattr(os, "sync"):
                os.sync()
        except Exception:
            pass

        drop_caches_path = "/proc/sys/vm/drop_caches"
        try:
            if os.path.exists(drop_caches_path) and os.access(drop_caches_path, os.W_OK):
                with open(drop_caches_path, "w", encoding="utf-8") as handle:
                    handle.write("3\n")
            else:
                logger.info("Linux cache drop requires write access to %s", drop_caches_path)
        except Exception as exc:
            logger.info("Linux cache drop not available: %s", exc)

    # ---- Execution sequence ----

    logger.info("Resetting benchmark environment (%s best-effort)...", platform.system())
    logger.info("Applying memory pressure...")
    _apply_memory_pressure(memory_pressure_gb)

    gc.collect()

    if IS_WINDOWS:
        logger.info("Trimming working set...")
        _trim_working_set()
    elif IS_LINUX:
        logger.info("Trimming allocator state...")
        _trim_linux_allocator()
        logger.info("Requesting Linux cache drop...")
        _drop_linux_caches()

    logger.info("Polluting CPU cache...")
    _pollute_cpu_cache(cpu_stress_seconds)

    logger.info("Cooldown...")
    time.sleep(2)

    logger.info("Environment reset complete.")
