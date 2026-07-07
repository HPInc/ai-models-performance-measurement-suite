import os
import time
import ctypes
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def reset_benchmark_environment(
    memory_pressure_gb: float = 8.0,
    cpu_stress_seconds: int = 5,
) -> None:
    """
    Reset the local system state to approximate a cold-start environment for benchmarking.

    This function is designed for Windows and attempts to reduce the effects of:
        - OS file/page cache (via memory pressure)
        - CPU cache and branch predictor warm-up
        - allocator and working set reuse

    NOTE:
        Windows does not provide a direct API to drop file system cache like Linux.
        This function uses indirect methods and provides best-effort behavior.

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

        This is the primary mechanism for approximating cache clearing on Windows.
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

        This can indirectly encourage system-wide memory reclamation.
        """
        try:
            kernel32 = ctypes.windll.kernel32
            h_process = kernel32.GetCurrentProcess()
            kernel32.SetProcessWorkingSetSize(h_process, -1, -1)
        except Exception:
            # Non-critical; ignore failures
            pass

    # ---- Execution sequence ----

    logger.info("Resetting benchmark environment (Windows best-effort)...")
    logger.info("Applying memory pressure...")
    _apply_memory_pressure(memory_pressure_gb)

    logger.info("Trimming working set...")
    _trim_working_set()

    logger.info("Polluting CPU cache...")
    _pollute_cpu_cache(cpu_stress_seconds)

    logger.info("Cooldown...")
    time.sleep(2)

    logger.info("Environment reset complete.")
