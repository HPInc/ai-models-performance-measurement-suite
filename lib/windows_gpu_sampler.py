"""Windows-specific GPU performance sampling utilities."""

import logging
import time

logger = logging.getLogger(__name__)

try:
    import win32pdh
except ImportError:
    raise RuntimeError("Please make sure win32pdh is installed.") from None

# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation


class GPUSampler:  # pylint: disable=too-few-public-methods
    """Sample GPU utilization and memory on Windows using PDH."""

    def __init__(self):
        self.q = None
        self.adapter_counters = {}
        self.mem_counters = {}
        self.linux_devices = []
        self.use_nvidia_smi = False
        self.ready = False
        self._init_pdh()

    def _init_pdh(self):  # pylint: disable=too-many-statements
        try:
            self.q = win32pdh.OpenQuery()
        except Exception as e:
            logger.error(f"[GPU] OpenQuery failed: {e}")
            self.ready = False
            return

        try:
            obj = "GPU Engine"
            _, instances = win32pdh.EnumObjectItems(
                None,
                None,
                obj,
                win32pdh.PERF_DETAIL_WIZARD
            )
        except Exception as e:
            logger.error(f"[GPU] EnumObjectItems for GPU Engine not ready: {e}")
            self.ready = False
            return

        for inst in instances:
            if "luid_" not in inst:
                continue
            luid_tokens = [t for t in inst.split("_") if t.startswith("luid")]
            if not luid_tokens:
                continue
            luid = luid_tokens[0]
            path = win32pdh.MakeCounterPath((None, obj, inst, None, 0, "Utilization Percentage"))
            try:
                ctr = win32pdh.AddCounter(self.q, path)
            except Exception as e:
                logger.warning(f"[GPU] AddCounter failed for {path}: {e}")
                continue
            self.adapter_counters.setdefault(luid, []).append(ctr)

        try:
            mem_obj = "GPU Adapter Memory"
            mem_counters, mem_insts = win32pdh.EnumObjectItems(
                None,
                None,
                mem_obj,
                win32pdh.PERF_DETAIL_WIZARD
            )
            if "Total Committed" in mem_counters:
                for inst in mem_insts:
                    luid_tokens = [t for t in inst.split("_") if t.startswith("luid")]
                    luid = luid_tokens[0] if luid_tokens else inst
                    p = win32pdh.MakeCounterPath((None, mem_obj, inst, None, 0, "Total Committed"))
                    try:
                        mctr = win32pdh.AddCounter(self.q, p)
                        self.mem_counters.setdefault(luid, []).append(mctr)
                    except Exception as e:
                        logger.warning(f"[GPU] AddCounter mem failed for {p}: {e}")
        except Exception as e:
            logger.warning(f"[GPU] EnumObjectItems for GPU Adapter Memory not available: {e}")

        if not self.adapter_counters:
            logger.error("[GPU] No engine counters found yet")
            self.ready = False
            return

        try:
            win32pdh.CollectQueryData(self.q)
            time.sleep(0.12)
            win32pdh.CollectQueryData(self.q)
        except Exception as e:
            logger.warning(f"[GPU] CollectQueryData priming failed: {e}")

        self.ready = True
        logger.debug(f"[GPU] PDH initialized, adapters: {list(self.adapter_counters.keys())}")

    def sample(self):
        if not self.ready:
            self._init_pdh()
            if not self.ready:
                return 0.0, 0.0, []

        try:
            win32pdh.CollectQueryData(self.q)
            time.sleep(0.12)
            win32pdh.CollectQueryData(self.q)

            adapter_stats = []
            overall_util = 0.0
            overall_mem_bytes = 0

            for luid, ctr_list in self.adapter_counters.items():
                engine_vals = []
                for c in ctr_list:
                    try:
                        _, val = win32pdh.GetFormattedCounterValue(c, win32pdh.PDH_FMT_DOUBLE)
                        if isinstance(val, (int, float)) and val >= 0:
                            engine_vals.append(float(val))
                    except Exception:
                        pass

                if engine_vals:
                    adapter_util = min(sum(engine_vals), 100.0)
                else:
                    adapter_util = 0.0

                mem_gb = 0.0
                mem_handles = self.mem_counters.get(luid, [])
                mem_bytes = 0
                for mc in mem_handles:
                    try:
                        _, v = win32pdh.GetFormattedCounterValue(mc, win32pdh.PDH_FMT_LARGE)
                        if isinstance(v, int) and v > 0:
                            mem_bytes += int(v)
                    except Exception:
                        pass

                if mem_bytes:
                    mem_gb = mem_bytes / (1024 ** 3)

                adapter_stats.append({
                    "adapter": luid,
                    "utilization_percent": round(adapter_util, 2),
                    "memory_gb": round(mem_gb, 3),
                })

                overall_util = max(overall_util, adapter_util)
                overall_mem_bytes += mem_bytes

            overall_mem_gb = overall_mem_bytes / (1024 ** 3) if overall_mem_bytes > 0 else 0.0
            return round(overall_util, 2), round(overall_mem_gb, 3), adapter_stats

        except Exception as e:
            logger.warning(f"[GPU] Sample failed: {e}")
            self.ready = False
            return 0.0, 0.0, []
