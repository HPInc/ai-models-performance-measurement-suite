"""
GPU performance sampling utilities using Windows PDH (Performance Data Helper).

What this module does
---------------------
- Exposes a single class, `GPUSampler`, for sampling GPU utilization and
  committed adapter memory using the Windows Performance Counter infrastructure
  (PDH) via `pywin32`'s `win32pdh` module.
- Discovers per-adapter GPU engine counters from the "GPU Engine" object and
  adapter memory counters from the "GPU Adapter Memory" object, then aggregates
  them into overall and per-adapter summaries.

Why PDH and not vendor APIs?
----------------------------
- PDH is vendor-agnostic and standardized on Windows across Intel, AMD, and NVIDIA
  drivers when the relevant performance objects are exposed by the OS/driver stack.
- PDH counter names and objects may vary slightly by OS version/driver, which is
  why this class is defensive about discovery and sampling failures.

Key behaviors and conventions
-----------------------------
- Adapter identity is inferred from instance names that contain a token "luid_*".
  The first token that starts with "luid" is used as the adapter key for grouping.
- Per-adapter utilization is calculated by summing all engine utilizations on that
  adapter and clamping to 100.0 to avoid inflated totals when multiple engines are
  busy simultaneously.
- Per-adapter committed memory is calculated by summing all "Total Committed"
  counters for that adapter (in bytes) and reporting GiB. The overall committed
  memory is the sum across all adapters.
- PDH rate counters require at least two samples for valid formatted values; the
  class performs a short priming sequence during init and before each read.

Resilience
----------
- PDH objects may not be available immediately after boot or driver changes; this
  class handles those cases by logging a concise message, marking itself as not
  ready, and retrying initialization on subsequent calls to `sample()`.

Usage
-----
    gpu_sampler = GPUSampler()
    gpu_util, gpu_mem, adapters = gpu_sampler.sample()

Returned values
---------------
- gpu_util (float): Peak utilization percentage across all adapters (0..100), rounded to 2 decimals.
- gpu_mem  (float): Total committed memory across all adapters (GiB), rounded to 3 decimals.
- adapters (List[Dict[str, Any]]): Per-adapter dictionaries with keys:
      "adapter"             -> adapter LUID-based key
      "utilization_percent" -> per-adapter sum of engine utilization (capped at 100.0)
      "memory_gb"           -> per-adapter sum of committed memory (GiB)

Platform requirements
---------------------
- Windows only, with GPU performance counters available and accessible.
- Requires `pywin32` and specifically `win32pdh` to be installed and importable.
"""
# Pylint assumes that multiline comments should only be used for file
# or function doc strings. I profoundly disagree.
# pylint: disable=pointless-string-statement

"""
Set a local logger that can be set from a higher level.
"""
import logging  # pylint: disable=wrong-import-position
logger = logging.getLogger(__name__)

"""
Time is used to introduce a small delay between consecutive PDH collections so
that rate-based counters (like utilization) can compute a meaningful delta.
"""
import time  # pylint: disable=wrong-import-position

"""
Import the PDH bindings. If unavailable, the module cannot function, so raise a
clear, early RuntimeError. This is preferable to deferring the failure to the
first method call, which would be harder to diagnose.
"""
try:
    import win32pdh
except ImportError:
    raise RuntimeError("Please make sure win32pdh is installed.") from None

"""
The module intentionally catches broad exceptions around PDH calls. In practice
many PDH failures are transient or environment-specific; the goal here is to
avoid crashing callers and to degrade gracefully by marking the sampler "not
ready" while logging a short diagnostic. Lint is disabled accordingly.
"""
# pylint: disable=broad-exception-caught

"""
Ideally all logging statements should use lazy formatting instead
of fstring interpolation, because lazy formatting is slightly more
efficient if the statement might not be printed. But since thiscode
is inherited, Your Humble Narrator did not want to change the logic
of the print statements that I converted to logging statements. It
also does not make any difference, because the only DEBUG conditional
logger statement in the code that might go slightly faster is not time
critical.
"""
# pylint: disable=logging-fstring-interpolation

class GPUSampler:  # pylint: disable=too-few-public-methods
    """
    Sample GPU utilization and adapter memory using Windows PDH counters.

    Construction behavior
    ---------------------
    - Initializes internal state and immediately attempts to open a PDH query,
      discover GPU counters, and "prime" them (two initial collections).
    - If initialization fails at any point, the instance logs a brief message
      and sets `ready=False`. The next call to `sample()` will attempt to
      initialize again so callers don't have to manage readiness externally.

    Attributes
    ----------
    q : Any
        PDH query handle returned by `win32pdh.OpenQuery()`. Holds all counters.
    adapter_counters : Dict[str, List[Any]]
        Mapping of adapter LUID -> list of engine utilization counters.
    mem_counters : Dict[str, List[Any]]
        Mapping of adapter LUID -> list of "Total Committed" memory counters.
    ready : bool
        True when counters are discovered and primed successfully.

    Notes on identity and grouping
    ------------------------------
    - Engine instances expose names that typically include a "luid_*" token.
      The first such token is used to group engines and memory counters under
      the same adapter key.
    - If a memory instance does not include a LUID token, the full instance
      name is used as the grouping key to preserve association and visibility.
    """

    def __init__(self):
        """
        Initialize the sampler and attempt to configure PDH counters.

        What happens here
        -----------------
        - Create default attributes and attempt `_init_pdh()`.
        - `_init_pdh()` handles query creation, counter discovery, and priming.
        - If anything fails, `ready` remains False; sampling will auto-retry.
        """
        self.q = None
        self.adapter_counters = {}
        self.mem_counters = {}
        self.ready = False
        self._init_pdh()

    def _init_pdh(self):   # pylint: disable=too-many-statements
        """
        Initialize PDH query and discover GPU-related counters.

        Steps
        -----
        1) Open a PDH query with `win32pdh.OpenQuery()`.
        2) Enumerate "GPU Engine" instances and bind "Utilization Percentage"
           counters for each relevant instance, grouping by adapter LUID.
        3) Enumerate "GPU Adapter Memory" instances. If "Total Committed"
           appears in the counters list, bind that counter for each instance,
           again grouped by LUID when possible.
        4) Perform a quick priming sequence (collect, sleep ~120ms, collect) so
           that rate counters produce meaningful formatted values on first read.
        5) Set `ready=True` only if at least one engine counter was discovered.

        Failure strategy
        ----------------
        - On any failure during these steps, print a concise diagnostic, mark
          the sampler not ready, and return. The next `sample()` call will
          attempt to reinitialize.
        """
        try:
            """
            Open a new PDH query handle. If this fails, PDH is likely not
            available or accessible in the current environment.
            """
            self.q = win32pdh.OpenQuery()
        except Exception as e:
            logger.error(f"[GPU] OpenQuery failed: {e}")
            self.ready = False
            return

        try:
            """
            Enumerate counters and instances for the "GPU Engine" object at the
            highest detail level. We primarily need instance names to locate
            individual engines and extract adapter LUIDs.
            """
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

        """
        For each engine instance, determine whether it appears to belong to an
        adapter by searching for a 'luid_*' token. If found, use the first such
        token as the adapter key. Build a counter path for 'Utilization Percentage'
        and add it to our query. Group added counters by adapter key.
        """
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
            """
            Attempt to discover and bind adapter memory counters from the
            'GPU Adapter Memory' object. This object may not exist on all
            systems; bind only if 'Total Committed' is advertised.
            """
            mem_obj = "GPU Adapter Memory"
            mem_counters, mem_insts = win32pdh.EnumObjectItems(
                None,
                None,
                mem_obj,
                win32pdh.PERF_DETAIL_WIZARD
            )
            if "Total Committed" in mem_counters:
                for inst in mem_insts:
                    """
                    Prefer to group memory counters using the same LUID token used
                    for engines. If no LUID token exists, fall back to the full
                    instance name so callers can still see per-instance memory.
                    """
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

        """
        If no engine counters were discovered, we cannot report utilization.
        This often means the object isn't populated yet; mark not ready so
        a subsequent call to `sample()` can retry initialization.
        """
        if not self.adapter_counters:
            logger.error("[GPU] No engine counters found yet")
            self.ready = False
            return

        try:
            """
            Prime the counters by performing two collections with a short delay
            in between. Rate-based counters require two points to compute a delta.
            """
            win32pdh.CollectQueryData(self.q)
            time.sleep(0.12)
            win32pdh.CollectQueryData(self.q)
        except Exception as e:
            logger.warning(f"[GPU] CollectQueryData priming failed: {e}")

        """
        Mark the sampler as ready for use and print a brief summary of discovered
        adapters to aid diagnostics and to confirm successful initialization.
        """
        self.ready = True
        logger.debug(f"[GPU] PDH initialized, adapters: {list(self.adapter_counters.keys())}")

    def sample(self):  #pylint: disable=too-many-branches,too-many-locals
        """
        Collect the latest GPU utilization and memory statistics.

        Returns
        -------
        Tuple[float, float, List[Dict[str, Any]]]
            overall_util :
                The maximum utilization percentage across all adapters (0..100),
                rounded to 2 decimals. Using max highlights the busiest adapter.
            overall_mem_gb :
                The sum of "Total Committed" memory across all adapters in GiB,
                rounded to 3 decimals.
            adapter_stats :
                A list of per-adapter dicts:
                    {
                        "adapter": <luid>,
                        "util_percent": <float>,  # sum of engines, clamped to 100
                        "mem_gb": <float>         # sum of memory counters (GiB)
                    }

        Behavior and rationale
        ----------------------
        - If not ready, try reinitialization before sampling so callers do not
          need to manage PDH readiness themselves.
        - Perform two collections with a small delay to ensure valid formatted
          values from rate-based counters.
        - Sum all engine utilization counters per adapter, then clamp to 100.0
          to avoid misleading values over 100% when multiple engines are active.
        - Sum all memory counters per adapter in bytes and convert to GiB
          using 1024^3 for consistency with system memory reporting.

        Failure mode
        ------------
        - Any unexpected PDH error results in a short diagnostic message,
          resetting `ready=False` and returning `(0.0, 0.0, [])`. A subsequent
          call will attempt to reinitialize.
        """
        if not self.ready:
            """
            Attempt to recover automatically if initialization failed earlier
            or if a previous sampling attempt marked the sampler as not ready.
            """
            self._init_pdh()
            if not self.ready:
                return 0.0, 0.0, []

        try:
            """
            Perform two collections separated by a brief delay so that rate
            counters have the two points required to compute a delta.
            """
            win32pdh.CollectQueryData(self.q)
            time.sleep(0.12)
            win32pdh.CollectQueryData(self.q)

            """
            Prepare aggregates:
            - adapter_stats accumulates per-adapter results for return.
            - overall_util tracks the peak adapter utilization across all GPUs.
            - overall_mem_bytes accumulates total committed memory in bytes.
            """
            adapter_stats = []
            overall_util = 0.0
            overall_mem_bytes = 0

            """
            Iterate adapters and compute per-adapter utilization and memory.
            Engine reads use PDH_FMT_DOUBLE (floats). Memory reads use
            PDH_FMT_LARGE (integers), which we later convert to GiB.
            """
            for luid, ctr_list in self.adapter_counters.items():
                engine_vals = []
                for c in ctr_list:
                    try:
                        _, val = win32pdh.GetFormattedCounterValue(c, win32pdh.PDH_FMT_DOUBLE)
                        if isinstance(val, (int, float)) and val >= 0:
                            engine_vals.append(float(val))
                    except Exception:
                        """
                        Ignore a single engine read failure; other engines on the
                        adapter can still provide useful data.
                        """

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
                        """
                        Ignore a single memory read failure; continue aggregating
                        other memory counters for this adapter.
                        """

                if mem_bytes:
                    """
                    Convert adapter memory from bytes to GiB with a binary base,
                    which is standard for system memory reporting.
                    """
                    mem_gb = mem_bytes / (1024 ** 3)

                adapter_stats.append({"adapter": luid,
                                      "utilization_percent": round(adapter_util, 2),
                                      "memory_gb": round(mem_gb, 3)})

                """
                Track the busiest adapter so callers can quickly detect any GPU
                saturation even in multi-GPU environments.
                """
                overall_util = max(overall_util, adapter_util)

                """
                Accumulate total committed memory in bytes across all adapters
                so we can return a single overall value along with the details.
                """
                overall_mem_bytes += mem_bytes

            """
            Convert the overall committed memory sum to GiB and return the
            complete snapshot: (peak utilization, total committed GiB, details).
            """
            overall_mem_gb = overall_mem_bytes / (1024 ** 3) if overall_mem_bytes > 0 else 0.0
            return round(overall_util, 2), round(overall_mem_gb, 3), adapter_stats

        except Exception as e:
            """
            Any PDH or formatting error here is treated as transient; log it and
            mark the sampler as not ready so the next call can retry init.
            """
            logger.warning(f"[GPU] Sample failed: {e}")
            self.ready = False
            return 0.0, 0.0, []

"""
Usage:
gpu_sampler = GPUSampler()

gpu_util, gpu_mem, adapters = gpu_sampler.sample()
"""
