"""
NPU performance sampling utilities using Windows PDH and D3DKMT.

What this module does
---------------------
- Exposes a single class, `NPUSampler`, for sampling NPU (Neural Processing
  Unit) committed adapter memory using the Windows Performance Counter
  infrastructure (PDH) via `pywin32`'s `win32pdh` module.
- Identifies NPU adapters by enumerating WDDM/MCDM adapters through the
  `D3DKMTEnumAdapters3` kernel API and checking the ``ComputeOnly`` flag in
  the adapter type.  Only compute-only adapters (i.e. not GPUs) are treated
  as NPUs.
- Once the NPU adapter LUID is known, binds PDH counters from the
  "GPU Adapter Memory" object that match that LUID.

Why D3DKMT for identification?
------------------------------
- PDH counter instance names contain adapter LUIDs but do not indicate whether
  an adapter is a GPU, NPU, or software renderer.
- The ``D3DKMTEnumAdapters3`` API with the ``IncludeComputeOnly`` filter is the
  only documented way to discover MCDM compute-only adapters and retrieve their
  LUIDs from user mode.
- ``D3DKMTEnumAdapters2`` does **not** return compute-only adapters.

Key behaviors and conventions
-----------------------------
- Adapter LUIDs are formatted as ``0xHIGH_0xLOW`` hex strings and matched
  against PDH instance names (which contain ``luid_0xHIGH_0xLOW`` tokens).
- NPU memory is reported as the "Total Committed" counter in GiB.
- PDH rate counters require at least two samples; the class performs a short
  priming sequence during initialization and before each read.

Resilience
----------
- If ``D3DKMTEnumAdapters3`` is unavailable (older Windows builds), the
  sampler marks itself as not ready and ``sample()`` returns zeros.
- If no compute-only adapters are found, the sampler silently returns zeros.
- PDH discovery or sampling failures are logged and the sampler degrades
  gracefully.

Usage
-----
    npu_sampler = NPUSampler()
    npu_mem = npu_sampler.sample()

Returned values
---------------
- npu_mem  (float): Total committed memory in GiB, rounded to 3 decimals.

Platform requirements
---------------------
- Windows 10/11 with MCDM-compatible NPU driver installed.
- Requires ``pywin32`` (``win32pdh``).
- Requires ``ctypes`` access to ``gdi32.dll`` (``D3DKMTEnumAdapters3``).
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
ctypes is used to call the D3DKMT kernel API for adapter enumeration.
time is used for the PDH priming delay.
"""
import ctypes               # pylint: disable=wrong-import-position
import ctypes.wintypes      # pylint: disable=wrong-import-position
import time                 # pylint: disable=wrong-import-position
from typing import List  # pylint: disable=wrong-import-position

"""
Import the PDH bindings. If unavailable, the module cannot function, so raise a
clear, early RuntimeError.
"""
try:
    import win32pdh
except ImportError:
    raise RuntimeError("Please make sure win32pdh is installed.") from None

"""
The module intentionally catches broad exceptions around PDH and D3DKMT calls.
Many failures are transient or environment-specific; the goal is to degrade
gracefully by marking the sampler "not ready" while logging a short diagnostic.
"""
# pylint: disable=broad-exception-caught

"""
Inherited code uses f-string logging; keeping for consistency with gpu_sampler.
"""
# pylint: disable=logging-fstring-interpolation


# -----------------------------------------------------------------------
# D3DKMT ctypes structures for adapter enumeration
# -----------------------------------------------------------------------

class _LUID(ctypes.Structure):
    """Windows LUID (Locally Unique Identifier) structure."""
    _fields_ = [
        ("LowPart", ctypes.wintypes.DWORD),
        ("HighPart", ctypes.wintypes.LONG),
    ]


class _D3DKMT_ADAPTERINFO(ctypes.Structure):
    """Adapter information returned by D3DKMTEnumAdapters2/3."""
    _fields_ = [
        ("hAdapter", ctypes.c_uint32),
        ("AdapterLuid", _LUID),
        ("NumOfSources", ctypes.c_uint32),
        ("bPresentMoveRegionsPreferred", ctypes.c_bool),
    ]


class _D3DKMT_ENUMADAPTERS3(ctypes.Structure):
    """Input/output structure for D3DKMTEnumAdapters3."""
    _fields_ = [
        ("Filter", ctypes.c_uint64),      # D3DKMT_ENUMADAPTERS_FILTER (ULONGLONG)
        ("NumAdapters", ctypes.c_uint32),
        ("pAdapters", ctypes.POINTER(_D3DKMT_ADAPTERINFO)),
    ]


class _D3DKMT_QUERYADAPTERINFO(ctypes.Structure):
    """Input structure for D3DKMTQueryAdapterInfo."""
    _fields_ = [
        ("hAdapter", ctypes.c_uint32),
        ("Type", ctypes.c_uint32),
        ("pPrivateDriverData", ctypes.c_void_p),
        ("PrivateDriverDataSize", ctypes.c_uint32),
    ]


class _D3DKMT_ADAPTERTYPE(ctypes.Structure):
    """Adapter type flags returned by D3DKMTQueryAdapterInfo."""
    _fields_ = [("Value", ctypes.c_uint32)]


class _D3DKMT_CLOSEADAPTER(ctypes.Structure):
    """Input structure for D3DKMTCloseAdapter."""
    _fields_ = [("hAdapter", ctypes.c_uint32)]


# D3DKMT constants
_KMTQAITYPE_ADAPTERTYPE = 15
_ADAPTER_FLAG_COMPUTE_ONLY = 0x800
_INCLUDE_COMPUTE_ONLY = 0x1  # D3DKMT_ENUMADAPTERS_FILTER bit 0


def _discover_npu_luids() -> List[str]:
    """
    Discover NPU adapter LUIDs via D3DKMTEnumAdapters3.

    Returns
    -------
    List[str]
        List of LUID strings in the format ``0xHIGH_0xLOW`` for adapters
        whose type flags include ``ComputeOnly``.  Empty if no NPU is found
        or if the API is unavailable.
    """
    try:
        gdi32 = ctypes.windll.gdi32
    except Exception as e:
        logger.debug(f"[NPU] Cannot load gdi32.dll: {e}")
        return []

    if not hasattr(gdi32, "D3DKMTEnumAdapters3"):
        logger.debug("[NPU] D3DKMTEnumAdapters3 not available on this Windows build")
        return []

    """
    First call: determine how many adapters are available. The Filter field
    includes IncludeComputeOnly so MCDM devices are returned.
    """
    try:
        enum_in = _D3DKMT_ENUMADAPTERS3()
        enum_in.Filter = _INCLUDE_COMPUTE_ONLY
        enum_in.NumAdapters = 0
        enum_in.pAdapters = None
        gdi32.D3DKMTEnumAdapters3(ctypes.byref(enum_in))
    except Exception as e:
        logger.debug(f"[NPU] D3DKMTEnumAdapters3 count call failed: {e}")
        return []

    if enum_in.NumAdapters == 0:
        return []

    """
    Second call: populate the adapter info array.
    """
    try:
        n = enum_in.NumAdapters
        buf = (_D3DKMT_ADAPTERINFO * n)()
        enum_in.pAdapters = ctypes.cast(buf, ctypes.POINTER(_D3DKMT_ADAPTERINFO))
        status = gdi32.D3DKMTEnumAdapters3(ctypes.byref(enum_in))
        if status != 0:
            logger.debug(f"[NPU] D3DKMTEnumAdapters3 enum call failed: {status:#x}")
            return []
    except Exception as e:
        logger.debug(f"[NPU] D3DKMTEnumAdapters3 enum failed: {e}")
        return []

    """
    For each returned adapter, query its type flags and check the ComputeOnly
    bit. Close the adapter handle after querying.
    """
    npu_luids: List[str] = []
    for i in range(enum_in.NumAdapters):
        adapter = buf[i]
        h = adapter.hAdapter
        try:
            adapter_type = _D3DKMT_ADAPTERTYPE()
            qai = _D3DKMT_QUERYADAPTERINFO()
            qai.hAdapter = h
            qai.Type = _KMTQAITYPE_ADAPTERTYPE
            qai.pPrivateDriverData = ctypes.addressof(adapter_type)
            qai.PrivateDriverDataSize = ctypes.sizeof(adapter_type)
            status = gdi32.D3DKMTQueryAdapterInfo(ctypes.byref(qai))
            if status == 0 and (adapter_type.Value & _ADAPTER_FLAG_COMPUTE_ONLY):
                luid_str = (f"0x{adapter.AdapterLuid.HighPart:08X}"
                            f"_0x{adapter.AdapterLuid.LowPart:08X}")
                npu_luids.append(luid_str)
        except Exception as e:
            logger.debug(f"[NPU] QueryAdapterInfo failed for adapter {i}: {e}")
        finally:
            try:
                close = _D3DKMT_CLOSEADAPTER(hAdapter=h)
                gdi32.D3DKMTCloseAdapter(ctypes.byref(close))
            except Exception:
                pass

    return npu_luids


class NPUSampler:  # pylint: disable=too-few-public-methods
    """
    Sample NPU committed memory using Windows PDH counters.

    Construction behavior
    ---------------------
    - Discovers NPU adapter LUIDs via D3DKMTEnumAdapters3.
    - If at least one ComputeOnly adapter is found, opens a PDH query and
      binds adapter memory counters filtered to that LUID.
    - If initialization fails, the instance sets ``ready=False``.  The next
      call to ``sample()`` will attempt to reinitialize.

    Attributes
    ----------
    npu_luids : List[str]
        LUID strings for discovered NPU adapters.
    q : Any
        PDH query handle.
    mem_counters : Dict[str, List[Any]]
        Mapping of LUID -> list of "Total Committed" memory counters.
    ready : bool
        True when counters are discovered and primed successfully.
    has_npu : bool
        True when at least one NPU adapter was found.
    """

    def __init__(self):
        """
        Initialize the sampler and attempt to configure PDH counters.
        """
        self.npu_luids: List[str] = []
        self.q = None
        self.mem_counters: dict = {}
        self.ready = False
        self.has_npu = False
        self._init_pdh()

    def _luid_matches_instance(self, instance: str, luid: str) -> bool:
        """
        Check whether a PDH instance name contains a matching LUID.

        PDH instance names use lowercase and underscores, e.g.:
            ``pid_1234_luid_0x00000000_0x481f6755_phys_0_eng_0``

        The LUID from D3DKMT is formatted as ``0x00000000_0x481F6755``
        (uppercase hex). Compare case-insensitively.
        """
        # The LUID in instance names appears as "luid_0xHIGH_0xLOW"
        luid_token = f"luid_{luid}".lower()
        return luid_token in instance.lower()

    def _init_pdh(self):   # pylint: disable=too-many-statements,too-many-branches
        """
        Initialize PDH query and discover NPU-related counters.

        Steps
        -----
        1) Discover NPU LUIDs via D3DKMTEnumAdapters3.
        2) Open a PDH query.
        3) Enumerate "GPU Adapter Memory" instances and bind "Total Committed"
           counters that match an NPU LUID.
        4) Perform a priming sequence if any counters were discovered.
        5) Set ``ready=True`` if at least memory counters were found.
        """
        self.npu_luids = _discover_npu_luids()
        if not self.npu_luids:
            logger.debug("[NPU] No compute-only (NPU) adapters found")
            self.has_npu = False
            self.ready = False
            return

        self.has_npu = True
        logger.debug(f"[NPU] Discovered NPU LUIDs: {self.npu_luids}")

        try:
            self.q = win32pdh.OpenQuery()
        except Exception as e:
            logger.error(f"[NPU] OpenQuery failed: {e}")
            self.ready = False
            return

        """
        Enumerate "GPU Adapter Memory" instances and bind "Total Committed"
        counters for instances whose LUID matches a discovered NPU adapter.
        """
        try:
            mem_obj = "GPU Adapter Memory"
            mem_counters_list, mem_insts = win32pdh.EnumObjectItems(
                None, None, mem_obj, win32pdh.PERF_DETAIL_WIZARD
            )
            if "Total Committed" in mem_counters_list:
                for inst in mem_insts:
                    for luid in self.npu_luids:
                        if not self._luid_matches_instance(inst, luid):
                            continue
                        p = win32pdh.MakeCounterPath(
                            (None, mem_obj, inst, None, 0, "Total Committed")
                        )
                        try:
                            mctr = win32pdh.AddCounter(self.q, p)
                            self.mem_counters.setdefault(luid, []).append(mctr)
                        except Exception as e:
                            logger.warning(f"[NPU] AddCounter mem failed: {e}")
        except Exception as e:
            logger.warning(f"[NPU] GPU Adapter Memory enumeration failed: {e}")

        """
        We consider the sampler ready if we found at least memory counters.
        """
        if not self.mem_counters:
            logger.debug("[NPU] No PDH counters found for NPU LUIDs")
            self.ready = False
            return

        """
        Prime the counters with two collections so rate-based counters
        produce valid values on the first real read.
        """
        try:
            win32pdh.CollectQueryData(self.q)
            time.sleep(0.12)
            win32pdh.CollectQueryData(self.q)
        except Exception as e:
            logger.warning(f"[NPU] CollectQueryData priming failed: {e}")

        self.ready = True
        mem_count = sum(len(v) for v in self.mem_counters.values())
        logger.debug(f"[NPU] PDH initialized: {mem_count} memory counter(s)")

    def sample(self) -> float:
        """
        Collect the latest NPU committed memory statistics.

        Returns
        -------
        float
            Total committed memory in GiB, rounded to 3 decimals.

        Behavior
        --------
        - If not ready and ``has_npu`` is True, try reinitialization.
        - If ``has_npu`` is False, return 0.0 immediately.
        - Perform two PDH collections with a short delay for rate counters.
        """
        if not self.has_npu:
            return 0.0

        if not self.ready:
            self._init_pdh()
            if not self.ready:
                return 0.0

        try:
            win32pdh.CollectQueryData(self.q)
            time.sleep(0.12)
            win32pdh.CollectQueryData(self.q)

            """
            Calculate committed memory from memory counters.
            """
            total_mem_bytes = 0
            for _luid, ctr_list in self.mem_counters.items():
                for mc in ctr_list:
                    try:
                        _, v = win32pdh.GetFormattedCounterValue(
                            mc, win32pdh.PDH_FMT_LARGE
                        )
                        if isinstance(v, int) and v > 0:
                            total_mem_bytes += int(v)
                    except Exception:
                        pass

            mem_gb = total_mem_bytes / (1024 ** 3) if total_mem_bytes > 0 else 0.0
            return round(mem_gb, 3)

        except Exception as e:
            logger.warning(f"[NPU] Sample failed: {e}")
            self.ready = False
            return 0.0
