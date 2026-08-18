"""Platform-specific NPU sampler loader."""

import platform

if platform.system() == "Windows":
    from windows_npu_sampler import NPUSampler
elif platform.system() == "Linux":
    from linux_npu_sampler import NPUSampler
else:
    class NPUSampler:  # pylint: disable=too-few-public-methods
        """Fallback sampler for unsupported platforms."""

        def __init__(self):
            self.has_npu = False

        def sample(self):
            return 0.0


__all__ = ["NPUSampler"]