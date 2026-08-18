"""Platform-specific GPU sampler loader."""

import platform

if platform.system() == "Windows":
    from windows_gpu_sampler import GPUSampler
elif platform.system() == "Linux":
    from linux_gpu_sampler import GPUSampler
else:
    class GPUSampler:  # pylint: disable=too-few-public-methods
        """Fallback sampler for unsupported platforms."""

        def sample(self):
            return 0.0, 0.0, []


__all__ = ["GPUSampler"]
