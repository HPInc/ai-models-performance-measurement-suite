"""Platform-specific power configuration loader."""

import platform

if platform.system() == "Windows":
    from windows_power_config import get_power_profile, get_power_mode, set_power_mode
elif platform.system() == "Linux":
    from linux_power_config import get_power_profile, get_power_mode, set_power_mode
else:
    def get_power_profile() -> str:
        return ""

    def get_power_mode() -> str:
        return ""

    def set_power_mode(mode: str) -> bool:
        _ = mode
        return False


__all__ = ["get_power_profile", "get_power_mode", "set_power_mode"]