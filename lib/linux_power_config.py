#!/usr/bin/env python3
"""
Power configuration utilities for Linux.

This module provides functions to query and set power settings on Linux systems
using the `powerprofilesctl` utility from the power-profiles-daemon service:
- Both Power Profile and Power Mode map directly to the system power profiles
  (e.g., "performance", "balanced", "power-saver").
- The module utilizes the existing platform_support library to execute commands.

Public API
----------
- get_power_profile() -> str: Get the active power profile name
- get_power_mode() -> str: Get the active power mode
- set_power_mode(mode: str) -> bool: Set the power mode
- POWER_MODE_PERFORMANCE: Constant for "Performance" mode
- POWER_MODE_BALANCED: Constant for "Balanced" mode
- POWER_MODE_POWER_SAVER: Constant for "Power Saver" mode
- VALID_POWER_MODES: List of valid power mode names for CLI validation
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Import platform_support
try:
    from platform_support import execute_command
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing platform_support.py library file.") from exc


# Power mode constants
POWER_MODE_PERFORMANCE = "Performance"
POWER_MODE_BALANCED = "Balanced"
POWER_MODE_POWER_SAVER = "Power Saver"

# Valid power mode names for CLI validation (case-insensitive matching)
VALID_POWER_MODES = [
    POWER_MODE_PERFORMANCE,
    POWER_MODE_BALANCED,
    POWER_MODE_POWER_SAVER,
]

# Mapping from powerprofilesctl output to public constants
_LINUX_TO_MODE = {
    "performance": POWER_MODE_PERFORMANCE,
    "balanced": POWER_MODE_BALANCED,
    "power-saver": POWER_MODE_POWER_SAVER,
}

# Mapping from input power mode name to powerprofilesctl profile names
_MODE_TO_LINUX = {
    POWER_MODE_PERFORMANCE.lower(): "performance",
    POWER_MODE_BALANCED.lower(): "balanced",
    POWER_MODE_POWER_SAVER.lower(): "power-saver",
}

def get_power_profile() -> str:
    """
    Get the active Linux power profile using powerprofilesctl.

    On Linux, the underlying profile and overlay power mode are unified under
    power-profiles-daemon, so this returns the same active profile as get_power_mode().

    Returns
    -------
    str
        The name of the active power plan (e.g., "Balanced", "Performance",
        "Power Saver"), or an empty string if detection fails.
    """
    return get_power_mode()


def get_power_mode() -> str:
    """
    Get the active Linux power mode using powerprofilesctl.

    Returns
    -------
    str
        The name of the active power mode (e.g., "Performance", "Balanced",
        "Power Saver"), or an empty string if detection fails.
    """
    # Query powerprofilesctl for the active profile
    cmd = ['powerprofilesctl', 'get']
    stdout, stderr, returncode = execute_command(cmd)
    
    if returncode == 0 and stdout:
        profile = stdout.strip().lower()
        if profile in _LINUX_TO_MODE:
            return _LINUX_TO_MODE[profile]
        if profile:
            return f"Unknown ({profile})"
    else:
        logger.error("Failed to query power profile: %s", stderr or stdout)
        
    return ""


def set_power_mode(mode: str) -> bool:
    """
    Set the Linux power mode using powerprofilesctl.

    Parameters
    ----------
    mode : str
        The power mode to set. Valid values (case-insensitive):
        - "Performance"
        - "Balanced"
        - "Power Saver"

    Returns
    -------
    bool
        True if the power mode was successfully set, False otherwise.
    """
    mode_lower = mode.lower()
    if mode_lower not in _MODE_TO_LINUX:
        logger.error("Invalid power mode: %s. Valid modes: %s",
                     mode, ", ".join(VALID_POWER_MODES))
        return False

    linux_profile = _MODE_TO_LINUX[mode_lower]

    # Use powerprofilesctl set to change the active profile
    cmd = ['powerprofilesctl', 'set', linux_profile]
    stdout, stderr, returncode = execute_command(cmd)

    if returncode != 0:
        logger.error("Failed to set power mode: %s", stderr or stdout)
        return False

    # Verify the change took effect
    new_mode = get_power_mode()
    expected_mode = _LINUX_TO_MODE[linux_profile]

    if new_mode == expected_mode:
        logger.info("Power mode set to: %s", expected_mode)
        return True

    logger.warning("Power mode may not have been set correctly. "
                   "Expected: %s, Got: %s", expected_mode, new_mode)
    return False
