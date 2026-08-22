#!/usr/bin/env python3
"""
Power configuration utilities.

This module provides functions to query and set power settings:
- On Windows, Power Profile is the active power scheme/plan (e.g.,
  "Balanced", "High Performance").
- On Windows, Power Mode is the overlay slider setting (e.g.,
  "Best Performance", "Balanced").
- On non-Windows platforms, the module degrades gracefully and returns empty
  values or ``False`` for unsupported operations.

The power mode is the slider shown in Windows Settings > System > Power & battery
that provides options like "Best Performance", "Balanced", and "Best Power Efficiency".
This is different from the power profile/scheme which is the underlying power plan.

Public API
----------
- get_power_profile() -> str: Get the active power plan name
- get_power_mode() -> str: Get the active power mode overlay
- set_power_mode(mode: str) -> bool: Set the power mode overlay
- POWER_MODE_PERFORMANCE: Constant for "Performance" mode,
                          which maps to Best Performance on Windows
- POWER_MODE_BALANCED: Constant for "Balanced" mode
- POWER_MODE_POWER_SAVER: Constant for "Power Saver" mode, which maps
                          to Best Power Efficiency on Windows
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

# Power mode overlay GUIDs
# Note: All-zeros GUID means no overlay is active, which is the default/Balanced mode
_OVERLAY_GUID_BEST_PERFORMANCE = "ded574b5-45a0-4f42-8737-46345c09c238"
_OVERLAY_GUID_BEST_POWER_EFFICIENCY = "961cc777-2547-4f9d-8174-7d86181b8a7a"
_OVERLAY_GUID_BALANCED = "00000000-0000-0000-0000-000000000000"

# Mapping from GUID to power mode name
_GUID_TO_MODE = {
    _OVERLAY_GUID_BEST_PERFORMANCE: POWER_MODE_PERFORMANCE,
    _OVERLAY_GUID_BALANCED: POWER_MODE_BALANCED,
    _OVERLAY_GUID_BEST_POWER_EFFICIENCY: POWER_MODE_POWER_SAVER
}

# Mapping from power mode name to GUID (case-insensitive keys)
_MODE_TO_GUID = {
    POWER_MODE_PERFORMANCE.lower(): _OVERLAY_GUID_BEST_PERFORMANCE,
    POWER_MODE_BALANCED.lower(): _OVERLAY_GUID_BALANCED,
    POWER_MODE_POWER_SAVER.lower(): _OVERLAY_GUID_BEST_POWER_EFFICIENCY
}

def get_power_profile() -> str:
    """
    Get the active Windows power profile using powercfg.

    The power profile is the power scheme/plan configured in Windows, such as
    "Balanced", "High Performance", or custom plans.

    Returns
    -------
    str
        The name of the active power plan (e.g., "Balanced", "High Performance"), 
        or an empty string if detection fails.
    """
    # Use powercfg /getactivescheme to get the currently active power plan
    # Output format: "Power Scheme GUID: <guid>  (Plan Name)"
    cmd = ['powercfg', '/getactivescheme']
    stdout, _, returncode = execute_command(cmd)
    if returncode == 0 and stdout:
        # Extract the plan name from parentheses
        # Example: "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)"
        # Note: Plan names may contain nested parentheses, e.g., "(Best Performance (Modern Standby))"
        # so we use the first '(' and last ')' to capture the full name.
        line = stdout.strip()
        if '(' in line and ')' in line:
            start = line.find('(') + 1
            end = line.rfind(')')
            if start < end:
                return line[start:end]
    return ""


def get_power_mode() -> str:
    """
    Get the active Windows power mode (slider setting) from the registry.

    The power mode is the overlay setting shown in Windows Settings > Power
    that provides options like "Best Performance", "Balanced", etc.
    This is different from the power profile/scheme.

    Returns
    -------
    str
        The name of the active power mode (e.g., "Best Performance",
        "Balanced", "Best Power Efficiency"), or an empty string if
        detection fails.
    """
    # Query the registry for the active power overlay (AC power)
    cmd = [
        'powershell', '-NoProfile', '-Command',
        '(Get-ItemProperty -Path '
        '"HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Power\\User\\PowerSchemes" '
        '-Name ActiveOverlayAcPowerScheme -ErrorAction SilentlyContinue).'
        'ActiveOverlayAcPowerScheme'
    ]
    stdout, _, returncode = execute_command(cmd)
    if returncode == 0 and stdout:
        guid = stdout.strip().lower()
        if guid in _GUID_TO_MODE:
            return _GUID_TO_MODE[guid]
        # Unknown GUID - return it for debugging
        if guid:
            return f"Unknown ({guid})"
    return ""


def set_power_mode(mode: str) -> bool:
    """
    Set the Windows power mode (slider setting).

    This changes the power overlay setting shown in Windows Settings > Power.

    Parameters
    ----------
    mode : str
        The power mode to set. Valid values (case-insensitive):
        - "Best Performance"
        - "Balanced"
        - "Best Power Efficiency"

    Returns
    -------
    bool
        True if the power mode was successfully set, False otherwise.

    Notes
    -----
    Uses powercfg /overlaysetactive to change the power mode overlay.
    This method does not require administrator privileges.
    """
    mode_lower = mode.lower()
    if mode_lower not in _MODE_TO_GUID:
        logger.error("Invalid power mode: %s. Valid modes: %s",
                     mode, ", ".join(VALID_POWER_MODES))
        return False

    guid = _MODE_TO_GUID[mode_lower]

    # Use powercfg /overlaysetactive to set the power overlay
    # This doesn't require admin privileges
    cmd = ['powercfg', '/overlaysetactive', guid]
    stdout, stderr, returncode = execute_command(cmd)

    if returncode != 0:
        logger.error("Failed to set power mode: %s", stderr or stdout)
        return False

    # Verify the change took effect
    new_mode = get_power_mode()
    expected_mode = _GUID_TO_MODE[guid]

    if new_mode == expected_mode:
        logger.info("Power mode set to: %s", expected_mode)
        return True

    logger.warning("Power mode may not have been set correctly. "
                   "Expected: %s, Got: %s", expected_mode, new_mode)
    return False
