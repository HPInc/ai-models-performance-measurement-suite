#!/usr/bin/env python3
"""
Helper utilities for resource_monitor.py.

This module contains shared utilities used by resource_monitor.py that need
to be separated to avoid circular import dependencies with benchmark_helpers.py.

Public API
----------
- sample_interval(candidate) -> float: Validate sample interval argument
"""
import argparse


def sample_interval(candidate):
    """
    Validate and convert the --sample-interval argument to float seconds.

    Ensures the value parses as a float and lies within [0.2, 1.0].
    Raises argparse.ArgumentTypeError for invalid input so argparse can render
    a helpful message.
    """
    try:
        candidate = float(candidate)
    except ValueError as exc:
        # Re-raise as ArgumentTypeError for argparse to handle
        raise argparse.ArgumentTypeError(
            f"{candidate!r} not a floating-point literal"
        ) from exc

    if candidate < 0.2 or candidate > 1.0:
        raise argparse.ArgumentTypeError(
            f"{candidate!r} not in range 0.2-1.0 inclusive")
    return candidate
