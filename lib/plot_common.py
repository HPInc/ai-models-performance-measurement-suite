#!/usr/bin/env python3
"""
Common utilities shared between PDF and browser plotting modules.

This module provides shared data structures, constants, and helper functions
used by both plot_pdf_stats.py and plot_stats_for_browser.py to avoid code
duplication.

Public API:
    - DatasetInfo: Dataclass for dataset metadata in overlay plots.
    - METRIC_DEFINITIONS: Standard metric definitions list.
    - OVERLAY_COLORS: Color palette for overlaying multiple datasets.
    - ensure_output_directory(output_dir) -> bool
    - sanitize_prefix(filename_prefix) -> str
    - compute_differential_labels(datasets) -> List[str]
    - extract_bar_chart_data(datasets, metric_key) -> tuple
    - extract_timestamps_values(data) -> tuple
    - filter_available_metrics(stats) -> List[tuple]
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:  # pylint: disable=too-many-instance-attributes
    """Information about a single dataset for overlay plotting."""
    label: str
    stats: Dict[str, Any]
    color: str
    llama_version: str = ""
    options: str = ""
    server_options: str = ""
    constant_server_options: str = ""
    fixed_options: str = ""
    hostname: str = ""
    model: str = ""
    description: str = ""
    power_mode: str = ""
    runs: str = ""
    variant: str = ""
    conv_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.conv_metrics is None:
            self.conv_metrics = {}


# Standard metric definitions used across plotting functions.
# Format: (data_key, metric_name, ylabel, color)
METRIC_DEFINITIONS = [
    ("cpu_utilization_pct", "CPU Utilization", "Util (%)", "blue"),
    ("gpu_utilization_pct", "GPU Utilization", "Util (%)", "orange"),
    ("gpu_memory_gb", "GPU Memory", "GiB", "purple"),
    ("npu_memory_gb", "NPU Memory", "GiB", "brown"),
    ("used_memory_gb", "System RAM", "GiB", "green"),
]


# Colors for overlaying multiple datasets
OVERLAY_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def ensure_output_directory(output_dir: str) -> bool:
    """
    Ensure the output directory exists, creating it if necessary.

    Parameters
    ----------
    output_dir : str
        Path to the output directory.

    Returns
    -------
    bool
        True if the directory exists or was created, False on error.
    """
    if os.path.isdir(output_dir):
        return True
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Created output directory: %s", output_dir)
        return True
    except OSError as exc:
        logger.error("Failed to create output directory '%s': %s", output_dir, exc)
        return False


def sanitize_prefix(filename_prefix: str) -> str:
    """
    Sanitize a filename prefix by replacing problematic characters.

    Parameters
    ----------
    filename_prefix : str
        The filename prefix to sanitize.

    Returns
    -------
    str
        The sanitized filename prefix.
    """
    return filename_prefix.replace(" ", "_").replace("/", "_")


def _truncate_string(value: str, max_len: int) -> str:
    """Truncate a string if it exceeds max_len, adding '...' suffix."""
    if len(value) > max_len:
        return value[:max_len - 3] + '...'
    return value


def _build_label_part(
    ds: DatasetInfo,
    hostname_differs: bool,
    version_differs: bool,
    server_options_differs: bool,
    options_differs: bool,
    cm_model_differs: bool = False,
    description_differs: bool = False,
    constant_server_options_differs: bool = False,
    fixed_options_differs: bool = False,
    power_mode_differs: bool = False,
    runs_differs: bool = False,
    variant_differs: bool = False,
    truncate: bool = True
) -> str:
    """Build a differential label part for a single dataset."""
    maybe_truncate = _truncate_string if truncate else (lambda value, _max_len: value)
    parts = []
    if hostname_differs and ds.hostname:
        parts.append(maybe_truncate(ds.hostname, 20))
    if description_differs and ds.description:
        parts.append(maybe_truncate(ds.description, 30))
    if cm_model_differs and ds.model:
        parts.append(maybe_truncate(ds.model, 25))
    if version_differs and ds.llama_version:
        version = 'b' + ds.llama_version if ds.llama_version.isdigit() else ds.llama_version
        parts.append(version)
    if server_options_differs and ds.server_options:
        parts.append(maybe_truncate(ds.server_options, 20))
    if options_differs and ds.options:
        parts.append(maybe_truncate(ds.options, 20))
    if constant_server_options_differs and ds.constant_server_options:
        parts.append(maybe_truncate(ds.constant_server_options, 20))
    if fixed_options_differs and ds.fixed_options:
        parts.append(maybe_truncate(ds.fixed_options, 20))
    if power_mode_differs and ds.power_mode:
        parts.append(ds.power_mode)
    if runs_differs and ds.runs:
        parts.append(ds.runs)
    if variant_differs and ds.variant:
        parts.append(ds.variant)
    return ' | '.join(parts) if parts else ds.label


def compute_differential_labels(datasets: List[DatasetInfo], truncate: bool = True) -> List[str]:
    """
    Compute legend labels showing only the differences between datasets.

    Analyzes hostnames, model names, llama.cpp versions, and option sets across
    all datasets and builds labels containing only the fields that differ.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of datasets to compare.

    Returns
    -------
    List[str]
        A list of differential labels, one per dataset, in the same order.
        If all fields are identical, returns the original labels.
    """
    if not datasets:
        return []
    if len(datasets) == 1:
        return [datasets[0].label]

    # Determine which categories have differences
    hostname_differs = len(set(ds.hostname for ds in datasets if ds.hostname)) > 1
    description_differs = len(set(ds.description for ds in datasets
                                  if ds.description)) > 1
    cm_model_differs = len(set(ds.model for ds in datasets if ds.model)) > 1
    version_differs = len(set(ds.llama_version for ds in datasets if ds.llama_version)) > 1
    server_options_differs = len(set(ds.server_options for ds in datasets
                                     if ds.server_options)) > 1
    options_differs = len(set(ds.options for ds in datasets if ds.options)) > 1
    constant_server_options_differs = len(set(ds.constant_server_options for ds in datasets
                                             if ds.constant_server_options)) > 1
    fixed_options_differs = len(set(ds.fixed_options for ds in datasets
                                    if ds.fixed_options)) > 1
    power_mode_differs = len(set(ds.power_mode for ds in datasets
                                 if ds.power_mode)) > 1
    runs_differs = len(set(ds.runs for ds in datasets if ds.runs)) > 1
    variant_differs = len(set(ds.variant for ds in datasets if ds.variant)) > 1

    # If nothing differs (all identical), fall back to original labels
    if not any([hostname_differs, description_differs,
                cm_model_differs, version_differs,
                server_options_differs, options_differs,
                constant_server_options_differs, fixed_options_differs,
                power_mode_differs, runs_differs, variant_differs]):
        return [ds.label for ds in datasets]

    return [_build_label_part(ds, hostname_differs, version_differs,
                              server_options_differs, options_differs,
                              cm_model_differs,
                              description_differs,
                              constant_server_options_differs,
                              fixed_options_differs,
                              power_mode_differs,
                              runs_differs,
                              variant_differs,
                              truncate)
            for ds in datasets]


def extract_bar_chart_data(
    datasets: List[DatasetInfo],
    metric_key: str
) -> tuple:
    """
    Extract valid datasets, values, and stddev values for bar chart rendering.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of datasets to extract data from.
    metric_key : str
        The key to look up in each dataset's conv_metrics.

    Returns
    -------
    tuple
        A tuple of (valid_datasets, values, stddev_values) where valid_datasets
        is a list of DatasetInfo objects with valid metric data, values is a list
        of the corresponding metric values, and stddev_values is a list of
        standard deviation values (or None for each entry if not available).
    """
    valid_datasets = []
    values = []
    stddev_values = []

    # Determine the corresponding stddev key
    stddev_key_map = {
        'ttft_ms': 'ttft_stddev_ms',
        'avg_ts': 'avg_ts_stddev',
        'throughput_samples_per_sec': 'throughput_samples_per_sec_stddev',
        'p50_batch_latency_ms': 'p50_batch_latency_ms_stddev',
    }
    stddev_key = stddev_key_map.get(metric_key)

    for ds in datasets:
        # Look in conv_metrics for llama-bench derived metrics (ttft_ms, avg_ts)
        value = ds.conv_metrics.get(metric_key)
        if value is not None and isinstance(value, (int, float)) and value > 0:
            valid_datasets.append(ds)
            values.append(value)
            # Extract stddev if available
            stddev = ds.conv_metrics.get(stddev_key) if stddev_key else None
            stddev_values.append(stddev)

    return valid_datasets, values, stddev_values


def extract_timestamps_values(data: List[List[Any]]) -> tuple:
    """
    Extract timestamps and values from time-series data.

    Parameters
    ----------
    data : List[List[Any]]
        Time-series data as [[timestamp, value], ...].

    Returns
    -------
    tuple
        A tuple of (timestamps, values) lists.
    """
    timestamps = [point[0] for point in data]
    values = [point[1] for point in data]
    return timestamps, values


def compute_avg_max_from_runtime_series(
    stats: Dict[str, Any],
    series_key: str
) -> Tuple[Optional[float], Optional[float]]:
    """Compute avg/max from a runtime time-series, excluding negative timestamps."""
    if not stats:
        return None, None

    series = stats.get(series_key)
    if not isinstance(series, list):
        return None, None

    values: List[float] = []
    for point in series:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue

        timestamp = point[0]
        value = point[1]
        if not isinstance(timestamp, (int, float)) or not isinstance(value, (int, float)):
            continue
        if timestamp < 0:
            continue

        values.append(float(value))

    if not values:
        return None, None

    avg_value = sum(values) / len(values)
    max_value = max(values)
    return avg_value, max_value


def filter_available_metrics(stats: Dict[str, Any]) -> List[tuple]:
    """
    Filter METRIC_DEFINITIONS to only include metrics with data.

    Parameters
    ----------
    stats : Dict[str, Any]
        Statistics dictionary containing metric data.

    Returns
    -------
    List[tuple]
        Filtered list of (data_key, metric_name, ylabel, color) tuples
        for metrics that have data in the stats dictionary.
    """
    return [(k, n, y, c) for k, n, y, c in METRIC_DEFINITIONS if stats.get(k)]


def filter_available_metrics_for_datasets(datasets: List[DatasetInfo]) -> List[tuple]:
    """
    Filter METRIC_DEFINITIONS to only include metrics with data in any dataset.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of datasets to check for available metrics.

    Returns
    -------
    List[tuple]
        Filtered list of (data_key, metric_name, ylabel, color) tuples
        for metrics that have data in at least one dataset.
    """
    return [(k, n, y, c) for k, n, y, c in METRIC_DEFINITIONS
            if any(ds.stats.get(k) for ds in datasets)]
