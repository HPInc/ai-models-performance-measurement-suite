#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
JSON file utilities for benchmark results.

This module provides functions for reading, writing, and processing
benchmark JSON files.

Public API
----------
- read_json_files_from_directory(directory) -> List[Dict]
- read_specific_json_files(file_paths) -> List[Dict]
- extract_label_from_json_data(data) -> str
- extract_dataset_metadata(data) -> Dict[str, Any]
- extract_runtime_stats(data) -> Optional[Dict]
- normalize_runtime_stats
"""
import glob
import json
import logging
import os
import platform
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

# Import power configuration utilities for setting power mode.
try:
    from power_config import get_power_mode
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing power_config.py library file.") from exc

def read_json_files_from_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Read all JSON files from the specified directory.

    Each JSON file is expected to contain benchmark results with at minimum
    a 'runtime_stats' key containing resource usage data.

    Parameters
    ----------
    directory : str
        Path to the directory containing JSON files.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, each containing the parsed contents of a JSON
        file along with a '_filename' key storing the source filename.
    """
    json_files = glob.glob(os.path.join(directory, '*.json'))
    results = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
                # Add filename and filepath for labeling in plots and artifact logging
                data['_filename'] = os.path.basename(json_file)
                data['_filepath'] = os.path.abspath(json_file)
                results.append(data)
                logger.debug("Loaded JSON file: %s", json_file)
        except (IOError, json.JSONDecodeError) as e:
            logger.warning("Failed to read JSON file %s: %s", json_file, e)

    logger.info("Loaded %d JSON file(s) from %s", len(results), directory)
    return results


def read_specific_json_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Read specific JSON files by their full paths.

    Each JSON file is expected to contain benchmark results with at minimum
    a 'runtime_stats' key containing resource usage data. Paths may contain
    wildcard characters (e.g., ``*.json``) which will be expanded via glob.

    Parameters
    ----------
    file_paths : List[str]
        List of full paths to JSON files to read. Glob patterns are supported.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, each containing the parsed contents of a JSON
        file along with a '_filename' key storing the source filename.
    """
    results = []

    # Expand any glob/wildcard patterns in the provided paths.
    expanded_paths: List[str] = []
    for path in file_paths:
        if any(ch in path for ch in ('*', '?', '[', ']')):
            matched = glob.glob(path)
            if not matched:
                logger.warning("No files matched pattern: %s", path)
            expanded_paths.extend(matched)
        else:
            expanded_paths.append(path)

    for json_file in expanded_paths:
        try:
            with open(json_file, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
                # Add filename and filepath for labeling in plots and artifact logging
                data['_filename'] = os.path.basename(json_file)
                data['_filepath'] = os.path.abspath(json_file)
                results.append(data)
                logger.debug("Loaded JSON file: %s", json_file)
        except (IOError, json.JSONDecodeError) as e:
            logger.warning("Failed to read JSON file %s: %s", json_file, e)

    logger.info("Loaded %d JSON file(s)", len(results))
    return results


def extract_label_from_json_data(data: Dict[str, Any]) -> str:
    """
    Extract a human-readable label from JSON benchmark data.

    The label is constructed from model filename, build number, and any
    options/power modes if available, falling back to the source filename.

    Parameters
    ----------
    data : Dict[str, Any]
        Parsed JSON data from a benchmark result file.

    Returns
    -------
    str
        A label suitable for use in plot legends.
    """
    label_parts = []

    # Try to extract model name and build number from llama_bench_results
    llama_bench_results = data.get('llama_bench_results', [])
    if llama_bench_results and isinstance(llama_bench_results, list):
        first_result = llama_bench_results[0]
        if isinstance(first_result, dict):
            if 'model_filename' in first_result:
                model_name = os.path.basename(first_result['model_filename'])
                # Shorten long model names
                if len(model_name) > 30:
                    model_name = model_name[:27] + '...'
                label_parts.append(model_name)
            if 'build_number' in first_result:
                label_parts.append('b' + str(first_result['build_number']))

    # Extract server_options from convenience_metrics if present
    conv_metrics = data.get('convenience_metrics', {})
    if isinstance(conv_metrics, dict):
        server_opts = conv_metrics.get('server_options', '')
        if server_opts:
            label_parts.append(server_opts)

    # Extract options from convenience_metrics
    if isinstance(conv_metrics, dict):
        options = conv_metrics.get('options', '')
        if options:
            label_parts.append(options)

    # Extract run label from convenience_metrics if present
    if isinstance(conv_metrics, dict):
        run_label = conv_metrics.get('runs', '')
        if run_label:
            label_parts.append(run_label)

    # Extract variant label from convenience_metrics if present
    if isinstance(conv_metrics, dict):
        variant = conv_metrics.get('variant', 0)
        if isinstance(variant, int) and variant > 0:
            label_parts.append(f'Variant {variant}')

    # Append power mode name from system_info if available
    system_info = data.get('system_info', {})
    if isinstance(system_info, dict):
        power_mode = system_info.get('Power Mode', '')
        if power_mode:
            label_parts.append(power_mode)

    if label_parts:
        return ' '.join(label_parts)

    # Fallback to filename
    filename = data.get('_filename', 'unknown')
    # Remove .json extension
    if filename.endswith('.json'):
        filename = filename[:-5]
    return filename


def extract_runtime_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract runtime statistics from JSON data, supporting multiple formats.

    This function supports JSON files from both:
    - mass_llama_bench.py (uses 'runtime_stats' key)
    - resource_monitor.py (uses 'stats' key)

    Parameters
    ----------
    data : Dict[str, Any]
        Parsed JSON data from a benchmark result file.

    Returns
    -------
    Dict[str, Any]
        The runtime statistics dictionary, or empty dict if not found.
    """
    # Try mass_llama_bench.py format first
    runtime_stats = data.get('runtime_stats', {})
    if runtime_stats:
        return runtime_stats

    # Fall back to resource_monitor.py format
    return data.get('stats', {})


def extract_dataset_metadata(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract model name, llama version, and options from JSON benchmark data.

    Parameters
    ----------
    data : Dict[str, Any]
        Parsed JSON data from a benchmark result file.

    Returns
    -------
    Dict[str, str]
        Dictionary with keys 'llama_version', 'options',
        'server_options', 'constant_server_options', 'fixed_options',
        'hostname', 'model', and 'power_mode'.
    """
    metadata = {
        'llama_version': '',
        'options': '',
        'server_options': '',
        'constant_server_options': '',
        'fixed_options': '',
        'hostname': '',
        'model': '',
        'description': '',
        'power_mode': '',
        'runs': '',
        'variant': '',
    }

    # Extract hostname from system_info
    system_info = data.get('system_info', {})
    if isinstance(system_info, dict):
        metadata['hostname'] = system_info.get('Host Name', '')

    # Extract from llama_bench_results or llama_benchy_results
    if 'llama_bench_results' in data:
        llama_bench_results = data['llama_bench_results']
    elif 'llama_benchy_results' in data:
        llama_bench_results = data['llama_benchy_results']
    else:
        llama_bench_results = []

    if llama_bench_results and isinstance(llama_bench_results, list):
        first_result = llama_bench_results[0]
        if isinstance(first_result, dict):
            if 'build_number' in first_result:
                metadata['llama_version'] = str(first_result['build_number'])

    # Extract fields from convenience_metrics if present
    conv_metrics = data.get('convenience_metrics', {})
    if isinstance(conv_metrics, dict):
        server_opts = conv_metrics.get('server_options', '')
        if server_opts:
            metadata['server_options'] = server_opts
        model = conv_metrics.get('model', '')
        if model:
            metadata['model'] = model
        description = conv_metrics.get('description', '')
        if description:
            metadata['description'] = description
        # Extract options from convenience_metrics (preferred source)
        options_from_conv = conv_metrics.get('options', '')
        if options_from_conv:
            metadata['options'] = options_from_conv
        constant_server_opts = conv_metrics.get('constant_server_options', '')
        if constant_server_opts:
            metadata['constant_server_options'] = constant_server_opts
        fixed_opts = conv_metrics.get('fixed_options', '')
        if fixed_opts:
            metadata['fixed_options'] = fixed_opts
        runs_label = conv_metrics.get('runs', '')
        if runs_label:
            metadata['runs'] = runs_label
        variant = conv_metrics.get('variant', 0)
        if isinstance(variant, int) and variant > 0:
            metadata['variant'] = f'Variant {variant}'

    # Extract power mode name from system_info for label differentiation
    system_info = data.get('system_info', {})
    if isinstance(system_info, dict):
        power_mode = system_info.get('Power Mode', '')
        if power_mode:
            metadata['power_mode'] = power_mode

    return metadata


def normalize_runtime_stats(runtime_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize runtime statistics by subtracting prestart values from each data point.

    This function modifies the time-series arrays (used_memory_gb, cpu_utilization_pct,
    gpu_utilization_pct, gpu_memory_gb) by subtracting the prestart value from each
    data point, making the data relative to the baseline.

    Parameters
    ----------
    runtime_stats : Dict[str, Any]
        Runtime statistics dictionary containing time-series data.

    Returns
    -------
    Dict[str, Any]
        A new dictionary with normalized time-series data.
    """
    if not runtime_stats:
        return runtime_stats

    # Create a copy to avoid modifying the original
    normalized = dict(runtime_stats)

    # List of time-series keys and their corresponding prestart keys
    series_keys = [
        ('used_memory_gb', 'prestart_used_memory_gb'),
        ('cpu_utilization_pct', 'prestart_cpu_utilization_pct'),
        ('gpu_utilization_pct', 'prestart_gpu_utilization_pct'),
        ('gpu_memory_gb', 'prestart_gpu_memory_gb'),
        ('npu_memory_gb', 'prestart_npu_memory_gb'),
    ]

    for series_key, prestart_key in series_keys:
        if series_key not in normalized:
            continue

        data = normalized[series_key]
        if not data:
            continue

        # Get prestart value - either from the prestart key or from the first element
        prestart_value = normalized.get(prestart_key)
        if prestart_value is None and data and data[0][0] < 0.0:
            prestart_value = data[0][1]

        if prestart_value is None:
            continue

        # Normalize each data point by subtracting the prestart value
        normalized_data = []
        for timestamp, value in data:
            normalized_data.append([timestamp, round(value - prestart_value, 3)])
        normalized[series_key] = normalized_data

    return normalized
