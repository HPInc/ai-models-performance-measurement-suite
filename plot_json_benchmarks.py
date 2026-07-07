#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
Generate overlay plots from JSON benchmark files produced by mass_llama_bench.py.

This script reads JSON benchmark result files from specified directories and/or
file paths, then generates overlay plots comparing resource usage across multiple
benchmark runs. It supports JSON files from both mass_llama_bench.py and
resource_monitor.py.

Usage
-----
    python plot_json_benchmarks.py -i /path/to/json/files -o /output/dir
    python plot_json_benchmarks.py -j file1.json -j file2.json -o /output/dir
    python plot_json_benchmarks.py -i /path/to/json/files -o /output/dir -b

High-level behavior
-------------------
- Reads JSON benchmark files from one or more directories or explicit file paths.
- Extracts runtime statistics (CPU, GPU, memory usage) and convenience metrics
  (TTFT, MTBT) from each JSON file.
- For a single JSON file, generates standard plots with end-of-benchmark markers.
- For multiple JSON files, overlays all datasets onto combined plots without
  end-of-benchmark markers for clarity.
- Generates TTFT and MTBT bar charts for comparing benchmark performance.
- Uses differential labeling (hostname, model name, llama.cpp version, options)
  to distinguish datasets in legends, showing only fields that differ.
- Supports both PDF output (default, via matplotlib) and interactive HTML output
  (via -b/--browser-plots flag).

Key design choices
------------------
- Supports combining JSON files from multiple directories and explicit paths
  in a single invocation.
- Automatically determines whether to generate single-file or overlay plots
  based on the number of input files.
- Extracts "Host Name" from system_info for cross-machine comparisons.
- PDF and browser output formats are mutually exclusive.

Expected environment
--------------------
- Requires `matplotlib` for generating PDF plots (default).
- Requires `plotly` for generating interactive HTML plots (optional).
- Requires library modules: json_utils, plot_pdf_stats, platform_support.
- Designed exclusively for Windows 10/11 environments.

Imports
-------
- argparse: CLI argument parsing.
- logging: Structured logging throughout the module.
- os, sys: Filesystem and path manipulation.
- platform: Detect Windows environment (via mass_bench_common).
- typing: Type annotations for readability and tooling support.
"""
import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Compute the directory containing this script to make relative imports robust.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Look in the lib directory for library files.
lib_dir = os.path.join(script_dir, 'lib')
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

# Import shared JSON utilities
try:
    from json_utils import (
        read_json_files_from_directory,
        read_specific_json_files,
        extract_label_from_json_data,
        extract_dataset_metadata,
        extract_runtime_stats,
        normalize_runtime_stats
    )
except ImportError:
    raise RuntimeError("Cannot find json_utils module.") from None

# Import shared utilities from plot_common module.
try:
    from plot_common import DatasetInfo, OVERLAY_COLORS
except ImportError:
    raise RuntimeError("Cannot find plot_common module.") from None

# Import shared benchmark utilities
try:
    from mass_bench_common import check_python_version, check_windows_platform
except ImportError:
    raise RuntimeError("Cannot find mass_bench_common module.") from None

# Import plotting utilities (PDF via matplotlib - default)
try:
    from plot_pdf_stats import (
        generate_all_pdf_plots,
        generate_overlay_pdf_plots,
        generate_ttft_bar_chart_pdf,
        generate_avg_ts_bar_chart_pdf,
        generate_combined_performance_pdf,
        generate_memory_usage_bar_chart_pdf,
        generate_gpu_memory_usage_bar_chart_pdf,
        generate_cpu_utilization_bar_chart_pdf,
        generate_gpu_utilization_bar_chart_pdf,
        generate_npu_memory_usage_bar_chart_pdf,
        generate_throughput_bar_chart_pdf,
        generate_p50_latency_bar_chart_pdf,
        generate_embeddings_combined_performance_pdf
    )
except ImportError:
    raise RuntimeError("Cannot find plot_pdf_stats module.") from None

# Import Plotly plotting utilities
try:
    from plot_stats_for_browser import (
        generate_all_plots_for_browser,
        generate_overlay_plots_for_browser,
        generate_ttft_bar_chart_for_browser,
        generate_avg_ts_bar_chart_for_browser,
        generate_combined_performance_for_browser,
        generate_memory_usage_bar_chart_for_browser,
        generate_gpu_memory_usage_bar_chart_for_browser,
        generate_cpu_utilization_bar_chart_for_browser,
        generate_gpu_utilization_bar_chart_for_browser,
        generate_npu_memory_usage_bar_chart_for_browser,
        generate_throughput_bar_chart_for_browser,
        generate_p50_latency_bar_chart_for_browser,
        generate_embeddings_combined_performance_for_browser
    )
except ImportError:
    raise RuntimeError("Cannot find plot_stats_for_browser module.") from None

# Import platform-specific helpers for directory creation.
try:
    from platform_support import create_output_directory_tree
except ImportError:
    raise RuntimeError("Cannot find platform_support module.") from None


def _generate_single_file_plots(
    data: Dict[str, Any],
    output_dir: str,
    create_browser_plots: bool = False,
    normalize_resource_data: bool = False,
    label_sort: bool = False
) -> int:
    """
    Generate plots for a single JSON file.

    Parameters
    ----------
    data : Dict[str, Any]
        Parsed JSON data from the benchmark file.
    output_dir : str
        Directory where plot files will be written.
    create_browser_plots : bool
        If True, generate interactive HTML plots for browser viewing.
        If False (default), generate PDF plots using matplotlib.
    normalize_resource_data : bool
        If True, normalize data by subtracting prestart values.

    Returns
    -------
    int
        Number of plots successfully generated.
    """
    runtime_stats = extract_runtime_stats(data)

    # Optionally normalize the data
    if runtime_stats and normalize_resource_data:
        runtime_stats = normalize_runtime_stats(runtime_stats)

    # Use the original filename (minus .json) as prefix
    filename = data.get('_filename', 'benchmark')
    if filename.endswith('.json'):
        filename = filename[:-5]

    # Use description from convenience_metrics as subtitle if available
    conv_metrics = data.get('convenience_metrics', {})
    subtitle = conv_metrics.get('description') if isinstance(conv_metrics, dict) else None

    plots_created = 0
    if runtime_stats:
        if create_browser_plots:
            plots_created = generate_all_plots_for_browser(runtime_stats, output_dir, filename,
                                                           subtitle=subtitle)
        else:
            plots_created = generate_all_pdf_plots(runtime_stats, output_dir, filename,
                                                   subtitle=subtitle)
    else:
        logger.warning("No runtime stats found in JSON file; skipping resource plots.")

    single_dataset = _build_dataset_from_json(data, 0, normalize_resource_data)
    if create_browser_plots:
        if generate_memory_usage_bar_chart_for_browser([single_dataset], output_dir, filename,
                                                       subtitle=subtitle,
                                                       label_sort=label_sort):
            plots_created += 1
        if generate_gpu_memory_usage_bar_chart_for_browser([single_dataset], output_dir,
                                                           filename, subtitle=subtitle,
                                                           label_sort=label_sort):
            plots_created += 1
        if generate_cpu_utilization_bar_chart_for_browser([single_dataset], output_dir,
                                                          filename, subtitle=subtitle,
                                                          label_sort=label_sort):
            plots_created += 1
        if generate_gpu_utilization_bar_chart_for_browser([single_dataset], output_dir,
                                                          filename, subtitle=subtitle,
                                                          label_sort=label_sort):
            plots_created += 1
        if generate_npu_memory_usage_bar_chart_for_browser([single_dataset], output_dir,
                                                           filename, subtitle=subtitle,
                                                           label_sort=label_sort):
            plots_created += 1
    else:
        if generate_memory_usage_bar_chart_pdf([single_dataset], output_dir, filename,
                                               subtitle=subtitle,
                                               label_sort=label_sort):
            plots_created += 1
        if generate_gpu_memory_usage_bar_chart_pdf([single_dataset], output_dir, filename,
                                                   subtitle=subtitle,
                                                   label_sort=label_sort):
            plots_created += 1
        if generate_cpu_utilization_bar_chart_pdf([single_dataset], output_dir, filename,
                                                  subtitle=subtitle,
                                                  label_sort=label_sort):
            plots_created += 1
        if generate_gpu_utilization_bar_chart_pdf([single_dataset], output_dir, filename,
                                                  subtitle=subtitle,
                                                  label_sort=label_sort):
            plots_created += 1
        if generate_npu_memory_usage_bar_chart_pdf([single_dataset], output_dir, filename,
                                                   subtitle=subtitle,
                                                   label_sort=label_sort):
            plots_created += 1

    logger.info("Generated %d plot(s) for single benchmark run.", plots_created)
    return plots_created


def _build_dataset_from_json(
    data: Dict[str, Any],
    idx: int,
    normalize_resource_data: bool = False
) -> DatasetInfo:
    """Build a DatasetInfo object from parsed JSON data."""
    runtime_stats = extract_runtime_stats(data)

    # Optionally normalize the data
    if runtime_stats and normalize_resource_data:
        runtime_stats = normalize_runtime_stats(runtime_stats)

    conv_metrics = data.get('convenience_metrics', {})
    label = extract_label_from_json_data(data)
    metadata = extract_dataset_metadata(data)
    color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]

    return DatasetInfo(
        label=label,
        stats=runtime_stats,
        color=color,
        llama_version=metadata['llama_version'],
        options=metadata['options'],
        server_options=metadata['server_options'],
        constant_server_options=metadata['constant_server_options'],
        fixed_options=metadata['fixed_options'],
        hostname=metadata['hostname'],
        model=metadata['model'],
        description=metadata.get('description', ''),
        power_mode=metadata.get('power_mode', ''),
        runs=metadata.get('runs', ''),
        variant=metadata.get('variant', ''),
        conv_metrics=conv_metrics
    )


def _build_datasets_from_json_list(
    json_data_list: List[Dict[str, Any]],
    normalize_resource_data: bool = False
) -> List[DatasetInfo]:
    """Build a list of DatasetInfo objects from parsed JSON data."""
    datasets = []
    for idx, data in enumerate(json_data_list):
        datasets.append(_build_dataset_from_json(data, idx, normalize_resource_data))

    _apply_multi_variant_labels(datasets)
    return datasets


def _extract_variant_index(dataset: DatasetInfo) -> int:
    """Extract the variant index from dataset metadata, defaulting to 0."""
    conv_variant = dataset.conv_metrics.get('variant') if dataset.conv_metrics else None
    if isinstance(conv_variant, int):
        return conv_variant
    if isinstance(conv_variant, str) and conv_variant.isdigit():
        return int(conv_variant)

    if dataset.variant.startswith('Variant '):
        suffix = dataset.variant[len('Variant '):]
        if suffix.isdigit():
            return int(suffix)

    if dataset.variant.startswith('Variant_'):
        suffix = dataset.variant[len('Variant_'):]
        if suffix.isdigit():
            return int(suffix)

    return 0


def _apply_multi_variant_labels(datasets: List[DatasetInfo]) -> None:
    """Ensure Variant_0 is set when overlays include more than one variant."""
    if len(datasets) < 2:
        return

    variant_indices = {_extract_variant_index(dataset) for dataset in datasets}
    if len(variant_indices) <= 1:
        return

    for dataset in datasets:
        dataset.variant = f'Variant {_extract_variant_index(dataset)}'


def _build_common_subtitle(datasets: List[DatasetInfo]) -> Optional[str]:
    """
    Build a subtitle string from label discriminators that are constant across datasets.

    Fields that have the same non-empty value across all datasets are collected
    into the subtitle. Description is listed first, followed by hostname, model,
    llama version, server options, and options.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of datasets to compare.

    Returns
    -------
    Optional[str]
        A subtitle string joining all constant discriminators, or None if none qualify.
    """
    if not datasets or len(datasets) < 2:
        return None

    parts = []

    # Description first
    descriptions = set(ds.description for ds in datasets if ds.description)
    if len(descriptions) == 1:
        parts.append(descriptions.pop())

    # Hostname
    hostnames = set(ds.hostname for ds in datasets if ds.hostname)
    if len(hostnames) == 1:
        parts.append(hostnames.pop())

    # Model
    models = set(ds.model for ds in datasets if ds.model)
    if len(models) == 1:
        parts.append(models.pop())

    # Llama version
    versions = set(ds.llama_version for ds in datasets if ds.llama_version)
    if len(versions) == 1:
        version = versions.pop()
        parts.append('b' + version if version.isdigit() else version)

    # Server options
    server_opts = set(ds.server_options for ds in datasets if ds.server_options)
    if len(server_opts) == 1:
        parts.append(server_opts.pop())

    # Options
    options = set(ds.options for ds in datasets if ds.options)
    if len(options) == 1:
        parts.append(options.pop())

    # Constant server options
    const_server_opts = set(ds.constant_server_options for ds in datasets
                            if ds.constant_server_options)
    if len(const_server_opts) == 1:
        parts.append(const_server_opts.pop())

    # Fixed options
    fixed_opts = set(ds.fixed_options for ds in datasets if ds.fixed_options)
    if len(fixed_opts) == 1:
        parts.append(fixed_opts.pop())

    # Power mode
    power_modes = set(ds.power_mode for ds in datasets if ds.power_mode)
    if len(power_modes) == 1:
        parts.append(power_modes.pop())

    # Run label
    runs = set(ds.runs for ds in datasets if ds.runs)
    if len(runs) == 1:
        parts.append(runs.pop())

    return ' | '.join(parts) if parts else None


def _generate_overlay_plots_from_data(
    json_data_list: List[Dict[str, Any]],
    output_dir: str,
    output_prefix: str,
    create_browser_plots: bool = False,
    normalize_resource_data: bool = False,
    label_sort: bool = False
) -> int:
    """
    Generate overlay plots from multiple JSON files.

    Parameters
    ----------
    json_data_list : List[Dict[str, Any]]
        List of parsed JSON data from benchmark files.
    output_dir : str
        Directory where plot files will be written.
    output_prefix : str
        Prefix for output plot filenames.
    create_browser_plots : bool
        If True, generate interactive HTML plots for browser viewing.
        If False (default), generate PDF plots using matplotlib.
    normalize_resource_data : bool
        If True, normalize data by subtracting prestart values.

    Returns
    -------
    int
        Number of plots successfully generated.
    """
    datasets = _build_datasets_from_json_list(json_data_list, normalize_resource_data)

    if not datasets:
        logger.warning("No valid datasets found for overlay plotting.")
        return 0

    # Select plotting functions based on output format
    if create_browser_plots:
        generate_overlay_plots = generate_overlay_plots_for_browser
        generate_ttft_bar_chart = generate_ttft_bar_chart_for_browser
        generate_avg_ts_bar_chart = generate_avg_ts_bar_chart_for_browser
        generate_combined_performance = generate_combined_performance_for_browser
        generate_memory_usage_bar_chart = generate_memory_usage_bar_chart_for_browser
        generate_gpu_memory_usage_bar_chart = generate_gpu_memory_usage_bar_chart_for_browser
        generate_cpu_utilization_bar_chart = generate_cpu_utilization_bar_chart_for_browser
        generate_gpu_utilization_bar_chart = generate_gpu_utilization_bar_chart_for_browser
        generate_npu_memory_usage_bar_chart = generate_npu_memory_usage_bar_chart_for_browser
        generate_throughput_bar_chart = generate_throughput_bar_chart_for_browser
        generate_p50_latency_bar_chart = generate_p50_latency_bar_chart_for_browser
        generate_embeddings_combined_performance = generate_embeddings_combined_performance_for_browser
    else:
        generate_overlay_plots = generate_overlay_pdf_plots
        generate_ttft_bar_chart = generate_ttft_bar_chart_pdf
        generate_avg_ts_bar_chart = generate_avg_ts_bar_chart_pdf
        generate_combined_performance = generate_combined_performance_pdf
        generate_memory_usage_bar_chart = generate_memory_usage_bar_chart_pdf
        generate_gpu_memory_usage_bar_chart = generate_gpu_memory_usage_bar_chart_pdf
        generate_cpu_utilization_bar_chart = generate_cpu_utilization_bar_chart_pdf
        generate_gpu_utilization_bar_chart = generate_gpu_utilization_bar_chart_pdf
        generate_npu_memory_usage_bar_chart = generate_npu_memory_usage_bar_chart_pdf
        generate_throughput_bar_chart = generate_throughput_bar_chart_pdf
        generate_p50_latency_bar_chart = generate_p50_latency_bar_chart_pdf
        generate_embeddings_combined_performance = generate_embeddings_combined_performance_pdf

    # Build subtitle from label discriminators that are constant across all datasets.
    # Description is listed first, followed by other invariant fields.
    subtitle = _build_common_subtitle(datasets)

    plots_created = 0

    # Generate overlay resource plots only for datasets that have runtime_stats
    datasets_with_stats = [ds for ds in datasets if ds.stats]
    if datasets_with_stats:
        plots_created = generate_overlay_plots(
            datasets=datasets_with_stats,
            output_dir=output_dir,
            filename_prefix=output_prefix,
            show_end_marker=False,
            subtitle=subtitle
        )
    else:
        logger.warning("No datasets with runtime stats; skipping resource overlay plots.")

    # Generate TTFT and avg_ts bar charts (these use convenience_metrics, not runtime_stats)
    if generate_ttft_bar_chart(datasets, output_dir, output_prefix, subtitle=subtitle,
                               label_sort=label_sort):
        plots_created += 1
    if generate_avg_ts_bar_chart(datasets, output_dir, output_prefix, subtitle=subtitle,
                                 label_sort=label_sort):
        plots_created += 1
    if generate_throughput_bar_chart(datasets, output_dir, output_prefix, subtitle=subtitle,
                                     label_sort=label_sort):
        plots_created += 1
    if generate_p50_latency_bar_chart(datasets, output_dir, output_prefix, subtitle=subtitle,
                                      label_sort=label_sort):
        plots_created += 1

    # Generate memory usage chart from runtime_stats
    if generate_memory_usage_bar_chart(datasets, output_dir, output_prefix, subtitle=subtitle,
                                       label_sort=label_sort):
        plots_created += 1

    # Generate GPU memory usage chart from runtime_stats (avg + max GPU memory)
    if generate_gpu_memory_usage_bar_chart(datasets, output_dir, output_prefix,
                                           subtitle=subtitle, label_sort=label_sort):
        plots_created += 1

    # Generate CPU utilization chart from runtime_stats (avg + max CPU utilization)
    if generate_cpu_utilization_bar_chart(datasets, output_dir, output_prefix,
                                          subtitle=subtitle, label_sort=label_sort):
        plots_created += 1

    # Generate GPU utilization chart from runtime_stats (avg + max GPU utilization)
    if generate_gpu_utilization_bar_chart(datasets, output_dir, output_prefix,
                                          subtitle=subtitle, label_sort=label_sort):
        plots_created += 1

    # Generate NPU memory usage chart from runtime_stats (avg + max NPU memory)
    if generate_npu_memory_usage_bar_chart(datasets, output_dir, output_prefix,
                                           subtitle=subtitle, label_sort=label_sort):
        plots_created += 1

    # Generate combined performance chart (TTFT + avg_ts)
    if generate_combined_performance(datasets, output_dir, output_prefix, subtitle=subtitle,
                                     label_sort=label_sort):
        plots_created += 1

    # Generate embeddings combined performance chart (throughput + P50 latency)
    if generate_embeddings_combined_performance(datasets, output_dir, output_prefix,
                                               subtitle=subtitle, label_sort=label_sort):
        plots_created += 1

    logger.info("Generated %d overlay plot(s) for %d benchmark runs.",
                plots_created, len(datasets))
    return plots_created


# pylint: disable=too-many-arguments,too-many-positional-arguments
def generate_plots_from_json_files(
    output_dir: str,
    output_prefix: str = "benchmark",
    directories: Optional[List[str]] = None,
    json_files: Optional[List[str]] = None,
    create_browser_plots: bool = False,
    normalize_resource_data: bool = False,
    label_sort: bool = False
) -> int:
    """
    Generate plots from JSON files specified by directories and/or file paths.

    If there is only one JSON file, generates standard plots with end-of-benchmark
    markers. If there are multiple JSON files, overlays all datasets onto a single
    set of plots without end-of-benchmark markers for clarity.

    Supports JSON files from both mass_llama_bench.py and resource_monitor.py.

    Parameters
    ----------
    output_dir : str
        Directory where plot files will be written.
    output_prefix : str
        Prefix for output plot filenames. Default is "benchmark".
    directories : Optional[List[str]]
        List of directories to search for JSON files. Default is None.
    json_files : Optional[List[str]]
        List of specific JSON file paths to include. Default is None.
    create_browser_plots : bool
        If True, generate interactive HTML plots for browser viewing.
        If False (default), generate PDF plots using matplotlib.
    normalize_resource_data : bool
        If True, normalize data by subtracting prestart values.

    Returns
    -------
    int
        Number of plots successfully generated.
    """
    json_data_list: List[Dict[str, Any]] = []

    # Read JSON files from specified directories
    if directories:
        for directory in directories:
            if os.path.isdir(directory):
                json_data_list.extend(read_json_files_from_directory(directory))
            else:
                logger.warning("Directory not found: %s", directory)

    # Read specific JSON files
    if json_files:
        json_data_list.extend(read_specific_json_files(json_files))

    if not json_data_list:
        logger.warning("No JSON files found, skipping plot generation.")
        return 0

    if len(json_data_list) == 1:
        return _generate_single_file_plots(json_data_list[0],
                                           output_dir,
                                           create_browser_plots,
                                           normalize_resource_data,
                                           label_sort)

    return _generate_overlay_plots_from_data(json_data_list,
                                             output_dir,
                                             output_prefix,
                                             create_browser_plots,
                                             normalize_resource_data,
                                             label_sort)


def main() -> int:
    """
    Parse arguments and generate plots from JSON benchmark files.
    """
    # This tool only runs in Windows environments. Exit early on other platforms.
    check_python_version()
    check_windows_platform()

    parser = argparse.ArgumentParser(
        description="Generate overlay plots from JSON benchmark files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate PDF plots from JSON files in a directory:
    python plot_json_benchmarks.py -i /path/to/json/files -o /output/dir

  Generate interactive HTML plots using Plotly:
    python plot_json_benchmarks.py -i /path/to/json/files -o /output/dir -b

  Generate plots from specific JSON files:
    python plot_json_benchmarks.py -j file1.json -j file2.json -o /output/dir

  Combine directory and specific files:
    python plot_json_benchmarks.py -i /path/to/dir -j extra_file.json -o /output/dir

  Generate plots with custom output prefix:
    python plot_json_benchmarks.py -i /path/to/json/files -o /output/dir -p comparison

  Verbose output:
    python plot_json_benchmarks.py -i /path/to/json/files -o /output/dir -v
"""
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Directory where plot files will be written.'
    )

    parser.add_argument(
        '-i', '--input-directory',
        type=str,
        action='append',
        help='Directory containing JSON benchmark files to process. '
             'Multiple -i parameters are allowed and will be combined.'
    )

    parser.add_argument(
        '-j', '--json-file',
        type=str,
        action='append',
        help='Path to a specific JSON benchmark file. '
             'Multiple -j parameters are allowed and will be combined.'
    )

    parser.add_argument(
        '-p', '--prefix',
        type=str,
        default='benchmark',
        help='Prefix for output plot filenames (default: "benchmark").'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose output.'
    )

    parser.add_argument(
        '-b', '--browser-plots',
        action='store_true',
        help='Generate interactive HTML plots using Plotly instead of PDF.'
    )

    parser.add_argument(
        '-n', '--normalize-resource-data',
        action='store_true',
        default=False,
        help='Normalize resource data by subtracting prestart values.'
    )

    parser.add_argument(
        '-l', '--label-sort',
        action='store_true',
        default=False,
        help='Sort bars in all bar charts by label instead of metric value.'
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s"
    )

    # Validate that at least one of -i or -j is provided
    if not args.input_directory and not args.json_file:
        parser.error("At least one of -i/--input-directory or -j/--json-file is required.")

    # Ensure the output directory exists (and any intermediate parents).
    if not create_output_directory_tree(args.output_dir):
        logger.error('Unable to create output directory tree: %s', args.output_dir)
        return 1

    # Generate plots
    plots_created = generate_plots_from_json_files(
        output_dir=args.output_dir,
        output_prefix=args.prefix,
        directories=args.input_directory,
        json_files=args.json_file,
        create_browser_plots=args.browser_plots,
        normalize_resource_data=args.normalize_resource_data,
        label_sort=args.label_sort
    )

    format_type = "HTML (browser)" if args.browser_plots else "PDF"
    if plots_created > 0:
        print(f"Successfully generated {plots_created} {format_type} plot(s) in {args.output_dir}")
    else:
        logger.info("No plots were generated (runtime stats may be absent).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
