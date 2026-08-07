#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
Generate PDF plots of performance statistics gathered by resource_monitor.py.

This module provides functions to visualize CPU, GPU, and memory utilization
time-series data collected during benchmark runs. Each plot includes:
    - Vertical markers at t=0 (worker start) and t=benchmark_duration_s (worker end)
    - All data points including pre-start (t<0) and post-end samples
    - X axis labeled in whole seconds with intelligent spacing for legibility

Public API:
    - generate_all_pdf_plots(stats: Dict, output_dir: str, filename_prefix: str = "benchmark")
    - generate_pdf_plots(stats: Dict, output_dir: str, filename_prefix: str = "benchmark")
    """
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
import warnings
from typing import Any, Dict, List, Optional

# Import shared utilities from plot_common module.
from plot_common import (
    DatasetInfo,
    METRIC_DEFINITIONS,
    ensure_output_directory,
    sanitize_prefix,
    compute_differential_labels,
    extract_bar_chart_data,
    extract_timestamps_values,
    compute_avg_max_from_runtime_series,
    filter_available_metrics,
    filter_available_metrics_for_datasets,
)

logger = logging.getLogger(__name__)

# Fail fast if matplotlib is not available.
try:
    from matplotlib import pyplot, ticker
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Please make sure matplotlib is installed.") from exc


@dataclass
class PlotConfig:
    """Configuration for a single metric plot."""
    metric_name: str
    ylabel: str
    output_path: str
    benchmark_duration_s: float
    color: str = "blue"
    subtitle: str = None


@dataclass
class OverlaySubplotConfig:
    """Configuration for overlay subplot rendering."""
    data_key: str
    ylabel: str
    title: str
    show_end_marker: bool
    differential_labels: List[str] = None


@dataclass
class BarChartConfig:
    """Configuration for bar chart generation."""
    metric_key: str
    metric_name: str
    ylabel: str
    output_path: str
    subtitle: str = None
    sort_ascending: bool = True
    label_sort: bool = False


@dataclass
class SingleOverlayConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for a single overlay plot."""
    data_key: str
    metric_name: str
    ylabel: str
    differential_labels: List[str]
    show_end_marker: bool
    subtitle: str
    output_path: str = ""


def _calculate_tick_interval(duration_s: float) -> float:
    """
    Calculate an appropriate tick interval based on the benchmark duration.

    Returns a tick interval that produces a reasonable number of ticks
    (approximately 5-15) for legibility.

    Parameters
    ----------
    duration_s : float
        The benchmark duration in seconds.

    Returns
    -------
    float
        The tick interval in seconds.
    """
    interval = 1.0  # Default interval

    if duration_s > 0:
        # Target approximately 8-12 ticks across the visible range
        target_ticks = 10
        raw_interval = duration_s / target_ticks

        # Round to a "nice" number: 1, 2, 5, 10, 20, 30, 60, etc.
        nice_intervals = [1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0]
        for nice in nice_intervals:
            if raw_interval <= nice:
                interval = nice
                break
        else:
            # For very long durations, use multiples of 60 seconds
            interval = math.ceil(raw_interval / 60) * 60

    return interval


def _tight_layout_with_auto_expand(
    fig,
    max_attempts: int = 4,
    width_scale: float = 1.2,
    height_scale: float = 1.15
) -> None:
    """Apply tight_layout and expand figure size automatically on layout warnings."""
    for _ in range(max_attempts):
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", UserWarning)
            pyplot.tight_layout()

        has_layout_warning = any(
            "Tight layout not applied" in str(warning.message)
            for warning in captured_warnings
        )
        if not has_layout_warning:
            return

        width_in, height_in = fig.get_size_inches()
        fig.set_size_inches(width_in * width_scale, height_in * height_scale, forward=True)

    logger.warning(
        "Could not fully resolve tight_layout constraints after auto-expanding figure size."
    )


def _expand_horizontal_bar_plot_width(
    fig,
    left_margin: float,
    right_margin: float = 0.99,
    min_plot_fraction: float = 0.62
) -> None:
    """Expand figure width when needed so horizontal bar plot area fills the page."""
    _expand_plot_width_to_page(
        fig,
        left_margin=left_margin,
        right_margin=right_margin,
        min_plot_fraction=min_plot_fraction,
    )


def _expand_plot_width_to_page(
    fig,
    left_margin: Optional[float] = None,
    right_margin: float = 0.99,
    bottom_margin: Optional[float] = None,
    min_plot_fraction: float = 0.62,
) -> None:
    """Expand figure width as needed, then apply subplot margins."""
    subplot_params = fig.subplotpars
    resolved_left = subplot_params.left if left_margin is None else left_margin
    resolved_bottom = subplot_params.bottom if bottom_margin is None else bottom_margin

    plot_fraction = max(0.05, right_margin - resolved_left)
    if plot_fraction < min_plot_fraction:
        width_in, height_in = fig.get_size_inches()
        width_scale = min_plot_fraction / plot_fraction
        fig.set_size_inches(width_in * width_scale, height_in, forward=True)

    fig.subplots_adjust(left=resolved_left, right=right_margin, bottom=resolved_bottom)


def plot_pdf_metric(data: List[List[Any]], config: PlotConfig) -> bool:
    """
    Generate a single metric plot and save it as a PDF.

    Parameters
    ----------
    data : List[List[Any]]
        Time-series data as [[timestamp, value], ...] where timestamps are
        relative to worker start (t=0).
    config : PlotConfig
        Configuration dataclass containing metric_name, ylabel, output_path,
        benchmark_duration_s, and color.

    Returns
    -------
    bool
        True if the plot was successfully created; False otherwise.
    """
    if not data:
        logger.warning("No data provided for metric '%s'", config.metric_name)
        return False

    try:
        # Extract timestamps and values
        timestamps = [point[0] for point in data]
        values = [point[1] for point in data]

        # Create the figure and axis
        fig, ax = pyplot.subplots(figsize=(12, 6))

        # Plot the data
        ax.plot(timestamps, values, color=config.color, linewidth=1.5, marker='o',
                markersize=3, label=config.metric_name)

        # Add vertical markers at t=0 and t=benchmark_duration_s
        ax.axvline(x=0, color='green', linestyle='--', linewidth=2,
                   label='Worker Start (t=0)')
        ax.axvline(x=config.benchmark_duration_s, color='red', linestyle='--',
                   linewidth=2,
                   label=f'Worker End (t={config.benchmark_duration_s:.1f}s)')

        # Configure X axis with whole second intervals
        tick_interval = _calculate_tick_interval(config.benchmark_duration_s)

        # Determine axis limits with padding
        min_ts = min(timestamps) if timestamps else 0
        max_ts = max(timestamps) if timestamps else config.benchmark_duration_s
        x_padding = max(abs(min_ts), abs(max_ts - config.benchmark_duration_s), 1.0) * 0.1
        ax.set_xlim(min_ts - x_padding, max_ts + x_padding)

        # Set major ticks at whole second intervals
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        # Labels and title
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel(config.ylabel, fontsize=12)
        # Add extra padding to title if subtitle will be shown
        title_pad = 20 if config.subtitle else 6
        ax.set_title(f'{config.metric_name} Over Time', fontsize=14,
                     fontweight='bold', pad=title_pad)

        # Add subtitle below title if provided (e.g., fixed options)
        if config.subtitle:
            ax.text(0.5, 1.0, config.subtitle,
                    transform=ax.transAxes, fontsize=10, style='italic',
                    ha='center', va='bottom')

        # Grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Tight layout and save
        _tight_layout_with_auto_expand(fig)
        _expand_plot_width_to_page(fig)
        pyplot.savefig(config.output_path, format='pdf', dpi=150, bbox_inches='tight')
        pyplot.close(fig)

        logger.info("Saved plot to %s", config.output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write plot for '%s': %s", config.metric_name, exc)
        return False
    # Catch matplotlib-specific errors (ValueError, TypeError, RuntimeError, etc.)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create plot for '%s': %s", config.metric_name, exc)
        return False


def generate_pdf_plots(
    stats: Dict[str, Any],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None
) -> int:
    """
    Generate PDF plots for all available metrics in the stats dictionary.

    Creates separate plots for CPU utilization, GPU utilization, GPU memory,
    and used RAM. Each plot includes vertical markers at t=0 and the benchmark
    end time, with all data points shown.

    Parameters
    ----------
    stats : Dict[str, Any]
        Statistics dictionary from ResourceMonitor.monitor_process(), containing:
            - benchmark_duration_s: float
            - cpu_utilization_pct: List[List[float]]
            - gpu_utilization_pct: List[List[float]]
            - gpu_memory_gb: List[List[float]]
            - used_memory_gb: List[List[float]]
    output_dir : str
        Directory path where PDF files will be written.
    filename_prefix : str
        Prefix for output filenames (default: "benchmark").
    subtitle : str, optional
        A subtitle to display below the main title (e.g., fixed options).

    Returns
    -------
    int
        Number of plots successfully generated.
    """
    if not stats:
        logger.error("No stats provided for plotting.")
        return 0

    benchmark_duration_s = stats.get("benchmark_duration_s", 0.0)
    if benchmark_duration_s <= 0:
        logger.warning("Invalid or missing benchmark_duration_s: %s", benchmark_duration_s)

    if not ensure_output_directory(output_dir):
        return 0

    plots_created = 0
    safe_prefix = sanitize_prefix(filename_prefix)

    for data_key, metric_name, ylabel, color in METRIC_DEFINITIONS:
        data = stats.get(data_key)
        if not data:
            logger.debug("Skipping metric '%s': no data available.", data_key)
            continue

        output_path = os.path.join(output_dir, f"{safe_prefix}_{data_key}.pdf")
        config = PlotConfig(metric_name, ylabel, output_path, benchmark_duration_s, color, subtitle)
        if plot_pdf_metric(data, config):
            plots_created += 1

    logger.info("Generated %d plot(s) in %s", plots_created, output_dir)
    return plots_created


@dataclass
class SubplotConfig:
    """Configuration for a subplot in combined plots."""
    metric_name: str
    ylabel: str
    color: str
    benchmark_duration_s: float


def _plot_subplot(ax: Any, data: List[List[Any]], config: SubplotConfig) -> None:
    """Plot a single metric on a subplot axis."""
    timestamps, values = extract_timestamps_values(data)

    ax.plot(timestamps, values, color=config.color, linewidth=1.5,
            marker='o', markersize=2)

    # Vertical markers
    ax.axvline(x=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=config.benchmark_duration_s, color='red', linestyle='--',
               linewidth=1.5, alpha=0.7)

    ax.set_ylabel(config.ylabel, fontsize=10)
    ax.set_title(config.metric_name, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


def _configure_bottom_axis(ax: Any, tick_interval: float) -> None:
    """Configure x-axis formatting for the bottom subplot."""
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.set_xlabel('Time (seconds)', fontsize=12)


def _add_legend_to_first_subplot(ax: Any, benchmark_duration_s: float) -> None:
    """Add legend with worker start/end markers to the first subplot."""
    ax.axvline(x=0, color='green', linestyle='--', linewidth=1.5,
               label='Worker Start')
    ax.axvline(x=benchmark_duration_s, color='red', linestyle='--',
               linewidth=1.5, label='Worker End')
    ax.legend(loc='upper right', fontsize=8)


def _render_combined_subplots(
    axes: List[Any],
    stats: Dict[str, Any],
    available_metrics: List[tuple],
    benchmark_duration_s: float,
    subtitle: str = None
) -> None:
    """Render all subplots for a combined plot."""
    tick_interval = _calculate_tick_interval(benchmark_duration_s)
    n_plots = len(available_metrics)

    for idx, (data_key, metric_name, ylabel, color) in enumerate(available_metrics):
        subplot_config = SubplotConfig(metric_name, ylabel, color, benchmark_duration_s)
        _plot_subplot(axes[idx], stats[data_key], subplot_config)
        if idx == n_plots - 1:
            _configure_bottom_axis(axes[idx], tick_interval)

    _add_legend_to_first_subplot(axes[0], benchmark_duration_s)

    if subtitle:
        axes[0].set_title(axes[0].get_title(), fontsize=11, fontweight='bold', pad=20)
        axes[0].text(0.5, 1.0, subtitle,
                     transform=axes[0].transAxes, fontsize=9, style='italic',
                     ha='center', va='bottom')


def generate_combined_pdf_plot(
    stats: Dict[str, Any],
    output_path: str,
    subtitle: str = None
) -> bool:
    """
    Generate a single PDF with all metrics as subplots.

    Parameters
    ----------
    stats : Dict[str, Any]
        Statistics dictionary from ResourceMonitor.monitor_process().
    output_path : str
        Full path to the output PDF file.
    subtitle : str, optional
        A subtitle to display below the main title (e.g., fixed options).

    Returns
    -------
    bool
        True if the combined plot was successfully created; False otherwise.
    """
    if not stats:
        logger.error("No stats provided for combined plot.")
        return False

    benchmark_duration_s = stats.get("benchmark_duration_s", 0.0)
    available_metrics = filter_available_metrics(stats)

    if not available_metrics:
        logger.error("No metric data available for combined plot.")
        return False

    try:
        n_plots = len(available_metrics)
        fig, axes = pyplot.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
        axes = [axes] if n_plots == 1 else axes

        _render_combined_subplots(axes, stats, available_metrics, benchmark_duration_s, subtitle)

        _tight_layout_with_auto_expand(fig)
        _expand_plot_width_to_page(fig)
        pyplot.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
        pyplot.close(fig)

        logger.info("Saved combined plot to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write combined plot: %s", exc)
        return False
    # Catch matplotlib-specific errors (ValueError, TypeError, RuntimeError, etc.)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create combined plot: %s", exc)
        return False


def generate_all_pdf_plots(
    stats: Dict[str, Any],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None
) -> int:
    """
    Generate all individual metric plots plus a combined plot.

    This is a convenience function that calls generate_pdf_plots() and
    generate_combined_pdf_plot().

    Parameters
    ----------
    stats : Dict[str, Any]
        Statistics dictionary from ResourceMonitor.monitor_process().
    output_dir : str
        Directory path where PDF files will be written.
    filename_prefix : str
        Prefix for output filenames.
    subtitle : str, optional
        A subtitle to display below the main title (e.g., fixed options).

    Returns
    -------
    int
        Total number of plots successfully generated.
    """
    count = generate_pdf_plots(stats, output_dir, filename_prefix, subtitle)

    # Also generate combined plot
    safe_prefix = filename_prefix.replace(" ", "_").replace("/", "_")
    combined_path = os.path.join(output_dir, f"{safe_prefix}_combined_resources.pdf")
    if generate_combined_pdf_plot(stats, combined_path, subtitle):
        count += 1

    return count


def _plot_overlay_subplot(
    ax: Any, datasets: List[DatasetInfo], config: OverlaySubplotConfig
) -> None:
    """
    Plot multiple datasets on a single subplot axis.

    Parameters
    ----------
    ax : Any
        The matplotlib axis to plot on.
    datasets : List[DatasetInfo]
        List of datasets to overlay.
    config : OverlaySubplotConfig
        Configuration for the subplot.
    """
    for idx, dataset in enumerate(datasets):
        data = dataset.stats.get(config.data_key)
        if not data:
            continue

        timestamps, values = extract_timestamps_values(data)

        # Use differential label if provided, otherwise fall back to dataset label
        diff_labels = config.differential_labels
        label = diff_labels[idx] if diff_labels and idx < len(diff_labels) else dataset.label

        ax.plot(timestamps, values, color=dataset.color, linewidth=1.5,
                marker='o', markersize=2, label=label, alpha=0.8)

        ax.axvline(x=0, color='green', linestyle='--', linewidth=1.5, alpha=0.5)

        if config.show_end_marker:
            benchmark_duration_s = dataset.stats.get("benchmark_duration_s", 0.0)
            ax.axvline(x=benchmark_duration_s, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.7)

    ax.set_ylabel(config.ylabel, fontsize=10)
    ax.set_title(config.title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


def generate_overlay_combined_plot(          # pylint: disable=too-many-locals
    datasets: List[DatasetInfo],
    output_path: str,
    show_end_marker: bool = True,
    subtitle: str = None
) -> bool:
    """
    Generate a single PDF with all metrics as subplots, overlaying multiple datasets.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of DatasetInfo objects containing label, stats, and color.
    output_path : str
        Full path to the output PDF file.
    show_end_marker : bool
        If True, show end-of-benchmark vertical lines. Default is True.
    subtitle : str, optional
        A subtitle to display below the main title (e.g., fixed options).

    Returns
    -------
    bool
        True if the combined plot was successfully created; False otherwise.

    Notes
    -----
    The legend only shows fields that differ between datasets (model names,
    llama.cpp versions, and/or option sets) for clarity.
    """
    if not datasets:
        logger.error("No datasets provided for overlay plot.")
        return False

    # Find metrics that have data in at least one dataset
    available_metrics = [
        (k, n, y) for k, n, y, _ in filter_available_metrics_for_datasets(datasets)
    ]

    if not available_metrics:
        logger.error("No metric data available for overlay plot.")
        return False

    differential_labels = compute_differential_labels(datasets)
    max_duration = max((ds.stats.get("benchmark_duration_s", 0.0) for ds in datasets), default=0.0)

    try:
        n_plots = len(available_metrics)
        fig, axes = pyplot.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
        axes = [axes] if n_plots == 1 else axes
        tick_interval = _calculate_tick_interval(max_duration)

        for idx, (data_key, metric_name, ylabel) in enumerate(available_metrics):
            config = OverlaySubplotConfig(data_key, ylabel, metric_name, show_end_marker,
                                          differential_labels)
            _plot_overlay_subplot(axes[idx], datasets, config)
            if idx == n_plots - 1:
                _configure_bottom_axis(axes[idx], tick_interval)

        axes[0].legend(loc='upper right', fontsize=8)

        if subtitle:
            axes[0].set_title(axes[0].get_title(), fontsize=11, fontweight='bold', pad=20)
            axes[0].text(0.5, 1.0, subtitle,
                         transform=axes[0].transAxes, fontsize=9, style='italic',
                         ha='center', va='bottom')

        _tight_layout_with_auto_expand(fig)
        _expand_plot_width_to_page(fig)
        pyplot.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
        pyplot.close(fig)

        logger.info("Saved overlay combined plot to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write overlay combined plot: %s", exc)
        return False
    # Catch matplotlib-specific errors (ValueError, TypeError, RuntimeError, etc.)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create overlay combined plot: %s", exc)
        return False


def _render_single_overlay_plot(
    ax: Any, datasets: List[DatasetInfo], config: SingleOverlayConfig
) -> None:
    """Render a single overlay plot on the given axis."""
    diff_labels = config.differential_labels
    for idx, dataset in enumerate(datasets):
        data = dataset.stats.get(config.data_key)
        if not data:
            continue

        timestamps, values = extract_timestamps_values(data)
        label = diff_labels[idx] if idx < len(diff_labels) else dataset.label

        ax.plot(timestamps, values, color=dataset.color, linewidth=1.5,
                marker='o', markersize=3, label=label, alpha=0.8)
        ax.axvline(x=0, color='green', linestyle='--', linewidth=1.5, alpha=0.5)

        if config.show_end_marker:
            duration = dataset.stats.get("benchmark_duration_s", 0.0)
            ax.axvline(x=duration, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    max_duration = max((ds.stats.get("benchmark_duration_s", 0.0) for ds in datasets), default=0.0)
    tick_interval = _calculate_tick_interval(max_duration)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel(config.ylabel, fontsize=12)
    title_pad = 20 if config.subtitle else 6
    ax.set_title(f'{config.metric_name} Over Time', fontsize=14, fontweight='bold', pad=title_pad)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    if config.subtitle:
        ax.text(0.5, 1.0, config.subtitle,
                transform=ax.transAxes, fontsize=10, style='italic', ha='center', va='bottom')


def generate_overlay_pdf_plots(             # pylint: disable=too-many-locals
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    show_end_marker: bool = True,
    subtitle: str = None
) -> int:
    """
    Generate individual overlay plots for each metric plus a combined overlay plot.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of DatasetInfo objects containing label, stats, and color.
    output_dir : str
        Directory path where PDF files will be written.
    filename_prefix : str
        Prefix for output filenames. Default is "benchmark".
    show_end_marker : bool
        If True, show end-of-benchmark vertical lines. Default is True.
    subtitle : str, optional
        A subtitle to display below the main title (e.g., fixed options).

    Returns
    -------
    int
        Total number of plots successfully generated.

    Notes
    -----
    The legend only shows fields that differ between datasets (model names,
    llama.cpp versions, and/or option sets) for clarity.
    """
    if not datasets:
        logger.error("No datasets provided for overlay plotting.")
        return 0

    if not ensure_output_directory(output_dir):
        return 0

    differential_labels = compute_differential_labels(datasets)
    plots_created = 0
    safe_prefix = sanitize_prefix(filename_prefix)

    for data_key, metric_name, ylabel, _ in METRIC_DEFINITIONS:
        if not any(ds.stats.get(data_key) for ds in datasets):
            logger.debug("Skipping metric '%s': no data available.", data_key)
            continue

        output_path = os.path.join(output_dir, f"{safe_prefix}_{data_key}.pdf")
        config = SingleOverlayConfig(data_key, metric_name, ylabel, differential_labels,
                                     show_end_marker, subtitle, output_path)

        try:
            fig, ax = pyplot.subplots(figsize=(12, 6))
            _render_single_overlay_plot(ax, datasets, config)
            _tight_layout_with_auto_expand(fig)
            _expand_plot_width_to_page(fig)
            pyplot.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
            pyplot.close(fig)
            logger.info("Saved overlay plot to %s", output_path)
            plots_created += 1
        except (IOError, OSError) as exc:
            logger.error("Failed to write overlay plot for '%s': %s", metric_name, exc)
        # Catch matplotlib-specific errors (ValueError, TypeError, RuntimeError, etc.)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to create overlay plot for '%s': %s", metric_name, exc)

    combined_path = os.path.join(output_dir, f"{safe_prefix}_combined_resources.pdf")
    if generate_overlay_combined_plot(datasets, combined_path, show_end_marker, subtitle):
        plots_created += 1

    logger.info("Generated %d overlay plot(s) in %s", plots_created, output_dir)
    return plots_created


def _render_pdf_bar_chart(
    ax: Any,
    valid_datasets: List[DatasetInfo],
    values: List[float],
    config: BarChartConfig,
    differential_labels: List[str],
    stddev_values: List[Optional[float]] = None
) -> None:
    """Render the bar chart on the given axis with optional error bars."""
    y_positions = range(len(valid_datasets))

    # Check if we have any non-zero stddev values for error bars
    has_error_bars = stddev_values and any(s is not None and s > 0 for s in stddev_values)
    xerr = None
    if has_error_bars:
        # Replace None values with 0 for error bar plotting
        xerr = [s if s is not None else 0 for s in stddev_values]

    bars = ax.barh(y_positions, values,
                   xerr=xerr,
                   color=[ds.color for ds in valid_datasets],
                   edgecolor='black', linewidth=0.5,
                   capsize=3 if has_error_bars else 0,
                   error_kw={'elinewidth': 1, 'capthick': 1} if has_error_bars else {})

    max_val = max(values)
    # Adjust text position if error bars are present
    max_error = max(xerr) if xerr else 0
    for idx, (bar_rect, value) in enumerate(zip(bars, values)):
        width = bar_rect.get_width()
        text_offset = max_val * 0.01 + max_error
        bar_center_y = bar_rect.get_y() + bar_rect.get_height() / 2

        if has_error_bars and xerr[idx] > 0:
            # Show value on top line, ± stddev below it
            ax.text(width + text_offset, bar_center_y - 0.12,
                    f'{value:.2f}', va='center', ha='left', fontsize=9)
            ax.text(width + text_offset, bar_center_y + 0.12,
                    f'±{xerr[idx]:.2f}', va='center', ha='left', fontsize=8, color='gray')
        else:
            ax.text(width + text_offset, bar_center_y,
                    f'{value:.2f}', va='center', ha='left', fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(differential_labels, fontsize=9)
    ax.set_xlabel(config.ylabel, fontsize=12)
    ax.invert_yaxis()

    title_pad = 20 if config.subtitle else 6
    ax.set_title(f'{config.metric_name} Comparison', fontsize=14, fontweight='bold', pad=title_pad)

    if config.subtitle:
        ax.text(0.5, 1.0, config.subtitle,
                transform=ax.transAxes, fontsize=10, style='italic', ha='center', va='bottom')

    ax.grid(True, axis='x', alpha=0.3)
    # Extend x-limit to accommodate error bars and labels
    ax.set_xlim(0, max_val * 1.15 + max_error)


def _pdf_left_margin_for_labels(labels: List[str]) -> float:
    """Compute a dynamic left margin for long y-axis bar-chart labels."""
    max_len = max((len(label) for label in labels), default=0)
    return min(0.62, max(0.20, 0.16 + (max_len * 0.006)))


def _pdf_bottom_margin_for_labels(labels: List[str]) -> float:
    """Compute a dynamic bottom margin for long rotated x-axis labels."""
    max_len = max((len(label) for label in labels), default=0)
    return min(0.58, max(0.20, 0.14 + (max_len * 0.005)))


def _grouped_bar_top_ylim(max_val: float) -> float:
    """Compute y-axis upper bound with extra top headroom for legend/value labels."""
    if max_val <= 0:
        return 1.0
    return max_val * 1.35


def generate_bar_chart(datasets: List[DatasetInfo], config: BarChartConfig) -> bool:
    """
    Generate a horizontal bar chart comparing a single metric across datasets.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of DatasetInfo objects containing stats with the metric.
    config : BarChartConfig
        Configuration for the bar chart including metric_key, metric_name,
        ylabel, output_path, and subtitle.

    Returns
    -------
    bool
        True if the bar chart was successfully created; False otherwise.
    """
    if not datasets:
        logger.warning("No datasets provided for bar chart.")
        return False

    valid_datasets, values, stddev_values = extract_bar_chart_data(datasets, config.metric_key)
    if not valid_datasets:
        logger.debug("No valid data found for metric '%s'.", config.metric_key)
        return False

    if config.label_sort:
        labels_for_sort = compute_differential_labels(valid_datasets, truncate=False)
        sorted_indices = sorted(range(len(labels_for_sort)),
                                key=lambda i: labels_for_sort[i].lower())
    else:
        # Sort datasets by value
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i],
                                reverse=not config.sort_ascending)
    valid_datasets = [valid_datasets[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    stddev_values = [stddev_values[i] for i in sorted_indices]

    differential_labels = compute_differential_labels(valid_datasets, truncate=False)

    try:
        fig, ax = pyplot.subplots(figsize=(10, max(4, len(valid_datasets) * 0.6)))
        _render_pdf_bar_chart(ax, valid_datasets, values, config, differential_labels, stddev_values)
        _tight_layout_with_auto_expand(fig)
        _expand_horizontal_bar_plot_width(
            fig,
            left_margin=_pdf_left_margin_for_labels(differential_labels)
        )
        pyplot.savefig(config.output_path, format='pdf', dpi=150, bbox_inches='tight')
        pyplot.close(fig)
        logger.info("Saved bar chart to %s", config.output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write bar chart for '%s': %s", config.metric_name, exc)
        return False
    # Catch matplotlib-specific errors (ValueError, TypeError, RuntimeError, etc.)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create bar chart for '%s': %s", config.metric_name, exc)
        return False


def generate_memory_usage_bar_chart_pdf(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """Generate a grouped bar chart comparing avg and max used memory per dataset."""
    if not datasets:
        logger.warning("No datasets provided for memory usage bar chart.")
        return False

    valid_datasets = []
    avg_values = []
    max_values = []
    for dataset in datasets:
        avg_used, max_used = compute_avg_max_from_runtime_series(
            dataset.stats, 'used_memory_gb'
        )
        if isinstance(avg_used, (int, float)) and isinstance(max_used, (int, float)):
            valid_datasets.append(dataset)
            avg_values.append(float(avg_used))
            max_values.append(float(max_used))

    if not valid_datasets:
        logger.warning("No valid runtime memory stats found for memory usage bar chart.")
        return False

    if label_sort:
        labels_for_sort = compute_differential_labels(valid_datasets, truncate=False)
        sorted_indices = sorted(range(len(labels_for_sort)),
                                key=lambda i: labels_for_sort[i].lower())
    else:
        sorted_indices = sorted(range(len(avg_values)), key=lambda i: avg_values[i])
    valid_datasets = [valid_datasets[i] for i in sorted_indices]
    avg_values = [avg_values[i] for i in sorted_indices]
    max_values = [max_values[i] for i in sorted_indices]

    differential_labels = compute_differential_labels(valid_datasets, truncate=False)
    safe_prefix = sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_memory_usage_bar.pdf")

    try:
        fig, ax = pyplot.subplots(figsize=(12, max(5, len(valid_datasets) * 0.8)))

        bar_width = 0.4
        base_positions = [float(i) for i in range(len(valid_datasets))]
        avg_positions = base_positions
        max_positions = [x + bar_width for x in base_positions]

        avg_bars = ax.bar(avg_positions, avg_values, width=bar_width,
                          color='green', edgecolor='black', linewidth=0.5,
                          label='avg_used_memory_gb')
        max_bars = ax.bar(max_positions, max_values, width=bar_width,
                          color='darkgreen', edgecolor='black', linewidth=0.5,
                          label='max_used_memory_gb')

        tick_positions = [x + (bar_width / 2.0) for x in base_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(differential_labels, rotation=25, ha='right', fontsize=9)

        max_val = max(max_values) if max_values else 0.0
        text_offset = max(max_val * 0.01, 0.02)
        for bar in list(avg_bars) + list(max_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + (bar.get_width() / 2.0), height + text_offset,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        title_pad = 20 if subtitle else 6
        ax.set_title('Memory Usage Comparison (avg vs max)', fontsize=14,
                     fontweight='bold', pad=title_pad)
        if subtitle:
            ax.text(0.5, 1.0, subtitle,
                    transform=ax.transAxes, fontsize=10, style='italic',
                    ha='center', va='bottom')

        ax.set_ylabel('Memory (GiB)', fontsize=12)
        ax.set_xlabel('Benchmark Run', fontsize=12)
        ax.set_ylim(0, _grouped_bar_top_ylim(max_val))
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper left')

        _tight_layout_with_auto_expand(fig)
        _expand_plot_width_to_page(
            fig,
            bottom_margin=_pdf_bottom_margin_for_labels(differential_labels)
        )
        pyplot.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
        pyplot.close(fig)
        logger.info("Saved memory usage bar chart to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write memory usage bar chart: %s", exc)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create memory usage bar chart: %s", exc)
        return False


def generate_gpu_memory_usage_bar_chart_pdf(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """Generate a grouped bar chart comparing avg and max GPU memory per dataset."""
    if not datasets:
        logger.warning("No datasets provided for GPU memory usage bar chart.")
        return False

    valid_datasets = []
    avg_values = []
    max_values = []
    for dataset in datasets:
        avg_used, max_used = compute_avg_max_from_runtime_series(
            dataset.stats, 'gpu_memory_gb'
        )
        if isinstance(avg_used, (int, float)) and isinstance(max_used, (int, float)):
            valid_datasets.append(dataset)
            avg_values.append(float(avg_used))
            max_values.append(float(max_used))

    if not valid_datasets:
        logger.warning("No valid runtime GPU memory stats found for GPU memory usage bar chart.")
        return False

    if label_sort:
        labels_for_sort = compute_differential_labels(valid_datasets, truncate=False)
        sorted_indices = sorted(range(len(labels_for_sort)),
                                key=lambda i: labels_for_sort[i].lower())
    else:
        sorted_indices = sorted(range(len(avg_values)), key=lambda i: avg_values[i])
    valid_datasets = [valid_datasets[i] for i in sorted_indices]
    avg_values = [avg_values[i] for i in sorted_indices]
    max_values = [max_values[i] for i in sorted_indices]

    differential_labels = compute_differential_labels(valid_datasets, truncate=False)
    safe_prefix = sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_gpu_memory_usage_bar.pdf")

    try:
        fig, ax = pyplot.subplots(figsize=(12, max(5, len(valid_datasets) * 0.8)))

        bar_width = 0.4
        base_positions = [float(i) for i in range(len(valid_datasets))]
        avg_positions = base_positions
        max_positions = [x + bar_width for x in base_positions]

        avg_bars = ax.bar(avg_positions, avg_values, width=bar_width,
                          color='purple', edgecolor='black', linewidth=0.5,
                          label='avg_gpu_memory_gb')
        max_bars = ax.bar(max_positions, max_values, width=bar_width,
                          color='indigo', edgecolor='black', linewidth=0.5,
                          label='max_gpu_memory_gb')

        tick_positions = [x + (bar_width / 2.0) for x in base_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(differential_labels, rotation=25, ha='right', fontsize=9)

        max_val = max(max_values) if max_values else 0.0
        text_offset = max(max_val * 0.01, 0.02)
        for bar in list(avg_bars) + list(max_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + (bar.get_width() / 2.0), height + text_offset,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        title_pad = 20 if subtitle else 6
        ax.set_title('GPU Memory Usage Comparison (avg vs max)', fontsize=14,
                     fontweight='bold', pad=title_pad)
        if subtitle:
            ax.text(0.5, 1.0, subtitle,
                    transform=ax.transAxes, fontsize=10, style='italic',
                    ha='center', va='bottom')

        ax.set_ylabel('Memory (GiB)', fontsize=12)
        ax.set_xlabel('Benchmark Run', fontsize=12)
        ax.set_ylim(0, _grouped_bar_top_ylim(max_val))
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper left')

        _tight_layout_with_auto_expand(fig)
        _expand_plot_width_to_page(
            fig,
            bottom_margin=_pdf_bottom_margin_for_labels(differential_labels)
        )
        pyplot.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
        pyplot.close(fig)
        logger.info("Saved GPU memory usage bar chart to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write GPU memory usage bar chart: %s", exc)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create GPU memory usage bar chart: %s", exc)
        return False


def _generate_grouped_runtime_stats_bar_chart_pdf(
    datasets: List[DatasetInfo],
    series_key: str,
    avg_label: str,
    max_label: str,
    title: str,
    y_axis_label: str,
    avg_color: str,
    max_color: str,
    output_path: str,
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """Generate a grouped avg/max bar chart computed from runtime time-series data."""
    if not datasets:
        logger.warning("No datasets provided for grouped runtime stats bar chart.")
        return False

    valid_datasets = []
    avg_values = []
    max_values = []
    for dataset in datasets:
        avg_used, max_used = compute_avg_max_from_runtime_series(dataset.stats, series_key)
        if isinstance(avg_used, (int, float)) and isinstance(max_used, (int, float)):
            valid_datasets.append(dataset)
            avg_values.append(float(avg_used))
            max_values.append(float(max_used))

    if not valid_datasets:
        logger.warning("No valid runtime stats found for grouped runtime stats bar chart.")
        return False

    if label_sort:
        labels_for_sort = compute_differential_labels(valid_datasets, truncate=False)
        sorted_indices = sorted(range(len(labels_for_sort)),
                                key=lambda i: labels_for_sort[i].lower())
    else:
        sorted_indices = sorted(range(len(avg_values)), key=lambda i: avg_values[i])
    valid_datasets = [valid_datasets[i] for i in sorted_indices]
    avg_values = [avg_values[i] for i in sorted_indices]
    max_values = [max_values[i] for i in sorted_indices]

    differential_labels = compute_differential_labels(valid_datasets, truncate=False)

    try:
        fig, ax = pyplot.subplots(figsize=(12, max(5, len(valid_datasets) * 0.8)))

        bar_width = 0.4
        base_positions = [float(i) for i in range(len(valid_datasets))]
        avg_positions = base_positions
        max_positions = [x + bar_width for x in base_positions]

        avg_bars = ax.bar(avg_positions, avg_values, width=bar_width,
                          color=avg_color, edgecolor='black', linewidth=0.5,
                          label=avg_label)
        max_bars = ax.bar(max_positions, max_values, width=bar_width,
                          color=max_color, edgecolor='black', linewidth=0.5,
                          label=max_label)

        tick_positions = [x + (bar_width / 2.0) for x in base_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(differential_labels, rotation=25, ha='right', fontsize=9)

        max_val = max(max_values) if max_values else 0.0
        text_offset = max(max_val * 0.01, 0.02)
        for bar in list(avg_bars) + list(max_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + (bar.get_width() / 2.0), height + text_offset,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        title_pad = 20 if subtitle else 6
        ax.set_title(title, fontsize=14, fontweight='bold', pad=title_pad)
        if subtitle:
            ax.text(0.5, 1.0, subtitle,
                    transform=ax.transAxes, fontsize=10, style='italic',
                    ha='center', va='bottom')

        ax.set_ylabel(y_axis_label, fontsize=12)
        ax.set_xlabel('Benchmark Run', fontsize=12)
        ax.set_ylim(0, _grouped_bar_top_ylim(max_val))
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper left')

        _tight_layout_with_auto_expand(fig)
        _expand_plot_width_to_page(
            fig,
            bottom_margin=_pdf_bottom_margin_for_labels(differential_labels)
        )
        pyplot.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
        pyplot.close(fig)
        logger.info("Saved grouped runtime stats bar chart to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write grouped runtime stats bar chart: %s", exc)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create grouped runtime stats bar chart: %s", exc)
        return False


def generate_cpu_utilization_bar_chart_pdf(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """Generate a grouped bar chart comparing avg and max CPU utilization per dataset."""
    safe_prefix = sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_cpu_utilization_bar.pdf")
    return _generate_grouped_runtime_stats_bar_chart_pdf(
        datasets=datasets,
        series_key='cpu_utilization_pct',
        avg_label='avg_cpu_utilization_pct',
        max_label='max_cpu_utilization_pct',
        title='CPU Utilization Comparison (avg vs max)',
        y_axis_label='Utilization (%)',
        avg_color='steelblue',
        max_color='navy',
        output_path=output_path,
        subtitle=subtitle,
        label_sort=label_sort
    )


def generate_gpu_utilization_bar_chart_pdf(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """Generate a grouped bar chart comparing avg and max GPU utilization per dataset."""
    safe_prefix = sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_gpu_utilization_bar.pdf")
    return _generate_grouped_runtime_stats_bar_chart_pdf(
        datasets=datasets,
        series_key='gpu_utilization_pct',
        avg_label='avg_gpu_utilization_pct',
        max_label='max_gpu_utilization_pct',
        title='GPU Utilization Comparison (avg vs max)',
        y_axis_label='Utilization (%)',
        avg_color='orange',
        max_color='darkorange',
        output_path=output_path,
        subtitle=subtitle,
        label_sort=label_sort
    )


def generate_npu_memory_usage_bar_chart_pdf(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """Generate a grouped bar chart comparing avg and max NPU memory per dataset."""
    safe_prefix = sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_npu_memory_usage_bar.pdf")
    return _generate_grouped_runtime_stats_bar_chart_pdf(
        datasets=datasets,
        series_key='npu_memory_gb',
        avg_label='avg_npu_memory_gb',
        max_label='max_npu_memory_gb',
        title='NPU Memory Usage Comparison (avg vs max)',
        y_axis_label='Memory (GiB)',
        avg_color='teal',
        max_color='darkslategray',
        output_path=output_path,
        subtitle=subtitle,
        label_sort=label_sort
    )


def generate_ttft_bar_chart_pdf(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """
    Generate a bar chart comparing TTFT (Time To First Token) across datasets.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of DatasetInfo objects containing stats with ttft_ms.
    output_dir : str
        Directory path where the PDF file will be written.
    filename_prefix : str
        Prefix for the output filename. Default is "benchmark".
    subtitle : str, optional
        A subtitle to display below the main title (e.g., fixed options).

    Returns
    -------
    bool
        True if the bar chart was successfully created; False otherwise.
    """
    safe_prefix = sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_ttft_bar.pdf")
    config = BarChartConfig(
        metric_key='ttft_ms',
        metric_name='Time To First Token (TTFT)',
        ylabel='TTFT (milliseconds)',
        output_path=output_path,
        subtitle=subtitle,
        sort_ascending=True,
        label_sort=label_sort
    )
    return generate_bar_chart(datasets, config)


def generate_avg_ts_bar_chart_pdf(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """
    Generate a bar chart comparing avg_ts (tokens/sec) across datasets.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of DatasetInfo objects containing conv_metrics with avg_ts.
    output_dir : str
        Directory path where the PDF file will be written.
    filename_prefix : str
        Prefix for the output filename. Default is "benchmark".
    subtitle : str, optional
        A subtitle to display below the main title (e.g., fixed options).

    Returns
    -------
    bool
        True if the bar chart was successfully created; False otherwise.
    """
    safe_prefix = sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_avg_ts_bar.pdf")
    config = BarChartConfig(
        metric_key='avg_ts',
        metric_name='Generation Throughput (avg_ts)',
        ylabel='avg_ts (tokens/sec)',
        output_path=output_path,
        subtitle=subtitle,
        sort_ascending=False,
        label_sort=label_sort
    )
    return generate_bar_chart(datasets, config)


def generate_combined_performance_pdf(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """
    Generate a combined performance PDF with TTFT and avg_ts bar charts as subplots.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of DatasetInfo objects containing conv_metrics.
    output_dir : str
        Directory path where the PDF file will be written.
    filename_prefix : str
        Prefix for the output filename. Default is "benchmark".
    subtitle : str, optional
        A subtitle to display below the main title.

    Returns
    -------
    bool
        True if the combined plot was successfully created; False otherwise.
    """
    if not datasets:
        logger.warning("No datasets provided for combined performance chart.")
        return False

    safe_prefix = sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_combined_performance.pdf")

    # Prepare data for both metrics
    metrics = [
        ('ttft_ms', 'Time To First Token (TTFT)', 'TTFT (milliseconds)'),
        ('avg_ts', 'Generation Throughput (avg_ts)', 'avg_ts (tokens/sec)'),
    ]

    chart_data = []
    for metric_key, metric_name, ylabel in metrics:
        valid_datasets, values, stddev_values = extract_bar_chart_data(datasets, metric_key)
        if valid_datasets:
            chart_data.append((valid_datasets, values, stddev_values, metric_key, metric_name, ylabel))

    if not chart_data:
        logger.debug("No valid data found for combined performance chart.")
        return False

    try:
        n_charts = len(chart_data)
        n_datasets = max(len(cd[0]) for cd in chart_data)
        fig, axes = pyplot.subplots(n_charts, 1,
                                     figsize=(10, max(4, n_datasets * 0.6) * n_charts))
        if n_charts == 1:
            axes = [axes]

        left_margin = 0.20

        for ax, (valid_ds, values, stddev_values, metric_key, metric_name, ylabel) in zip(axes, chart_data):
            # Sort: TTFT ascending (lower is better), avg_ts descending (higher is better)
            sort_asc = metric_key != 'avg_ts'
            if label_sort:
                labels_for_sort = compute_differential_labels(valid_ds, truncate=False)
                sorted_indices = sorted(range(len(labels_for_sort)),
                                        key=lambda i: labels_for_sort[i].lower())
            else:
                sorted_indices = sorted(range(len(values)), key=lambda i: values[i],
                                        reverse=not sort_asc)
            valid_ds = [valid_ds[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            stddev_values = [stddev_values[i] for i in sorted_indices]

            differential_labels = compute_differential_labels(valid_ds, truncate=False)
            left_margin = max(left_margin, _pdf_left_margin_for_labels(differential_labels))
            config = BarChartConfig(
                metric_key=metric_key,
                metric_name=metric_name,
                ylabel=ylabel,
                output_path=output_path,
                subtitle=subtitle,
                sort_ascending=sort_asc
            )
            _render_pdf_bar_chart(ax, valid_ds, values, config, differential_labels, stddev_values)

        _tight_layout_with_auto_expand(fig)
        _expand_horizontal_bar_plot_width(fig, left_margin=left_margin)
        pyplot.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
        pyplot.close(fig)
        logger.info("Saved combined performance chart to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write combined performance chart: %s", exc)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create combined performance chart: %s", exc)
        return False


def generate_throughput_bar_chart_pdf(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """
    Generate a bar chart comparing throughput (samples/sec) across datasets.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of DatasetInfo objects containing conv_metrics with throughput_samples_per_sec.
    output_dir : str
        Directory path where the PDF file will be written.
    filename_prefix : str
        Prefix for the output filename. Default is "benchmark".
    subtitle : str, optional
        A subtitle to display below the main title.
    label_sort : bool, optional
        If True, sort bars by label instead of metric value.

    Returns
    -------
    bool
        True if the bar chart was successfully created; False otherwise.
    """
    safe_prefix = sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_throughput_bar.pdf")
    config = BarChartConfig(
        metric_key='throughput_samples_per_sec',
        metric_name='Embedding Throughput',
        ylabel='throughput (samples/sec)',
        output_path=output_path,
        subtitle=subtitle,
        sort_ascending=False,
        label_sort=label_sort
    )
    return generate_bar_chart(datasets, config)


def generate_p50_latency_bar_chart_pdf(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """
    Generate a bar chart comparing P50 batch latency (ms) across datasets.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of DatasetInfo objects containing conv_metrics with p50_batch_latency_ms.
    output_dir : str
        Directory path where the PDF file will be written.
    filename_prefix : str
        Prefix for the output filename. Default is "benchmark".
    subtitle : str, optional
        A subtitle to display below the main title.
    label_sort : bool, optional
        If True, sort bars by label instead of metric value.

    Returns
    -------
    bool
        True if the bar chart was successfully created; False otherwise.
    """
    safe_prefix = sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_p50_latency_bar.pdf")
    config = BarChartConfig(
        metric_key='p50_batch_latency_ms',
        metric_name='P50 Batch Latency',
        ylabel='P50 batch latency (milliseconds)',
        output_path=output_path,
        subtitle=subtitle,
        sort_ascending=True,
        label_sort=label_sort
    )
    return generate_bar_chart(datasets, config)


def generate_embeddings_combined_performance_pdf(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """
    Generate a combined performance PDF with throughput and P50 latency bar charts as subplots.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of DatasetInfo objects containing conv_metrics.
    output_dir : str
        Directory path where the PDF file will be written.
    filename_prefix : str
        Prefix for the output filename. Default is "benchmark".
    subtitle : str, optional
        A subtitle to display below the main title.
    label_sort : bool, optional
        If True, sort bars by label instead of metric value.

    Returns
    -------
    bool
        True if the combined plot was successfully created; False otherwise.
    """
    if not datasets:
        logger.warning("No datasets provided for embeddings combined performance chart.")
        return False

    safe_prefix = sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_embeddings_combined_performance.pdf")

    metrics = [
        ('throughput_samples_per_sec', 'Embedding Throughput', 'throughput (samples/sec)'),
        ('p50_batch_latency_ms', 'P50 Batch Latency', 'P50 batch latency (milliseconds)'),
    ]

    chart_data = []
    for metric_key, metric_name, ylabel in metrics:
        valid_datasets, values, stddev_values = extract_bar_chart_data(datasets, metric_key)
        if valid_datasets:
            chart_data.append((valid_datasets, values, stddev_values, metric_key, metric_name, ylabel))

    if not chart_data:
        logger.debug("No valid data found for embeddings combined performance chart.")
        return False

    try:
        n_charts = len(chart_data)
        n_datasets = max(len(cd[0]) for cd in chart_data)
        fig, axes = pyplot.subplots(n_charts, 1,
                                     figsize=(10, max(4, n_datasets * 0.6) * n_charts))
        if n_charts == 1:
            axes = [axes]

        left_margin = 0.20

        for ax, (valid_ds, values, stddev_values, metric_key, metric_name, ylabel) in zip(axes, chart_data):
            sort_asc = metric_key != 'throughput_samples_per_sec'
            if label_sort:
                labels_for_sort = compute_differential_labels(valid_ds, truncate=False)
                sorted_indices = sorted(range(len(labels_for_sort)),
                                        key=lambda i: labels_for_sort[i].lower())
            else:
                sorted_indices = sorted(range(len(values)), key=lambda i: values[i],
                                        reverse=not sort_asc)
            valid_ds = [valid_ds[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            stddev_values = [stddev_values[i] for i in sorted_indices]

            differential_labels = compute_differential_labels(valid_ds, truncate=False)
            left_margin = max(left_margin, _pdf_left_margin_for_labels(differential_labels))
            config = BarChartConfig(
                metric_key=metric_key,
                metric_name=metric_name,
                ylabel=ylabel,
                output_path=output_path,
                subtitle=subtitle,
                sort_ascending=sort_asc
            )
            _render_pdf_bar_chart(ax, valid_ds, values, config, differential_labels, stddev_values)

        _tight_layout_with_auto_expand(fig)
        _expand_horizontal_bar_plot_width(fig, left_margin=left_margin)
        pyplot.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
        pyplot.close(fig)
        logger.info("Saved embeddings combined performance chart to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write embeddings combined performance chart: %s", exc)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create embeddings combined performance chart: %s", exc)
        return False
