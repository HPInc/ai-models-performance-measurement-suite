#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
Generate interactive HTML plots using Plotly for performance statistics.

This module provides Plotly-based equivalents of the matplotlib plotting
functions in plot_pdf_stats.py. It generates interactive HTML files instead of
static PDFs, enabling zooming, panning, and hover tooltips.

High-level behavior
-------------------
- Generates interactive HTML plots for CPU, GPU, and memory utilization.
- Supports both single-dataset and multi-dataset overlay plots.
- Includes vertical markers at t=0 (worker start) and benchmark end.
- Generates TTFT and avg_ts bar charts for benchmark comparisons.

Expected environment
--------------------
- Requires `plotly` for generating interactive HTML plots.

Public API
----------
- generate_all_plots_for_browser(stats, output_dir, filename_prefix, subtitle) -> int
- generate_overlay_plots_for_browser(datasets, output_dir, filename_prefix, show_end_marker,
                                     subtitle) -> int
- generate_ttft_bar_chart_for_browser(datasets, output_dir, filename_prefix, subtitle) -> bool
- generate_avg_ts_bar_chart_for_browser(datasets, output_dir, filename_prefix, subtitle) -> bool
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, List, Optional

# Import shared utilities from plot_common module.
from plot_common import (
    DatasetInfo,
    METRIC_DEFINITIONS,
    ensure_output_directory as _ensure_output_directory,
    sanitize_prefix as _sanitize_prefix,
    compute_differential_labels as _compute_differential_labels,
    extract_bar_chart_data as _extract_bar_chart_data,
    extract_timestamps_values,
    compute_avg_max_from_runtime_series,
    filter_available_metrics,
)

logger = logging.getLogger(__name__)

# Fail fast if plotly is not available.
try:
    from plotly import graph_objects
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Please make sure plotly is installed.") from exc


def _grouped_bar_top_ylim(max_val: float) -> float:
    """Compute y-axis upper bound with extra top headroom for legend/value labels."""
    if max_val <= 0:
        return 1.0
    return max_val * 1.35


@dataclass
class PlotlyMetricConfig:
    """Configuration for a single Plotly metric plot."""
    metric_name: str
    ylabel: str
    benchmark_duration_s: float
    color: str
    subtitle: Optional[str] = None
    output_path: Optional[str] = None


@dataclass
class PlotlyOverlayConfig:
    """Configuration for overlay metric plots."""
    data_key: str
    metric_name: str
    ylabel: str
    differential_labels: List[str]
    show_end_marker: bool
    subtitle: Optional[str]
    output_path: str


@dataclass
class PlotlyBarChartConfig:
    """Configuration for Plotly bar chart generation."""
    metric_key: str
    metric_name: str
    ylabel: str
    output_path: str
    subtitle: Optional[str] = None
    sort_ascending: bool = True
    label_sort: bool = False


def _create_single_plot(
    data: List[List[Any]],
    config: PlotlyMetricConfig
) -> graph_objects.Figure:
    """
    Create a single metric Plotly figure.

    Parameters
    ----------
    data : List[List[Any]]
        Time-series data as [[timestamp, value], ...].
    config : PlotlyMetricConfig
        Configuration containing metric_name, ylabel, benchmark_duration_s,
        color, and optional subtitle.

    Returns
    -------
    graph_objects.Figure
        The Plotly figure object.
    """
    timestamps, values = extract_timestamps_values(data)

    fig = graph_objects.Figure()

    # Add the main data trace
    fig.add_trace(graph_objects.Scatter(
        x=timestamps,
        y=values,
        mode='lines+markers',
        name=config.metric_name,
        line={"color": config.color, "width": 2},
        marker={"size": 4}
    ))

    # Add vertical lines for worker start and end
    fig.add_vline(x=0, line={"color": "green", "dash": "dash", "width": 2},
                  annotation_text='Worker Start', annotation_position='top left')
    fig.add_vline(x=config.benchmark_duration_s,
                  line={"color": "red", "dash": "dash", "width": 2},
                  annotation_text='Worker End', annotation_position='top right')

    # Build title with optional subtitle
    title_text = f'<b>{config.metric_name} Over Time</b>'
    if config.subtitle:
        title_text += f'<br><sub>{config.subtitle}</sub>'

    fig.update_layout(
        title={"text": title_text, "x": 0.5, "xanchor": "center"},
        xaxis_title='Time (seconds)',
        yaxis_title=config.ylabel,
        hovermode='x unified',
        template='plotly_white',
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99}
    )

    return fig


def plot_metric_for_browser(data: List[List[Any]], config: PlotlyMetricConfig) -> bool:
    """
    Generate a single metric plot and save it as an HTML file.

    Parameters
    ----------
    data : List[List[Any]]
        Time-series data as [[timestamp, value], ...].
    config : PlotlyMetricConfig
        Configuration containing metric_name, ylabel, output_path,
        benchmark_duration_s, color, and optional subtitle.

    Returns
    -------
    bool
        True if the plot was successfully created; False otherwise.
    """
    if not data:
        logger.warning("No data provided for metric '%s'", config.metric_name)
        return False

    try:
        fig = _create_single_plot(data, config)
        fig.write_html(config.output_path)
        logger.info("Saved browser plot to %s", config.output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write browser plot for '%s': %s", config.metric_name, exc)
        return False
    # Catch Plotly-specific errors (ValueError, TypeError, etc.)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create browser plot for '%s': %s", config.metric_name, exc)
        return False


def _add_combined_subplot(
    fig: graph_objects.Figure,
    stats: Dict[str, Any],
    metric_tuple: tuple,
    row_idx: int,
    benchmark_duration_s: float
) -> None:
    """Add a single metric subplot to a combined figure."""
    data_key, metric_name, ylabel, color = metric_tuple
    data = stats[data_key]
    timestamps, values = extract_timestamps_values(data)

    fig.add_trace(graph_objects.Scatter(
        x=timestamps,
        y=values,
        mode='lines+markers',
        name=metric_name,
        line={"color": color, "width": 1.5},
        marker={"size": 3}
    ), row=row_idx, col=1)

    fig.add_vline(x=0, line={"color": "green", "dash": "dash", "width": 1.5},
                  row=row_idx, col=1)
    fig.add_vline(x=benchmark_duration_s,
                  line={"color": "red", "dash": "dash", "width": 1.5},
                  row=row_idx, col=1)
    fig.update_yaxes(title_text=ylabel, row=row_idx, col=1)


def generate_combined_plot_for_browser(
    stats: Dict[str, Any],
    output_path: str,
    subtitle: str = None
) -> bool:
    """
    Generate a single HTML file with all metrics as subplots.

    Parameters
    ----------
    stats : Dict[str, Any]
        Statistics dictionary from ResourceMonitor.monitor_process().
    output_path : str
        Full path to the output HTML file.
    subtitle : str, optional
        Subtitle to display below the title.

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
        fig = make_subplots(
            rows=n_plots, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[m[1] for m in available_metrics]
        )

        for idx, metric_tuple in enumerate(available_metrics, 1):
            _add_combined_subplot(fig, stats, metric_tuple, idx, benchmark_duration_s)

        fig.update_xaxes(title_text='Time (seconds)', row=n_plots, col=1)

        title_text = '<b>Resource Usage Over Time</b>'
        if subtitle:
            title_text += f'<br><sub>{subtitle}</sub>'

        fig.update_layout(
            title={"text": title_text, "x": 0.5, "xanchor": "center"},
            height=300 * n_plots,
            hovermode='x unified',
            template='plotly_white',
            showlegend=False
        )

        fig.write_html(output_path)
        logger.info("Saved browser combined plot to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write browser combined plot: %s", exc)
        return False
    # Catch Plotly-specific errors (ValueError, TypeError, etc.)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create browser combined plot: %s", exc)
        return False


def generate_plots_for_browser(
    stats: Dict[str, Any],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None
) -> int:
    """
    Generate HTML plots for all available metrics in the stats dictionary.

    Parameters
    ----------
    stats : Dict[str, Any]
        Statistics dictionary from ResourceMonitor.monitor_process().
    output_dir : str
        Directory path where HTML files will be written.
    filename_prefix : str
        Prefix for output filenames.
    subtitle : str, optional
        Subtitle to display below the title.

    Returns
    -------
    int
        Number of plots successfully generated.
    """
    if not stats:
        logger.error("No stats provided for plotting.")
        return 0

    benchmark_duration_s = stats.get("benchmark_duration_s", 0.0)

    if not _ensure_output_directory(output_dir):
        return 0

    plots_created = 0
    safe_prefix = _sanitize_prefix(filename_prefix)

    for data_key, metric_name, ylabel, color in METRIC_DEFINITIONS:
        data = stats.get(data_key)
        if not data:
            logger.debug("Skipping metric '%s': no data available.", data_key)
            continue

        output_path = os.path.join(output_dir, f"{safe_prefix}_{data_key}.html")
        config = PlotlyMetricConfig(
            metric_name=metric_name,
            ylabel=ylabel,
            benchmark_duration_s=benchmark_duration_s,
            color=color,
            subtitle=subtitle,
            output_path=output_path
        )
        if plot_metric_for_browser(data, config):
            plots_created += 1

    logger.info("Generated %d browser plot(s) in %s", plots_created, output_dir)
    return plots_created


def generate_all_plots_for_browser(
    stats: Dict[str, Any],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None
) -> int:
    """
    Generate all individual metric plots plus a combined plot.

    Parameters
    ----------
    stats : Dict[str, Any]
        Statistics dictionary from ResourceMonitor.monitor_process().
    output_dir : str
        Directory path where HTML files will be written.
    filename_prefix : str
        Prefix for output filenames.
    subtitle : str, optional
        Subtitle to display below the title.

    Returns
    -------
    int
        Total number of plots successfully generated.
    """
    count = generate_plots_for_browser(stats, output_dir, filename_prefix, subtitle)

    safe_prefix = _sanitize_prefix(filename_prefix)
    combined_path = os.path.join(output_dir, f"{safe_prefix}_combined_resources.html")
    if generate_combined_plot_for_browser(stats, combined_path, subtitle):
        count += 1

    return count


@dataclass
class OverlayTraceConfig:
    """Configuration for adding an overlay dataset trace."""
    data_key: str
    label: str
    show_end_marker: bool
    is_first: bool


def _add_overlay_dataset_trace(
    fig: graph_objects.Figure,
    dataset: DatasetInfo,
    config: OverlayTraceConfig
) -> None:
    """Add a single dataset trace to an overlay figure."""
    data = dataset.stats.get(config.data_key)
    if not data:
        return

    timestamps, values = extract_timestamps_values(data)

    fig.add_trace(graph_objects.Scatter(
        x=timestamps,
        y=values,
        mode='lines+markers',
        name=config.label,
        line={"color": dataset.color, "width": 1.5},
        marker={"size": 3},
        opacity=0.8
    ))

    if config.is_first:
        fig.add_vline(x=0, line={"color": "green", "dash": "dash", "width": 1.5},
                      annotation_text='Start', annotation_position='top left')

    if config.show_end_marker:
        duration = dataset.stats.get("benchmark_duration_s", 0.0)
        fig.add_vline(x=duration,
                      line={"color": dataset.color, "dash": "dot", "width": 1},
                      opacity=0.5)


def _render_overlay_metric_plot(
    datasets: List[DatasetInfo],
    config: PlotlyOverlayConfig
) -> bool:
    """Render a single overlay metric plot and save it."""
    try:
        fig = graph_objects.Figure()

        for idx, dataset in enumerate(datasets):
            if idx < len(config.differential_labels):
                label = config.differential_labels[idx]
            else:
                label = dataset.label
            trace_config = OverlayTraceConfig(
                data_key=config.data_key,
                label=label,
                show_end_marker=config.show_end_marker,
                is_first=(idx == 0)
            )
            _add_overlay_dataset_trace(fig, dataset, trace_config)

        title_text = f'<b>{config.metric_name} Over Time</b>'
        if config.subtitle:
            title_text += f'<br><sub>{config.subtitle}</sub>'

        fig.update_layout(
            title={"text": title_text, "x": 0.5, "xanchor": "center"},
            xaxis_title='Time (seconds)',
            yaxis_title=config.ylabel,
            hovermode='x unified',
            template='plotly_white',
            legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99}
        )

        fig.write_html(config.output_path)
        logger.info("Saved browser overlay plot to %s", config.output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write browser overlay plot for '%s': %s",
                     config.metric_name, exc)
        return False
    # Catch Plotly-specific errors (ValueError, TypeError, etc.)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create browser overlay plot for '%s': %s",
                     config.metric_name, exc)
        return False


def generate_overlay_plots_for_browser(
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
        Directory path where HTML files will be written.
    filename_prefix : str
        Prefix for output filenames. Default is "benchmark".
    show_end_marker : bool
        If True, show end-of-benchmark vertical lines.
    subtitle : str, optional
        Subtitle to display below the title.

    Returns
    -------
    int
        Total number of plots successfully generated.
    """
    if not datasets:
        logger.error("No datasets provided for overlay plotting.")
        return 0

    if not _ensure_output_directory(output_dir):
        return 0

    differential_labels = _compute_differential_labels(datasets)
    plots_created = 0
    safe_prefix = _sanitize_prefix(filename_prefix)

    # Generate individual metric overlay plots
    for data_key, metric_name, ylabel, _ in METRIC_DEFINITIONS:
        if not any(ds.stats.get(data_key) for ds in datasets):
            logger.debug("Skipping metric '%s': no data available.", data_key)
            continue

        output_path = os.path.join(output_dir, f"{safe_prefix}_{data_key}.html")
        config = PlotlyOverlayConfig(
            data_key=data_key,
            metric_name=metric_name,
            ylabel=ylabel,
            differential_labels=differential_labels,
            show_end_marker=show_end_marker,
            subtitle=subtitle,
            output_path=output_path
        )
        if _render_overlay_metric_plot(datasets, config):
            plots_created += 1

    # Generate combined overlay plot
    combined_path = os.path.join(output_dir, f"{safe_prefix}_combined_resources.html")
    if _generate_overlay_combined_plot(datasets, combined_path, subtitle, differential_labels):
        plots_created += 1

    logger.info("Generated %d browser overlay plot(s) in %s", plots_created, output_dir)
    return plots_created


@dataclass
class CombinedOverlayTraceConfig:
    """Configuration for adding a combined overlay dataset trace."""
    data_key: str
    label: str
    show_legend: bool
    row_idx: int


def _add_combined_overlay_dataset_trace(
    fig: graph_objects.Figure,
    dataset: DatasetInfo,
    config: CombinedOverlayTraceConfig
) -> None:
    """Add a single dataset trace to a combined overlay subplot."""
    data = dataset.stats.get(config.data_key)
    if not data:
        return

    timestamps, values = extract_timestamps_values(data)

    fig.add_trace(graph_objects.Scatter(
        x=timestamps,
        y=values,
        mode='lines+markers',
        name=config.label,
        line={"color": dataset.color, "width": 1.5},
        marker={"size": 2},
        opacity=0.8,
        showlegend=config.show_legend,
        legendgroup=config.label
    ), row=config.row_idx, col=1)


def _get_subplot_titles(datasets: List[DatasetInfo]) -> List[str]:
    """Build subplot titles from metric names for metrics with data."""
    subplot_titles = []
    for data_key, metric_name, _, _ in METRIC_DEFINITIONS:
        if any(ds.stats.get(data_key) for ds in datasets):
            subplot_titles.append(metric_name)
    return subplot_titles


@dataclass
class SubplotTracesConfig:
    """Configuration for adding subplot traces."""
    differential_labels: List[str]
    data_key: str
    ylabel: str
    plot_idx: int


def _add_subplot_traces(
    fig: graph_objects.Figure,
    datasets: List[DatasetInfo],
    config: SubplotTracesConfig
) -> None:
    """Add all dataset traces for a single subplot."""
    for ds_idx, dataset in enumerate(datasets):
        if ds_idx < len(config.differential_labels):
            label = config.differential_labels[ds_idx]
        else:
            label = dataset.label
        trace_config = CombinedOverlayTraceConfig(
            data_key=config.data_key,
            label=label,
            show_legend=(config.plot_idx == 1),
            row_idx=config.plot_idx
        )
        _add_combined_overlay_dataset_trace(fig, dataset, trace_config)

    fig.add_vline(x=0, line={"color": "green", "dash": "dash", "width": 1.5},
                  row=config.plot_idx, col=1)
    fig.update_yaxes(title_text=config.ylabel, row=config.plot_idx, col=1)


def _generate_overlay_combined_plot(
    datasets: List[DatasetInfo],
    output_path: str,
    subtitle: str,
    differential_labels: List[str]
) -> bool:
    """Generate a combined overlay plot with all metrics as subplots."""
    available_metrics = [(k, y) for k, _, y, _ in METRIC_DEFINITIONS
                         if any(ds.stats.get(k) for ds in datasets)]

    if not available_metrics:
        logger.error("No metric data available for overlay combined plot.")
        return False

    try:
        n_plots = len(available_metrics)
        subplot_titles = _get_subplot_titles(datasets)

        fig = make_subplots(
            rows=n_plots, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=subplot_titles
        )

        for plot_idx, (data_key, ylabel) in enumerate(available_metrics, 1):
            subplot_config = SubplotTracesConfig(
                differential_labels=differential_labels,
                data_key=data_key,
                ylabel=ylabel,
                plot_idx=plot_idx
            )
            _add_subplot_traces(fig, datasets, subplot_config)

        fig.update_xaxes(title_text='Time (seconds)', row=n_plots, col=1)

        title_text = '<b>Resource Usage Comparison</b>'
        if subtitle:
            title_text += f'<br><sub>{subtitle}</sub>'

        fig.update_layout(
            title={"text": title_text, "x": 0.5, "xanchor": "center"},
            height=300 * n_plots,
            hovermode='x unified',
            template='plotly_white',
            legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99}
        )

        fig.write_html(output_path)
        logger.info("Saved browser overlay combined plot to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write browser overlay combined plot: %s", exc)
        return False
    # Catch Plotly-specific errors (ValueError, TypeError, etc.)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create browser overlay combined plot: %s", exc)
        return False


def generate_ttft_bar_chart_for_browser(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """
    Generate a bar chart comparing TTFT across datasets.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of DatasetInfo objects containing stats with ttft_ms.
    output_dir : str
        Directory path where the HTML file will be written.
    filename_prefix : str
        Prefix for the output filename. Default is "benchmark".
    subtitle : str, optional
        Subtitle to display below the title.

    Returns
    -------
    bool
        True if the bar chart was successfully created; False otherwise.
    """
    safe_prefix = _sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_ttft_ms_bar.html")
    config = PlotlyBarChartConfig(
        metric_key='ttft_ms',
        metric_name='Time To First Token (TTFT)',
        ylabel='TTFT (milliseconds)',
        output_path=output_path,
        subtitle=subtitle,
        sort_ascending=True,
        label_sort=label_sort
    )
    return _generate_bar_chart(datasets, config)


def generate_memory_usage_bar_chart_for_browser(
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
        labels_for_sort = _compute_differential_labels(valid_datasets, truncate=False)
        sorted_indices = sorted(range(len(labels_for_sort)),
                                key=lambda i: labels_for_sort[i].lower())
    else:
        sorted_indices = sorted(range(len(avg_values)), key=lambda i: avg_values[i])
    valid_datasets = [valid_datasets[i] for i in sorted_indices]
    avg_values = [avg_values[i] for i in sorted_indices]
    max_values = [max_values[i] for i in sorted_indices]

    differential_labels = _compute_differential_labels(valid_datasets, truncate=False)
    safe_prefix = _sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_memory_usage_bar.html")

    try:
        fig = graph_objects.Figure()

        fig.add_trace(graph_objects.Bar(
            x=differential_labels,
            y=avg_values,
            name='avg_used_memory_gb',
            marker={"color": "green", "line": {"color": "black", "width": 1}},
            text=[f'{v:.2f}' for v in avg_values],
            textposition='outside'
        ))

        fig.add_trace(graph_objects.Bar(
            x=differential_labels,
            y=max_values,
            name='max_used_memory_gb',
            marker={"color": "darkgreen", "line": {"color": "black", "width": 1}},
            text=[f'{v:.2f}' for v in max_values],
            textposition='outside'
        ))

        title_text = '<b>Memory Usage Comparison (avg vs max)</b>'
        if subtitle:
            title_text += f'<br><sub>{subtitle}</sub>'

        max_val = max(max_values) if max_values else 0.0
        fig.update_layout(
            title={"text": title_text, "x": 0.5, "xanchor": "center"},
            xaxis_title='Benchmark Run',
            yaxis_title='Memory (GiB)',
            barmode='group',
            bargap=0.25,
            bargroupgap=0.0,
            yaxis={"range": [0, _grouped_bar_top_ylim(max_val)]},
            template='plotly_white',
            margin={"b": min(650, max(160, 100 + (max((len(lbl)
                                                          for lbl in differential_labels),
                                                         default=0) * 5)))},
            height=max(450, len(valid_datasets) * 65)
        )

        fig.write_html(output_path)
        logger.info("Saved browser memory usage bar chart to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write browser memory usage bar chart: %s", exc)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create browser memory usage bar chart: %s", exc)
        return False


def generate_gpu_memory_usage_bar_chart_for_browser(
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
        labels_for_sort = _compute_differential_labels(valid_datasets, truncate=False)
        sorted_indices = sorted(range(len(labels_for_sort)),
                                key=lambda i: labels_for_sort[i].lower())
    else:
        sorted_indices = sorted(range(len(avg_values)), key=lambda i: avg_values[i])
    valid_datasets = [valid_datasets[i] for i in sorted_indices]
    avg_values = [avg_values[i] for i in sorted_indices]
    max_values = [max_values[i] for i in sorted_indices]

    differential_labels = _compute_differential_labels(valid_datasets, truncate=False)
    safe_prefix = _sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_gpu_memory_usage_bar.html")

    try:
        fig = graph_objects.Figure()

        fig.add_trace(graph_objects.Bar(
            x=differential_labels,
            y=avg_values,
            name='avg_gpu_memory_gb',
            marker={"color": "purple", "line": {"color": "black", "width": 1}},
            text=[f'{v:.2f}' for v in avg_values],
            textposition='outside'
        ))

        fig.add_trace(graph_objects.Bar(
            x=differential_labels,
            y=max_values,
            name='max_gpu_memory_gb',
            marker={"color": "indigo", "line": {"color": "black", "width": 1}},
            text=[f'{v:.2f}' for v in max_values],
            textposition='outside'
        ))

        title_text = '<b>GPU Memory Usage Comparison (avg vs max)</b>'
        if subtitle:
            title_text += f'<br><sub>{subtitle}</sub>'

        max_val = max(max_values) if max_values else 0.0
        fig.update_layout(
            title={"text": title_text, "x": 0.5, "xanchor": "center"},
            xaxis_title='Benchmark Run',
            yaxis_title='Memory (GiB)',
            barmode='group',
            bargap=0.25,
            bargroupgap=0.0,
            yaxis={"range": [0, _grouped_bar_top_ylim(max_val)]},
            template='plotly_white',
            margin={"b": min(650, max(160, 100 + (max((len(lbl)
                                                          for lbl in differential_labels),
                                                         default=0) * 5)))},
            height=max(450, len(valid_datasets) * 65)
        )

        fig.write_html(output_path)
        logger.info("Saved browser GPU memory usage bar chart to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write browser GPU memory usage bar chart: %s", exc)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create browser GPU memory usage bar chart: %s", exc)
        return False


def _generate_grouped_runtime_stats_bar_chart_for_browser(
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
    """Generate a grouped avg/max bar chart from runtime time-series data for browser output."""
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
        labels_for_sort = _compute_differential_labels(valid_datasets, truncate=False)
        sorted_indices = sorted(range(len(labels_for_sort)),
                                key=lambda i: labels_for_sort[i].lower())
    else:
        sorted_indices = sorted(range(len(avg_values)), key=lambda i: avg_values[i])
    valid_datasets = [valid_datasets[i] for i in sorted_indices]
    avg_values = [avg_values[i] for i in sorted_indices]
    max_values = [max_values[i] for i in sorted_indices]

    differential_labels = _compute_differential_labels(valid_datasets, truncate=False)

    try:
        fig = graph_objects.Figure()

        fig.add_trace(graph_objects.Bar(
            x=differential_labels,
            y=avg_values,
            name=avg_label,
            marker={"color": avg_color, "line": {"color": "black", "width": 1}},
            text=[f'{v:.2f}' for v in avg_values],
            textposition='outside'
        ))

        fig.add_trace(graph_objects.Bar(
            x=differential_labels,
            y=max_values,
            name=max_label,
            marker={"color": max_color, "line": {"color": "black", "width": 1}},
            text=[f'{v:.2f}' for v in max_values],
            textposition='outside'
        ))

        title_text = f'<b>{title}</b>'
        if subtitle:
            title_text += f'<br><sub>{subtitle}</sub>'

        max_val = max(max_values) if max_values else 0.0
        fig.update_layout(
            title={"text": title_text, "x": 0.5, "xanchor": "center"},
            xaxis_title='Benchmark Run',
            yaxis_title=y_axis_label,
            barmode='group',
            bargap=0.25,
            bargroupgap=0.0,
            yaxis={"range": [0, _grouped_bar_top_ylim(max_val)]},
            template='plotly_white',
            margin={"b": min(650, max(160, 100 + (max((len(lbl)
                                                          for lbl in differential_labels),
                                                         default=0) * 5)))},
            height=max(450, len(valid_datasets) * 65)
        )

        fig.write_html(output_path)
        logger.info("Saved browser grouped runtime stats bar chart to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write browser grouped runtime stats bar chart: %s", exc)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create browser grouped runtime stats bar chart: %s", exc)
        return False


def generate_cpu_utilization_bar_chart_for_browser(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """Generate a grouped bar chart comparing avg and max CPU utilization per dataset."""
    safe_prefix = _sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_cpu_utilization_bar.html")
    return _generate_grouped_runtime_stats_bar_chart_for_browser(
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


def generate_gpu_utilization_bar_chart_for_browser(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """Generate a grouped bar chart comparing avg and max GPU utilization per dataset."""
    safe_prefix = _sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_gpu_utilization_bar.html")
    return _generate_grouped_runtime_stats_bar_chart_for_browser(
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


def generate_npu_memory_usage_bar_chart_for_browser(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """Generate a grouped bar chart comparing avg and max NPU memory per dataset."""
    safe_prefix = _sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_npu_memory_usage_bar.html")
    return _generate_grouped_runtime_stats_bar_chart_for_browser(
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


def generate_avg_ts_bar_chart_for_browser(
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
        Directory path where the HTML file will be written.
    filename_prefix : str
        Prefix for the output filename. Default is "benchmark".
    subtitle : str, optional
        Subtitle to display below the title.

    Returns
    -------
    bool
        True if the bar chart was successfully created; False otherwise.
    """
    safe_prefix = _sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_avg_ts_bar.html")
    config = PlotlyBarChartConfig(
        metric_key='avg_ts',
        metric_name='Generation Throughput (avg_ts)',
        ylabel='avg_ts (tokens/sec)',
        output_path=output_path,
        subtitle=subtitle,
        sort_ascending=False,
        label_sort=label_sort
    )
    return _generate_bar_chart(datasets, config)


def generate_combined_performance_for_browser(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """
    Generate a combined performance HTML plot with TTFT and avg_ts bar charts as subplots.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of DatasetInfo objects containing conv_metrics.
    output_dir : str
        Directory path where the HTML file will be written.
    filename_prefix : str
        Prefix for the output filename. Default is "benchmark".
    subtitle : str, optional
        Subtitle to display below the title.

    Returns
    -------
    bool
        True if the combined plot was successfully created; False otherwise.
    """
    if not datasets:
        logger.warning("No datasets provided for combined performance chart.")
        return False

    safe_prefix = _sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_combined_performance.html")

    metrics = [
        ('ttft_ms', 'Time To First Token (TTFT)', 'TTFT (milliseconds)'),
        ('avg_ts', 'Generation Throughput (avg_ts)', 'avg_ts (tokens/sec)'),
    ]

    chart_data = []
    for metric_key, metric_name, ylabel in metrics:
        valid_datasets, values, stddev_values = _extract_bar_chart_data(datasets, metric_key)
        if valid_datasets:
            chart_data.append((valid_datasets, values, stddev_values, metric_key, metric_name, ylabel))

    if not chart_data:
        logger.debug("No valid data found for combined performance chart.")
        return False

    try:
        n_charts = len(chart_data)
        subplot_titles = [cd[4] + ' Comparison' for cd in chart_data]
        fig = make_subplots(
            rows=n_charts, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15
        )

        for row_idx, (valid_ds, values, stddev_values, metric_key, metric_name, ylabel) in enumerate(chart_data, 1):
            # Sort: TTFT ascending (lower is better), avg_ts descending (higher is better)
            sort_asc = metric_key != 'avg_ts'
            if label_sort:
                labels_for_sort = _compute_differential_labels(valid_ds, truncate=False)
                sorted_indices = sorted(range(len(labels_for_sort)),
                                        key=lambda i: labels_for_sort[i].lower())
            else:
                sorted_indices = sorted(range(len(values)), key=lambda i: values[i],
                                        reverse=not sort_asc)
            valid_ds = [valid_ds[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            stddev_values = [stddev_values[i] for i in sorted_indices]

            differential_labels = _compute_differential_labels(valid_ds, truncate=False)

            has_error_bars = stddev_values and any(s is not None and s > 0 for s in stddev_values)
            error_x = None
            if has_error_bars:
                error_array = [s if s is not None else 0 for s in stddev_values]
                error_x = {"type": "data", "array": error_array}

            fig.add_trace(graph_objects.Bar(
                y=differential_labels,
                x=values,
                orientation='h',
                marker={
                    "color": [ds.color for ds in valid_ds],
                    "line": {"color": "black", "width": 1}
                },
                text=None if has_error_bars else [f'{v:.2f}' for v in values],
                textposition='outside' if not has_error_bars else None,
                error_x=error_x,
                showlegend=False
            ), row=row_idx, col=1)

            fig.update_xaxes(title_text=ylabel, row=row_idx, col=1)
            fig.update_yaxes(autorange="reversed", automargin=True, row=row_idx, col=1)

        title_text = '<b>Performance Comparison</b>'
        if subtitle:
            title_text += f'<br><sub>{subtitle}</sub>'

        n_datasets = max(len(cd[0]) for cd in chart_data)
        fig.update_layout(
            title={"text": title_text, "x": 0.5, "xanchor": "center"},
            height=max(400, n_datasets * 50) * n_charts,
            template='plotly_white',
            margin={"l": min(650, max(200, 120 + (max((len(lbl)
                                                          for cd in chart_data
                                                          for lbl in _compute_differential_labels(cd[0], truncate=False)),
                                                         default=0) * 7)))}
        )

        fig.write_html(output_path)
        logger.info("Saved combined performance chart to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write combined performance chart: %s", exc)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create combined performance chart: %s", exc)
        return False


def generate_throughput_bar_chart_for_browser(
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
        Directory path where the HTML file will be written.
    filename_prefix : str
        Prefix for the output filename. Default is "benchmark".
    subtitle : str, optional
        Subtitle to display below the title.
    label_sort : bool, optional
        If True, sort bars by label instead of metric value.

    Returns
    -------
    bool
        True if the bar chart was successfully created; False otherwise.
    """
    safe_prefix = _sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_throughput_bar.html")
    config = PlotlyBarChartConfig(
        metric_key='throughput_samples_per_sec',
        metric_name='Embedding Throughput',
        ylabel='throughput (samples/sec)',
        output_path=output_path,
        subtitle=subtitle,
        sort_ascending=False,
        label_sort=label_sort
    )
    return _generate_bar_chart(datasets, config)


def generate_p50_latency_bar_chart_for_browser(
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
        Directory path where the HTML file will be written.
    filename_prefix : str
        Prefix for the output filename. Default is "benchmark".
    subtitle : str, optional
        Subtitle to display below the title.
    label_sort : bool, optional
        If True, sort bars by label instead of metric value.

    Returns
    -------
    bool
        True if the bar chart was successfully created; False otherwise.
    """
    safe_prefix = _sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_p50_latency_bar.html")
    config = PlotlyBarChartConfig(
        metric_key='p50_batch_latency_ms',
        metric_name='P50 Batch Latency',
        ylabel='P50 batch latency (milliseconds)',
        output_path=output_path,
        subtitle=subtitle,
        sort_ascending=True,
        label_sort=label_sort
    )
    return _generate_bar_chart(datasets, config)


def generate_embeddings_combined_performance_for_browser(
    datasets: List[DatasetInfo],
    output_dir: str,
    filename_prefix: str = "benchmark",
    subtitle: str = None,
    label_sort: bool = False
) -> bool:
    """
    Generate a combined performance HTML plot with throughput and P50 latency bar charts as subplots.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of DatasetInfo objects containing conv_metrics.
    output_dir : str
        Directory path where the HTML file will be written.
    filename_prefix : str
        Prefix for the output filename. Default is "benchmark".
    subtitle : str, optional
        Subtitle to display below the title.
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

    safe_prefix = _sanitize_prefix(filename_prefix)
    output_path = os.path.join(output_dir, f"{safe_prefix}_embeddings_combined_performance.html")

    metrics = [
        ('throughput_samples_per_sec', 'Embedding Throughput', 'throughput (samples/sec)'),
        ('p50_batch_latency_ms', 'P50 Batch Latency', 'P50 batch latency (milliseconds)'),
    ]

    chart_data = []
    for metric_key, metric_name, ylabel in metrics:
        valid_datasets, values, stddev_values = _extract_bar_chart_data(datasets, metric_key)
        if valid_datasets:
            chart_data.append((valid_datasets, values, stddev_values, metric_key, metric_name, ylabel))

    if not chart_data:
        logger.debug("No valid data found for embeddings combined performance chart.")
        return False

    try:
        n_charts = len(chart_data)
        subplot_titles = [cd[4] + ' Comparison' for cd in chart_data]
        fig = make_subplots(
            rows=n_charts, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15
        )

        for row_idx, (valid_ds, values, stddev_values, metric_key, metric_name, ylabel) in enumerate(chart_data, 1):
            sort_asc = metric_key != 'throughput_samples_per_sec'
            if label_sort:
                labels_for_sort = _compute_differential_labels(valid_ds, truncate=False)
                sorted_indices = sorted(range(len(labels_for_sort)),
                                        key=lambda i: labels_for_sort[i].lower())
            else:
                sorted_indices = sorted(range(len(values)), key=lambda i: values[i],
                                        reverse=not sort_asc)
            valid_ds = [valid_ds[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            stddev_values = [stddev_values[i] for i in sorted_indices]

            differential_labels = _compute_differential_labels(valid_ds, truncate=False)

            has_error_bars = stddev_values and any(s is not None and s > 0 for s in stddev_values)
            error_x = None
            if has_error_bars:
                error_array = [s if s is not None else 0 for s in stddev_values]
                error_x = {"type": "data", "array": error_array}

            fig.add_trace(graph_objects.Bar(
                y=differential_labels,
                x=values,
                orientation='h',
                marker={
                    "color": [ds.color for ds in valid_ds],
                    "line": {"color": "black", "width": 1}
                },
                text=None if has_error_bars else [f'{v:.2f}' for v in values],
                textposition='outside' if not has_error_bars else None,
                error_x=error_x,
                showlegend=False
            ), row=row_idx, col=1)

            fig.update_xaxes(title_text=ylabel, row=row_idx, col=1)
            fig.update_yaxes(autorange="reversed", automargin=True, row=row_idx, col=1)

        title_text = '<b>Embeddings Performance Comparison</b>'
        if subtitle:
            title_text += f'<br><sub>{subtitle}</sub>'

        n_datasets = max(len(cd[0]) for cd in chart_data)
        fig.update_layout(
            title={"text": title_text, "x": 0.5, "xanchor": "center"},
            height=max(400, n_datasets * 50) * n_charts,
            template='plotly_white',
            margin={"l": min(650, max(200, 120 + (max((len(lbl)
                                                          for cd in chart_data
                                                          for lbl in _compute_differential_labels(cd[0], truncate=False)),
                                                         default=0) * 7)))}
        )

        fig.write_html(output_path)
        logger.info("Saved embeddings combined performance chart to %s", output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write embeddings combined performance chart: %s", exc)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create embeddings combined performance chart: %s", exc)
        return False


def _render_browser_bar_chart(
    fig: graph_objects.Figure,
    valid_datasets: List[DatasetInfo],
    values: List[float],
    differential_labels: List[str],
    config: PlotlyBarChartConfig,
    stddev_values: List[Optional[float]] = None
) -> None:
    """Render bar chart content onto the figure with optional error bars."""
    # Check if we have any non-zero stddev values for error bars
    has_error_bars = stddev_values and any(s is not None and s > 0 for s in stddev_values)
    error_x = None
    error_array = None
    if has_error_bars:
        # Replace None values with 0 for error bar plotting
        error_array = [s if s is not None else 0 for s in stddev_values]
        error_x = {"type": "data", "array": error_array}

    # When error bars are present, use annotations for text positioning
    # to place text after the error bar, not just after the bar
    fig.add_trace(graph_objects.Bar(
        y=differential_labels,
        x=values,
        orientation='h',
        marker={
            "color": [ds.color for ds in valid_datasets],
            "line": {"color": "black", "width": 1}
        },
        text=None if has_error_bars else [f'{v:.2f}' for v in values],
        textposition='outside' if not has_error_bars else None,
        error_x=error_x
    ))

    # Add annotations for text labels positioned after error bars
    if has_error_bars:
        max_val = max(values)
        for i, (val, label) in enumerate(zip(values, differential_labels)):
            err = error_array[i] if error_array else 0
            fig.add_annotation(
                x=val + err + max_val * 0.02,
                y=label,
                text=f'{val:.2f}',
                showarrow=False,
                xanchor='left',
                font={"size": 12}
            )

    title_text = f'<b>{config.metric_name} Comparison</b>'
    if config.subtitle:
        title_text += f'<br><sub>{config.subtitle}</sub>'

    fig.update_layout(
        title={"text": title_text, "x": 0.5, "xanchor": "center"},
        xaxis_title=config.ylabel,
        yaxis={"autorange": "reversed", "automargin": True},
        template='plotly_white',
        margin={"l": min(650, max(200, 120 + (max((len(lbl)
                                                      for lbl in differential_labels),
                                                     default=0) * 7)))},
        height=max(400, len(valid_datasets) * 50)
    )


def _generate_bar_chart(datasets: List[DatasetInfo], config: PlotlyBarChartConfig) -> bool:
    """Generate a horizontal bar chart for the specified metric."""
    if not datasets:
        logger.warning("No datasets provided for bar chart.")
        return False

    valid_datasets, values, stddev_values = _extract_bar_chart_data(datasets, config.metric_key)

    if not valid_datasets:
        logger.debug("No valid data found for metric '%s'.", config.metric_key)
        return False

    if config.label_sort:
        labels_for_sort = _compute_differential_labels(valid_datasets, truncate=False)
        sorted_indices = sorted(range(len(labels_for_sort)),
                                key=lambda i: labels_for_sort[i].lower())
    else:
        # Sort datasets by value
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i],
                                reverse=not config.sort_ascending)
    valid_datasets = [valid_datasets[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    stddev_values = [stddev_values[i] for i in sorted_indices]

    differential_labels = _compute_differential_labels(valid_datasets, truncate=False)

    try:
        fig = graph_objects.Figure()
        _render_browser_bar_chart(fig, valid_datasets, values, differential_labels, config,
                                  stddev_values)
        fig.write_html(config.output_path)
        logger.info("Saved browser bar chart to %s", config.output_path)
        return True

    except (IOError, OSError) as exc:
        logger.error("Failed to write browser bar chart for '%s': %s",
                     config.metric_name, exc)
        return False
    # Catch Plotly-specific errors (ValueError, TypeError, etc.)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create browser bar chart for '%s': %s",
                     config.metric_name, exc)
        return False

