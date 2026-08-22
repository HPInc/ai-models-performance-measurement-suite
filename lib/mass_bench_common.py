#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
Shared utilities for mass benchmark scripts.

This module consolidates code that was duplicated across mass_benchy.py,
mass_lemonade_benchy.py, mass_llama_server_benchy.py, and mass_llama_bench.py.

Contents
--------
- DEFAULT_SAMPLE_INTERVAL: Default sampling cadence for resource monitoring.
- BenchmarkArgumentParser: argparse.ArgumentParser subclass with customized
  error/help behavior; scripts subclass to add their own examples.
- extract_benchy_convenience_metrics(): Maps llama-benchy JSON output to the
  standard convenience_metrics dictionary.
- parse_model_spec(): Splits a possibly comma-separated model specification
  into (server_model, benchy_model).
- setup_logging(): Configure logging level and format.
- check_python_version(): Exit early if Python version is too old.
"""
from __future__ import annotations

import argparse
import json as _json
import logging
import sys
import urllib.request

from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default sample interval for resource monitoring (seconds)
DEFAULT_SAMPLE_INTERVAL = 0.2


class BenchmarkArgumentParser(argparse.ArgumentParser):
    """Argument parser with customized error/help behavior.

    Subclasses should override ``_get_example_lines()`` to append
    script-specific usage examples to the help text.
    """

    def error(self, message):
        """Print a concise error and usage, then exit with status 2."""
        sys.stderr.write(f'\nerror: {message}\n\n')
        self.print_usage(sys.stderr)
        sys.exit(2)


def extract_benchy_convenience_metrics(
    benchy_results: Dict[str, Any],
    server_options: Optional[List[str]] = None,
    extra_benchy_options: Optional[List[str]] = None,
    model: Optional[str] = None,
    description: Optional[str] = None,
    fixed_server_options: Optional[List[str]] = None,
    fixed_benchy_options: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Extract convenience metrics from llama-benchy JSON output.

    Maps llama-benchy fields to the standard convenience_metrics format:
    - benchmarks[0]:e2e_ttft:mean => ttft_ms
    - benchmarks[0]:e2e_ttft:std => ttft_stddev_ms
    - benchmarks[0]:tg_throughput:mean => avg_ts
    - benchmarks[0]:tg_throughput:std => avg_ts_stddev
    - model => model name used for the benchmark
    - description => custom description for the benchmark run
    - options => extra benchy options (from -e, excluding -f)
    - server_options => variable server options (from -s, excluding -c)
    - fixed_options => fixed benchy options (from -f)
    - constant_server_options => constant server options (from -c)

    Args:
        benchy_results: Parsed JSON output from llama-benchy.
        server_options: Variable server options (from -s, not -c).
        extra_benchy_options: Extra benchy options (from -e, not -f).
        model: Model name used for the benchmark.
        description: Optional custom description for the benchmark run.
        fixed_server_options: Fixed server options (from -c).
        fixed_benchy_options: Fixed benchy options (from -f).

    Returns:
        Dictionary with mapped convenience metrics.
    """
    conv_metrics: Dict[str, Any] = {}

    # Get the benchmarks list and extract the first element
    benchmarks_list = benchy_results.get('benchmarks')
    if not benchmarks_list or not isinstance(benchmarks_list, list) or len(benchmarks_list) == 0:
        logger.warning("No 'benchmarks' list found in llama-benchy results")
        return conv_metrics

    benchmark = benchmarks_list[0]
    if not isinstance(benchmark, dict):
        logger.warning("First benchmark entry is not a dictionary")
        return conv_metrics

    # Extract TTFT from benchmarks[0]:e2e_ttft:mean (end-to-end time to first token)
    if 'e2e_ttft' in benchmark and isinstance(benchmark['e2e_ttft'], dict):
        e2e_ttft = benchmark['e2e_ttft']
        if 'mean' in e2e_ttft and e2e_ttft['mean'] is not None:
            conv_metrics['ttft_ms'] = round(float(e2e_ttft['mean']), 3)
        if 'std' in e2e_ttft and e2e_ttft['std'] is not None:
            conv_metrics['ttft_stddev_ms'] = round(float(e2e_ttft['std']), 3)

    # Extract avg_ts from benchmarks[0]:tg_throughput:mean (token generation throughput)
    if 'tg_throughput' in benchmark and isinstance(benchmark['tg_throughput'], dict):
        tg_tp = benchmark['tg_throughput']
        if 'mean' in tg_tp and tg_tp['mean'] is not None:
            conv_metrics['avg_ts'] = round(float(tg_tp['mean']), 2)
        if 'std' in tg_tp and tg_tp['std'] is not None:
            conv_metrics['avg_ts_stddev'] = round(float(tg_tp['std']), 2)

    # Store the model name
    if model:
        conv_metrics['model'] = model

    # Store the description
    if description:
        conv_metrics['description'] = description

    # Build server_options string from variable server options (-s only)
    # Excludes constant server options (-c)
    if server_options:
        conv_metrics['server_options'] = ' '.join(server_options)

    # Build options string from extra benchy options (-e) only
    # Variable server options (-s) are already stored in server_options
    if extra_benchy_options:
        conv_metrics['options'] = ' '.join(extra_benchy_options)

    # Store constant server options (-c) separately
    if fixed_server_options:
        conv_metrics['constant_server_options'] = ' '.join(fixed_server_options)

    # Build fixed_options string from fixed benchy options (-f) only
    if fixed_benchy_options:
        conv_metrics['fixed_options'] = ' '.join(fixed_benchy_options)

    return conv_metrics


def query_server_model(base_url: str, timeout: float = 5.0) -> Optional[str]:
    """
    Query an OpenAI-compatible chat/completion server for its model name.

    Sends a GET request to the ``/v1/models`` endpoint and returns the id
    of the first model listed in the response.

    Args:
        base_url: The base URL of the server (e.g., ``http://localhost:8080``).
        timeout: Request timeout in seconds.

    Returns:
        The model id string reported by the server, or None on failure.
    """
    url = base_url.rstrip('/')
    if not url.endswith('/v1'):
        url += '/v1'
    url += '/models'
    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode('utf-8'))
            models = data.get('data', [])
            if models and isinstance(models, list) and 'id' in models[0]:
                found_model = models[0]['id']
                logger.info("Model ID found from first element of v1/models endpoint = %s", found_model)
                if len(models) > 1:
                    logger.info("Number of models in v1/models list = %d", len(models))
                return found_model
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Failed to query server model at %s: %s", url, exc)
    return None


def parse_model_spec(model_spec: str) -> Tuple[str, Optional[str]]:
    """
    Parse a model specification which may be comma-separated.

    The model spec can be:
    - A single value: used for both server and benchy
    - Two comma-separated values: first for server, second for benchy

    Args:
        model_spec: Model specification string (possibly comma-separated).

    Returns:
        Tuple of (server_model, benchy_model) where benchy_model may be None
        if only one model was specified.
    """
    parts = [p.strip() for p in model_spec.split(',')]
    if len(parts) >= 2:
        return (parts[0], parts[1])
    return (parts[0], None)


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging level and format.

    Args:
        verbose: If True, use DEBUG level; otherwise INFO.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(message)s - (%(funcName)s in %(filename)s:%(lineno)d)",
        force=True
    )


# Minimum Python version required by this toolset.
# Python 3.10 is needed for:
#   - Union type syntax (X | Y) used in type hints (PEP 604)
#   - Dictionary merge operator (dict | dict) (PEP 584, Python 3.9+)
MIN_PYTHON_VERSION = (3, 10)


def check_python_version() -> None:
    """Exit with a message if the Python version is below the minimum."""
    if sys.version_info < MIN_PYTHON_VERSION:
        min_ver = '.'.join(str(v) for v in MIN_PYTHON_VERSION)
        cur_ver = '.'.join(str(v) for v in sys.version_info[:3])
        print(f'Python {min_ver} or later is required (current: {cur_ver}).')
        sys.exit(1)


# Power mode CLI format -> Description
POWER_MODE_MAP = {
    "performance": "Performance",
    "balanced": "Balanced",
    "power-saver": "Power Saver",
}


def iter_power_modes(power_modes: Optional[List[str]]):
    """
    Iterate over power modes, yielding (cli_name, api_name) tuples.

    If no power modes are specified, yields a single (None, None) tuple
    to indicate that the current power mode should be used without changes.

    Parameters
    ----------
    power_modes : Optional[List[str]]
        List of power mode CLI names (e.g., ['performance', 'balanced']),
        or None to use the current power mode.

    Yields
    ------
    Tuple[Optional[str], Optional[str]]
        (cli_name, api_name) for each power mode, or (None, None) if no modes specified.
    """
    if not power_modes:
        # No power modes specified - run once with current settings
        yield (None, None)
    else:
        for cli_mode in power_modes:
            api_mode = POWER_MODE_MAP.get(cli_mode)
            if api_mode:
                yield (cli_mode, api_mode)
