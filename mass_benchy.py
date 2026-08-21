#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
Run llama-benchy benchmarks against one or more external endpoints.

This script runs llama-benchy across multiple endpoint URLs and llama-benchy
parameter sets, capturing benchmark results for later analysis.  Unlike
mass_llama_server_benchy.py it does **not** launch llama-server — the
endpoints are assumed to be already running and reachable.

High-level behavior
-------------------
- Iterates over endpoint URLs (-u) and llama-benchy parameter sets (-e).
- Runs llama-benchy against each endpoint with the specified parameters.
- Monitors system resources (CPU, RAM, GPU) during each llama-benchy run.
- Saves JSON benchmark results with runtime_stats and convenience_metrics.

Key design choices
------------------
- Supports multiple option sets via ``-e`` flags for llama-benchy runs.
- Fixed options via ``-f`` are applied to all llama-benchy runs.
- Model parameter (``-m``) specifies the model name passed to llama-benchy.
- Resource monitoring via ResourceMonitor captures CPU, RAM, and GPU usage.

Expected environment
--------------------
- Requires ``psutil`` for process management.
- Requires library modules: platform_support, dashed_options, resource_monitor.
- Supports Windows and Linux, with optional Windows-only power mode changes.

Imports
-------
- argparse: CLI argument parsing.
- json: Serialize combined results to JSON files.
- logging: Structured logging throughout the module.
- multiprocessing: Worker processes for resource monitoring.
- os, sys: Filesystem and path manipulation.
- platform: Detect Windows environment.
- re: Regular expression helpers.
- socket: Retrieve hostname for output filenames.
- subprocess: Launch llama-benchy process.
- textwrap: Format help text in argument parser.
- typing: Type annotations for readability and tooling support.
- urllib.parse: Parse endpoint URLs for filename generation.
"""
import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import textwrap

from typing import List, Optional
from urllib.parse import urlparse

# Configure a module-level logger. The main() routine will set the global
# logging level and format via logging.basicConfig().
logger = logging.getLogger(__name__)

# Pylint assumes that multiline comments should only be used for file
# or function doc strings. I profoundly disagree.
# pylint: disable=pointless-string-statement

"""
The Pylint warnings logging-not-lazy and consider-using-f-string conflict
because they represent opposing best practices in different contexts:
lazy logging for logging specific performance versus the general preference
for f-strings for readability and common performance. Since we will be
using logging exclusively instead of print, we will use lazy logging
instead of f-strings.
"""
# pylint: disable=consider-using-f-string

# Compute the directory containing this script to make relative imports robust.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Look in the lib directory for library files.
# Prepend our local ./lib to sys.path so local support modules can be imported
# without requiring site-wide installation.
lib_dir = os.path.join(script_dir, 'lib')
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

# Import platform-specific helpers for directory creation.
# Keep failures explicit for easier troubleshooting.
try:
    from platform_support import (
        create_output_directory_tree,
        get_nonconflicting_output_path,
        extract_variant_number_from_path
    )
except ImportError:
    raise RuntimeError("Cannot find platform_support module.") from None

# Import option parsing utilities
# that are provided as free-form text in arguments.
try:
    from dashed_options import (
        create_options_list, iter_option_sets, combine_options, preprocess_dash_value_args
    )
except ImportError:
    raise RuntimeError("Cannot find dashed_options module.") from None

try:
    from power_config import get_power_profile, get_power_mode, set_power_mode
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing power_config.py library file.") from exc
        
# Import shared benchmark utilities to avoid code duplication across mass_* scripts.
try:
    from mass_bench_common import (
        DEFAULT_SAMPLE_INTERVAL,
        BenchmarkArgumentParser,
        POWER_MODE_MAP,
        iter_power_modes as _iter_power_modes,
        extract_benchy_convenience_metrics as _extract_convenience_metrics,
        query_server_model as _query_server_model,
        setup_logging,
        check_python_version,
    )
except ImportError:
    raise RuntimeError("Cannot find mass_bench_common module.") from None

# Lazy import ResourceMonitor to avoid circular dependency.
# Import happens inside functions that need it.


def _sanitize_url_for_filename(url: str) -> str:
    """
    Convert a URL into a string safe for use in filenames.

    Extracts the hostname and port from the URL, replacing characters
    that are problematic in filenames.

    Args:
        url: The endpoint URL (e.g., "http://localhost:8080/v1").

    Returns:
        A sanitized string suitable for inclusion in a filename
        (e.g., "localhost_8080").
    """
    parsed = urlparse(url)
    host = parsed.hostname or 'unknown'
    port = parsed.port
    if port:
        return '%s_%d' % (host, port)
    return host


def _build_output_filename(
    hostname: str,
    endpoint_url: str,
    model: Optional[str],
    extra_benchy_options: Optional[List[str]] = None,
    power_mode_label: Optional[str] = None
) -> str:
    """
    Build a descriptive output filename from benchmark parameters.

    The filename includes the local hostname, sanitized endpoint URL,
    and extra benchy options used for the run.

    Args:
        hostname: The machine hostname.
        endpoint_url: The endpoint URL being benchmarked.
        model: Model name passed to llama-benchy (may be None).
        extra_benchy_options: Extra benchy options from -e (may be None).
        power_mode_label: Optional shortened power mode name for filename.

    Returns:
        The constructed filename with .json extension.
    """
    parts = ["endpoint_" + hostname]

    # Append sanitized endpoint identifier
    endpoint_id = _sanitize_url_for_filename(endpoint_url)
    if endpoint_id:
        parts.append(endpoint_id)

    # Append extra benchy options (if any)
    if extra_benchy_options:
        for option in extra_benchy_options:
            # Clean up option for filename (remove leading dashes)
            clean_opt = option.lstrip('-')
            if clean_opt:
                parts.append(clean_opt)

    # Append power mode label (if specified)
    if power_mode_label:
        parts.append(power_mode_label)

    # Join parts with underscore
    filename = '_'.join(parts) + '.json'
    return filename


# _extract_convenience_metrics is imported from mass_bench_common
# (as extract_benchy_convenience_metrics) and aliased above.


def _worker_llama_benchy(
    cmd: List[str],
    temp_output_path: str,
    verbose: bool,
    result_queue,
) -> None:
    """Run llama-benchy in a child process and push results to a queue."""
    import logging as _logging  # pylint: disable=import-outside-toplevel

    log_level = _logging.DEBUG if verbose else _logging.INFO
    _logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(message)s - (%(funcName)s in %(filename)s:%(lineno)d)",
        force=True
    )

    result = {
        'success': False,
        'temp_output_path': temp_output_path,
        'stdout': '',
        'stderr': '',
        'returncode': -1
    }

    try:
        logger.info("Running llama-benchy command: %s", ' '.join(cmd))

        proc_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        result['stdout'] = proc_result.stdout
        result['stderr'] = proc_result.stderr
        result['returncode'] = proc_result.returncode

        if proc_result.returncode == 0:
            result['success'] = True
            if verbose:
                logger.debug("llama-benchy stdout: %s", proc_result.stdout)
        else:
            logger.error("llama-benchy failed with return code %d", proc_result.returncode)
            logger.error("stdout: %s", proc_result.stdout)
            logger.error("stderr: %s", proc_result.stderr)
    except FileNotFoundError:
        logger.error("llama-benchy not found. Ensure it is installed and in PATH.")
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to run llama-benchy: %s", exc)

    result_queue.put(result)


def run_llama_benchy_with_monitoring(
    endpoint_url: str,
    model: Optional[str],
    benchy_options: List[str],
    output_dir: str,
    hostname: str,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    extra_benchy_options: Optional[List[str]] = None,
    verbose: bool = False,
    power_mode_label: Optional[str] = None,
    fixed_benchy_options: Optional[List[str]] = None,
    description: Optional[str] = None
) -> Optional[str]:
    """
    Run llama-benchy with resource monitoring and save combined results.

    This function wraps llama-benchy execution with ResourceMonitor to capture
    CPU, RAM, and GPU usage during the benchmark. Results are saved as a JSON
    file containing three elements:
    - llama_benchy_results: Raw output from llama-benchy
    - convenience_metrics: Mapped metrics (ttft_ms, avg_ts, options)
    - runtime_stats: Resource monitoring data

    Args:
        endpoint_url: URL of the endpoint to benchmark.
        model: Optional model name to pass to llama-benchy.
        benchy_options: Additional command-line options for llama-benchy.
        output_dir: Directory to save benchmark results.
        hostname: Machine hostname for filename.
        sample_interval: Seconds between resource monitor samples.
        extra_benchy_options: Extra benchy options from -e (for filename/options).
        verbose: If True, show verbose output.
        power_mode_label: Optional shortened power mode name for filename.
        fixed_benchy_options: Fixed benchy options from -f.

    Returns:
        Path to the output JSON file if successful, None otherwise.
    """
    # Lazy import ResourceMonitor to avoid circular dependency
    # pylint: disable=import-outside-toplevel
    import tempfile
    from resource_monitor import ResourceMonitor

    # Generate descriptive output filename
    output_filename = _build_output_filename(
        hostname, endpoint_url, model, extra_benchy_options,
        power_mode_label=power_mode_label
    )
    output_path = os.path.join(output_dir, output_filename)
    output_path = get_nonconflicting_output_path(output_path)

    # Create a temporary file for llama-benchy to write its JSON output
    # We use delete=False so we can read the file after the subprocess completes
    temp_fd, temp_output_path = tempfile.mkstemp(suffix='.json', prefix='llama_benchy_')
    os.close(temp_fd)  # Close the file descriptor; llama-benchy will write to it

    try:
        logger.debug("Using endpoint URL: %s", endpoint_url)

        # Construct the llama-benchy command with --save-result to temp file
        cmd = ['llama-benchy', '--base-url', endpoint_url, '--format', 'json',
               '--save-result', temp_output_path]

        if model:
            cmd.extend(['--model', model])

        # Add user-specified options
        cmd.extend(benchy_options)

        logger.info("Running llama-benchy with monitoring: %s", ' '.join(cmd))

        # Run llama-benchy with resource monitoring
        worker_result, runtime_stats, system_info = ResourceMonitor.monitor_process(
            worker_fn=_worker_llama_benchy,
            worker_args=(cmd, temp_output_path, verbose),
            sample_interval_s=sample_interval,
            normalize_resource_data=False
        )

        if not worker_result or not worker_result.get('success'):
            logger.error("llama-benchy failed")
            return None

        # Read the llama-benchy JSON output from the temp file
        if not os.path.isfile(temp_output_path):
            logger.error("llama-benchy did not create output file: %s", temp_output_path)
            return None

        try:
            with open(temp_output_path, 'r', encoding='utf-8') as f:
                benchy_results = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse llama-benchy JSON output: %s", exc)
            return None

        # Extract convenience metrics from llama-benchy results
        conv_metrics = _extract_convenience_metrics(
            benchy_results, extra_benchy_options=extra_benchy_options,
            model=model,
            description=description,
            fixed_benchy_options=fixed_benchy_options
        )
        conv_metrics['variant'] = extract_variant_number_from_path(output_path)

        # Query the server for its actual model name
        reported_model = _query_server_model(endpoint_url)
        if reported_model:
            conv_metrics['model'] = reported_model

        # Build the combined JSON structure (same format as mass_llama_bench.py)
        combined_results = {
            'llama_benchy_results': benchy_results,
            'convenience_metrics': conv_metrics,
            'runtime_stats': runtime_stats,
            'system_info': system_info
        }

        # Write combined JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2)

        logger.info("Benchmark results saved to: %s", output_path)

        return output_path

    except RuntimeError as exc:
        logger.error("Resource monitoring failed: %s", exc)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to run llama-benchy with monitoring: %s", exc)
        return None
    finally:
        # Clean up the temporary file
        if os.path.isfile(temp_output_path):
            try:
                os.remove(temp_output_path)
            except OSError:
                pass  # Ignore cleanup errors


def _run_benchmarks_for_endpoint(
    endpoint_url: str,
    model: Optional[str],
    fixed_benchy_options: List[str],
    extra_benchy_option_sets: Optional[List[str]],
    output_dir: str,
    hostname: str,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    verbose: bool = False,
    power_mode_label: Optional[str] = None,
    description: Optional[str] = None
) -> List[str]:
    """
    Run all llama-benchy configurations for a single endpoint.

    Args:
        endpoint_url: URL of the endpoint to benchmark.
        model: Optional model name for llama-benchy.
        fixed_benchy_options: Benchy options applied to all runs.
        extra_benchy_option_sets: List of additional benchy option strings.
        output_dir: Directory to save benchmark results.
        hostname: Machine hostname for filenames.
        sample_interval: Seconds between resource monitor samples.
        verbose: If True, show verbose output.
        power_mode_label: Optional shortened power mode name for filename.

    Returns:
        List of paths to generated JSON files.
    """
    written_json_files: List[str] = []

    # Iterate over all llama-benchy option sets
    for extra_options in iter_option_sets(extra_benchy_option_sets):
        combined_benchy_options = combine_options(fixed_benchy_options, extra_options) or []

        output_file = run_llama_benchy_with_monitoring(
            endpoint_url=endpoint_url,
            model=model,
            benchy_options=combined_benchy_options,
            output_dir=output_dir,
            hostname=hostname,
            sample_interval=sample_interval,
            extra_benchy_options=extra_options,
            verbose=verbose,
            power_mode_label=power_mode_label,
            fixed_benchy_options=fixed_benchy_options,
            description=description
        )

        if output_file:
            written_json_files.append(output_file)

    return written_json_files


def benchmark_endpoints(parser_args) -> bool:
    """
    Run benchmarks across all selected endpoints, models, and option sets.

    Iterates over the Cartesian product of (endpoint × model × benchy-params).
    Each combination runs llama-benchy against the endpoint and records the
    results with resource monitoring.

    Args:
        parser_args:
            Namespace from ``argparse.parse_args()`` with fields:
              - ``output_dir``: Output directory for benchmark results.
              - ``url``: One or more endpoint URLs.
              - ``model``: Optional model name(s) for llama-benchy.
              - ``extra_options``: llama-benchy option sets.
              - ``fixed_options``: llama-benchy options for all runs.
              - ``sample_interval``: Seconds between resource monitor samples.

    Returns:
        bool: True if at least one benchmark succeeded, False otherwise.
    """
    # Ensure the output directory exists (and any intermediate parents).
    if not create_output_directory_tree(parser_args.output_dir):
        logger.error('Unable to create output directory tree.')
        return False

    # Parse fixed benchy options
    fixed_benchy_options = []
    if parser_args.fixed_options:
        fixed_benchy_options = create_options_list(parser_args.fixed_options)
        logger.debug('Fixed benchy options: %s', fixed_benchy_options)

    # Get hostname for filenames
    hostname = socket.gethostname()

    # Build the list of models (may be None if not specified)
    models = parser_args.model if parser_args.model else [None]

    # Collect all JSON files written during this invocation
    written_json_files: List[str] = []

    # Get power modes (if any)
    power_modes = getattr(parser_args, 'power_mode', None)

    # Iterate: power-modes × endpoints × models × benchy-params
    for power_cli, power_api in _iter_power_modes(power_modes):
        # Set power mode if specified
        if power_api:
            logger.info("Setting power mode to: %s", power_api)
            if not set_power_mode(power_api):
                logger.warning("Failed to set power mode to %s. Continuing anyway.",
                               power_api)
            else:
                current_mode = get_power_mode()
                logger.debug("Current power mode: %s", current_mode)

        # Build power mode label for filenames
        power_mode_label = POWER_MODE_MAP.get(power_cli, power_cli) if power_cli else None

        for endpoint_url in parser_args.url:
            logger.info("Benchmarking endpoint: %s", endpoint_url)
            logger.debug("Resolved endpoint URL: %s", endpoint_url)

            for model in models:
                if model:
                    logger.info("Using model: %s", model)

                files_from_endpoint = _run_benchmarks_for_endpoint(
                    endpoint_url=endpoint_url,
                    model=model,
                    fixed_benchy_options=fixed_benchy_options,
                    extra_benchy_option_sets=parser_args.extra_options,
                    output_dir=parser_args.output_dir,
                    hostname=hostname,
                    sample_interval=parser_args.sample_interval,
                    verbose=parser_args.verbose,
                    power_mode_label=power_mode_label,
                    description=getattr(parser_args, 'description', None)
                )
                written_json_files.extend(files_from_endpoint)

    logger.info("All benchmarks complete. Generated %d JSON file(s).", len(written_json_files))

    # Return True if at least one benchmark produced output
    return len(written_json_files) > 0


class _BenchmarkArgumentParser(BenchmarkArgumentParser):
    """Argument parser with mass_benchy-specific examples."""

    def format_help(self):
        """Append practical examples to the default help text."""
        help_text = super().format_help()
        helplines = help_text.splitlines()
        helplines.append(' ')
        helplines.append('EXAMPLES: ')
        helplines.append(' ')
        helplines.append(r'  Run llama-benchy against a single endpoint:')
        helplines.append(' ')
        helplines.append(r'    python mass_benchy.py -u http://localhost:8080/v1 -o \json\testrun')
        helplines.append(' ')
        helplines.append(r'  Run against multiple endpoints:')
        helplines.append(' ')
        helplines.append(r'    python mass_benchy.py -u http://host1:8080/v1 -u http://host2:8080/v1 -o \json\testrun')
        helplines.append(' ')
        helplines.append(r'  Run with a model name and multiple option sets:')
        helplines.append(' ')
        helplines.append(r'    python mass_benchy.py -u http://localhost:8080/v1 -m my-model -o \json\testrun -e "--pp 128" -e "--pp 256"')
        helplines.append(' ')
        helplines.append(r'  Run with fixed options applied to all runs:')
        helplines.append(' ')
        helplines.append(r'    python mass_benchy.py -u http://localhost:8080/v1 -o \json\testrun -f "--runs 5"')
        helplines.append('\n')

        return os.linesep.join(helplines)


# Options that accept values which may start with dashes
_DASH_VALUE_OPTIONS = {
    '-e', '--extra-options',
    '-f', '--fixed-options'
}

# Options that may have quoted values needing quote stripping
_QUOTED_VALUE_OPTIONS = {
    '-p', '--power-mode',
    '-e', '--extra-options',
    '-f', '--fixed-options'
}


def _create_argument_parser() -> _BenchmarkArgumentParser:
    """
    Create and configure the argument parser for mass_benchy.

    Returns:
        Configured argument parser with all benchmark options.
    """
    parser = _BenchmarkArgumentParser(
        description="Run llama-benchy benchmarks against one or more " +
                    "external endpoints.",
        formatter_class=argparse.RawTextHelpFormatter)

    # Reorganize default argparse groups
    parser._action_groups.pop()  # pylint: disable=protected-access
    required = parser.add_argument_group("Required arguments")
    optional = parser.add_argument_group('Optional arguments')

    # Output directory is mandatory
    required.add_argument('-o', '--output-dir', type=str, required=True,
                          help=textwrap.dedent('''\
                          Output directory to store JSON files.
                          The directory will be created if it
                          does not exist.
                          '''))

    # Endpoint URLs
    required.add_argument('-u', '--url', type=str, action='append', required=True,
        help=textwrap.dedent('''\
        URL of an endpoint to benchmark (e.g., http://localhost:8080/v1).
        Multiple -u parameters are allowed, which accumulate,
        and will create separate benchmark runs for each endpoint.
        '''))

    # Model name for llama-benchy
    optional.add_argument('-m', '--model', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Model name to pass to llama-benchy via --model.
        Multiple -m parameters are allowed, which accumulate,
        and will create separate benchmark runs for each model.
        '''))

    # Extra llama-benchy options (create separate benchy runs)
    optional.add_argument('-e', '--extra-options', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Additional options to pass to llama-benchy, surrounded by quotes.
        Multiple -e parameters are allowed, which accumulate,
        and will be used for successive llama-benchy runs.
        Example: -e "--pp 128 --tg 32" -e "--pp 256 --tg 64"
        '''))

    # Fixed llama-benchy options (applied to all benchy runs)
    optional.add_argument('-f', '--fixed-options', type=str, required=False,
        help=textwrap.dedent('''\
        Fixed options to pass to llama-benchy for all runs, surrounded
        by quotes. Unlike -e, this does not create additional benchmark
        runs; the options are applied to every llama-benchy run.
        Only one -f parameter is allowed.
        Example: -f "--runs 5 --latency-mode generation"
        '''))

    # Sample interval for resource monitoring
    optional.add_argument('-i', '--sample-interval', type=float,
        default=DEFAULT_SAMPLE_INTERVAL, required=False,
        help=textwrap.dedent(f'''\
        Interval in seconds between resource monitoring samples.
        Default is {DEFAULT_SAMPLE_INTERVAL} seconds.
        '''))

    # Verbose logging
    optional.add_argument('-v', '--verbose', action='store_true',
        help='Print verbose output.')

    # Custom description for the benchmark run.
    optional.add_argument('-d', '--description', type=str, required=False,
        default='endpoint',
        help=textwrap.dedent('''
        Custom description for the benchmark run.
        Stored in the convenience_metrics field of the output JSON
        and used as a label differentiator in plots.
        Default: "endpoint"
        '''))

    # Set Windows power mode before running benchmarks.
    optional.add_argument('-p', '--power-mode', type=str, action='append', required=False,
        choices=['performance', 'balanced', 'power-saver'],
        help=textwrap.dedent('''\
        Set Windows power mode before running benchmarks (Windows only).
        Multiple -p parameters are allowed, which accumulate,
        and will create separate benchmark runs for each power mode.
        Choices: performance, balanced, power-saver.
        '''))

    return parser


def main():
    """
    Parse arguments, set up logging, and run benchmarks.
    """
    check_python_version()

    # Preprocess arguments to handle values starting with dashes
    preprocessed_args = preprocess_dash_value_args(
        sys.argv[1:], _DASH_VALUE_OPTIONS, _QUOTED_VALUE_OPTIONS
    )
    parser = _create_argument_parser()
    args = parser.parse_args(preprocessed_args)

    setup_logging(args.verbose)

    # Save the initial power mode so we can restore it when done.
    initial_power_mode = get_power_mode()
    power_modes_requested = getattr(args, 'power_mode', None)

    try:
        # Run the benchmarks
        success = benchmark_endpoints(args)
    finally:
        # Restore the initial power mode if we changed it during benchmarking
        if power_modes_requested and initial_power_mode:
            logger.info("Restoring power mode to: %s", initial_power_mode)
            if not set_power_mode(initial_power_mode):
                logger.warning("Failed to restore power mode.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Only execute main() when invoked as a script; importing this module will
    # not trigger argument parsing or benchmarking.
    main()
