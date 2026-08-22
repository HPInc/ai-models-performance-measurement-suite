#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
Run llama-benchy benchmarks over one or more models via lemonade server.

This script orchestrates loading models into lemonade server and running
llama-benchy against them, capturing benchmark results for later analysis.

High-level behavior
-------------------
- Checks that lemonade server is reachable via ``lemonade status``.
- Iterates over models (-m) and server parameter sets (-s).
- For each llama-benchy parameter set (-e), loads the model with the specified
  parameters.
- Runs llama-benchy against the lemonade endpoint with the specified parameters.
- Monitors system resources (CPU, RAM, GPU) during model load, llama-benchy,
  and model unload for each run.
- Unloads the model from lemonade server after each llama-benchy run.
- Saves JSON benchmark results with runtime_stats and convenience_metrics.

Key design choices
------------------
- Lemonade server is assumed to be a persistent service (default port 13305).
  The script does not start or stop the service itself.
- Models are loaded via ``lemonade load`` and unloaded via ``lemonade unload``.
- If a model is already loaded, the load command succeeds without error.
- Constant server parameters via ``-c`` are applied to all ``lemonade load`` calls.
- Server parameters via ``-s`` create separate per-run load/unload cycles.
- Model parameter (-m) supports comma-separated pairs: first for server, second
  for benchy.
- Resource monitoring via ResourceMonitor captures CPU, RAM, and GPU usage.

Expected environment
--------------------
- Requires ``psutil`` for process management.
- Requires library modules: platform_support, dashed_options, resource_monitor.
- Requires ``lemonade`` CLI to be installed and in PATH.
- Requires lemonade server service to be running.
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
- subprocess: Launch llama-benchy and lemonade CLI processes.
- textwrap: Format help text in argument parser.
- typing: Type annotations for readability and tooling support.
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

# Import platform-specific helpers for command execution, directory changes, and
# platform detection. Keep failures explicit for easier troubleshooting.
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
        create_options_list, iter_option_sets, combine_options, preprocess_dash_value_args,
        parse_joint_options
    )
except ImportError:
    raise RuntimeError("Cannot find dashed_options module.") from None

try:
    from power_config import get_power_mode, set_power_mode
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Missing power_config.py library file.") from exc

try:
    from benchmark_environment import reset_benchmark_environment
except ImportError:
    raise RuntimeError("Cannot find benchmark_environment module.") from None

# Import shared benchmark utilities to avoid code duplication across mass_* scripts.
try:
    from mass_bench_common import (
        DEFAULT_SAMPLE_INTERVAL,
        BenchmarkArgumentParser,
        POWER_MODE_MAP,
        iter_power_modes as _iter_power_modes,
        extract_benchy_convenience_metrics as _extract_convenience_metrics,
        parse_model_spec,
        setup_logging,
        check_python_version,
    )
except ImportError:
    raise RuntimeError("Cannot find mass_bench_common module.") from None

# Lazy import ResourceMonitor to avoid circular dependency.
# Import happens inside functions that need it.

# Constants
DEFAULT_LEMONADE_HOST = '127.0.0.1'
DEFAULT_LEMONADE_PORT = 13305


def check_lemonade_server(host: str, port: int) -> bool:
    """
    Check whether lemonade server is reachable via ``lemonade status``.

    Args:
        host: Lemonade server host address.
        port: Lemonade server port number.

    Returns:
        True if the server is reachable, False otherwise.
    """
    cmd = ['lemonade', '--host', host, '--port', str(port), 'status']
    logger.debug("Checking lemonade server: %s", ' '.join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            logger.info("Lemonade server is reachable at %s:%d", host, port)
            return True
        logger.error("Lemonade server not reachable at %s:%d: %s",
                     host, port, result.stderr.strip())
        return False
    except FileNotFoundError:
        logger.error("lemonade CLI not found. Ensure it is installed and in PATH.")
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to check lemonade server status: %s", exc)
        return False

def load_lemonade_model(
    model_name: str,
    server_options: List[str],
    host: str,
    port: int
) -> bool:
    """
    Load a model into lemonade server via ``lemonade load``.

    If the model is already loaded, lemonade server accepts the load
    command without error.

    Args:
        model_name: Name of the model to load (e.g., "Qwen3-0.6B-GGUF").
        server_options: Additional options for the load command
            (e.g., ["--ctx-size", "8192"]).
        host: Lemonade server host address.
        port: Lemonade server port number.

    Returns:
        True if the model was loaded successfully, False otherwise.
    """
    cmd = ['lemonade', '--host', host, '--port', str(port),
           'load', model_name] + server_options

    logger.info("Loading model into lemonade server: %s", ' '.join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            logger.info("Model '%s' loaded successfully.", model_name)
            return True
        logger.error("Failed to load model '%s': %s",
                     model_name, result.stderr.strip())
        return False
    except FileNotFoundError:
        logger.error("lemonade CLI not found. Ensure it is installed and in PATH.")
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to load model '%s': %s", model_name, exc)
        return False


def unload_lemonade_model(
    model_name: str,
    host: str,
    port: int
) -> bool:
    """
    Unload a model from lemonade server via ``lemonade unload``.

    Args:
        model_name: Name of the model to unload. If empty, unloads all models.
        host: Lemonade server host address.
        port: Lemonade server port number.

    Returns:
        True if the model was unloaded successfully, False otherwise.
    """
    cmd = ['lemonade', '--host', host, '--port', str(port), 'unload']
    if model_name:
        cmd.append(model_name)

    logger.info("Unloading model from lemonade server: %s", ' '.join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            logger.info("Model unloaded successfully.")
            return True
        logger.warning("Failed to unload model: %s", result.stderr.strip())
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to unload model: %s", exc)
        return False


# parse_model_spec is imported from mass_bench_common.


def _build_output_filename(
    hostname: str,
    server_model: str,
    server_options: Optional[List[str]],
    extra_benchy_options: Optional[List[str]] = None,
    power_mode_label: Optional[str] = None
) -> str:
    """
    Build a descriptive output filename from benchmark parameters.

    The filename includes hostname, model name, variable server options,
    and extra benchy options used for the run.

    Args:
        hostname: The machine hostname.
        server_model: Model name loaded into lemonade server.
        server_options: Variable server options (may be None).
        extra_benchy_options: Extra benchy options from -e (may be None).
        power_mode_label: Optional shortened power mode name for filename.

    Returns:
        The constructed filename with .json extension.
    """
    parts = ["lemonade_server_" + hostname]

    # Append model name
    if server_model:
        parts.append(server_model)

    # Append variable server options (if any)
    if server_options:
        for option in server_options:
            # Clean up option for filename (remove leading dashes)
            clean_opt = option.lstrip('-')
            if clean_opt:
                parts.append(clean_opt)

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


def _worker_load_run_unload_lemonade_benchy(
    host: str,
    port: int,
    server_model: str,
    all_server_options: List[str],
    benchy_model: Optional[str],
    benchy_options: List[str],
    temp_output_path: str,
    verbose: bool,
    result_queue,
) -> None:
    """Load model, run llama-benchy, unload model, and publish worker result."""
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
        'returncode': -1,
        'reported_model': None
    }

    model_loaded = False
    server_url = 'http://%s:%d/api/v1' % (host, port)
    logger.debug("Constructed lemonade server URL: %s", server_url)

    try:
        if not load_lemonade_model(server_model, all_server_options, host, port):
            logger.error(
                "Failed to load model '%s' for benchmark run.", server_model)
            result_queue.put(result)
            return

        model_loaded = True
        result['reported_model'] = server_model

        cmd = ['llama-benchy', '--base-url', server_url, '--format', 'json',
               '--save-result', temp_output_path]
        if benchy_model:
            cmd.extend(['--model', benchy_model])
        cmd.extend(benchy_options)

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
            logger.error(
                "llama-benchy failed with return code %d", proc_result.returncode)
            logger.error("stdout: %s", proc_result.stdout)
            logger.error("stderr: %s", proc_result.stderr)
    except FileNotFoundError:
        logger.error("llama-benchy not found. Ensure it is installed and in PATH.")
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Failed to run lemonade load+bench+unload worker: %s", exc)
    finally:
        if model_loaded:
            unload_lemonade_model(server_model, host, port)

    result_queue.put(result)


def run_llama_benchy_with_monitoring(
    benchy_model: Optional[str],
    benchy_options: List[str],
    output_dir: str,
    hostname: str,
    host: str,
    port: int,
    all_server_options: List[str],
    server_model: str,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    server_options: Optional[List[str]] = None,
    extra_benchy_options: Optional[List[str]] = None,
    verbose: bool = False,
    power_mode_label: Optional[str] = None,
    constant_server_options: Optional[List[str]] = None,
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
        benchy_model: Optional model name to pass to llama-benchy.
        benchy_options: Additional command-line options for llama-benchy.
        output_dir: Directory to save benchmark results.
        hostname: Machine hostname for filename.
        host: Lemonade server host address.
        port: Lemonade server port number.
        all_server_options: Full server options used for lemonade load.
        server_model: Model name loaded into lemonade server (for filename).
        sample_interval: Seconds between resource monitor samples.
        server_options: Variable server options used (from -s, for filename/options).
        extra_benchy_options: Extra benchy options from -e (for filename/options).
        verbose: If True, show verbose output.
        power_mode_label: Optional shortened power mode name for filename.
        constant_server_options: Constant server options from -c.
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
        hostname, server_model, server_options, extra_benchy_options,
        power_mode_label=power_mode_label
    )
    output_path = os.path.join(output_dir, output_filename)
    output_path = get_nonconflicting_output_path(output_path)

    # Create a temporary file for llama-benchy to write its JSON output
    # We use delete=False so we can read the file after the subprocess completes
    temp_fd, temp_output_path = tempfile.mkstemp(suffix='.json', prefix='llama_benchy_')
    os.close(temp_fd)  # Close the file descriptor; llama-benchy will write to it

    try:
        logger.info("Running lemonade load + llama-benchy + unload with monitoring")

        # Run model load + llama-benchy + unload with resource monitoring
        worker_result, runtime_stats, system_info = ResourceMonitor.monitor_process(
            worker_fn=_worker_load_run_unload_lemonade_benchy,
            worker_args=(
                host,
                port,
                server_model,
                all_server_options,
                benchy_model,
                benchy_options,
                temp_output_path,
                verbose
            ),
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
            benchy_results, server_options, extra_benchy_options,
            model=os.path.basename(server_model),
            description=description,
            fixed_server_options=constant_server_options,
            fixed_benchy_options=fixed_benchy_options
        )
        conv_metrics['variant'] = extract_variant_number_from_path(output_path)

        # Use model reported while the model was loaded in the worker.
        reported_model = worker_result.get('reported_model') if worker_result else None
        if reported_model:
            conv_metrics['model'] = reported_model

        # Build the combined JSON structure
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


def _run_benchmarks_for_server_config(
    server_model: str,
    benchy_model: Optional[str],
    constant_server_options: List[str],
    variable_server_options: Optional[List[str]],
    fixed_benchy_options: List[str],
    extra_benchy_option_sets: Optional[List[str]],
    output_dir: str,
    hostname: str,
    host: str,
    port: int,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    verbose: bool = False,
    power_mode_label: Optional[str] = None,
    description: Optional[str] = None,
    reset_environment: bool = False
 ) -> List[str]:
    """
    Run all llama-benchy configurations for a single server configuration.

    For each benchy option set, loads the model into lemonade server with the
    combined constant and variable server options, runs llama-benchy, then
    unloads the model.

    Args:
        server_model: Model name for lemonade server.
        benchy_model: Optional model name for llama-benchy.
        constant_server_options: Server options applied to all runs.
        variable_server_options: Server options specific to this configuration.
        fixed_benchy_options: Benchy options applied to all runs.
        extra_benchy_option_sets: List of additional benchy option strings.
        output_dir: Directory to save benchmark results.
        hostname: Machine hostname for filenames.
        host: Lemonade server host address.
        port: Lemonade server port number.
        sample_interval: Seconds between resource monitor samples.
        verbose: If True, show verbose output.
        power_mode_label: Optional shortened power mode name for filename.
        description: Optional diagnostic description for this benchmark run.

    Returns:
        List of paths to generated JSON files.
    """
    written_json_files: List[str] = []

    # Combine constant and variable server options
    all_server_options = combine_options(constant_server_options, variable_server_options) or []

    # Reset benchmark environment before loading the model.
    if reset_environment:
        reset_benchmark_environment()

    # Iterate over all llama-benchy option sets; each run does its own model load/unload.
    for extra_options in iter_option_sets(extra_benchy_option_sets):
        combined_benchy_options = combine_options(fixed_benchy_options, extra_options) or []

        output_file = run_llama_benchy_with_monitoring(
            benchy_model=benchy_model,
            benchy_options=combined_benchy_options,
            output_dir=output_dir,
            hostname=hostname,
            host=host,
            port=port,
            all_server_options=all_server_options,
            server_model=server_model,
            sample_interval=sample_interval,
            server_options=variable_server_options,
            extra_benchy_options=extra_options,
            verbose=verbose,
            power_mode_label=power_mode_label,
            constant_server_options=constant_server_options,
            fixed_benchy_options=fixed_benchy_options,
            description=description
        )

        if output_file:
            written_json_files.append(output_file)

    return written_json_files


def benchmark_models(parser_args) -> bool:
    """
    Run benchmarks across all selected models and option sets.

    Iterates over the Cartesian product of (model × server-params × benchy-params).
    Each combination loads a model into lemonade server and runs llama-benchy
    against it. Resource usage (CPU, RAM, GPU) is monitored during each benchmark.

    Args:
        parser_args:
            Namespace from ``argparse.parse_args()`` with fields:
              - ``output_dir``: Output directory for benchmark results.
              - ``model``: Model name(s), possibly comma-separated pairs.
              - ``host``: Lemonade server host address.
              - ``port``: Lemonade server port number.
              - ``constant_server_parms``: Server options for all runs.
              - ``server_parms``: Server options that vary between runs.
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

    # Get the list of models to benchmark
    if not parser_args.model:
        logger.error('No models specified, please use -m to specify models.')
        return False

    # Check that lemonade server is reachable
    if not check_lemonade_server(parser_args.host, parser_args.port):
        logger.error('Lemonade server is not reachable. '
                     'Please ensure the lemonade service is running.')
        return False

    # Parse constant server options
    constant_server_options = []
    if parser_args.constant_server_parms:
        constant_server_options = create_options_list(parser_args.constant_server_parms)
        logger.debug('Constant server options: %s', constant_server_options)

    # Parse fixed benchy options
    fixed_benchy_options = []
    if parser_args.fixed_options:
        fixed_benchy_options = create_options_list(parser_args.fixed_options)
        logger.debug('Fixed benchy options: %s', fixed_benchy_options)

    # Get hostname for filenames
    hostname = socket.gethostname()

    # Collect all JSON files written during this invocation
    written_json_files: List[str] = []

    # Get power modes (if any)
    power_modes = getattr(parser_args, 'power_mode', None)

    # Iterate: power-modes × models × server-params
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

        for model_spec in parser_args.model:
            server_model, benchy_model = parse_model_spec(model_spec)
            logger.info("Benchmarking model: server=%s, benchy=%s",
                        server_model, benchy_model or "(auto-detect)")

            # Iterate over variable server parameters
            for server_opts in iter_option_sets(parser_args.server_parms):
                logger.debug("Server variable options: %s", server_opts)

                files_from_config = _run_benchmarks_for_server_config(
                    server_model=server_model,
                    benchy_model=benchy_model,
                    constant_server_options=constant_server_options,
                    variable_server_options=server_opts,
                    fixed_benchy_options=fixed_benchy_options,
                    extra_benchy_option_sets=parser_args.extra_options,
                    output_dir=parser_args.output_dir,
                    hostname=hostname,
                    host=parser_args.host,
                    port=parser_args.port,
                    sample_interval=parser_args.sample_interval,
                    verbose=parser_args.verbose,
                    power_mode_label=power_mode_label,
                    description=getattr(parser_args, 'description', None),
                    reset_environment=parser_args.reset_environment
                )
                written_json_files.extend(files_from_config)

            # Iterate over joint server+benchy option pairs
            for joint_server, joint_benchy in parse_joint_options(
                    parser_args.joint_options):
                logger.debug("Joint server options: %s, benchy options: %s",
                             joint_server, joint_benchy)

                # Joint benchy options are passed as extra_benchy_option_sets
                # (a list with one element) so iter_option_sets yields them.
                joint_benchy_sets = None
                if joint_benchy:
                    joint_benchy_sets = [' '.join(joint_benchy)]

                files_from_config = _run_benchmarks_for_server_config(
                    server_model=server_model,
                    benchy_model=benchy_model,
                    constant_server_options=constant_server_options,
                    variable_server_options=joint_server,
                    fixed_benchy_options=fixed_benchy_options,
                    extra_benchy_option_sets=joint_benchy_sets,
                    output_dir=parser_args.output_dir,
                    hostname=hostname,
                    host=parser_args.host,
                    port=parser_args.port,
                    sample_interval=parser_args.sample_interval,
                    verbose=parser_args.verbose,
                    power_mode_label=power_mode_label,
                    description=getattr(parser_args, 'description', None),
                    reset_environment=parser_args.reset_environment
                )
                written_json_files.extend(files_from_config)

    logger.info("All benchmarks complete. Generated %d JSON file(s).", len(written_json_files))

    # Return True if at least one benchmark produced output
    return len(written_json_files) > 0


class _BenchmarkArgumentParser(BenchmarkArgumentParser):
    """Argument parser with mass_lemonade_benchy-specific examples."""

    def format_help(self):
        """Append practical examples to the default help text."""
        help_text = super().format_help()
        helplines = help_text.splitlines()
        helplines.append(' ')
        helplines.append('EXAMPLES: ')
        helplines.append(' ')
        helplines.append(r'  Run llama-benchy against lemonade server with a single model:')
        helplines.append(' ')
        helplines.append(r'    python mass_lemonade_benchy.py -m Qwen3-0.6B-GGUF -o \json\testrun')
        helplines.append(' ')
        helplines.append(r'  Run with server model and separate benchy model specification:')
        helplines.append(' ')
        helplines.append(r'    python mass_lemonade_benchy.py -m "Qwen3-0.6B-GGUF,my-model-name" -o \json\testrun')
        helplines.append(' ')
        helplines.append(r'  Run with constant server options and variable server options:')
        helplines.append(' ')
        helplines.append(r'    python mass_lemonade_benchy.py -m Qwen3-0.6B-GGUF -o \json\testrun -c "--ctx-size 8192" -s "--llamacpp vulkan" -s "--llamacpp cpu"')
        helplines.append(' ')
        helplines.append(r'  Run with multiple llama-benchy option sets:')
        helplines.append(' ')
        helplines.append(r'    python mass_lemonade_benchy.py -m Qwen3-0.6B-GGUF -o \json\testrun -e "--pp 128" -e "--pp 256"')
        helplines.append('\n')

        return os.linesep.join(helplines)


# Options that accept values which may start with dashes
_DASH_VALUE_OPTIONS = {
    '-e', '--extra-options',
    '-f', '--fixed-options',
    '-c', '--constant-server-parms',
    '-s', '--server-parms',
    '-j', '--joint-options'
}

# Options that may have quoted values needing quote stripping
_QUOTED_VALUE_OPTIONS = {
    '-p', '--power-mode',
    '-e', '--extra-options',
    '-f', '--fixed-options',
    '-c', '--constant-server-parms',
    '-s', '--server-parms',
    '-j', '--joint-options'
}


def _create_argument_parser() -> _BenchmarkArgumentParser:
    """
    Create and configure the argument parser for mass_lemonade_benchy.

    Returns:
        Configured argument parser with all benchmark options.
    """
    parser = _BenchmarkArgumentParser(
        description="Run llama-benchy benchmarks against lemonade server " +
                    "for multiple models and configurations.",
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

    # Model specification (supports comma-separated pairs)
    required.add_argument('-m', '--model', type=str, action='append', required=True,
        help=textwrap.dedent('''\
        Lemonade model name, optionally with a comma-separated
        model name for llama-benchy. Format: "ModelName" or
        "ModelName,model-name-for-benchy".
        The first value is passed to lemonade load.
        The second value (if present) is passed to llama-benchy as --model.
        Multiple -m parameters are allowed and will create separate runs.
        '''))

    # Lemonade server host
    optional.add_argument('--host', type=str, default=DEFAULT_LEMONADE_HOST,
        help=textwrap.dedent(f'''\
        Lemonade server host address.
        Default is {DEFAULT_LEMONADE_HOST}.
        '''))

    # Lemonade server port
    optional.add_argument('--port', type=int, default=DEFAULT_LEMONADE_PORT,
        help=textwrap.dedent(f'''\
        Lemonade server port number.
        Default is {DEFAULT_LEMONADE_PORT}.
        '''))

    # Constant server parameters (applied to all lemonade load calls)
    optional.add_argument('-c', '--constant-server-parms', type=str, required=False,
        help=textwrap.dedent('''\
        Options to pass to lemonade load for all runs, surrounded by quotes.
        These options are appended to every lemonade load command line.
        Only one -c parameter is allowed.
        Example: -c "--ctx-size 8192 --llamacpp vulkan"
        '''))

    # Variable server parameters (create separate load/unload cycles)
    optional.add_argument('-s', '--server-parms', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Additional options to pass to lemonade load, surrounded by quotes.
        Multiple -s parameters are allowed, which accumulate,
        and will create separate model load/unload cycles for each.
        Example: -s "--llamacpp vulkan" -s "--llamacpp cpu" will run
        benchmarks twice, once with each backend.
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

    # Joint server+benchy options (paired per iteration)
    optional.add_argument('-j', '--joint-options', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Combined server and benchy options in one string, separated by a
        comma.  The part before the comma is passed to lemonade load; the
        part after the comma is passed to llama-benchy.  If no comma is
        present, the entire string is treated as llama-benchy options.
        Multiple -j parameters are allowed, which accumulate, and will
        create separate benchmark runs for each.
        Example: -j "--llamacpp vulkan,--pp 128" -j "--llamacpp cpu,--pp 64"
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
        default='lemonade_server',
        help=textwrap.dedent('''
        Custom description for the benchmark run.
        Stored in the convenience_metrics field of the output JSON
        and used as a label differentiator in plots.
        Default: "lemonade_server"
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

    # Control whether benchmark environment reset is performed before each server launch.
    optional.add_argument('--reset-environment',
        action='store_true',
        help='Enable benchmark environment reset before each model load.')

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
    # This is important when -p option is used to run benchmarks across multiple
    # power modes - we want to leave the system in the same state as when we started.
    initial_power_mode = get_power_mode()
    power_modes_requested = getattr(args, 'power_mode', None)
    success = False

    try:
        # Run the benchmarks
        success = benchmark_models(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Unloading model(s) and exiting.")
    finally:
        logger.info("Ensuring lemonade server models are unloaded before exit.")
        unload_lemonade_model('', args.host, args.port)

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
