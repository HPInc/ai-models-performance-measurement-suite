#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
Run llama-benchy benchmarks over one or more models via llama-server.

This script orchestrates running llama-server and llama-benchy across multiple
models, server configurations, and llama-benchy parameter sets, capturing
benchmark results for later analysis.

High-level behavior
-------------------
- Iterates over llama.cpp installations (-l), models (-m), and server parameters (-s).
- Launches llama-server with the specified parameters and captures the server URL.
- For each server instance, iterates over llama-benchy parameter sets (-e).
- Runs llama-benchy against the server endpoint with the specified parameters.
- Monitors system resources (CPU, RAM, GPU) during each llama-benchy run.
- Shuts down llama-server after all llama-benchy runs for that configuration complete.
- Saves JSON benchmark results with runtime_stats and convenience_metrics.

Key design choices
------------------
- Uses subprocess to manage llama-server lifecycle (start, capture URL, shutdown).
- Supports multiple option sets via `-e` flags for llama-benchy runs.
- Fixed options via `-f` are applied to all llama-benchy runs.
- Constant server parameters via `-c` are applied to all llama-server launches.
- Server parameters via `-s` create separate server instances for iteration.
- Model parameter (-m) supports comma-separated pairs: first for server, second for benchy.
- Resource monitoring via ResourceMonitor captures CPU, RAM, and GPU usage.

Expected environment
--------------------
- Requires `psutil` for process management.
- Requires library modules: platform_support, dashed_options, resource_monitor.
- Designed exclusively for Windows 10/11 environments.

Imports
-------
- argparse: CLI argument parsing.
- json: Serialize combined results to JSON files.
- logging: Structured logging throughout the module.
- multiprocessing: Worker processes for resource monitoring.
- os, sys: Filesystem and path manipulation.
- platform: Detect Windows environment.
- re: Regular expression for URL extraction.
- signal: Process signal handling.
- socket: Retrieve hostname for output filenames.
- subprocess: Launch and manage llama-server process.
- textwrap: Format help text in argument parser.
- time: Delays for server startup.
- typing: Type annotations for readability and tooling support.
"""
import argparse
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import textwrap
import time

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

# Import power configuration utilities for setting power mode.
try:
    from power_config import set_power_mode, get_power_mode
except ImportError:
    raise RuntimeError("Cannot find power_config module.") from None

# Import power configuration utilities for setting power mode.
try:
    from benchmark_environment import reset_benchmark_environment
except ImportError:
    raise RuntimeError("Cannot find benchmark_environment module.") from None

# Import shared benchmark utilities to avoid code duplication across mass_* scripts.
try:
    from mass_bench_common import (
        DEFAULT_SAMPLE_INTERVAL,
        BenchmarkArgumentParser,
        POWER_MODE_SHORT,
        iter_power_modes as _iter_power_modes,
        worker_llama_benchy as _worker_llama_benchy,
        extract_benchy_convenience_metrics as _extract_convenience_metrics,
        query_server_model as _query_server_model,
        parse_model_spec,
        setup_logging,
        check_python_version,
        check_windows_platform,
    )
except ImportError:
    raise RuntimeError("Cannot find mass_bench_common module.") from None

# Lazy import ResourceMonitor to avoid circular dependency.
# Import happens inside functions that need it.

# Constants
DEFAULT_SERVER_STARTUP_TIMEOUT = 60  # seconds
SERVER_URL_PATTERN = re.compile(r'http://[^\s]+:\d+')


def find_llama_server_installs(llama_dirs: Optional[List[str]]) -> List[str]:
    """
    Return validated llama-server install directories or default to CWD.

    The function checks the user-provided list of directories, retains those that
    exist and contain llama-server, warns for those that do not, and falls back
    to the current working directory when none are provided or none validate.

    Args:
        llama_dirs: Paths to llama.cpp installations (where `llama-server` resides).

    Returns:
        A list of directories to use when invoking `llama-server`.
    """
    server_installs = []
    if llama_dirs:
        for install_dir in llama_dirs:
            if os.path.isdir(install_dir):
                # Check for llama-server executable
                server_path = os.path.join(install_dir, 'llama-server.exe')
                if os.path.isfile(server_path):
                    server_installs.append(install_dir)
                else:
                    logger.warning('llama-server.exe not found in %s', install_dir)
            else:
                logger.warning('Directory %s not found.', install_dir)

    if not server_installs:
        # Pick the current working directory as the default.
        cwd = os.getcwd()
        if os.path.isfile(os.path.join(cwd, 'llama-server.exe')):
            server_installs.append(cwd)
        else:
            logger.warning('llama-server.exe not found in current directory.')

    return server_installs


def check_existing_llama_server_running() -> bool:
    """Detect an existing llama-server.exe process and abort before launching a new one."""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq llama-server.exe'],
            capture_output=True,
            text=True,
            check=False
        )
        if 'llama-server.exe' not in result.stdout:
            return False

        logger.warning('llama-server.exe is already running. Terminating benchmark launch.')
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning('Failed to check existing llama-server.exe: %s', exc)
        return False


# parse_model_spec is imported from mass_bench_common.


class LlamaServerProcess:
    """
    Manages the lifecycle of a llama-server process.

    This class handles starting llama-server, capturing its URL from output,
    and cleanly shutting it down.
    """

    def __init__(
        self,
        install_dir: str,
        model_path: Optional[str],
        server_options: List[str],
        verbose: bool = False,
        startup_timeout: int = DEFAULT_SERVER_STARTUP_TIMEOUT
    ):
        """
        Initialize the server process manager.

        Args:
            install_dir: Directory containing llama-server executable.
            model_path: Path to the GGUF model file, or None if the model
                is specified via server_options.
            server_options: Additional command-line options for the server.
            verbose: If True, append -v to llama-server command.
            startup_timeout: Seconds to wait for server to start.
        """
        self.install_dir = install_dir
        self.model_path = model_path
        self.server_options = server_options
        self.verbose = verbose
        self.startup_timeout = startup_timeout
        self.process: Optional[subprocess.Popen] = None
        self.url: Optional[str] = None

    def start(self) -> Optional[str]:
        """
        Start the llama-server process and wait for it to be ready.

        Returns:
            The server URL if successful, None otherwise.
        """
        server_exe = os.path.join(self.install_dir, 'llama-server.exe')
        if self.model_path:
            cmd = [server_exe, '-m', self.model_path] + self.server_options
        else:
            cmd = [server_exe] + self.server_options

        if self.verbose:
            cmd.append('-v')

        logger.info("Changed directory to %s", self.install_dir)
        logger.info("Starting llama-server: %s", ' '.join(cmd))

        try:
            # Start the server process with combined stdout/stderr
            # Use CREATE_NEW_PROCESS_GROUP on Windows for proper signal handling
            # pylint: disable=subprocess-popen-preexec-fn
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.install_dir,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )

            # Wait for the server to output its URL
            start_time = time.time()
            while time.time() - start_time < self.startup_timeout:
                if self.process.poll() is not None:
                    # Process has exited
                    remaining_output = self.process.stdout.read()
                    logger.error("llama-server exited unexpectedly. Output: %s",
                                 remaining_output)
                    return None

                # Read available output
                line = self.process.stdout.readline()
                if line:
                    logger.debug("llama-server: %s", line.strip())
                    # Look for URL pattern in output
                    match = SERVER_URL_PATTERN.search(line)
                    if match:
                        self.url = match.group(0)
                        logger.info("llama-server started at: %s", self.url)
                        return self.url

                time.sleep(0.1)

            logger.error("Timeout waiting for llama-server to start.")
            self.stop()
            return None

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to start llama-server: %s", exc)
            self.stop()
            return None

    def stop(self) -> None:
        """Stop the llama-server process gracefully."""
        if self.process is None:
            return

        logger.info("Stopping llama-server...")

        try:
            # Send CTRL_BREAK_EVENT for graceful shutdown (Windows)
            self.process.send_signal(signal.CTRL_BREAK_EVENT)

            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=10)
                logger.debug("llama-server stopped gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning("llama-server did not stop gracefully, killing...")
                self.process.kill()
                self.process.wait()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error stopping llama-server: %s", exc)
            try:
                self.process.kill()
            except Exception:  # pylint: disable=broad-except
                pass

        self.process = None
        self.url = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, _, __, ___):
        """Context manager exit."""
        self.stop()
        return False


def _build_output_filename(
    hostname: str,
    install_dir: str,
    server_model: Optional[str],
    server_options: Optional[List[str]],
    extra_benchy_options: Optional[List[str]] = None,
    power_mode_label: Optional[str] = None,
    run_number: int = 1,
    total_runs: int = 1
) -> str:
    """
    Build a descriptive output filename from benchmark parameters.

    The filename includes hostname, llama.cpp build identifier, model name,
    variable server options, and extra benchy options used for the run.

    Args:
        hostname: The machine hostname.
        install_dir: Directory containing llama-server (used to extract build info).
        server_model: Path to the GGUF model file.
        server_options: Variable server options (may be None).
        extra_benchy_options: Extra benchy options from -e (may be None).
        power_mode_label: Optional shortened power mode name for filename.
        run_number: Run number for this benchmark run (1-based index).
        total_runs: Total configured runs for this benchmark configuration.

    Returns:
        The constructed filename with .json extension.
    """
    parts = ["llama-server_" + hostname]

    # Extract build identifier from install directory name
    # e.g., "llama-b6876-bin-win-vulkan-x64" -> "b6876"
    install_name = os.path.basename(install_dir.rstrip(os.sep))
    if install_name:
        # Look for build number pattern (b followed by digits)
        build_match = re.search(r'b\d+', install_name)
        if build_match:
            parts.append(build_match.group(0))
        else:
            # Use the directory name if no build number found
            parts.append(install_name)

    # Extract model filename (without path and extension)
    if server_model:
        model_basename = os.path.basename(server_model)
        if model_basename:
            parts.append(model_basename)

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

    # Append run number label only for multi-run configurations.
    if total_runs > 1:
        parts.append(f'Run_{run_number}')

    # Join parts with underscore
    filename = '_'.join(parts) + '.json'
    return filename


# _extract_convenience_metrics is imported from mass_bench_common
# (as extract_benchy_convenience_metrics) and aliased above.


# _worker_llama_benchy is imported from mass_bench_common
# (as worker_llama_benchy) and aliased above.


def run_llama_benchy_with_monitoring(
    server_url: str,
    benchy_model: Optional[str],
    benchy_options: List[str],
    output_dir: str,
    hostname: str,
    install_dir: str,
    server_model: Optional[str] = None,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    server_options: Optional[List[str]] = None,
    extra_benchy_options: Optional[List[str]] = None,
    verbose: bool = False,
    power_mode_label: Optional[str] = None,
    constant_server_options: Optional[List[str]] = None,
    fixed_benchy_options: Optional[List[str]] = None,
    description: Optional[str] = None,
    run_number: int = 1,
    total_runs: int = 1
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
        server_url: URL of the llama-server endpoint.
        benchy_model: Optional model name to pass to llama-benchy.
        benchy_options: Additional command-line options for llama-benchy.
        output_dir: Directory to save benchmark results.
        hostname: Machine hostname for filenames.
        install_dir: Directory containing llama-server (for filename).
        server_model: Path to the GGUF model (for filename).
        sample_interval: Seconds between resource monitor samples.
        server_options: Variable server options used (from -s, for filename/options).
        extra_benchy_options: Extra benchy options from -e (for filename/options).
        verbose: If True, show verbose output.
        power_mode_label: Optional shortened power mode name for filename.
        constant_server_options: Constant server options from -c.
        fixed_benchy_options: Fixed benchy options from -f.
        description: Optional custom description for the benchmark run.

    Returns:
        Path to the output JSON file if successful, None otherwise.
    """
    # Lazy import ResourceMonitor to avoid circular dependency
    # pylint: disable=import-outside-toplevel
    import tempfile
    from resource_monitor import ResourceMonitor

    # Generate descriptive output filename
    output_filename = _build_output_filename(
        hostname, install_dir, server_model, server_options, extra_benchy_options,
        power_mode_label=power_mode_label,
        run_number=run_number,
        total_runs=total_runs
    )
    output_path = os.path.join(output_dir, output_filename)
    output_path = get_nonconflicting_output_path(output_path)

    # Create a temporary file for llama-benchy to write its JSON output
    # We use delete=False so we can read the file after the subprocess completes
    temp_fd, temp_output_path = tempfile.mkstemp(suffix='.json', prefix='llama_benchy_')
    os.close(temp_fd)  # Close the file descriptor; llama-benchy will write to it

    try:
        # Construct the llama-benchy command with --save-result to temp file
        cmd = ['llama-benchy', '--base-url', server_url, '--format', 'json',
               '--save-result', temp_output_path]

        if benchy_model:
            cmd.extend(['--model', benchy_model])

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
            benchy_results, server_options, extra_benchy_options,
            model=os.path.basename(server_model) if server_model else None,
            description=description,
            fixed_server_options=constant_server_options,
            fixed_benchy_options=fixed_benchy_options
        )
        if total_runs > 1:
            conv_metrics['runs'] = f'Run #{run_number}'
        conv_metrics['variant'] = extract_variant_number_from_path(output_path)

        # Query the server for its actual model name
        reported_model = _query_server_model(server_url)
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


def _run_benchmarks_for_server_config(
    install_dir: str,
    server_model: Optional[str],
    benchy_model: Optional[str],
    constant_server_options: List[str],
    variable_server_options: Optional[List[str]],
    fixed_benchy_options: List[str],
    extra_benchy_option_sets: Optional[List[str]],
    output_dir: str,
    hostname: str,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    verbose: bool = False,
    power_mode_label: Optional[str] = None,
    description: Optional[str] = None,
    runs: int = 1,
    reset_environment: bool = False
) -> List[str]:
    """
    Run all llama-benchy configurations for a single server configuration.

    Args:
        install_dir: Directory containing llama-server.
        server_model: Path to the GGUF model for llama-server.
        benchy_model: Optional model name for llama-benchy.
        constant_server_options: Server options applied to all runs.
        variable_server_options: Server options specific to this configuration.
        fixed_benchy_options: Benchy options applied to all runs.
        extra_benchy_option_sets: List of additional benchy option strings.
        output_dir: Directory to save benchmark results.
        hostname: Machine hostname for filenames.
        sample_interval: Seconds between resource monitor samples.
        verbose: If True, show verbose output.
        power_mode_label: Optional shortened power mode name for filename.
        description: Optional custom description for the benchmark run.
        runs: Number of full repeated runs per configuration.

    Returns:
        List of paths to generated JSON files.
    """
    written_json_files: List[str] = []

    # Ensure llama-server cache RAM behavior defaults to disabled unless explicitly set.
    effective_constant_server_options = list(constant_server_options) if constant_server_options else []
    all_input_server_options = effective_constant_server_options + (variable_server_options or [])
    has_cache_ram = any(
        opt in ('-cram', '--cache-ram')
        for opt in all_input_server_options
    )
    if not has_cache_ram:
        effective_constant_server_options.extend(['-cram', '0'])

    # Combine constant and variable server options
    all_server_options = combine_options(
        effective_constant_server_options,
        variable_server_options
    ) or []

    for run_number in range(1, runs + 1):
        logger.info("Starting run %d of %d for current server configuration.", run_number, runs)

        # Reset benchmark environment before starting the server process.
        if reset_environment:
            reset_benchmark_environment()

        # Start llama-server with this configuration
        if check_existing_llama_server_running():
            return written_json_files
        with LlamaServerProcess(
            install_dir,
            server_model,
            all_server_options,
            verbose=verbose
        ) as server:
            if server.url is None:
                logger.error("Failed to start llama-server, skipping this configuration.")
                return written_json_files

            # If no explicit model was provided, verify the server loaded one
            if not server_model:
                reported_model = _query_server_model(server.url)
                if not reported_model:
                    logger.error("No model loaded on llama-server. When -m is omitted, "
                                 "the model must be specified via -c, -s, or -j options.")
                    return written_json_files
                logger.info("Server loaded model: %s", reported_model)

            # Iterate over all llama-benchy option sets
            for extra_options in iter_option_sets(extra_benchy_option_sets):
                combined_benchy_options = combine_options(fixed_benchy_options, extra_options) or []

                output_file = run_llama_benchy_with_monitoring(
                    server_url=server.url,
                    benchy_model=benchy_model,
                    benchy_options=combined_benchy_options,
                    output_dir=output_dir,
                    hostname=hostname,
                    install_dir=install_dir,
                    server_model=server_model,
                    sample_interval=sample_interval,
                    server_options=variable_server_options,
                    extra_benchy_options=extra_options,
                    verbose=verbose,
                    power_mode_label=power_mode_label,
                    constant_server_options=effective_constant_server_options,
                    fixed_benchy_options=fixed_benchy_options,
                    description=description,
                    run_number=run_number,
                    total_runs=runs
                )

                if output_file:
                    written_json_files.append(output_file)

    return written_json_files


def benchmark_models(parser_args) -> bool:
    """
    Run benchmarks across all selected models, option sets, and installs.

    Discovers GGUF models and llama-server installations from CLI arguments, then
    iterates over the Cartesian product of (install × model × server-params × benchy-params).
    Each combination launches a server instance and runs llama-benchy against it.
    Resource usage (CPU, RAM, GPU) is monitored during each benchmark.

    Args:
        parser_args:
            Namespace from ``argparse.parse_args()`` with fields:
              - ``output_dir``: Output directory for benchmark results.
              - ``model``: Model path(s), possibly comma-separated pairs.
              - ``llamacpp_dir``: One or more llama.cpp install roots.
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

    # Get the list of models to benchmark.
    # If no models specified, use [None] so the loop runs once without an
    # explicit --model flag (the model must come from -c, -s, or -j).
    model_specs = parser_args.model if parser_args.model else [None]

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

    # Find llama-server installations
    server_installs = find_llama_server_installs(parser_args.llamacpp_dir)
    if not server_installs:
        logger.error('No valid llama-server installations found.')
        return False

    # Get hostname for filenames
    hostname = socket.gethostname()

    # Collect all JSON files written during this invocation
    written_json_files: List[str] = []

    # Get power modes (if any)
    power_modes = getattr(parser_args, 'power_mode', None)

    # Iterate: power-modes × installs × models × server-params
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
        power_mode_label = POWER_MODE_SHORT.get(power_cli, power_cli) if power_cli else None

        for install_dir in server_installs:
            logger.info("Using llama-server installation: %s", install_dir)

            for model_spec in model_specs:
                if model_spec is not None:
                    server_model, benchy_model = parse_model_spec(model_spec)
                    logger.info("Benchmarking model: server=%s, benchy=%s",
                                server_model, benchy_model or "(auto-detect)")
                else:
                    server_model = None
                    benchy_model = None
                    logger.info("No explicit model; expecting model from "
                                "server options (-c, -s, or -j).")

                # Iterate over variable server parameters
                for server_opts in iter_option_sets(parser_args.server_parms):
                    logger.debug("Server variable options: %s", server_opts)

                    files_from_config = _run_benchmarks_for_server_config(
                        install_dir=install_dir,
                        server_model=server_model,
                        benchy_model=benchy_model,
                        constant_server_options=constant_server_options,
                        variable_server_options=server_opts,
                        fixed_benchy_options=fixed_benchy_options,
                        extra_benchy_option_sets=parser_args.extra_options,
                        output_dir=parser_args.output_dir,
                        hostname=hostname,
                        sample_interval=parser_args.sample_interval,
                        verbose=parser_args.verbose,
                        power_mode_label=power_mode_label,
                        description=getattr(parser_args, 'description', None),
                        runs=parser_args.runs,
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
                        # Reconstruct as a single string for iter_option_sets
                        joint_benchy_sets = [' '.join(joint_benchy)]

                    files_from_config = _run_benchmarks_for_server_config(
                        install_dir=install_dir,
                        server_model=server_model,
                        benchy_model=benchy_model,
                        constant_server_options=constant_server_options,
                        variable_server_options=joint_server,
                        fixed_benchy_options=fixed_benchy_options,
                        extra_benchy_option_sets=joint_benchy_sets,
                        output_dir=parser_args.output_dir,
                        hostname=hostname,
                        sample_interval=parser_args.sample_interval,
                        verbose=parser_args.verbose,
                        power_mode_label=power_mode_label,
                        description=getattr(parser_args, 'description', None),
                        runs=parser_args.runs,
                        reset_environment=parser_args.reset_environment
                    )
                    written_json_files.extend(files_from_config)

    logger.info("All benchmarks complete. Generated %d JSON file(s).", len(written_json_files))

    # Return True if at least one benchmark produced output
    return len(written_json_files) > 0


class _BenchmarkArgumentParser(BenchmarkArgumentParser):
    """Argument parser with mass_llama_server_benchy-specific examples."""

    def format_help(self):
        """Append practical examples to the default help text."""
        help_text = super().format_help()
        helplines = help_text.splitlines()
        helplines.append(' ')
        helplines.append('EXAMPLES: ')
        helplines.append(' ')
        helplines.append(r'  Run llama-benchy against llama-server with a single model:')
        helplines.append(' ')
        helplines.append(r'    python mass_llama_server_benchy.py -m \models\model.gguf -l \llama-b6876-bin-win-vulkan-x64 -o \json\testrun')
        helplines.append(' ')
        helplines.append(r'  Run with server model and separate benchy model specification:')
        helplines.append(' ')
        helplines.append(r'    python mass_llama_server_benchy.py -m "\models\model.gguf,my-model-name" -l \llama-dir -o \json\testrun')
        helplines.append(' ')
        helplines.append(r'  Run with constant server options and variable server options:')
        helplines.append(' ')
        helplines.append(r'    python mass_llama_server_benchy.py -m \models\model.gguf -l \llama-dir -o \json\testrun -c "-ngl 99" -s "-fa 0" -s "-fa 1"')
        helplines.append(' ')
        helplines.append(r'  Run with multiple llama-benchy option sets:')
        helplines.append(' ')
        helplines.append(r'    python mass_llama_server_benchy.py -m \models\model.gguf -l \llama-dir -o \json\testrun -e "--pp 128" -e "--pp 256"')
        helplines.append('\n')

        return os.linesep.join(helplines)


# Options that accept values which may start with dashes
_DASH_VALUE_OPTIONS = {
    '-r', '--runs',
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
    Create and configure the argument parser for mass_llama_server_benchy.

    Returns:
        Configured argument parser with all benchmark options.
    """
    parser = _BenchmarkArgumentParser(
        description="Run llama-benchy benchmarks against llama-server " +
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
    optional.add_argument('-m', '--model', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Path to a GGUF LLM file, optionally with a comma-separated
        model name for llama-benchy. Format: "path/to/model.gguf" or
        "path/to/model.gguf,model-name-for-benchy".
        The first value is passed to llama-server as --model.
        The second value (if present) is passed to llama-benchy as --model.
        Multiple -m parameters are allowed and will create separate runs.
        If omitted, the model must be specified via -c, -s, or -j options.
        '''))

    # llama.cpp installation directories
    optional.add_argument('-l', '--llamacpp-dir', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Path to directory that contains an installation of llama.cpp
        with llama-server.exe.
        Multiple -l parameters are allowed, which accumulate,
        and will create separate benchmark runs.
        Default is current working directory.
        '''))

    # Constant server parameters (applied to all server launches)
    optional.add_argument('-c', '--constant-server-parms', type=str, required=False,
        help=textwrap.dedent('''\
        Options to pass to llama-server for all runs, surrounded by quotes.
        These options are appended to every llama-server command line.
        Only one -c parameter is allowed.
        Example: -c "-ngl 99 --port 8080"
        '''))

    # Variable server parameters (create separate server instances)
    optional.add_argument('-s', '--server-parms', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Additional options to pass to llama-server, surrounded by quotes.
        Multiple -s parameters are allowed, which accumulate,
        and will create separate server instances for each.
        Example: -s "-fa 0" -s "-fa 1" will run benchmarks twice,
        once with flash attention disabled, once enabled.
        '''))

    # Extra llama-benchy options (create separate benchy runs)
    optional.add_argument('-e', '--extra-options', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Additional options to pass to llama-benchy, surrounded by quotes.
        Multiple -e parameters are allowed, which accumulate,
        and will be used for successive llama-benchy runs.
        Example: -e "--pp 128 --tg 32" -e "--pp 256 --tg 64"
        '''))

    # Fixed llama-benchy options (applied to all runs)
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
        comma.  The part before the comma is passed to llama-server; the
        part after the comma is passed to llama-benchy.  If no comma is
        present, the entire string is treated as llama-benchy options.
        Multiple -j parameters are allowed, which accumulate, and will
        create separate benchmark runs for each.
        Example: -j "-ngl 99,--pp 128" -j "-ngl 0,--pp 64"
        '''))

    # Sample interval for resource monitoring
    optional.add_argument('-i', '--sample-interval', type=float,
        default=DEFAULT_SAMPLE_INTERVAL, required=False,
        help=textwrap.dedent(f'''\
        Interval in seconds between resource monitoring samples.
        Default is {DEFAULT_SAMPLE_INTERVAL} seconds.
        '''))

    # Number of repeated full runs per configuration.
    optional.add_argument('-r', '--runs', type=int, required=False, default=1,
        help=textwrap.dedent('''\
        Number of full repeated runs per configuration.
        Each run performs environment reset, launches llama-server,
        executes llama-benchy option sets, and terminates llama-server.
        Default: 1
        '''))

    # Verbose logging
    optional.add_argument('-v', '--verbose', action='store_true',
        help='Print verbose output.')

    # Custom description for the benchmark run.
    optional.add_argument('-d', '--description', type=str, required=False,
        default='llama-server',
        help=textwrap.dedent('''
        Custom description for the benchmark run.
        Stored in the convenience_metrics field of the output JSON
        and used as a label differentiator in plots.
        Default: "llama-server"
        '''))

    # Set Windows power mode before running benchmarks.
    optional.add_argument('-p', '--power-mode', type=str, action='append', required=False,
        choices=['best-performance', 'balanced', 'best-power-efficiency'],
        help=textwrap.dedent('''\
        Set Windows power mode before running benchmarks.
        Multiple -p parameters are allowed, which accumulate,
        and will create separate benchmark runs for each power mode.
        Choices: best-performance, balanced, best-power-efficiency.
        '''))

    # Control whether benchmark environment reset is performed before each server launch.
    optional.add_argument('--reset-environment',
        action='store_true',
        help='Enable benchmark environment reset before each server launch.')

    return parser


def main():
    """
    Parse arguments, set up logging, and run benchmarks.
    """
    check_python_version()
    check_windows_platform()

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
    success = False

    try:
        # Run the benchmarks
        success = benchmark_models(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Stopping llama-server and exiting.")
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
