#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
Run embedding benchmarks over one or more models via llama-server.

This script orchestrates running llama-server (with the --embedding flag) and
benchmark_embeddings.py across multiple models, server configurations, and
benchmark parameter sets, capturing results for later analysis.

High-level behavior
-------------------
- Iterates over llama.cpp installations (-l), models (-m), and server parameters (-s).
- Launches llama-server with --embedding and the specified parameters, then captures the
  server URL.
- For each server instance, iterates over benchmark_embeddings parameter sets (-e).
- Runs benchmark_embeddings.py against the v1/embeddings endpoint with the specified
  parameters.
- Monitors system resources (CPU, RAM, GPU) during each benchmark_embeddings run.
- Shuts down llama-server after all benchmark runs for that configuration complete.
- Saves JSON benchmark results with runtime_stats and convenience_metrics.

Key design choices
------------------
- Uses subprocess to manage llama-server lifecycle (start, capture URL, shutdown).
- Supports multiple option sets via `-e` flags for benchmark_embeddings runs.
- Fixed options via `-f` are applied to all benchmark_embeddings runs.
- Constant server parameters via `-c` are applied to all llama-server launches.
- Server parameters via `-s` create separate server instances for iteration.
- --embedding is always injected into every llama-server command line.
- Resource monitoring via ResourceMonitor captures CPU, RAM, and GPU usage.

Expected environment
--------------------
- Requires `psutil` for process management.
- Requires library modules: platform_support, dashed_options, resource_monitor,
  benchmark_embeddings, mass_bench_common.
- Designed exclusively for Windows 10/11 environments.

Imports
-------
- argparse: CLI argument parsing.
- json: Serialize combined results to JSON files.
- logging: Structured logging throughout the module.
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

# Import benchmark environment reset utility.
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

# Path to the benchmark_embeddings script, located in the lib directory.
_BENCHMARK_EMBEDDINGS_SCRIPT = os.path.join(lib_dir, 'benchmark_embeddings.py')

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
                server_path = os.path.join(install_dir, 'llama-server.exe')
                if os.path.isfile(server_path):
                    server_installs.append(install_dir)
                else:
                    logger.warning('llama-server.exe not found in %s', install_dir)
            else:
                logger.warning('Directory %s not found.', install_dir)

    if not server_installs:
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
                --embedding is already included in server_options by the caller.
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
            # Start the server process with combined stdout/stderr.
            # Use CREATE_NEW_PROCESS_GROUP on Windows for proper signal handling.
            # pylint: disable=subprocess-popen-preexec-fn
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.install_dir,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )

            # Wait for the server to output its URL.
            start_time = time.time()
            while time.time() - start_time < self.startup_timeout:
                if self.process.poll() is not None:
                    remaining_output = self.process.stdout.read()
                    logger.error("llama-server exited unexpectedly. Output: %s",
                                 remaining_output)
                    return None

                line = self.process.stdout.readline()
                if line:
                    logger.debug("llama-server: %s", line.strip())
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
            self.process.send_signal(signal.CTRL_BREAK_EVENT)

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
    extra_bench_options: Optional[List[str]] = None,
    power_mode_label: Optional[str] = None,
    run_number: int = 1,
    total_runs: int = 1
) -> str:
    """
    Build a descriptive output filename from benchmark parameters.

    The filename includes hostname, llama.cpp build identifier, model name,
    variable server options, and extra benchmark options used for the run.

    Args:
        hostname: The machine hostname.
        install_dir: Directory containing llama-server (used to extract build info).
        server_model: Path to the GGUF model file.
        server_options: Variable server options (may be None).
        extra_bench_options: Extra benchmark_embeddings options from -e (may be None).
        power_mode_label: Optional shortened power mode name for filename.
        run_number: Run number for this benchmark run (1-based index).
        total_runs: Total configured runs for this benchmark configuration.

    Returns:
        The constructed filename with .json extension.
    """
    parts = ["emb-server_" + hostname]

    install_name = os.path.basename(install_dir.rstrip(os.sep))
    if install_name:
        build_match = re.search(r'b\d+', install_name)
        if build_match:
            parts.append(build_match.group(0))
        else:
            parts.append(install_name)

    if server_model:
        model_basename = os.path.basename(server_model)
        if model_basename:
            parts.append(model_basename)

    if server_options:
        for option in server_options:
            clean_opt = option.lstrip('-')
            if clean_opt:
                parts.append(clean_opt)

    if extra_bench_options:
        for option in extra_bench_options:
            clean_opt = option.lstrip('-')
            if clean_opt:
                parts.append(clean_opt)

    if power_mode_label:
        parts.append(power_mode_label)

    if total_runs > 1:
        parts.append(f'Run_{run_number}')

    filename = '_'.join(parts) + '.json'
    return filename


def _extract_embeddings_convenience_metrics(
    bench_results: dict,
    server_options: Optional[List[str]] = None,
    extra_bench_options: Optional[List[str]] = None,
    model: Optional[str] = None,
    description: Optional[str] = None,
    fixed_server_options: Optional[List[str]] = None,
    fixed_bench_options: Optional[List[str]] = None
) -> dict:
    """
    Build convenience metrics for a benchmark_embeddings run.

    Promotes ``throughput_samples_per_sec`` from the first batch result.
    All other benchmark result data
    lives in the ``embedding_bench_results`` key of the combined JSON output
    and is not duplicated here.

    Fields stored:
        - throughput_samples_per_sec (from results[0], if present)
        - throughput_samples_per_sec_stddev (from results[0], if present)
        - p50_batch_latency_ms (from results[0], if present)
        - p50_batch_latency_ms_stddev (from results[0], if present)
        - model, description, server_options, options,
          constant_server_options, fixed_options

    Args:
        bench_results: Parsed JSON object written to stdout by benchmark_embeddings.py
            (keys: ``target``, ``results``).
        server_options: Variable server options (from -s, not -c).
        extra_bench_options: Extra benchmark options (from -e, not -f).
        model: Model name used for the benchmark.
        description: Optional custom description for the benchmark run.
        fixed_server_options: Fixed server options (from -c).
        fixed_bench_options: Fixed benchmark options (from -f).

    Returns:
        Dictionary with convenience metrics and run metadata.
    """
    conv_metrics: dict = {}

    results_list = bench_results.get('results')
    if results_list and isinstance(results_list, list) and len(results_list) > 0:
        first = results_list[0]
        if isinstance(first, dict):
            value = first.get('throughput_samples_per_sec')
            if value is not None:
                conv_metrics['throughput_samples_per_sec'] = value
            value = first.get('throughput_samples_per_sec_stddev')
            if value is not None:
                conv_metrics['throughput_samples_per_sec_stddev'] = value
            value = first.get('p50_batch_latency_ms')
            if value is not None:
                conv_metrics['p50_batch_latency_ms'] = value
            value = first.get('p50_batch_latency_ms_stddev')
            if value is not None:
                conv_metrics['p50_batch_latency_ms_stddev'] = value

    if model:
        conv_metrics['model'] = model

    if description:
        conv_metrics['description'] = description

    if server_options:
        conv_metrics['server_options'] = ' '.join(server_options)

    if extra_bench_options:
        conv_metrics['options'] = ' '.join(extra_bench_options)

    if fixed_server_options:
        conv_metrics['constant_server_options'] = ' '.join(fixed_server_options)

    if fixed_bench_options:
        conv_metrics['fixed_options'] = ' '.join(fixed_bench_options)

    return conv_metrics


def _worker_benchmark_embeddings(
    cmd: List[str],
    verbose: bool,
    result_queue,
) -> None:
    """
    Run benchmark_embeddings.py in a child process and push results to a queue.

    This worker is executed inside a ResourceMonitor child process so that
    resource sampling runs concurrently with the benchmark.  The script output
    is captured from stdout (JSON when --output-format json is used).

    Args:
        cmd: The complete command as a list of strings.
        verbose: If True, echo subprocess output to the console.
        result_queue: Queue to put the result dictionary into.
    """
    import logging as _logging  # pylint: disable=import-outside-toplevel
    log_level = _logging.DEBUG if verbose else _logging.INFO
    _logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(message)s - (%(funcName)s in %(filename)s:%(lineno)d)",
        force=True
    )

    result = {
        'success': False,
        'stdout': '',
        'stderr': '',
        'returncode': -1
    }

    try:
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
                _logging.getLogger(__name__).debug(
                    "benchmark_embeddings stdout: %s", proc_result.stdout)
        else:
            _logging.getLogger(__name__).error(
                "benchmark_embeddings failed with return code %d", proc_result.returncode)
            _logging.getLogger(__name__).error("stdout: %s", proc_result.stdout)
            _logging.getLogger(__name__).error("stderr: %s", proc_result.stderr)

    except FileNotFoundError:
        _logging.getLogger(__name__).error(
            "benchmark_embeddings script not found at: %s", cmd[1] if len(cmd) > 1 else '?')
    except Exception as exc:  # pylint: disable=broad-except
        _logging.getLogger(__name__).error("Failed to run benchmark_embeddings: %s", exc)

    result_queue.put(result)


def run_benchmark_embeddings_with_monitoring(
    server_url: str,
    bench_options: List[str],
    output_dir: str,
    hostname: str,
    install_dir: str,
    server_model: Optional[str] = None,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    server_options: Optional[List[str]] = None,
    extra_bench_options: Optional[List[str]] = None,
    verbose: bool = False,
    power_mode_label: Optional[str] = None,
    constant_server_options: Optional[List[str]] = None,
    fixed_bench_options: Optional[List[str]] = None,
    description: Optional[str] = None,
    run_number: int = 1,
    total_runs: int = 1
) -> Optional[str]:
    """
    Run benchmark_embeddings.py with resource monitoring and save combined results.

    This function wraps benchmark_embeddings execution with ResourceMonitor to
    capture CPU, RAM, and GPU usage during the benchmark.  Results are saved as a
    JSON file containing:
    - ``embedding_bench_results``: Raw output from benchmark_embeddings.py
    - ``convenience_metrics``: Run metadata (model, description, options)
    - ``runtime_stats``: Resource monitoring data
    - ``system_info``: System information

    Args:
        server_url: Base URL of the llama-server endpoint
            (e.g., ``http://localhost:8080``).
        bench_options: Additional command-line options for benchmark_embeddings.py.
        output_dir: Directory to save benchmark results.
        hostname: Machine hostname for filenames.
        install_dir: Directory containing llama-server (for filename).
        server_model: Path to the GGUF model (for filename).
        sample_interval: Seconds between resource monitor samples.
        server_options: Variable server options used (from -s, for filename/options).
        extra_bench_options: Extra benchmark options from -e (for filename/options).
        verbose: If True, show verbose output.
        power_mode_label: Optional shortened power mode name for filename.
        constant_server_options: Constant server options from -c.
        fixed_bench_options: Fixed benchmark options from -f.
        description: Optional custom description for the benchmark run.
        run_number: Current run number (1-based) within repeated runs.
        total_runs: Total number of configured repeated runs.

    Returns:
        Path to the output JSON file if successful, None otherwise.
    """
    # Lazy import ResourceMonitor to avoid circular dependency.
    # pylint: disable=import-outside-toplevel
    from resource_monitor import ResourceMonitor

    output_filename = _build_output_filename(
        hostname, install_dir, server_model, server_options, extra_bench_options,
        power_mode_label=power_mode_label,
        run_number=run_number,
        total_runs=total_runs
    )
    output_path = os.path.join(output_dir, output_filename)
    output_path = get_nonconflicting_output_path(output_path)

    # Build the endpoint URL for the embeddings endpoint.
    embeddings_url = server_url.rstrip('/') + '/v1/embeddings'

    cmd = [
        sys.executable,
        _BENCHMARK_EMBEDDINGS_SCRIPT,
        '--endpoint-url', embeddings_url,
        '--output-format', 'json',
    ]

    cmd.extend(bench_options)

    logger.info("Running benchmark_embeddings with monitoring: %s", ' '.join(cmd))

    try:
        worker_result, runtime_stats, system_info = ResourceMonitor.monitor_process(
            worker_fn=_worker_benchmark_embeddings,
            worker_args=(cmd, verbose),
            sample_interval_s=sample_interval,
            normalize_resource_data=False
        )

        if not worker_result or not worker_result.get('success'):
            logger.error("benchmark_embeddings failed")
            if worker_result:
                logger.error("stderr: %s", worker_result.get('stderr', ''))
            return None

        raw_stdout = worker_result.get('stdout', '')
        if not raw_stdout.strip():
            logger.error("benchmark_embeddings produced no stdout output")
            return None

        try:
            bench_results = json.loads(raw_stdout)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse benchmark_embeddings JSON output: %s", exc)
            logger.error("stdout was: %s", raw_stdout)
            return None

        # Extract convenience metrics.
        reported_model = _query_server_model(server_url)
        model_name = reported_model or (
            os.path.basename(server_model) if server_model else None)

        conv_metrics = _extract_embeddings_convenience_metrics(
            bench_results,
            server_options=server_options,
            extra_bench_options=extra_bench_options,
            model=model_name,
            description=description,
            fixed_server_options=constant_server_options,
            fixed_bench_options=fixed_bench_options,
        )
        if total_runs > 1:
            conv_metrics['runs'] = 'Run #%d' % run_number
        conv_metrics['variant'] = extract_variant_number_from_path(output_path)

        combined_results = {
            'embedding_bench_results': bench_results,
            'convenience_metrics': conv_metrics,
            'runtime_stats': runtime_stats,
            'system_info': system_info
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2)

        logger.info("Benchmark results saved to: %s", output_path)

        return output_path

    except RuntimeError as exc:
        logger.error("Resource monitoring failed: %s", exc)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to run benchmark_embeddings with monitoring: %s", exc)
        return None


def _run_benchmarks_for_server_config(
    install_dir: str,
    server_model: Optional[str],
    constant_server_options: List[str],
    variable_server_options: Optional[List[str]],
    fixed_bench_options: List[str],
    extra_bench_option_sets: Optional[List[str]],
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
    Run all benchmark_embeddings configurations for a single server configuration.

    Args:
        install_dir: Directory containing llama-server.
        server_model: Path to the GGUF model for llama-server.
        constant_server_options: Server options applied to all runs.
        variable_server_options: Server options specific to this configuration.
        fixed_bench_options: Benchmark options applied to all runs.
        extra_bench_option_sets: List of additional benchmark option strings.
        output_dir: Directory to save benchmark results.
        hostname: Machine hostname for filenames.
        sample_interval: Seconds between resource monitor samples.
        verbose: If True, show verbose output.
        power_mode_label: Optional shortened power mode name for filename.
        description: Optional custom description for the benchmark run.
        runs: Number of full repeated runs per configuration.
        reset_environment: If True, reset the benchmark environment before each run.

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

    # Combine constant and variable server options, then inject --embedding.
    base_server_options = combine_options(
        effective_constant_server_options,
        variable_server_options
    ) or []
    if '--embedding' not in base_server_options:
        base_server_options = base_server_options + ['--embedding']

    for run_number in range(1, runs + 1):
        logger.info("Starting run %d of %d for current server configuration.", run_number, runs)

        if reset_environment:
            reset_benchmark_environment()

        if check_existing_llama_server_running():
            return written_json_files

        with LlamaServerProcess(
            install_dir,
            server_model,
            base_server_options,
            verbose=verbose
        ) as server:
            if server.url is None:
                logger.error("Failed to start llama-server, skipping this configuration.")
                return written_json_files

            if not server_model:
                reported_model = _query_server_model(server.url)
                if not reported_model:
                    logger.error("No model loaded on llama-server. When -m is omitted, "
                                 "the model must be specified via -c, -s, or -j options.")
                    return written_json_files
                logger.info("Server loaded model: %s", reported_model)

            for extra_options in iter_option_sets(extra_bench_option_sets):
                combined_bench_options = combine_options(fixed_bench_options, extra_options) or []

                output_file = run_benchmark_embeddings_with_monitoring(
                    server_url=server.url,
                    bench_options=combined_bench_options,
                    output_dir=output_dir,
                    hostname=hostname,
                    install_dir=install_dir,
                    server_model=server_model,
                    sample_interval=sample_interval,
                    server_options=variable_server_options,
                    extra_bench_options=extra_options,
                    verbose=verbose,
                    power_mode_label=power_mode_label,
                    constant_server_options=effective_constant_server_options,
                    fixed_bench_options=fixed_bench_options,
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
    iterates over the Cartesian product of (install × model × server-params ×
    bench-params).  Each combination launches a server instance (with --embedding)
    and runs benchmark_embeddings.py against it.  Resource usage (CPU, RAM, GPU) is
    monitored during each benchmark.

    Args:
        parser_args:
            Namespace from ``argparse.parse_args()`` with fields:
              - ``output_dir``: Output directory for benchmark results.
              - ``model``: Model path(s).
              - ``llamacpp_dir``: One or more llama.cpp install roots.
              - ``constant_server_parms``: Server options for all runs.
              - ``server_parms``: Server options that vary between runs.
              - ``extra_options``: benchmark_embeddings option sets.
              - ``fixed_options``: benchmark_embeddings options for all runs.
              - ``sample_interval``: Seconds between resource monitor samples.

    Returns:
        bool: True if at least one benchmark succeeded, False otherwise.
    """
    if not create_output_directory_tree(parser_args.output_dir):
        logger.error('Unable to create output directory tree.')
        return False

    model_specs = parser_args.model if parser_args.model else [None]

    constant_server_options = []
    if parser_args.constant_server_parms:
        constant_server_options = create_options_list(parser_args.constant_server_parms)
        logger.debug('Constant server options: %s', constant_server_options)

    fixed_bench_options = []
    if parser_args.fixed_options:
        fixed_bench_options = create_options_list(parser_args.fixed_options)
        logger.debug('Fixed benchmark options: %s', fixed_bench_options)

    server_installs = find_llama_server_installs(parser_args.llamacpp_dir)
    if not server_installs:
        logger.error('No valid llama-server installations found.')
        return False

    hostname = socket.gethostname()

    written_json_files: List[str] = []

    power_modes = getattr(parser_args, 'power_mode', None)

    for power_cli, power_api in _iter_power_modes(power_modes):
        if power_api:
            logger.info("Setting power mode to: %s", power_api)
            if not set_power_mode(power_api):
                logger.warning("Failed to set power mode to %s. Continuing anyway.",
                               power_api)
            else:
                current_mode = get_power_mode()
                logger.debug("Current power mode: %s", current_mode)

        power_mode_label = POWER_MODE_SHORT.get(power_cli, power_cli) if power_cli else None

        for install_dir in server_installs:
            logger.info("Using llama-server installation: %s", install_dir)

            for model_spec in model_specs:
                if model_spec is not None:
                    server_model, _ = parse_model_spec(model_spec)
                    logger.info("Benchmarking model: %s", server_model)
                else:
                    server_model = None
                    logger.info("No explicit model; expecting model from "
                                "server options (-c, -s, or -j).")

                # Iterate over variable server parameters.
                for server_opts in iter_option_sets(parser_args.server_parms):
                    logger.debug("Server variable options: %s", server_opts)

                    files_from_config = _run_benchmarks_for_server_config(
                        install_dir=install_dir,
                        server_model=server_model,
                        constant_server_options=constant_server_options,
                        variable_server_options=server_opts,
                        fixed_bench_options=fixed_bench_options,
                        extra_bench_option_sets=parser_args.extra_options,
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

                # Iterate over joint server+bench option pairs.
                for joint_server, joint_bench in parse_joint_options(
                        parser_args.joint_options):
                    logger.debug("Joint server options: %s, bench options: %s",
                                 joint_server, joint_bench)

                    joint_bench_sets = None
                    if joint_bench:
                        joint_bench_sets = [' '.join(joint_bench)]

                    files_from_config = _run_benchmarks_for_server_config(
                        install_dir=install_dir,
                        server_model=server_model,
                        constant_server_options=constant_server_options,
                        variable_server_options=joint_server,
                        fixed_bench_options=fixed_bench_options,
                        extra_bench_option_sets=joint_bench_sets,
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

    return len(written_json_files) > 0


class _BenchmarkArgumentParser(BenchmarkArgumentParser):
    """Argument parser with mass_benchmark_embeddings-specific examples."""

    def format_help(self):
        """Append practical examples to the default help text."""
        help_text = super().format_help()
        helplines = help_text.splitlines()
        helplines.append(' ')
        helplines.append('EXAMPLES: ')
        helplines.append(' ')
        helplines.append(r'  Benchmark a single model:')
        helplines.append(' ')
        helplines.append(
            r'    python mass_benchmark_embeddings.py '
            r'-m \models\model.gguf -l \llama-b6876-bin-win-vulkan-x64 -o \json\testrun')
        helplines.append(' ')
        helplines.append(r'  Run with constant server options and variable server options:')
        helplines.append(' ')
        helplines.append(
            r'    python mass_benchmark_embeddings.py '
            r'-m \models\model.gguf -l \llama-dir -o \json\testrun -c "-ngl 99" '
            r'-s "-fa 0" -s "-fa 1"')
        helplines.append(' ')
        helplines.append(r'  Run with multiple benchmark option sets:')
        helplines.append(' ')
        helplines.append(
            r'    python mass_benchmark_embeddings.py '
            r'-m \models\model.gguf -l \llama-dir -o \json\testrun '
            r'-e "--samples 500" -e "--samples 2000"')
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
    Create and configure the argument parser for mass_benchmark_embeddings.

    Returns:
        Configured argument parser with all benchmark options.
    """
    parser = _BenchmarkArgumentParser(
        description="Run embedding benchmarks against llama-server "
                    "for multiple models and configurations.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser._action_groups.pop()  # pylint: disable=protected-access
    required = parser.add_argument_group("Required arguments")
    optional = parser.add_argument_group('Optional arguments')

    required.add_argument('-o', '--output-dir', type=str, required=True,
                          help=textwrap.dedent('''\
                          Output directory to store JSON files.
                          The directory will be created if it
                          does not exist.
                          '''))

    optional.add_argument('-m', '--model', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Path to a GGUF model file to pass to llama-server via --model.
        Multiple -m parameters are allowed and will create separate runs.
        If omitted, the model must be specified via -c, -s, or -j options.
        '''))

    optional.add_argument('-l', '--llamacpp-dir', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Path to directory that contains an installation of llama.cpp
        with llama-server.exe.
        Multiple -l parameters are allowed, which accumulate,
        and will create separate benchmark runs.
        Default is current working directory.
        '''))

    optional.add_argument('-c', '--constant-server-parms', type=str, required=False,
        help=textwrap.dedent('''\
        Options to pass to llama-server for all runs, surrounded by quotes.
        These options are appended to every llama-server command line.
        --embedding is always injected automatically and does not need to be
        specified here.
        Only one -c parameter is allowed.
        Example: -c "-ngl 99 --port 8080"
        '''))

    optional.add_argument('-s', '--server-parms', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Additional options to pass to llama-server, surrounded by quotes.
        Multiple -s parameters are allowed, which accumulate,
        and will create separate server instances for each.
        Example: -s "-fa 0" -s "-fa 1" will run benchmarks twice,
        once with flash attention disabled, once enabled.
        '''))

    optional.add_argument('-e', '--extra-options', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Additional options to pass to benchmark_embeddings.py, surrounded by quotes.
        Multiple -e parameters are allowed, which accumulate,
        and will be used for successive benchmark runs.
        Example: -e "--samples 500 --warmup 5" -e "--samples 2000 --warmup 20"
        '''))

    optional.add_argument('-f', '--fixed-options', type=str, required=False,
        help=textwrap.dedent('''\
        Fixed options to pass to benchmark_embeddings.py for all runs, surrounded
        by quotes. Unlike -e, this does not create additional benchmark
        runs; the options are applied to every benchmark_embeddings run.
        Only one -f parameter is allowed.
        Example: -f "--concurrency 4 --request-timeout-sec 120"
        '''))

    optional.add_argument('-j', '--joint-options', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Combined server and benchmark options in one string, separated by a
        comma.  The part before the comma is passed to llama-server; the
        part after the comma is passed to benchmark_embeddings.py.  If no
        comma is present, the entire string is treated as benchmark options.
        Multiple -j parameters are allowed, which accumulate, and will
        create separate benchmark runs for each.
        Example: -j "-ngl 99,--samples 1000" -j "-ngl 0,--samples 500"
        '''))

    optional.add_argument('-i', '--sample-interval', type=float,
        default=DEFAULT_SAMPLE_INTERVAL, required=False,
        help=textwrap.dedent(f'''\
        Interval in seconds between resource monitoring samples.
        Default is {DEFAULT_SAMPLE_INTERVAL} seconds.
        '''))

    optional.add_argument('-r', '--runs', type=int, required=False, default=1,
        help=textwrap.dedent('''\
        Number of full repeated runs per configuration.
        Each run performs environment reset, launches llama-server,
        executes benchmark_embeddings option sets, and terminates llama-server.
        Default: 1
        '''))

    optional.add_argument('-v', '--verbose', action='store_true',
        help='Print verbose output.')

    optional.add_argument('-d', '--description', type=str, required=False,
        default='benchmark-embeddings',
        help=textwrap.dedent('''
        Custom description for the benchmark run.
        Stored in the convenience_metrics field of the output JSON
        and used as a label differentiator in plots.
        Default: "benchmark-embeddings"
        '''))

    optional.add_argument('-p', '--power-mode', type=str, action='append', required=False,
        choices=['best-performance', 'balanced', 'best-power-efficiency'],
        help=textwrap.dedent('''\
        Set Windows power mode before running benchmarks.
        Multiple -p parameters are allowed, which accumulate,
        and will create separate benchmark runs for each power mode.
        Choices: best-performance, balanced, best-power-efficiency.
        '''))

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

    preprocessed_args = preprocess_dash_value_args(
        sys.argv[1:], _DASH_VALUE_OPTIONS, _QUOTED_VALUE_OPTIONS
    )
    parser = _create_argument_parser()
    args = parser.parse_args(preprocessed_args)

    setup_logging(args.verbose)

    initial_power_mode = get_power_mode()
    power_modes_requested = getattr(args, 'power_mode', None)
    success = False

    try:
        success = benchmark_models(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Stopping llama-server and exiting.")
    finally:
        if power_modes_requested and initial_power_mode:
            logger.info("Restoring power mode to: %s", initial_power_mode)
            if not set_power_mode(initial_power_mode):
                logger.warning("Failed to restore power mode.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Only execute main() when invoked as a script; importing this module will
    # not trigger argument parsing or benchmarking.
    main()
