#!/usr/bin/env python3
"""
Helper functions for mass_llama_bench.py.

This module contains utility functions for benchmark execution, file I/O,
flash attention handling, and resource monitoring coordination. These
functions are separated from the main module to improve maintainability.

Public API
----------
- find_llama_bench_installs(llama_bench_dirs) -> List[str]
- benchmark_model(gguf_model_file, options, verbose) -> str
- build_output_filename_base(prefix, test_results, options) -> str
- write_json_file(outdir, prefix, output_string, options) -> Optional[str]
- was_flash_attention_activated(test_results) -> bool
- deactivate_flash_attention(options) -> List[str]
- benchmark_model_and_monitor_resources(...) -> tuple
- run_benchmark_and_write(config) -> List[str]
- BenchmarkMetadata: Dataclass for benchmark metadata
- BenchmarkRunConfig: Dataclass for benchmark run configuration
"""
from dataclasses import dataclass
import json
import logging
import multiprocessing
import os
from typing import Any, Dict, List, Optional

try:
    from json_repair import repair_json
except ImportError as e:
    raise RuntimeError(
        "Required package 'json-repair' is not installed. "
        "Install it with: pip install json-repair"
    ) from e

from platform_support import (
    execute_command,
    get_nonconflicting_output_path,
    extract_variant_number_from_path
)
from dashed_options import (
    find_dashed_options,
    convert_options_dict_to_options_list
)
from convenience_metrics import calculate_convenience_metrics

logger = logging.getLogger(__name__)


def find_llama_bench_installs(llama_bench_dirs: List[str]) -> List[str]:
    """
    Return validated llama-bench install directories or default to CWD.

    The function checks the user-provided list of directories, retains those that
    exist, warns for those that do not, and falls back to the current working
    directory when none are provided or none validate.

    Args:
        llama_bench_dirs: Paths to llama.cpp installations (where `llama-bench` resides).

    Returns:
        A list of directories to use when invoking `llama-bench`.
    """
    # Find all the installations of llama-bench.
    bench_installs = []
    if llama_bench_dirs:
        for bench_install in llama_bench_dirs:
            if os.path.isdir(bench_install):
                bench_installs.append(bench_install)
            else:
                logger.warning('Llama bench install %s not found.', bench_install)

    if not bench_installs:

        # Pick the current working directory as the default.
        bench_installs.append(os.getcwd())

    return bench_installs


def benchmark_model(gguf_model_file: str, options: List[str], verbose: bool) -> str:
    """
    Run `llama-bench` in the current working directory for a single model.

    The command is constructed with JSON output enabled and any additional
    options appended as provided.

    Args:
        gguf_model_file: Path to the model file.
        options: A flat list of CLI tokens (each dashed option and its value, if any).
        verbose: If True, echo stderr to console during execution.

    Returns:
        The `llama-bench` stdout as a string on success; otherwise, None.
    """
    # Construct the command to run llama-bench with JSON output.
    cmd = ['llama-bench', '-m', gguf_model_file, '-o', 'json']

    # Add options (if any)
    if options:
        cmd += options

    # Enable llama-bench verbose output when requested.
    if verbose:
        cmd.append('-v')

    # Log the command being executed for diagnostics.
    logger.info('Running: %s', ' '.join(cmd))

    # Run the llama-bench command.
    results, stderr, error_code = execute_command(cmd, echo_stderr = verbose)
    retval = None
    if not error_code:
        # Validate that results is valid JSON; sometimes output may be truncated.
        try:
            json.loads(results)
            retval = results
        except json.JSONDecodeError:
            # Try to repair truncated or malformed JSON.
            repaired = repair_json(results)
            try:
                json.loads(repaired)
                retval = repaired
                logger.debug('Repaired malformed JSON output')
            except json.JSONDecodeError:
                logger.warning('Invalid JSON output from llama-bench for %s', gguf_model_file)
                retval = None
    else:
        logger.warning('Unable to run %s', gguf_model_file)
        logger.warning(stderr)

    return retval


def build_output_filename_base(prefix: str,
                               test_results: List[Dict],
                               options: List[str]) -> str:
    """
    Build a descriptive base filename (without extension) from benchmark metadata.

    The filename is dynamically constructed from the prefix (typically hostname),
    build number, model filename, and any CLI options used for the benchmark run.

    Args:
        prefix: The base filename prefix (e.g., hostname).
        test_results: A list of test result dictionaries from llama-bench JSON output.
        options: A flat list of llama-bench options to append to the filename.

    Returns:
        The constructed base filename without any file extension.
    """
    name = prefix

    # Extract metadata from the first test result entry.
    if test_results and isinstance(test_results, list):
        first_result = test_results[0]
        if isinstance(first_result, dict):
            if 'build_number' in first_result:
                build_str = str(first_result['build_number'])
                if build_str.isdigit():
                    build_str = 'b' + build_str
                name += '_' + build_str
            if 'model_filename' in first_result:
                # Extract just the filename part, not the full path.
                base_name = os.path.basename(first_result['model_filename'])
                name += '_' + base_name
        else:
            logging.error('Expecting a dictionary in test_results')
    else:
        logging.error('Expecting a non-empty list for test_results')

    # Append options (if any) to capture impactful flags (e.g., `-fa 0`).
    if options:
        for option in options:
            name += '_' + option

    return name


def write_json_file(outdir: str,
                    prefix: str,
                    output_string: str,
                    options: List[str]
                    ) -> Optional[str]:
    """
    Write the composed JSON results to a file with a descriptive name.

    The filename is dynamically built from fields present in the first entry of
    the `test_results` list (e.g., build number and model filename) as well as any
    options used to run the benchmark.

    Args:
        outdir: The output directory for the file.
        prefix: The base filename prefix (e.g., hostname).
        output_string: A string containing the JSON data to write.
        options: A flat list of llama-bench options appended to the filename.

    Returns:
        The full path to the written JSON file on success; otherwise, None.

    Side Effects:
        Creates `outdir` if required (by upstream caller), then writes a single
        `.json` file whose name encodes build/model/options for traceability.
    """
    # Deserialize the JSON string to extract llama_bench_results for filename building.
    json_object = json.loads(output_string)

    test_results = []
    if json_object and isinstance(json_object, dict):
        if 'llama_bench_results' in json_object:
            test_results = json_object['llama_bench_results']
    else:
        logging.error('Expecting a dictionary')

    # Build the base filename and add the .json extension.
    name = build_output_filename_base(prefix, test_results, options)
    name += '.json'

    # Build the absolute path to the output file.
    file_path = os.path.join(outdir, name)
    file_path = get_nonconflicting_output_path(file_path)

    # Include variant number in convenience_metrics based on final file path.
    variant_number = extract_variant_number_from_path(file_path)
    if isinstance(json_object, dict):
        conv_metrics = json_object.get('convenience_metrics')
        if isinstance(conv_metrics, dict):
            conv_metrics['variant'] = variant_number

    # Serialize after variant is finalized.
    output_string = json.dumps(json_object, indent=2)

    # --- Write the file ---
    try:
        # Use UTF-8 to be robust to non-ASCII model filenames or fields.
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(output_string)
        logger.info('Wrote JSON data to %s', file_path)
        return file_path
    except IOError as e:
        logger.error(e)
        return None


def _worker_benchmark(
    gguf_model_file: str,
    options: List[str],
    verbose: bool,
    result_queue: "multiprocessing.queues.Queue[str]",
) -> None:
    """
    Run a single benchmark in a child process and push the JSON results to a queue.
    """
    # Configure logging in the child process. On Windows, multiprocessing uses
    # 'spawn' which creates a fresh interpreter without the parent's logging config.
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(message)s - (%(funcName)s in %(filename)s:%(lineno)d)",
        force=True
    )

    # The separate process ensures isolated execution while allowing the
    # ResourceMonitor to sample this process's resource usage independently.
    result = benchmark_model(gguf_model_file, options, verbose)
    result_queue.put(result)


def benchmark_model_and_monitor_resources(
    gguf_model_file: str,
    options: List[str],
    incoming_sample_interval: float,
    verbose: bool,
    normalize_resource_data: bool = False
) -> tuple:
    """
    Execute a benchmark and capture resource statistics.

    This function launches the benchmark in a worker process and concurrently
    collects runtime statistics (RAM/GPU, etc.) via ResourceMonitor.

    Args:
        gguf_model_file: Path to the model file.
        options: CLI options for `llama-bench` as a flat token list.
        incoming_sample_interval: Seconds between resource monitor samples.
        verbose: If True, echo stderr to console during benchmark execution.
        normalize_resource_data: If True, subtract prestart values from data.

    Returns:
        A tuple of (worker_result, stats, system_info), where:
          - worker_result is the raw JSON string from `llama-bench` (or None).
          - stats is a dictionary of sampled runtime statistics.
          - system_info is a dictionary of system information from systeminfo.
    """
    # Lazy import to avoid circular dependency with resource_monitor.py
    # pylint: disable=import-outside-toplevel,import-error
    from resource_monitor import ResourceMonitor

    worker_result, stats, system_info = ResourceMonitor.monitor_process(
        worker_fn=_worker_benchmark,
        worker_args=(gguf_model_file, options, verbose),
        sample_interval_s=incoming_sample_interval,
        normalize_resource_data=normalize_resource_data
    )

    return worker_result, stats, system_info


@dataclass
class BenchmarkMetadata:
    """Metadata extracted from benchmark results for display and file naming."""
    model_name: Optional[str] = None
    llama_version: Optional[str] = None
    options: Optional[List[str]] = None
    hostname: Optional[str] = None
    output_dir: Optional[str] = None


@dataclass
class BenchmarkRunConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for a single benchmark run."""
    model: str
    options: Optional[List[str]]
    display_options: Optional[List[str]]
    hostname: str
    output_dir: str
    sample_interval: float
    verbose: bool
    normalize_resource_data: bool = False
    description: Optional[str] = None
    fixed_options: Optional[List[str]] = None
    power_mode_label: Optional[str] = None
    run_number: int = 1
    total_runs: int = 1


def _extract_benchmark_metadata(
    test_results: List[Dict],
    options: Optional[List[str]],
    hostname: str,
    output_dir: str
) -> BenchmarkMetadata:
    """
    Extract model name and llama version from test results.

    Parameters
    ----------
    test_results : List[Dict]
        List of test result dictionaries from llama-bench JSON output.
    options : Optional[List[str]]
        Command-line options used for the benchmark run.
    hostname : str
        Hostname for filename prefix.
    output_dir : str
        Output directory path.

    Returns
    -------
    BenchmarkMetadata
        Dataclass containing extracted metadata.
    """
    metadata = BenchmarkMetadata(options=options, hostname=hostname, output_dir=output_dir)
    if test_results and isinstance(test_results[0], dict):
        first_result = test_results[0]
        if 'model_filename' in first_result:
            metadata.model_name = os.path.basename(first_result['model_filename'])
        if 'build_number' in first_result:
            metadata.llama_version = str(first_result['build_number'])
    return metadata


def _write_benchmark_outputs(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    test_results: List[Dict],
    runtime_stats: Dict[str, Any],
    conv_metrics: Dict[str, Any],
    system_info: Dict[str, str],
    metadata: BenchmarkMetadata,
    run_number: int = 1,
    total_runs: int = 1
) -> Optional[str]:
    """
    Write JSON results for a benchmark run.

    Parameters
    ----------
    test_results : List[Dict]
        Parsed test results from llama-bench.
    runtime_stats : Dict[str, Any]
        Runtime statistics including resource usage.
    conv_metrics : Dict[str, Any]
        Convenience metrics including TTFT, MTBT, and options.
    system_info : Dict[str, str]
        System information dictionary.
    metadata : BenchmarkMetadata
        Extracted benchmark metadata including hostname and output_dir.

    Returns
    -------
    Optional[str]
        The full path to the written JSON file on success; otherwise, None.

    Notes
    -----
    Plot generation is deferred until all benchmarks complete. See
    generate_benchmark_plots() for the post-benchmark plotting logic.
    """
    # Build and write JSON file
    new_json_dict = {
        'llama_bench_results': test_results,
        'convenience_metrics': conv_metrics,
        'runtime_stats': runtime_stats,
        'system_info': system_info
    }
    output_string = json.dumps(new_json_dict, indent=2)
    file_prefix = "llama-bench_" + metadata.hostname
    json_file_path = write_json_file(metadata.output_dir, file_prefix,
                                     output_string, metadata.options)

    return json_file_path


def run_benchmark_and_write(config: BenchmarkRunConfig) -> List[str]:
    """
    Run a single benchmark for a given model and write results to JSON.

    The function runs the benchmark with resource monitoring, enriches the output
    with convenience metrics (TTFT, MTBT), and writes a consolidated JSON file.
    The benchmark is executed exactly once per configuration.

    Args:
        config:
            A BenchmarkRunConfig dataclass containing:
            - model: The model identifier or path to be benchmarked.
            - options: List of CLI-like option tokens (may be None); used for the run.
            - display_options: List of CLI-like option tokens to include in the
              output filename for traceability and plot legends.
            - hostname: Hostname used as a filename prefix/label.
            - output_dir: Directory to write JSON files to.
            - sample_interval: Seconds between resource monitoring samples.
            - verbose: Whether to enable verbose logging.

    Returns:
        List[str]
            A list of JSON file paths written during this invocation. May contain
            one or more paths depending on caller iteration.

    Side Effects:
        - Writes one or more JSON files to ``output_dir``.
        - Emits log messages via the ``logging`` module.

    Notes:
        - Convenience metrics are stored in the `convenience_metrics` object.
        - The underlying `llama-bench` output is kept intact under `llama_bench_results`.
    """
    written_files: List[str] = []

    # Run the model and monitor resource consumption
    test_results_str, runtime_stats, system_info = \
        benchmark_model_and_monitor_resources(config.model,
                                              config.options,
                                              config.sample_interval,
                                              config.verbose,
                                              config.normalize_resource_data)

    if not test_results_str or not runtime_stats:
        logging.error('Model results and runtime stats not found.')
        return written_files

    # Convert the JSON test results from a string to a Python data structure.
    test_results = json.loads(test_results_str)

    # Extract model name from the first result's model_filename field
    model_for_metrics = ''
    if test_results and isinstance(test_results, list):
        first_result = test_results[0]
        if isinstance(first_result, dict) and 'model_filename' in first_result:
            model_for_metrics = os.path.basename(first_result['model_filename'])

    # Calculate TTFT/MTBT and format options as convenience metrics
    # Use display_options (only -e params) for convenience_metrics:options
    conv_metrics = calculate_convenience_metrics(
        test_results_str, config.display_options,
        model=model_for_metrics,
        description=config.description,
        fixed_options=config.fixed_options
    )
    if config.total_runs > 1:
        conv_metrics['runs'] = f'Run #{config.run_number}'

    # Llama-bench writes lists, so let's make sure the structure is a list.
    if not test_results or not isinstance(test_results, list):
        logger.error("Error deserializing json object.")
        return written_files

    # Build filename options: display_options plus power_mode_label (if any)
    filename_options = list(config.display_options) if config.display_options else []
    if config.power_mode_label:
        filename_options.append(config.power_mode_label)
    if config.total_runs > 1:
        filename_options.append(f'Run_{config.run_number}')
    filename_options = filename_options if filename_options else None

    # Extract metadata and write all outputs
    metadata = _extract_benchmark_metadata(
        test_results, filename_options, config.hostname, config.output_dir
    )
    json_file_path = _write_benchmark_outputs(test_results, runtime_stats, conv_metrics,
                                              system_info, metadata,
                                              run_number=config.run_number,
                                              total_runs=config.total_runs)
    if json_file_path:
        written_files.append(json_file_path)

    return written_files


# Re-export combine_options from dashed_options for backward compatibility.
# combine_options was originally defined here but has been moved to dashed_options
# to be shared across multiple scripts.
from dashed_options import combine_options  # noqa: E402  # pylint: disable=wrong-import-position


def run_warmup(model: str, bench_install: str) -> None:
    """
    Run a short benchmark to warm up the model and GPU.

    Parameters
    ----------
    model : str
        Path to the model file.
    bench_install : str
        Path to the llama.cpp installation (for logging).
    """
    cmd = ['llama-bench', '-m', model, '-r', '1', '--n_prompt', '1', '--n_gen', '0']
    _, _, return_code = execute_command(cmd)
    if return_code != 0:
        logger.warning("Warm-up run failed for model %s in install %s.",
            model, bench_install)
        logger.warning("Continuing with benchmark runs. Error code: %d", return_code)
