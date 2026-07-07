#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
Run llama-bench over one or more models in one or more llama.cpp installations.

This script orchestrates running `llama-bench` across multiple models and multiple
llama.cpp installations, capturing the raw benchmark output along with machine
resource usage (RAM/CPU/GPU) sampled during each run. It writes enriched JSON
results to disk for later analysis.

High-level behavior
-------------------
- Discovers GGUF model files from specified directories and/or explicit paths.
- Runs `llama-bench` across multiple llama.cpp installations in sequence.
- Monitors system resources (RAM, CPU, GPU utilization and memory) during each
  benchmark run via the ResourceMonitor module.
- Calculates convenience metrics (TTFT, MTBT) from benchmark results.
- Writes enriched JSON results combining llama-bench output, resource statistics,
  convenience metrics, and system information.

Key design choices
------------------
- Uses multiprocessing to isolate benchmark execution and allow independent
  resource monitoring without blocking the main process.
- Supports multiple option sets via `-e` flags, creating separate benchmark
  runs for each combination.
- Fixed options via `-f` are applied to all runs without creating additional
  benchmark combinations.
Expected environment
--------------------
- Requires `psutil` for system metrics (via ResourceMonitor).
- Requires `pywin32` for Windows GPU monitoring via PDH.
- Requires library modules: resource_monitor, platform_support,
  dashed_options, convenience_metrics, json_utils.
- Designed exclusively for Windows 10/11 environments.

Imports
-------
- argparse: CLI argument parsing.
- dataclasses: Configuration and metadata containers.
- json: Serialize and deserialize benchmark results.
- logging: Structured logging throughout the module.
- multiprocessing: Isolate benchmark execution from resource monitoring.
- os, sys: Filesystem and path manipulation.
- platform: Detect Windows environment.
- socket: Retrieve hostname for output filenames.
- textwrap: Format help text in argument parser.
- typing: Type annotations for readability and tooling support.
"""
import argparse
import os
import socket
import sys
import textwrap

from typing import List

# Configure a module-level logger. The main() routine will set the global
# logging level and format via logging.basicConfig().
import logging
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
    from platform_support import temporarily_change_directory, create_output_directory_tree
except ImportError:
    raise RuntimeError("Cannot find platform_support module.") from None

# Import option parsing utilities to support dashed CLI-style option strings
# that are provided as free-form text in arguments.
try:
    from dashed_options import create_options_list, iter_option_sets, preprocess_dash_value_args
except ImportError:
    raise RuntimeError("Cannot find dashed_options module.") from None

# Import helper functions for benchmark execution and file I/O.
try:
    from llama_bench_helpers import (
        find_llama_bench_installs,
        run_benchmark_and_write,
        combine_options,
        run_warmup,
        BenchmarkRunConfig
    )
except ImportError:
    raise RuntimeError("Cannot find llama_bench_helpers module.") from None

# Import benchmark environment reset helper.
try:
    from benchmark_environment import reset_benchmark_environment
except ImportError:
    raise RuntimeError("Cannot find benchmark_environment module.") from None

# Import sample_interval from resource_monitor_helpers to avoid circular imports.
try:
    from resource_monitor_helpers import sample_interval
except ImportError:
    raise RuntimeError("Cannot find resource_monitor_helpers module.") from None

# Import power configuration utilities for setting power mode.
try:
    from power_config import set_power_mode, get_power_mode
except ImportError:
    raise RuntimeError("Cannot find power_config module.") from None

# Import shared benchmark utilities to avoid code duplication across mass_* scripts.
try:
    from mass_bench_common import (
        BenchmarkArgumentParser,
        POWER_MODE_SHORT,
        iter_power_modes as _iter_power_modes,
        setup_logging,
        check_python_version,
        check_windows_platform,
    )
except ImportError:
    raise RuntimeError("Cannot find mass_bench_common module.") from None


def _run_benchmarks_for_install(
    bench_install: str,
    parser_args,
    hostname: str,
    fixed_options_list: List[str]
) -> List[str]:
    """Run all model/option-set/power-mode combinations for a single install directory."""
    written_json_files: List[str] = []
    power_modes = getattr(parser_args, 'power_mode', None)

    with temporarily_change_directory(bench_install):
        logger.info('Changed directory to %s', bench_install)

        for power_cli, power_api in _iter_power_modes(power_modes):
            # Set power mode if specified
            if power_api:
                logger.info("Setting power mode to: %s", power_api)
                if not set_power_mode(power_api):
                    logger.warning("Failed to set power mode to %s. Continuing anyway.",
                                   power_api)
                else:
                    # Verify the change
                    current_mode = get_power_mode()
                    logger.debug("Current power mode: %s", current_mode)

            for model in parser_args.model:
                for run_number in range(1, getattr(parser_args, 'runs', 1) + 1):
                    logger.info("Starting run %d for model %s", run_number, model)

                    # Reset benchmark environment before running llama-bench.
                    if getattr(parser_args, 'reset_environment', True):
                        reset_benchmark_environment()

                    run_warmup(model, bench_install)

                    for extra_options in iter_option_sets(parser_args.extra_options):
                        combined_options = combine_options(fixed_options_list, extra_options)

                        # display_opts contains only -e options (for convenience_metrics)
                        display_opts = list(extra_options) if extra_options else None

                        # Power mode label is kept separate for filename generation
                        power_mode_label = POWER_MODE_SHORT.get(
                            power_cli, power_cli) if power_cli else None

                        config = BenchmarkRunConfig(
                            model=model,
                            options=combined_options,
                            display_options=display_opts,
                            hostname=hostname,
                            output_dir=parser_args.output_dir,
                            sample_interval=parser_args.sample_interval,
                            verbose=parser_args.verbose,
                            normalize_resource_data=False,
                            description=getattr(parser_args, 'description', None),
                            fixed_options=fixed_options_list,
                            power_mode_label=power_mode_label,
                            run_number=run_number,
                            total_runs=getattr(parser_args, 'runs', 1)
                        )
                        files_from_run = run_benchmark_and_write(config)
                        written_json_files.extend(files_from_run)

    return written_json_files


def benchmark_models(parser_args) -> bool:
    """
    Run benchmarks across all selected models, option sets, and installs.

    Discovers GGUF models and llama-bench installations from CLI arguments, then
    iterates over the Cartesian product of (install × model × option-set). Each
    combination is run from within the target install directory, and a JSON file
    is written to the output directory named with hostname, build, model, and
    options.

    Args:
        parser_args:
            Namespace from ``argparse.parse_args()`` with fields:
              - ``output_dir``: Output directory for benchmark results.
              - ``gguf_dir``: Directory containing GGUF model files.
              - ``model``: Optional model path(s) or filters.
              - ``llamacpp_dir``: One or more llama.cpp install roots.
              - ``extra_options``: Additional llama-bench option sets (strings).

    Returns:
        bool: True if at least one benchmark succeeded, False otherwise.

    Side Effects:
        - Creates the output directory tree if needed.
        - Temporarily changes CWD to each install while running.
        - Writes one or more result JSON files to disk.
        - Logs progress and any issues.

    """
    # Ensure the output directory exists (and any intermediate parents).
    if not create_output_directory_tree(parser_args.output_dir):
        logger.error('Unable to create output directory tree.')
        return False

    # Parse fixed options (if provided) into a list that will be applied to all runs.
    fixed_options_list = []
    if hasattr(parser_args, 'fixed_options') and parser_args.fixed_options:
        fixed_options_list = create_options_list(parser_args.fixed_options)
        logger.debug('Fixed options: %s', fixed_options_list)

    hostname = socket.gethostname()
    llama_bench_installs = find_llama_bench_installs(parser_args.llamacpp_dir)

    # Collect all JSON files written during this invocation.
    written_json_files: List[str] = []

    for bench_install in llama_bench_installs:
        files_from_install = _run_benchmarks_for_install(
            bench_install, parser_args, hostname, fixed_options_list
        )
        written_json_files.extend(files_from_install)

    logger.info("All benchmarks complete. Generated %d JSON file(s).",
                 len(written_json_files))

    # Return True if at least one benchmark produced output
    return len(written_json_files) > 0


class _BenchmarkArgumentParser(BenchmarkArgumentParser):
    """Argument parser with mass_llama_bench-specific examples."""

    def format_help(self):
        """Append practical examples to the default help text."""
        help_text = super().format_help()
        helplines = help_text.splitlines()
        helplines.append(' ')
        helplines.append('EXAMPLES: ')
        helplines.append(' ')
        helplines.append(r'  Run llama-bench from \llama-b6876-bin-win-vulkan-x64 over all ')
        helplines.append(r'  gguf model files in \models. Disable flash attention for all ')
        helplines.append(r'  runs and save json files in the \json\testrun directory:')
        helplines.append(' ')
        helplines.append(r'    python mass_llama_bench.py -g \models -l \llama-b6876-bin-win-vulkan-x64 -o \json\testrun -e "-fa 0"')  # pylint: disable=line-too-long
        helplines.append('\n')

        return os.linesep.join(helplines)


# Options that accept values which may start with dashes
_DASH_VALUE_OPTIONS = {
    '-r', '--runs',
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
    Create and configure the argument parser for mass_llama_bench.

    Returns:
        Configured argument parser with all benchmark options.
    """
    # Create the parser with a succinct description of the tool's purpose.
    parser = _BenchmarkArgumentParser(
        description="Run llama-bench benchmarks for multiple " +
                    "models and/or llama.cpp releases.",
        formatter_class=argparse.RawTextHelpFormatter)

    # Reorganize default argparse groups so we can present arguments in the
    # desired sections (required/at-least-one/optional).
    parser._action_groups.pop()     # pylint: disable=protected-access
    required = parser.add_argument_group("Required argument")
    optional = parser.add_argument_group('Optional arguments')

    # Output directory is mandatory and will be created if missing.
    required.add_argument('-o', '--output-dir', type=str, required=True,
                          help=textwrap.dedent('''\
                          Output directory to store JSON files.
                          The directory will be created if it
                          does not exist.
                          '''))

    # One or both of these arguments have to be supplied or else no traces can be decoded.
    # Allow specifying one or more explicit model paths; may be combined with -g.
    required.add_argument('-m', '--model', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Path to a GGUF LLM file.
        Multiple -m parameters are allowed, which accumulate,
        and will be used for successive runs of llama-bench.
        '''))

    # Specify one or more llama.cpp installation directories to run from.
    optional.add_argument('-l', '--llamacpp-dir', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Path to directory that contains an installation of llama.cpp.
        Multiple -l parameters are allowed, which accumulate,
        and will be used for successive runs of llama-bench.
        Default is current working directory.
        '''))

    # Pass through extra (quoted) llama-bench arguments. Multiple -e flags accumulate
    # and are expanded into all combinations via iter_option_sets().
    optional.add_argument('-e', '--extra-options', type=str, action='append', required=False,
        help=textwrap.dedent('''\
        Additional options to pass to llama-bench, surrounded by quotes.
        Multiple -e parameters are allowed, which accumulate,
        and will be used for successive runs of llama-bench.
        '''))

    # Pass through fixed llama-bench arguments applied to all runs.
    optional.add_argument('-f', '--fixed-options', type=str, required=False,
        help=textwrap.dedent('''\
        Fixed options to pass to llama-bench for all runs, surrounded
        by quotes. Unlike -e, this does not create additional benchmark
        runs; the options are applied to every run. Only one -f parameter
        is allowed.
        '''))

    # Sampling interval for resource monitoring; validated by sample_interval().
    optional.add_argument('-s', '--sample-interval', type=sample_interval, required=False,
        help=textwrap.dedent('''\
        Number of seconds between resource monitoring samples,
        range 0.2 to 1.0 inclusive.
        '''))

    # Number of repeated full runs per configuration.
    optional.add_argument('-r', '--runs', type=int, required=False, default=1,
        help=textwrap.dedent('''\
        Number of full repeated runs per configuration.
        Each run performs optional environment reset and executes llama-bench.
        Default: 1
        '''))

    # Verbose logging includes debug-level messages and callsite information.
    optional.add_argument('-v', '--verbose', action='store_true',
        help='Print verbose output.')

    # Custom description for the benchmark run.
    optional.add_argument('-d', '--description', type=str, required=False,
        default='llama-bench',
        help=textwrap.dedent('''
        Custom description for the benchmark run.
        Stored in the convenience_metrics field of the output JSON
        and used as a label differentiator in plots.
        Default: "llama-bench"
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
        help='Enable benchmark environment reset before each llama-bench run.')

    return parser


def main():
    """
    Parse arguments, set up logging, and benchmark models.
    """
    check_python_version()
    check_windows_platform()

    # Grab the arguments and stash them away.
    # Preprocess arguments to handle values starting with dashes (e.g., -e "-fa 1")
    preprocessed_args = preprocess_dash_value_args(
        sys.argv[1:], _DASH_VALUE_OPTIONS, _QUOTED_VALUE_OPTIONS
    )
    parser = _create_argument_parser()
    args = parser.parse_args(preprocessed_args)
    if not args.sample_interval:
        # Default to a relatively fine-grained sampling cadence for resource stats.
        args.sample_interval = 0.2

    setup_logging(args.verbose)

    # Save the initial power mode so we can restore it when done.
    # This is important when -p option is used to run benchmarks across multiple
    # power modes - we want to leave the system in the same state as when we started.
    initial_power_mode = get_power_mode()
    power_modes_requested = getattr(args, 'power_mode', None)

    try:
        # Kick off the end-to-end benchmark process.
        success = benchmark_models(args)
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
