#!/usr/bin/env python3
"""
Convenience metrics calculation for llama-bench results.

This module provides functions to calculate derived performance metrics
such as TTFT (Time To First Token) and generation throughput (avg_ts, tokens/sec)
from llama-bench JSON output.

NOTE ON STANDARD DEVIATION CALCULATIONS
---------------------------------------
At this writing, the stddev_ns and stddev_ts values reported by llama.cpp
are unreliable. llama.cpp uses a flawed incremental variance calculation
that accumulates floating-point errors and can produce stddev values that
are orders of magnitude larger than the actual sample variance. For example,
samples with a range of ~400M nanoseconds may report stddev_ns of ~3.7B
nanoseconds.

This module recalculates standard deviations directly from the samples_ns
and samples_ts arrays using the corrected sample standard deviation formula
(Bessel's correction, dividing by n-1), which provides accurate results.

Public API
----------
- calculate_convenience_metrics(test_results_str, options) -> Dict[str, Any]
- get_non_zero_int_from_test_results(field_name, test_results) -> Optional[int]
- get_non_zero_float_from_test_results(field_name, test_results) -> Optional[float]
- calculate_stddev_from_samples(samples) -> Optional[float]
"""
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def calculate_stddev_from_samples(
    samples: List[Union[int, float]]
) -> Optional[float]:
    """
    Calculate the sample standard deviation from a list of numeric samples.

    This function computes the corrected sample standard deviation using
    Bessel's correction (dividing by n-1) rather than the population
    standard deviation (dividing by n). This is the appropriate formula
    when the samples represent a subset of all possible measurements.

    This replaces the flawed stddev values reported by llama.cpp which,
    at this writing, uses an incremental variance algorithm that accumulates
    significant floating-point errors, often producing stddev values that
    are orders of magnitude larger than the true sample standard deviation.

    Parameters
    ----------
    samples : List[Union[int, float]]
        A list of numeric sample values (integers or floats).

    Returns
    -------
    float or None
        The sample standard deviation if calculable (requires at least 2
        samples); otherwise, None.

    Examples
    --------
    >>> calculate_stddev_from_samples([12140144300, 12308847400, 12225136900])
    84678987.5  # Approximate, actual calculation varies
    """
    if not samples or len(samples) < 2:
        return None

    n = len(samples)
    mean = sum(samples) / n
    sum_squared_diff = sum((x - mean) ** 2 for x in samples)
    # Use sample standard deviation (Bessel's correction: n-1)
    variance = sum_squared_diff / (n - 1)
    return math.sqrt(variance)


def get_non_zero_int_from_test_results(field_name: str,
                                       test_results: Dict[str, Any]) -> Optional[int]:
    """
    Return a positive non-zero integer field from a test results dict.

    The function checks that a field exists, is of type int, and is > 0. Returns
    the value if valid; otherwise, None.

    Parameters
    ----------
    field_name : str
        The key to look up in the test results dictionary.
    test_results : Dict[str, Any]
        A dictionary containing test result fields and values.

    Returns
    -------
    int or None
        The integer value associated with `field_name` if it exists and is
        greater than zero; otherwise, None.
    """
    retval = None
    if field_name in test_results:
        field_value = test_results[field_name]
        if isinstance(field_value, int):
            # The field value can be zero if only one type of test was
            # run, so don't log a warning if it's not a positive number.
            # Just return None instead.
            if field_value > 0:
                retval = field_value
        else:
            logger.warning('Field value is not an integer')
    else:
        logger.warning('Unable to find %s in test_results dictionary.', field_name)

    return retval


def get_non_zero_float_from_test_results(field_name: str,
                                         test_results: Dict[str, Any]) -> Optional[float]:
    """
    Return a positive non-zero float field from a test results dict.

    The function checks that a field exists, is of type float, and is > 0.0.
    Returns the value if valid; otherwise, None.

    Parameters
    ----------
    field_name : str
        The key to look up in the test results dictionary.
    test_results : Dict[str, Any]
        A dictionary containing test result fields and values.

    Returns
    -------
    float or None
        The floating-point value associated with `field_name` if it exists
        and is greater than zero; otherwise, None.
    """
    retval = None
    if field_name in test_results:
        field_value = test_results[field_name]
        if isinstance(field_value, float):
            if field_value > 0.0:
                retval = field_value
            else:
                logger.warning('Field value is not positive')
        else:
            logger.warning('Field value is not a float.')
    else:
        logger.warning('Unable to find %s in test_results dictionary.', field_name)

    return retval


def calculate_ttft(test_results: List[Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate TTFT (time to first token, ms) and its standard deviation from test results.

    This function scans a list of result dicts and looks for the first entry
    that has both a positive non-zero `n_prompt` and a positive non-zero
    `avg_ns` (average nanoseconds). It then converts `avg_ns` to milliseconds
    and rounds to three decimals.

    Standard deviation is calculated directly from `samples_ns` rather than
    using the pre-computed `stddev_ns` from llama.cpp which, at this writing,
    uses a flawed incremental variance algorithm that produces unreliable
    results.

    Parameters
    ----------
    test_results : List[Any]
        A list of test result entries, typically dictionaries containing
        performance metrics.

    Returns
    -------
    Tuple[float or None, float or None]
        A tuple of (TTFT in milliseconds, stddev in milliseconds) if calculable;
        otherwise, (None, None).
    """
    for result in test_results:
        if isinstance(result, dict):
            n_prompt = get_non_zero_int_from_test_results('n_prompt', result)
            if n_prompt:
                # `avg_ns` is expected to be an integer number of nanoseconds.
                avg_ns = get_non_zero_int_from_test_results('avg_ns', result)
                if avg_ns:
                    ttft = round((float(avg_ns) / 1000000.0), 3)
                    # Calculate stddev from samples_ns instead of using the flawed
                    # stddev_ns value from llama.cpp. At this writing, the llama.cpp
                    # incremental variance calculation accumulates floating-point errors.
                    ttft_stddev = None
                    samples_ns = result.get('samples_ns')
                    if samples_ns and isinstance(samples_ns, list) and len(samples_ns) >= 2:
                        stddev_ns = calculate_stddev_from_samples(samples_ns)
                        if stddev_ns is not None:
                            ttft_stddev = round(stddev_ns / 1000000.0, 3)
                    return (ttft, ttft_stddev)
        else:
            logger.error('Expecting a dictionary of test results')

    return (None, None)


def calculate_avg_ts(test_results: List[Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract avg_ts (average tokens/sec) and its standard deviation from the first generation step.

    This function scans result dicts looking for `n_gen` > 0 and `avg_ts` > 0.0
    (tokens/sec). It returns avg_ts rounded to two decimals.

    Standard deviation is calculated directly from `samples_ts` rather than
    using the pre-computed `stddev_ts` from llama.cpp which, at this writing,
    uses a flawed incremental variance algorithm that produces unreliable
    results.

    Parameters
    ----------
    test_results : List[Any]
        A list of test result entries, typically dictionaries containing
        performance metrics.

    Returns
    -------
    Tuple[float or None, float or None]
        A tuple of (avg_ts in tokens/sec, stddev_ts) if present; otherwise, (None, None).
    """
    for result in test_results:
        if isinstance(result, dict):
            n_gen = get_non_zero_int_from_test_results('n_gen', result)
            if n_gen:
                avg_ts = get_non_zero_float_from_test_results('avg_ts', result)
                if avg_ts:
                    avg_ts_rounded = round(avg_ts, 2)
                    # Calculate stddev from samples_ts instead of using the flawed
                    # stddev_ts value from llama.cpp. At this writing, the llama.cpp
                    # incremental variance calculation accumulates floating-point errors.
                    stddev_ts_rounded = None
                    samples_ts = result.get('samples_ts')
                    if samples_ts and isinstance(samples_ts, list) and len(samples_ts) >= 2:
                        stddev_ts = calculate_stddev_from_samples(samples_ts)
                        if stddev_ts is not None:
                            stddev_ts_rounded = round(stddev_ts, 2)
                    return (avg_ts_rounded, stddev_ts_rounded)
        else:
            logger.error('Expecting a dictionary of test results')

    return (None, None)


def calculate_convenience_metrics(
    test_results: str,
    options: Optional[List[str]] = None,
    model: Optional[str] = None,
    description: Optional[str] = None,
    fixed_options: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate derived convenience metrics (TTFT, avg_ts) and format options.

    The function parses a JSON string containing test results, calculates TTFT and
    extracts avg_ts (tokens/sec) from the first step where n_gen is non-zero.
    Results are returned in a dictionary suitable for the convenience_metrics
    field in the output JSON. Standard deviations are also included if available.

    Parameters
    ----------
    test_results : str
        A JSON-encoded string representing a list of test result objects.
    options : Optional[List[str]]
        Optional list of variable CLI options used for the benchmark run (from -e).
    model : Optional[str]
        Model name/identifier for the benchmark.
    description : Optional[str]
        Descriptive label for the benchmark run.
    fixed_options : Optional[List[str]]
        Optional list of fixed CLI options applied to all runs (from -f).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing ttft_ms, ttft_stddev_ms, avg_ts, avg_ts_stddev,
        options, fixed_options (if available). Returns an empty dictionary if no
        metrics can be calculated.
    """
    conv_metrics: Dict[str, Any] = {}
    test_results_object = json.loads(test_results)
    if isinstance(test_results_object, list):
        ttft, ttft_stddev = calculate_ttft(test_results_object)
        if ttft:
            conv_metrics['ttft_ms'] = ttft
        if ttft_stddev:
            conv_metrics['ttft_stddev_ms'] = ttft_stddev
        avg_ts, avg_ts_stddev = calculate_avg_ts(test_results_object)
        if avg_ts:
            conv_metrics['avg_ts'] = avg_ts
        if avg_ts_stddev:
            conv_metrics['avg_ts_stddev'] = avg_ts_stddev
        if options:
            conv_metrics['options'] = ' '.join(options)
        if fixed_options:
            conv_metrics['fixed_options'] = ' '.join(fixed_options)
        if model:
            conv_metrics['model'] = model
        if description:
            conv_metrics['description'] = description
    else:
        logger.error('Expecting a list')

    return conv_metrics
