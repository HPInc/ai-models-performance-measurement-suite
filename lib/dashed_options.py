"""
Functions for manipulating command line argument lists where each argument
starts with either '-' or '--'.

Public API
----------
- find_dashed_options(options_string) -> Dict[str, Tuple[str, Any]]
- convert_options_dict_to_options_list(options_dict) -> List[str]
- create_options_list(options_string) -> List[str]
- iter_option_sets(option_sets) -> Iterator[Optional[List[str]]]
- combine_options(constant_options, variable_options) -> Optional[List[str]]
- parse_joint_options(joint_options) -> Iterator[Tuple[Optional[List[str]], Optional[List[str]]]]
- preprocess_dash_value_args(argv, dash_value_options, quoted_value_options) -> List[str]
"""
from typing import Dict, Optional, Set
from typing import List
from typing import Tuple

import logging
logger = logging.getLogger(__name__)

def find_dashed_options(options_string: str) -> Dict[str, Tuple[str, any]]:
    """
    Parses a string of dashed options (e.g., "-o value --flag -f another_val")
    into a dictionary of options and their values.

    Args:
        options_string: A string containing space-separated dashed options.

    Returns:
        A dictionary where keys are option names (without the dash) and
        values are tuples of (prefix, value) where prefix is '-' or '--'
        and value is the subsequent argument (string) or None if no value
        follows.
    """
    parts = options_string.split()
    key = ''
    prefix = ''
    options = {}
    for part in parts:
        if part.startswith('--'):
            key = part[2:]  # Remove leading '--'
            prefix = '--'
            if key:
                options[key] = (prefix, None)
        elif part.startswith('-'):
            key = part[1:]  # Remove leading '-'
            prefix = '-'
            if key:
                options[key] = (prefix, None)
        else:
            if key:
                options[key] = (prefix, part)
            key = ''
            prefix = ''
    logger.debug('find_dashed_options: %s', options)
    return options


def convert_options_dict_to_options_list(options_dict: Dict[str, Tuple[str, any]]) -> List[str]:
    """
    Convert a dictionary of option names and values into a flattened list
    of command-line-style options.

    Each key in the dictionary is converted into a dashed option using the
    stored prefix ('-' or '--'). If the corresponding value is truthy, it
    is appended immediately after the dashed option. The options are
    processed in sorted key order.

    Example:
        >>> convert_options_dict_to_options_list({'a': ('-', '1'), 'use_mmap': ('--', '0')})
        ['-a', '1', '--use_mmap', '0']

    Args:
        options_dict (Dict[str, Tuple[str, Any]]): A dictionary where keys are
            option names (without leading dashes) and values are tuples of
            (prefix, value) where prefix is '-' or '--'.

    Returns:
        List[str]: A list of strings representing the command-line options
        and their associated values, formatted for sequential use.
    """
    options_list = []
    for option in sorted(options_dict):
        prefix, value = options_dict[option]
        dashed_option = prefix + option
        options_list.append(dashed_option)
        if value:
            options_list.append(value)
    return options_list

def create_options_list(extra_options: str) -> List[str]:
    """
    Parse a string of dashed options and return a flattened list of CLI tokens.

    Parses the input string to extract dashed options, removes any -o or -m
    options (reserved for output and model specification), and converts the
    remaining options to a list format suitable for command-line use.

    Args:
        extra_options: A string containing space-separated dashed options
            (e.g., "-fa 0 --use_mmap 1").

    Returns:
        A list of CLI tokens representing the parsed options and their values
        (e.g., ["-fa", "0", "--use_mmap", "1"]), excluding -o and -m options.
    """
    options_list = []
    if extra_options:
        # Strip any surrounding quotes that may be included when using -f="options"
        # or -e="options" syntax on Windows command line.
        extra_options = extra_options.strip('"').strip("'")
        options_dict = find_dashed_options(extra_options)
        if options_dict:
            bad_options = ['o', 'm']
            for bad_option in bad_options:
                options_dict.pop(bad_option, None)
            options_list = convert_options_dict_to_options_list(options_dict)

    return options_list

def iter_option_sets(extra_options: List[str]):
    """
    Yield a normalized stream of options lists:
    - If extra_options is provided, convert each to an options list.
    - Otherwise, yield a single None to indicate 'no extra options'.
    """
    if extra_options:
        for opt in extra_options:
            yield create_options_list(opt)
    else:
        yield None


def combine_options(
    constant_options: List[str],
    variable_options: Optional[List[str]]
) -> Optional[List[str]]:
    """
    Combine constant and variable options into a single list.

    Args:
        constant_options: Options applied to every run (may be empty).
        variable_options: Options specific to this run (may be None).

    Returns:
        Combined options list, or None if both inputs are empty/None.
    """
    if not constant_options and not variable_options:
        return None
    combined = []
    if constant_options:
        combined.extend(constant_options)
    if variable_options:
        combined.extend(variable_options)
    return combined


def parse_joint_options(joint_options: Optional[List[str]]):
    """
    Parse joint option strings into (server_options, benchy_options) pairs.

    Each joint option string may contain up to two substrings separated by a
    comma.  If a comma is present, the first substring is treated as server
    parameters and the second as llama-benchy parameters.  If no comma is
    present, the entire string is treated as llama-benchy parameters (server
    parameters will be None).

    Args:
        joint_options: List of joint option strings from -j arguments,
                       or None if no -j was specified.

    Yields:
        Tuples of (server_options_list_or_None, benchy_options_list_or_None).
        If joint_options is None or empty, yields nothing.
    """
    if not joint_options:
        return
    for joint_str in joint_options:
        if ',' in joint_str:
            server_part, benchy_part = joint_str.split(',', 1)
            server_opts = create_options_list(server_part.strip()) if server_part.strip() else None
            benchy_opts = create_options_list(benchy_part.strip()) if benchy_part.strip() else None
        else:
            server_opts = None
            benchy_opts = create_options_list(joint_str.strip()) if joint_str.strip() else None
        yield (server_opts, benchy_opts)


def preprocess_dash_value_args(
    argv: List[str],
    dash_value_options: Set[str],
    quoted_value_options: Optional[Set[str]] = None
) -> List[str]:
    """
    Preprocess command-line arguments to handle values that start with dashes.

    When using ``-e "-fa 1"`` or ``-f "-ngl 99"``, argparse interprets the
    value as a new option because it starts with ``-``. This function converts
    such patterns to the ``=`` syntax (e.g., ``-e="-fa 1"``) which argparse
    handles correctly. Also strips quotes from values when using ``=`` syntax.

    Args:
        argv: The original argument list (typically ``sys.argv[1:]``).
        dash_value_options: Set of option flags whose values may start with
            dashes (e.g., ``{'-e', '--extra-options', '-f', '--fixed-options'}``).
        quoted_value_options: Set of option flags whose values may need quote
            stripping when using ``=`` syntax. Defaults to *dash_value_options*
            when not provided.

    Returns:
        A modified argument list with dash-value arguments converted to ``=``
        syntax.
    """
    if quoted_value_options is None:
        quoted_value_options = dash_value_options

    result = []
    i = 0
    while i < len(argv):
        arg = argv[i]

        # Handle options using = syntax that may have quoted values
        if '=' in arg:
            opt_part, val_part = arg.split('=', 1)
            if opt_part in quoted_value_options:
                # Strip surrounding quotes from the value
                val_part = val_part.strip('"').strip("'")
                result.append(f'{opt_part}={val_part}')
                i += 1
                continue

        # Check if this is one of our special options (without = syntax)
        if arg in dash_value_options:
            # Check if there's a next argument that looks like a value
            if i + 1 < len(argv):
                next_arg = argv[i + 1]
                # If the next arg starts with - but isn't a known option flag,
                # it's likely a value like "-fa 1". Combine using = syntax.
                if next_arg.startswith('-') and not next_arg.startswith('--'):
                    # Combine into = syntax so argparse treats it as a value
                    result.append(f'{arg}={next_arg}')
                    i += 2
                    continue
        result.append(arg)
        i += 1

    return result
