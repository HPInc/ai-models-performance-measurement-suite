#!/usr/bin/env python3
"""
Operating system support functions for command execution, environment detection,
temporary directory changes, and simple filesystem utilities.

This module centralizes a few cross-platform helpers:

- Detect whether the runtime is native Windows or Windows Subsystem for Linux.
- Execute external commands with optional real-time stderr echoing.
- Find files using wildcard patterns while filtering out directories.
- Temporarily change the working directory within a context manager.
- Create output directory trees safely.
- Combine file path lists while eliminating duplicates.

All functions are intentionally small and dependency-light to be reusable
across tooling and scripts.
"""
import os
import re
import subprocess

from contextlib import contextmanager
from typing import List, Tuple

import logging
logger = logging.getLogger(__name__)

def execute_command(command: List[str],
                    shell: bool = False,
                    echo_stderr: bool = False) -> Tuple[str, str, int]:
    """
    Execute a subprocess and optionally echo stderr to the console in real time.

    Parameters
    ----------
    command
        The command to execute as a list of arguments (preferred). If providing a
        single string, set ``shell=True`` to allow shell parsing.
    shell
        Execute through the operating system shell when True. Use with care to avoid
        shell injection and quoting pitfalls.
    echo_stderr
        When True, the child process inherits the parent's stderr and error output
        is displayed live in the console (stderr is not captured and will be "").
        When False, stderr is captured and returned as a string.

    Returns
    -------
    Tuple[str, str, int]
        A tuple of ``(stdout, stderr, returncode)``. ``stdout`` is always captured
        as text; ``stderr`` is captured unless ``echo_stderr=True``.

    Notes
    -----
    - ``subprocess.run`` is used for simplicity and reliability (blocking until
      the command finishes).
    - For long-running commands where real-time stdout is also required, use
      ``subprocess.Popen`` with incremental reads.
    """
    # Build a readable representation for logging, without altering the command used.
    delimiter = ' '
    command_str = delimiter.join(command) if isinstance(command, list) else str(command)
    logger.debug('Executing command: %s', command_str)

    try:
        # If echo_stderr is requested, let the child inherit the console's stderr
        # by passing stderr=None. Otherwise, capture it so it can be returned.
        result = subprocess.run(
            command,
            check=False,                    # Do not raise on non-zero exit
            stdout=subprocess.PIPE,         # Always capture stdout (text)
            stderr=None if echo_stderr else subprocess.PIPE,
            text=True,                      # Decode to str
            shell=shell
        )

        # Normalize captured outputs to empty strings when None to keep return type stable.
        captured_stderr = "" if echo_stderr else (result.stderr or "")
        return result.stdout or "", captured_stderr, result.returncode

    except FileNotFoundError:
        # Common failure when the executable is not on PATH or the command name is wrong.
        return "", f"Error: Command not found or could not be executed: {command}", 127
    except OSError as e:
        # OS-level errors (e.g., permission denied, resource exhaustion).
        return "", f"OS error occurred: {e}", 1
    except subprocess.SubprocessError as e:
        # Subprocess-specific errors (timeout, etc.).
        return "", f"Subprocess error occurred: {e}", 1
    # Catch any other unexpected errors to prevent caller crashes
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Defensive catch to surface unexpected subprocess issues without
        # crashing callers.
        return "", f"An unexpected error occurred: {e}", 1


@contextmanager
def temporarily_change_directory(path: str):
    """
    Change the current working directory within a ``with`` block and restore it.

    This context manager is robust to exceptions inside the block; the original
    working directory is restored in the ``finally`` clause.

    Examples
    --------
    >>> with temporarily_change_directory("/tmp"):
    ...     # do work in /tmp
    """
    # Save the current working directory so we can restore it later.
    prev = os.getcwd()

    # Switch to the requested directory for the duration of the context.
    os.chdir(path)
    try:
        # Yield control back to the caller's with-block.
        yield
    finally:
        # Always restore the previous working directory, even on exceptions.
        os.chdir(prev)


def create_output_directory_tree(outdir: str) -> bool:
    """
    Create ``outdir`` and all missing parent directories, returning success.

    Parameters
    ----------
    outdir
        The directory path (which may include multiple missing parents) to create.

    Returns
    -------
    bool
        True on success; False if an ``OSError`` occurs (also logged as an error).
    """
    retval = True
    try:
        # Strip any surrounding quotes that may be included when using -o="path" syntax
        # on Windows command line.
        outdir = outdir.strip('"').strip("'")
        # Create the directories recursively, ignoring the error if they already exist.
        os.makedirs(outdir, exist_ok=True)
    except OSError as e:
        # Log the specific OS error for troubleshooting (e.g., permissions, invalid path).
        logger.error(e)
        retval = False
    return retval


def get_nonconflicting_output_path(path: str) -> str:
    """
    Return a non-conflicting path by appending ``_Variant_n`` if needed.

    If ``path`` does not exist, it is returned unchanged. Otherwise,
    this function probes ``<stem>_Variant_1<ext>``, ``_Variant_2``, etc.
    and returns the first candidate that does not already exist.

    Parameters
    ----------
    path
        Desired output path (file path with extension).

    Returns
    -------
    str
        A path guaranteed not to collide with an existing file at call time.
    """
    if not path or not os.path.exists(path):
        return path

    stem, ext = os.path.splitext(path)
    variant = 1
    while True:
        candidate = f"{stem}_Variant_{variant}{ext}"
        if not os.path.exists(candidate):
            return candidate
        variant += 1


def extract_variant_number_from_path(path: str) -> int:
    """
    Extract Variant suffix number from a file path, defaulting to 0.

    Expected suffix format is ``_Variant_<n>`` immediately before extension.
    If absent, returns 0.
    """
    if not path:
        return 0

    stem = os.path.splitext(path)[0]
    match = re.search(r'_Variant_(\d+)$', stem)
    if not match:
        return 0

    try:
        return int(match.group(1))
    except ValueError:
        return 0
