#!/usr/bin/env python3
"""Stop llama-server from a pid file."""

from __future__ import annotations

import argparse
import os
import platform
import sys

from llama_server_process import stop_process_by_pid


IS_WINDOWS = platform.system() == "Windows"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stop llama-server using PID from a file.")
    parser.add_argument("--pid-file", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not os.path.isfile(args.pid_file):
        return 0

    try:
        with open(args.pid_file, "r", encoding="utf-8") as pid_file:
            pid = int(pid_file.read().strip())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to read pid file {args.pid_file}: {exc}", file=sys.stderr)
        return 1

    try:
        stop_process_by_pid(pid, is_windows=IS_WINDOWS)
    except ProcessLookupError:
        pass
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to stop process {pid}: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            os.remove(args.pid_file)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
