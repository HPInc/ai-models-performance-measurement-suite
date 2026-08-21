#!/usr/bin/env python3
"""Restart llama-server from explicit launch arguments and wait for readiness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request

from llama_server_process import build_process_group_popen_kwargs, stop_process_by_pid


IS_WINDOWS = platform.system() == "Windows"


def _parse_args() -> argparse.Namespace:
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_log_dir = os.path.join(workspace_dir, "logs")
    default_log_file = os.path.join(
        default_log_dir,
        "restart_llama_server.log",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Terminate current llama-server process, relaunch it with the provided "
            "command, then wait until the server responds."
        )
    )
    parser.add_argument("--server-pid", type=int, required=True, help="Current llama-server PID.")
    parser.add_argument("--server-cwd", required=True, help="Working directory for server relaunch.")
    parser.add_argument(
        "--server-cmd-json",
        required=True,
        help="JSON array containing the full server command to relaunch.",
    )
    parser.add_argument(
        "--server-url",
        required=True,
        help="Server base URL that should become healthy again.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for server readiness (default: %(default)s).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between readiness checks (default: %(default)s).",
    )
    parser.add_argument(
        "--log-file",
        default=default_log_file,
        help="Path to append timestamped execution logs (default: %(default)s).",
    )
    return parser.parse_args()


def _log_message(log_file: str, message: str) -> None:
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"[{timestamp}] {message}\n")


def _reset_log_file(log_file: str) -> None:
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_file, "w", encoding="utf-8"):
        pass


def _parse_server_cmd(server_cmd_json: str) -> list[str]:
    value = json.loads(server_cmd_json)
    if not isinstance(value, list) or not value:
        raise ValueError("server command JSON must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError("server command JSON must only contain non-empty strings")
    return value


def _format_command_for_log(command: list[str]) -> str:
    if IS_WINDOWS:
        return subprocess.list2cmdline(command)
    return " ".join(json.dumps(part) for part in command)


def _start_server(server_cmd: list[str], server_cwd: str) -> int:
    popen_kwargs = build_process_group_popen_kwargs(
        is_windows=IS_WINDOWS,
        cwd=server_cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=False,
    )
    process = subprocess.Popen(server_cmd, **popen_kwargs)
    return process.pid


def _wait_until_server_ready(base_url: str, timeout_s: float, poll_interval_s: float) -> None:
    deadline = time.time() + timeout_s
    models_url = base_url.rstrip("/") + "/v1/models"

    last_error = "unknown"
    while time.time() < deadline:
        try:
            request = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(request, timeout=10.0) as response:  # nosec B310
                if 200 <= response.status < 500:
                    _ = response.read().decode("utf-8", errors="replace")
                    return
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except urllib.error.URLError as exc:
            last_error = str(exc)
        except Exception as exc:  # pylint: disable=broad-except
            # During restart windows, low-level socket errors (for example,
            # WinError 10054 connection reset) can be raised directly.
            # Treat them as transient and keep polling until timeout.
            last_error = str(exc)

        time.sleep(poll_interval_s)

    raise TimeoutError(
        f"Timed out waiting for server readiness at {models_url}. Last error: {last_error}"
    )


def main() -> int:
    args = _parse_args()

    _reset_log_file(args.log_file)

    _log_message(args.log_file, f"Restart request for server {args.server_url}")

    try:
        server_cmd = _parse_server_cmd(args.server_cmd_json)
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Invalid server command payload: {exc}"
        print(message, file=sys.stderr)
        _log_message(args.log_file, f"ERROR: {message}")
        return 2

    try:
        stop_process_by_pid(args.server_pid, is_windows=IS_WINDOWS)
        _log_message(args.log_file, f"Stopped llama-server PID {args.server_pid}")
    except Exception as exc:  # pylint: disable=broad-except
        _log_message(
            args.log_file,
            f"WARNING: Failed to stop llama-server PID {args.server_pid}: {exc}. Continuing with relaunch.",
        )

    try:
        new_pid = _start_server(server_cmd, args.server_cwd)
        _log_message(
            args.log_file,
            f"Relaunched llama-server PID {new_pid} with command: {_format_command_for_log(server_cmd)}",
        )
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Failed to relaunch llama-server: {exc}"
        print(message, file=sys.stderr)
        _log_message(args.log_file, f"ERROR: {message}")
        return 1

    try:
        _wait_until_server_ready(args.server_url, args.timeout, args.poll_interval)
    except Exception as exc:  # pylint: disable=broad-except
        message = f"Server did not become ready after restart: {exc}"
        print(message, file=sys.stderr)
        _log_message(args.log_file, f"ERROR: {message}")
        return 1

    _log_message(args.log_file, "llama-server restart completed successfully")
    print("llama-server restart completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
