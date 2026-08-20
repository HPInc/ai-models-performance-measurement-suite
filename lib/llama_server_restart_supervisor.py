#!/usr/bin/env python3
"""Supervise llama-server and restart it when it exits unexpectedly."""

from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import sys
import threading
import time

from typing import List, Optional

from llama_server_process import build_process_group_popen_kwargs, stop_popen_process

IS_WINDOWS = platform.system() == "Windows"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run llama-server under a restart supervisor."
    )
    parser.add_argument(
        "--pid-file",
        required=True,
        help="Path to write the active llama-server child PID.",
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=0.5,
        help="Seconds to wait before restarting llama-server (default: %(default)s).",
    )
    parser.add_argument(
        "server_cmd",
        nargs=argparse.REMAINDER,
        help="Server command after '--' (example: -- llama-server -m model.gguf).",
    )
    args = parser.parse_args()

    command = args.server_cmd
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        parser.error("Missing server command. Pass it after '--'.")

    args.server_cmd = command
    return args


class _Supervisor:
    def __init__(self, server_cmd: List[str], pid_file: str, restart_delay: float):
        self.server_cmd = server_cmd
        self.pid_file = pid_file
        self.restart_delay = restart_delay
        self.stop_requested = False
        self.child: Optional[subprocess.Popen] = None

    def _write_pid_file(self, pid: int) -> None:
        with open(self.pid_file, "w", encoding="utf-8") as pid_out:
            pid_out.write(f"{pid}\n")

    def _clear_pid_file(self) -> None:
        try:
            if os.path.isfile(self.pid_file):
                os.remove(self.pid_file)
        except OSError:
            pass

    def _stream_child_output(self, child: subprocess.Popen) -> None:
        if child.stdout is None:
            return
        for line in iter(child.stdout.readline, ""):
            if not line:
                break
            sys.stdout.write(line)
            sys.stdout.flush()

    def _terminate_child(self, child: subprocess.Popen) -> None:
        stop_popen_process(child, is_windows=IS_WINDOWS)

    def request_stop(self, *_args) -> None:
        self.stop_requested = True
        if self.child is not None:
            self._terminate_child(self.child)

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.request_stop)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self.request_stop)

        while not self.stop_requested:
            popen_kwargs = build_process_group_popen_kwargs(
                is_windows=IS_WINDOWS,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            self.child = subprocess.Popen(self.server_cmd, **popen_kwargs)
            self._write_pid_file(self.child.pid)

            output_thread = threading.Thread(
                target=self._stream_child_output,
                args=(self.child,),
                daemon=True,
            )
            output_thread.start()

            while not self.stop_requested:
                if self.child.poll() is not None:
                    break
                time.sleep(0.1)

            if self.stop_requested:
                self._terminate_child(self.child)
                break

            self.child.wait()
            time.sleep(self.restart_delay)

        self._clear_pid_file()
        return 0


def main() -> int:
    args = _parse_args()
    supervisor = _Supervisor(
        server_cmd=args.server_cmd,
        pid_file=args.pid_file,
        restart_delay=args.restart_delay,
    )
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
