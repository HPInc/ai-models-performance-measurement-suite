#!/usr/bin/env python3
"""Shared llama-server process lifecycle management."""

import logging
import os
import signal
import subprocess
import time

from typing import Callable, Dict, List, Optional, Pattern


def is_process_alive(pid: int) -> bool:
    """Return True if the process id is alive."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_process_exit(pid: int, timeout_s: float, poll_interval_s: float = 0.1) -> bool:
    """Wait for a process id to exit and return True if it exits in time."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not is_process_alive(pid):
            return True
        time.sleep(poll_interval_s)
    return not is_process_alive(pid)


def build_process_group_popen_kwargs(
    is_windows: bool,
    cwd: Optional[str] = None,
    stdout=None,
    stderr=None,
    text: bool = True,
    bufsize: int = -1,
) -> Dict[str, object]:
    """Build subprocess.Popen kwargs for creating a dedicated process group."""
    kwargs: Dict[str, object] = {
        'stdout': stdout,
        'stderr': stderr,
        'text': text,
        'bufsize': bufsize,
    }
    if cwd is not None:
        kwargs['cwd'] = cwd

    if is_windows:
        kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs['start_new_session'] = True
    return kwargs


def stop_popen_process(
    process: subprocess.Popen,
    is_windows: bool,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Stop a Popen process group with graceful shutdown and escalation."""
    try:
        if process.poll() is not None:
            return

        if is_windows:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)

        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass

        if is_windows:
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)

        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass

        if is_windows:
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    except Exception as exc:  # pylint: disable=broad-except
        if logger:
            logger.error("Error stopping process: %s", exc)
        try:
            process.kill()
        except Exception:  # pylint: disable=broad-except
            pass


def stop_process_by_pid(
    pid: int,
    is_windows: bool,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Stop a process group by pid with graceful shutdown and escalation."""
    if is_windows:
        os.kill(pid, signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(pid, signal.SIGINT)

    if wait_for_process_exit(pid, 10.0):
        return

    if is_windows:
        subprocess.run(
            ['taskkill', '/PID', str(pid), '/T'],
            capture_output=True,
            text=True,
            check=False,
        )
        if wait_for_process_exit(pid, 5.0):
            return

        subprocess.run(
            ['taskkill', '/PID', str(pid), '/T', '/F'],
            capture_output=True,
            text=True,
            check=False,
        )
        wait_for_process_exit(pid, 5.0)
        return

    os.killpg(pid, signal.SIGTERM)
    if wait_for_process_exit(pid, 5.0):
        return
    os.killpg(pid, signal.SIGKILL)
    wait_for_process_exit(pid, 5.0)


class LlamaServerProcess:
    """Manages starting, URL capture, and shutdown for llama-server."""

    def __init__(
        self,
        install_dir: str,
        model_path: Optional[str],
        server_options: List[str],
        find_server_binary_fn: Callable[[str], Optional[str]],
        server_url_pattern: Pattern[str],
        is_windows: bool,
        verbose: bool = False,
        startup_timeout: int = 300,
        logger: Optional[logging.Logger] = None,
        command_prefix: Optional[List[str]] = None,
    ):
        self.install_dir = install_dir
        self.model_path = model_path
        self.server_options = server_options
        self.find_server_binary_fn = find_server_binary_fn
        self.server_url_pattern = server_url_pattern
        self.is_windows = is_windows
        self.verbose = verbose
        self.startup_timeout = startup_timeout
        self.logger = logger or logging.getLogger(__name__)
        self.command_prefix = command_prefix or []
        self.process: Optional[subprocess.Popen] = None
        self.url: Optional[str] = None

    def start(self) -> Optional[str]:
        """Start llama-server and return detected URL, or None on failure."""
        server_exe = self.find_server_binary_fn(self.install_dir)
        if not server_exe:
            self.logger.error("llama-server binary not found in %s", self.install_dir)
            return None

        if self.model_path:
            server_cmd = [server_exe, '-m', self.model_path] + self.server_options
        else:
            server_cmd = [server_exe] + self.server_options

        cmd = self.command_prefix + server_cmd

        if self.verbose:
            cmd.append('-v')

        self.logger.info("Changed directory to %s", self.install_dir)
        self.logger.info("Starting llama-server: %s", ' '.join(cmd))

        try:
            # pylint: disable=subprocess-popen-preexec-fn
            popen_kwargs = build_process_group_popen_kwargs(
                is_windows=self.is_windows,
                cwd=self.install_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.process = subprocess.Popen(cmd, **popen_kwargs)

            start_time = time.time()
            while time.time() - start_time < self.startup_timeout:
                if self.process.poll() is not None:
                    remaining_output = self.process.stdout.read()
                    self.logger.error("llama-server exited unexpectedly. Output: %s",
                                      remaining_output)
                    return None

                line = self.process.stdout.readline()
                if line:
                    self.logger.debug("llama-server: %s", line.strip())
                    match = self.server_url_pattern.search(line)
                    if match:
                        self.url = match.group(0)
                        self.logger.debug("Captured llama-server URL: %s", self.url)
                        self.logger.info("llama-server started at: %s", self.url)
                        return self.url

                time.sleep(0.1)

            self.logger.error("Timeout waiting for llama-server to start.")
            self.stop()
            return None

        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Failed to start llama-server: %s", exc)
            self.stop()
            return None

    def stop(self) -> None:
        """Stop the llama-server process gracefully."""
        if self.process is None:
            return

        self.logger.info("Stopping llama-server...")
        stop_popen_process(self.process, is_windows=self.is_windows, logger=self.logger)

        self.process = None
        self.url = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _, __, ___):
        self.stop()
        return False
