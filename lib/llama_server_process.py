#!/usr/bin/env python3
"""Shared llama-server process lifecycle management."""

import logging
import os
import signal
import subprocess
import time

from typing import Callable, List, Optional, Pattern


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
        self.process: Optional[subprocess.Popen] = None
        self.url: Optional[str] = None

    def start(self) -> Optional[str]:
        """Start llama-server and return detected URL, or None on failure."""
        server_exe = self.find_server_binary_fn(self.install_dir)
        if not server_exe:
            self.logger.error("llama-server binary not found in %s", self.install_dir)
            return None

        if self.model_path:
            cmd = [server_exe, '-m', self.model_path] + self.server_options
        else:
            cmd = [server_exe] + self.server_options

        if self.verbose:
            cmd.append('-v')

        self.logger.info("Changed directory to %s", self.install_dir)
        self.logger.info("Starting llama-server: %s", ' '.join(cmd))

        try:
            # pylint: disable=subprocess-popen-preexec-fn
            popen_kwargs = {
                'stdout': subprocess.PIPE,
                'stderr': subprocess.STDOUT,
                'text': True,
                'cwd': self.install_dir,
            }
            if self.is_windows:
                popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs['start_new_session'] = True
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

        try:
            if self.is_windows:
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(self.process.pid, signal.SIGINT)

            try:
                self.process.wait(timeout=10)
                self.logger.debug("llama-server stopped gracefully.")
            except subprocess.TimeoutExpired:
                self.logger.warning(
                    "llama-server did not stop after SIGINT, sending SIGTERM..."
                )
                if self.is_windows:
                    self.process.terminate()
                else:
                    os.killpg(self.process.pid, signal.SIGTERM)

                try:
                    self.process.wait(timeout=5)
                    self.logger.debug("llama-server stopped after SIGTERM.")
                except subprocess.TimeoutExpired:
                    self.logger.warning("llama-server did not stop gracefully, killing...")
                    if self.is_windows:
                        self.process.kill()
                    else:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Error stopping llama-server: %s", exc)
            try:
                self.process.kill()
            except Exception:  # pylint: disable=broad-except
                pass

        self.process = None
        self.url = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _, __, ___):
        self.stop()
        return False
