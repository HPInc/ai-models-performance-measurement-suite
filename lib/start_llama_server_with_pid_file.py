#!/usr/bin/env python3
"""Start llama-server, write its pid to a file, and wait for readiness."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import tempfile
import urllib.error
import urllib.request
from typing import Optional

from llama_server_process import build_process_group_popen_kwargs, stop_process_by_pid


IS_WINDOWS = platform.system() == "Windows"
_LLAMA_SERVER_CANDIDATES = ("llama-server.exe", "llama-server")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start llama-server and persist its PID.")
    parser.add_argument("--install-dir", required=True)
    parser.add_argument("--server-model", default=None)
    parser.add_argument("--server-options-json", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--warmup-prompt", default="Hello")
    parser.add_argument("--warmup-max-tokens", type=int, default=16)
    return parser.parse_args()


def _parse_server_options(server_options_json: str) -> list[str]:
    value = json.loads(server_options_json)
    if not isinstance(value, list):
        raise ValueError("server options JSON must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("server options JSON must only contain strings")
    return value


def _find_server_binary(install_dir: str) -> str:
    for name in _LLAMA_SERVER_CANDIDATES:
        candidate = os.path.join(install_dir, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"llama-server binary not found in {install_dir}")


def _wait_until_ready(base_url: str, timeout_s: float, poll_interval_s: float) -> None:
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
            last_error = str(exc)

        time.sleep(poll_interval_s)

    raise TimeoutError(f"Timed out waiting for llama-server readiness at {models_url}: {last_error}")


def _fetch_first_model_id(base_url: str) -> Optional[str]:
    models_url = base_url.rstrip("/") + "/v1/models"
    request = urllib.request.Request(models_url, method="GET")
    with urllib.request.urlopen(request, timeout=10.0) as response:  # nosec B310
        payload = response.read().decode("utf-8", errors="replace")

    data = json.loads(payload)
    models = data.get("data") if isinstance(data, dict) else None
    if isinstance(models, list):
        for model in models:
            if isinstance(model, dict):
                model_id = model.get("id")
                if isinstance(model_id, str) and model_id:
                    return model_id
    return None


def _run_warmup_request(base_url: str, prompt: str, max_tokens: int) -> None:
    model_id = _fetch_first_model_id(base_url)
    if not model_id:
        raise RuntimeError("Unable to determine model id from /v1/models for warm-up request")

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    completions_url = base_url.rstrip("/") + "/v1/chat/completions"
    request = urllib.request.Request(
        completions_url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120.0) as response:  # nosec B310
        if not (200 <= response.status < 300):
            raise RuntimeError(f"Warm-up request failed with HTTP {response.status}")
        _ = response.read()


def _tail_text_file(path: str, max_chars: int = 4000) -> str:
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file_obj:
            content = file_obj.read()
        if len(content) <= max_chars:
            return content
        return content[-max_chars:]
    except Exception:  # pylint: disable=broad-except
        return ""


def main() -> int:
    args = _parse_args()

    try:
        server_options = _parse_server_options(args.server_options_json)
        server_exe = _find_server_binary(args.install_dir)
    except Exception as exc:  # pylint: disable=broad-except
        print(str(exc), file=sys.stderr)
        return 2

    server_cmd = [server_exe]
    if args.server_model:
        server_cmd.extend(["-m", args.server_model])
    server_cmd.extend(server_options)

    process = None
    server_stderr_path = ""
    server_stderr_file = None
    try:
        pid_abs_path = os.path.abspath(args.pid_file)
        pid_dir = os.path.dirname(pid_abs_path)
        if pid_dir:
            os.makedirs(pid_dir, exist_ok=True)

        server_stderr_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            prefix="llama-server-stderr-",
            suffix=".log",
            dir=pid_dir if pid_dir else None,
        )
        server_stderr_path = server_stderr_file.name

        process = subprocess.Popen(
            server_cmd,
            **build_process_group_popen_kwargs(
                is_windows=IS_WINDOWS,
                cwd=args.install_dir,
                stdout=subprocess.DEVNULL,
                stderr=server_stderr_file,
                text=False,
            ),
        )

        server_stderr_file.close()
        server_stderr_file = None

        with open(args.pid_file, "w", encoding="utf-8") as pid_file:
            pid_file.write(str(process.pid))

        _wait_until_ready(args.server_url, args.timeout, args.poll_interval)
        _run_warmup_request(
            base_url=args.server_url,
            prompt=args.warmup_prompt,
            max_tokens=args.warmup_max_tokens,
        )
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        if server_stderr_file is not None:
            try:
                server_stderr_file.close()
            except Exception:  # pylint: disable=broad-except
                pass

        stderr_tail = _tail_text_file(server_stderr_path)
        if stderr_tail:
            print(
                f"llama-server stderr log: {server_stderr_path}\n"
                f"--- stderr tail ---\n{stderr_tail}",
                file=sys.stderr,
            )
        elif server_stderr_path:
            print(f"llama-server stderr log: {server_stderr_path}", file=sys.stderr)

        if process is not None:
            try:
                stop_process_by_pid(process.pid, is_windows=IS_WINDOWS)
            except Exception:  # pylint: disable=broad-except
                pass
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
