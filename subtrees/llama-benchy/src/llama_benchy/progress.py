"""Optional progress-event emitter for llama-benchy.

When the user passes ``--emit-progress PATH``, llama-benchy writes a stream
of newline-delimited JSON events to PATH (or stdout when PATH is ``-``).
Consumers — separate tools, e.g. live TUIs, web dashboards, post-hoc
visualizers — parse that stream and render whatever they like.

Schema spec:        docs/progress-schema.md
Schema version tag: ``llama-benchy-progress.v1``

This module is intentionally small. It carries no UI, no rendering, no
optional deps. Anything fancier lives in a separate consumer repo.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import IO, Optional

SCHEMA_VERSION = "llama-benchy-progress.v1"


class ProgressEmitter:
    """Append-only JSONL writer for benchmark progress events.

    Thread-safe (a lock guards the underlying file write). Methods are
    no-throwing — emit failures are silently dropped so a broken consumer
    can't take down a benchmark run.
    """

    def __init__(self, target: str, *, llama_benchy_version: str = "unknown") -> None:
        self._target = target
        self._lock = threading.Lock()
        self._stream: Optional[IO[str]] = None
        self._owns_stream = False
        self._open(llama_benchy_version)

    def _open(self, llama_benchy_version: str) -> None:
        if self._target == "-":
            # Caller is expected to have already redirected sys.stdout to
            # sys.stderr (see __main__) so llama-benchy's regular status
            # prints don't corrupt the JSONL stream we emit here.
            self._stream = sys.__stdout__
            self._owns_stream = False
        else:
            self._stream = open(self._target, "w", buffering=1)  # line-buffered
            self._owns_stream = True
        self._write(
            {
                "schema": SCHEMA_VERSION,
                "type": "header",
                "ts": time.time(),
                "llama_benchy_version": llama_benchy_version,
            }
        )

    # event API
    def request_start(
        self,
        *,
        request_id: int,
        model: str,
        base_url: str,
        prompt_size: int,
        response_size: int,
        context_size: int,
        concurrency: int,
        run_index: int,
        target_label: str = "",
    ) -> None:
        self._emit(
            "request_start",
            request_id=request_id,
            model=model,
            base_url=base_url,
            prompt_size=prompt_size,
            response_size=response_size,
            context_size=context_size,
            concurrency=concurrency,
            run_index=run_index,
            target_label=target_label,
        )

    def request_first_response(self, *, request_id: int, ttfr_s: float) -> None:
        """First chunk of any kind arrived (may be empty / role-only)."""
        self._emit("request_first_response", request_id=request_id, ttfr_s=ttfr_s)

    def request_first_token(self, *, request_id: int, ttft_s: float) -> None:
        """First content-bearing token arrived (== e2e_ttft)."""
        self._emit("request_first_token", request_id=request_id, ttft_s=ttft_s)

    def latency_measured(self, *, latency_s: float, mode: str) -> None:
        """Network latency probe complete (used to derive est_ppt = ttfr − latency)."""
        self._emit("latency_measured", latency_s=latency_s, mode=mode)

    def tokens(self, *, request_id: int, count: int, snippet: str = "", estimated: bool = False) -> None:
        if count <= 0 and not snippet:
            return
        fields = {"request_id": request_id, "count": count, "snippet": snippet}
        if estimated:
            fields["estimated"] = True
        self._emit("tokens", **fields)

    def request_end(
        self,
        *,
        request_id: int,
        total_tokens: int,
        prompt_tokens: int,
        decode_seconds: float,
        error: str = "",
    ) -> None:
        self._emit(
            "request_end",
            request_id=request_id,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            decode_seconds=decode_seconds,
            error=error,
        )

    def bench_complete(self, status: str = "ok") -> None:
        self._emit("bench_complete", status=status)

    def close(self) -> None:
        with self._lock:
            if self._stream is not None and self._owns_stream:
                try:
                    self._stream.close()
                except Exception:
                    pass
            self._stream = None

    def __enter__(self) -> "ProgressEmitter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # internals
    def _emit(self, event_type: str, **fields) -> None:
        self._write(
            {
                "schema": SCHEMA_VERSION,
                "type": event_type,
                "ts": time.time(),
                **fields,
            }
        )

    def _write(self, obj: dict) -> None:
        if self._stream is None:
            return
        try:
            line = json.dumps(obj, separators=(",", ":"))
        except (TypeError, ValueError):
            return
        with self._lock:
            try:
                self._stream.write(line)
                self._stream.write("\n")
                self._stream.flush()
            except Exception:
                # Consumer hung up / disk full — don't crash the benchmark.
                pass
