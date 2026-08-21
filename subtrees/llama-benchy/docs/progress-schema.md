# `--emit-progress` JSONL schema

When llama-benchy is invoked with `--emit-progress PATH` (or
`--emit-progress -` for stdout), it writes a stream of newline-delimited
JSON events to PATH that an external visualizer can consume in real time.

This document is the contract between llama-benchy (the producer) and any
consumer (live TUIs, web dashboards, post-hoc analyzers). The producer
side is intentionally tiny — no UI, no rendering, no visualization-specific
dependencies — so consumers can be implemented in any language.

## Versioning

Every line carries a `schema` field with the value:

```
"llama-benchy-progress.v1"
```

Forward compatibility rules:

- New fields may be added to existing event types without bumping the version.
  Consumers MUST ignore unknown fields.
- New event types may be added without bumping the version. Consumers SHOULD
  ignore unknown `type` values.
- Renaming or removing a field, or changing a field's semantics, requires
  bumping the version (`v2`, `v3`, …). Consumers SHOULD reject lines whose
  `schema` field they don't understand.

## Common fields

Every line is a JSON object with these fields:

| Field    | Type   | Description                                    |
|----------|--------|------------------------------------------------|
| `schema` | string | Always `"llama-benchy-progress.v1"`.           |
| `type`   | string | Event discriminator (see below).               |
| `ts`     | number | Unix timestamp (seconds, fractional) at emit.  |

## Event types

### `header`

Emitted once at the start of the run. Lets a consumer detect the schema
version and the producer version before any data events arrive.

| Field                  | Type   | Description                          |
|------------------------|--------|--------------------------------------|
| `llama_benchy_version` | string | Version string of the producer.      |

### `request_start`

Emitted by the runner before each individual HTTP request. One event per
in-flight request; multiple may be in-flight simultaneously when the user
configures concurrency > 1.

| Field            | Type    | Description                                                            |
|------------------|---------|------------------------------------------------------------------------|
| `request_id`     | integer | Stable per-request identifier (monotonically increasing per process).  |
| `model`          | string  | Model name as configured (e.g. `meta-llama/Llama-3.1-8B-Instruct`).    |
| `base_url`       | string  | Endpoint URL the request is being sent to.                             |
| `prompt_size`    | integer | User-facing `--pp` for this request (the value the user typed). The actual number of tokens sent to the server may differ slightly when llama-benchy's `--adapt-prompt` is on; see `prompt_tokens` on `request_end`. |
| `response_size`  | integer | User-facing `--tg` for this request.                                   |
| `context_size`   | integer | User-facing `--depth` for this request.                                |
| `concurrency`    | integer | Concurrency level for this batch.                                      |
| `run_index`      | integer | Zero-based run index within the current `(pp, tg, depth)` cell.        |
| `target_label`   | string  | Optional human label; empty in v1 (reserved for future multi-target).  |

### `request_first_response`

Emitted the moment the first response chunk of any kind (even an empty
role-only chunk) arrives for a given request. Distinct from
`request_first_token` because some servers send an empty header chunk
before the first content-bearing chunk.

| Field        | Type    | Description                                                     |
|--------------|---------|-----------------------------------------------------------------|
| `request_id` | integer | Matches the request's `request_start`.                          |
| `ttfr_s`     | number  | Time from request start to first response chunk (seconds).      |

### `request_first_token`

Emitted the moment the first content-bearing token arrives for a given
request. This is `e2e_ttft` (end-to-end time to first token).

| Field        | Type    | Description                                                     |
|--------------|---------|-----------------------------------------------------------------|
| `request_id` | integer | Matches the request's `request_start`.                          |
| `ttft_s`     | number  | Time from request start to first content token (seconds, includes network). |

### `latency_measured`

Emitted once after llama-benchy completes its network-latency probe.
Consumers use it to compute `est_ppt = ttfr − latency` (estimated pure
prompt-processing time, with the network round-trip subtracted).

| Field        | Type    | Description                                                     |
|--------------|---------|-----------------------------------------------------------------|
| `latency_s`  | number  | Measured / assumed network latency (seconds). Zero for `--latency-mode none`. |
| `mode`       | string  | `"api"`, `"generation"`, or `"none"`.                           |

### `tokens`

Emitted per streaming chunk during decode. Multiple `tokens` events occur
between `request_first_token` and `request_end`.

| Field        | Type    | Description                                            |
|--------------|---------|--------------------------------------------------------|
| `request_id` | integer | Matches the request's `request_start`.                 |
| `count`      | integer | Number of generated tokens in this chunk (≥ 1).        |
| `snippet`    | string  | Decoded text for this chunk (may be empty).            |
| `estimated`  | boolean | Optional. Present and `true` when the server did not return `token_ids` and `count` is a best-effort estimate (one-per-chunk fallback). Absent otherwise. The authoritative total is reported on `request_end`. |

A single `tokens` event may carry multiple tokens when the server streams
multi-token chunks (e.g. speculative decoding / MTP). Consumers computing
tok/s should sum `count` across events where `estimated` is absent or false,
not assume one-per-event. Consumers SHOULD treat `estimated: true` counts as
arrival/progress hints only and use `request_end.total_tokens` as the
authoritative total.

### `request_end`

Emitted exactly once per request (including failed ones).

| Field            | Type    | Description                                                        |
|------------------|---------|--------------------------------------------------------------------|
| `request_id`     | integer | Matches the request's `request_start`.                             |
| `total_tokens`   | integer | Total generated tokens for this request.                           |
| `prompt_tokens`  | integer | Server-reported prompt token count (0 if the server didn't say).   |
| `decode_seconds` | number  | Time from `first_token_ts` to request end (seconds).               |
| `error`          | string  | Empty on success, otherwise the error message.                     |

### `bench_complete`

Emitted exactly once at the end of the entire benchmark suite. Consumers
SHOULD treat this as the final frame and exit cleanly.

| Field    | Type   | Notes                                                                                   |
|----------|--------|-----------------------------------------------------------------------------------------|
| `status` | string | `"ok"` — suite ran to completion. `"interrupted"` — Ctrl+C / SIGINT. `"error"` — unhandled exception. |

Consumers SHOULD treat unknown `status` values as `"error"`.

## Ordering invariants

For any single `request_id`:

1. `request_start` always precedes any other event for that id.
2. `request_first_response` (if it fires at all) precedes
   `request_first_token`, all `tokens`, and `request_end` for that id.
3. `request_first_token` (if it fires at all) precedes any `tokens` and
   `request_end` for that id, and follows `request_first_response`.
4. `tokens` events for a given id arrive in stream order (chunk by chunk).
5. `request_end` is the last event for that id.

Across requests, `latency_measured` fires once per benchmark run, before
any `request_start` events.

Across requests:

- `request_id` values are strictly monotonic per process (not sparse, not
  reused). Consumers can rely on the fact that a higher id was started
  later than a lower id.
- `request_start` events for concurrent requests can interleave with each
  other and with `tokens` / `request_end` events of earlier requests.
- `bench_complete` is always the last event in the stream.

## Stream destination

`--emit-progress PATH` writes to `PATH`, line-buffered.

`--emit-progress -` writes to stdout. To prevent llama-benchy's normal
status output from corrupting the JSONL stream, in this mode all status
prints are routed to stderr instead. Pipe consumers can rely on stdout
being clean JSONL.

## What's *not* in v1

By design, v1 does NOT include:

- Hardware metrics (GPU%, VRAM, RAM, temp, power). Visualizers can sample
  these on their own from the host they're running on.
- Aggregate / smoothed throughput. Consumers compute this themselves
  (rolling window over `tokens.count`).
- Phase labels (`PREFILL` / `DECODE` / `DONE`). Phases are derivable: a
  request is `PREFILL` between `request_start` and `request_first_token`,
  `DECODE` between `request_first_token` and `request_end`, and
  `DONE`/`ERROR` after `request_end`.
- Display modes / chart settings / colors. Those are visualizer concerns.

These may appear in a later schema version if a real consumer needs them.

## Example

A two-token, single-request bench produces:

```jsonl
{"schema":"llama-benchy-progress.v1","type":"header","ts":1714969500.123,"llama_benchy_version":"0.3.8"}
{"schema":"llama-benchy-progress.v1","type":"request_start","ts":1714969500.456,"request_id":0,"model":"meta-llama/Llama-3.1-8B-Instruct","base_url":"http://localhost:8000/v1","prompt_size":2048,"response_size":2,"context_size":0,"concurrency":1,"run_index":0,"target_label":""}
{"schema":"llama-benchy-progress.v1","type":"request_first_token","ts":1714969500.890,"request_id":0,"ttft_s":0.434}
{"schema":"llama-benchy-progress.v1","type":"tokens","ts":1714969500.911,"request_id":0,"count":1,"snippet":"Hello"}
{"schema":"llama-benchy-progress.v1","type":"tokens","ts":1714969500.932,"request_id":0,"count":1,"snippet":" world"}
{"schema":"llama-benchy-progress.v1","type":"request_end","ts":1714969500.945,"request_id":0,"total_tokens":2,"prompt_tokens":2048,"decode_seconds":0.055,"error":""}
{"schema":"llama-benchy-progress.v1","type":"bench_complete","ts":1714969501.000,"status":"ok"}
```
