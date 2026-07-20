"""Replay chat-completions traffic against the GORGO proxy.

Runs as a Modal function pinned to the same region as the proxy (and the
engine). Three data sources are supported via the ``--source`` flag:

- ``prod`` (default): the production ClickHouse export. Parquet shards under
  ``GORGO-completions`` (mounted at ``/data``) with ``timestamp`` plus a
  JSON ``request`` column. Supports ISO ``--start-time`` / ``--end-time``
  filtering against the row timestamps.
- ``hf``: any Hugging Face ``save_to_disk`` chat dataset (LMSYS-Chat-1M,
  WildChat-4.8M, etc.). Reads the first non-empty among the ``conversation``
  / ``messages`` / ``conversations`` columns and assembles an OpenAI-style
  chat body. ``--preset lmsys`` or ``--preset wildchat`` fills in default
  disk paths under the mounted dataset volumes; ``--data-path`` overrides.
  HF rows have no native timestamp, so ``--start-time`` / ``--end-time`` are
  ignored for this source.

Sources stream rows lazily and feed a bounded asyncio queue, so memory stays
``O(concurrency)`` regardless of dataset size. The inter-request gap from the
original timeline is *not* preserved -- the ``concurrency`` knob alone
determines how fast the dataset is consumed.

Usage::

    # production (default source).
    modal run proxy/workload.py --proxy-url https://...modal.host \\
        --start-time 2026-04-01T12:00:00 \\
        --end-time   2026-04-01T13:00:00 \\
        --concurrency 32

    # LMSYS-Chat-1M.
    modal run proxy/workload.py --proxy-url https://... \\
        --source hf --preset lmsys --num-requests 1000

    # WildChat-4.8M from a custom path.
    modal run proxy/workload.py --proxy-url https://... \\
        --source hf --data-path /datasets/datasets/allenai__WildChat-4.8M

All knobs are also kwargs on ``replay`` for programmatic invocation.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Iterator

import httpx
import modal

from app import (
    app,
    bench_results_volume,
    completions_volume,
    hf_datasets_volume,
    lmsys_chat_1m_volume,
)
from gorgo.measure import consume_sse_stream


def _ts() -> str:
    """Millisecond-precision ISO 8601 UTC timestamp string. Matches the
    format used by ``proxy/modal_proxy.py``::``_log`` so workload and
    proxy lines sort/correlate cleanly when interleaved."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _log(message: str) -> None:
    """Emit a ``[workload]`` log line prefixed with an ISO 8601 UTC
    timestamp. ``flush=True`` so progress is visible in real time when
    streamed via ``modal run`` (Modal's stdout otherwise line-buffers
    inside the container)."""
    print(f"{_ts()} [workload] {message}", flush=True)


# We want to launch the workload client in the same region as the proxy server
# in order to minimize the variable latency of crossing regions. REGION strings
# can also contain a zone like 1.
REGION = os.getenv("REGION", "us-east-1")

image = (
    modal.Image.debian_slim()
    .pip_install("httpx[http2]", "pyarrow", "datasets>=3.0", "transformers", "jinja2")
    .add_local_python_source("app", "proxy")
)

# Auto-detected context_length, plus a constant safety margin for chat
# template tokens / role markers that the gpt-4o tokenizer doesn't see
# but the model's actual tokenizer does. Empirically ~256 covers
# ``<|im_start|>role\n`` boundaries on Qwen-style chat templates with
# room to spare.
CONTEXT_LENGTH_SAFETY_MARGIN_TOKENS = 256

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"

# HF chat datasets store the conversation under one of these columns; we
# pick the first non-empty match. Order matters: ``conversation`` is the
# canonical role/content list on LMSYS / WildChat; ``messages`` and
# ``conversations`` are fallbacks for other shapes.
HF_MESSAGE_COLUMNS = ("conversation", "messages", "conversations")

# Roles SGLang's chat-completions endpoint accepts. Anything else is mapped
# to ``user`` so the request still validates.
_VALID_ROLES = frozenset({"system", "user", "assistant", "tool", "function"})

# ``--source`` choices.
SOURCE_PROD = "prod"
SOURCE_HF = "hf"
SOURCE_MOONCAKE = "mooncake"
SUPPORTED_SOURCES = (SOURCE_PROD, SOURCE_HF, SOURCE_MOONCAKE)

# Defaults wired up to volumes mounted on ``replay``. Match the layouts used
# by ``data_processing/build_hf_prefix_trie.py``.
PROD_DEFAULT_PATH = "/data"
HF_PRESETS = {
    "lmsys": "/datasets/datasets/lmsys__lmsys-chat-1m",
    "wildchat": "/datasets/datasets/allenai__WildChat-4.8M",
}


_TIKTOKEN_ENCODER = None


def _get_encoder():
    """Cached cl100k tiktoken encoder. Module-level cache so the encoder
    isn't reinitialized per row during ``_iter_bodies`` filtering."""
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        import tiktoken

        _TIKTOKEN_ENCODER = tiktoken.encoding_for_model("gpt-4o")
    return _TIKTOKEN_ENCODER


def _approx_input_tokens(messages) -> int:
    """Quick, directional token count for a chat-completions ``messages``
    payload. Mirrors :func:`proxy.modal_proxy.tokenize_input` so the
    workload's pre-filter sees the same numbers the proxy uses for
    routing decisions.

    Tiktoken's BPE diverges from Qwen's vocab (typically within ±20% on
    English-heavy chat data, more divergent on code / non-English), so
    callers feeding this into a context-length filter should leave a
    safety margin -- :data:`CONTEXT_LENGTH_SAFETY_MARGIN_TOKENS` is
    applied automatically by :func:`_resolve_input_token_cap`."""
    if not isinstance(messages, list) or not messages:
        return 0
    enc = _get_encoder()
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                total += len(enc.encode(content, disallowed_special=()))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text") or ""
                        if text:
                            total += len(enc.encode(text, disallowed_special=()))
        elif isinstance(msg, str):
            total += len(enc.encode(msg, disallowed_special=()))
    return total


def _parse_iso(s: str | None) -> datetime | None:
    """Parse an ISO 8601 string into a naive (tz-stripped) datetime."""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d.replace(tzinfo=None) if d.tzinfo else d


def _to_naive_dt(ts) -> datetime | None:
    """ClickHouse timestamps come back from pyarrow as datetimes most of the
    time; defensively handle ISO strings too."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None) if ts.tzinfo else ts
    if isinstance(ts, str):
        try:
            d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
        return d.replace(tzinfo=None) if d.tzinfo else d
    return None


def _select_files(
    data_dir: str,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> list[str]:
    """Pick parquets in ``[start_dt, end_dt)`` (filename-timestamp wise),
    plus the file immediately before ``start_dt`` because chunks may straddle
    the boundary -- the row-level timestamp filter handles the rest."""
    names = sorted(
        f for f in os.listdir(data_dir) if f.startswith("llm_responses_") and f.endswith(".parquet")
    )

    dated: list[tuple[datetime, str]] = []
    for name in names:
        stem = name[len("llm_responses_") : -len(".parquet")]
        try:
            ts = datetime.strptime(stem, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        dated.append((ts, name))
    dated.sort()

    selected: list[str] = []
    last_before_start: str | None = None
    for ts, name in dated:
        if start_dt is not None and ts < start_dt:
            last_before_start = name
            continue
        if end_dt is not None and ts >= end_dt:
            break
        selected.append(name)

    if start_dt is not None and last_before_start is not None:
        selected.insert(0, last_before_start)

    return selected


# A "row source" is any iterator that yields ``(timestamp_or_none, chat_completion_body)``
# tuples. ``_iter_bodies`` layers offset / limit / overrides / stream_options
# injection on top of one. Both built-in sources (production parquet, HF
# ``save_to_disk``) follow this contract; adding a new dataset only requires
# implementing another generator.
RowSource = Iterator[tuple["datetime | None", dict]]


def _iter_prod_rows(
    data_dir: str,
    *,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> RowSource:
    """Yield ``(timestamp, chat_completion_body)`` from production parquet shards.

    Streams via ``iter_batches`` so memory stays bounded even on a multi-day
    window. The optional time range is applied at the row level; rows with
    missing or unparseable timestamps are dropped.
    """
    import pyarrow.parquet as pq

    files = _select_files(data_dir, start_dt, end_dt)
    if not files:
        raise SystemExit(
            f"no production parquet files match the requested time range under {data_dir!r}"
        )
    for filename in files:
        path = os.path.join(data_dir, filename)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=4096, columns=["timestamp", "request"]):
            ts_col = batch.column("timestamp").to_pylist()
            req_col = batch.column("request").to_pylist()
            rows = sorted(zip(ts_col, req_col), key=lambda r: r[0] or datetime.min)
            for ts, raw in rows:
                ts_dt = _to_naive_dt(ts)
                if ts_dt is None:
                    continue
                if start_dt is not None and ts_dt < start_dt:
                    continue
                if end_dt is not None and ts_dt >= end_dt:
                    return
                try:
                    body = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, ValueError):
                    continue
                if not isinstance(body, dict):
                    continue
                msgs = body.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    continue
                yield ts_dt, body


def _hf_content_to_str(content) -> str:
    """Coerce an OpenAI-style ``content`` value to a plain string.

    HF chat datasets are usually plain strings, but some shapes (e.g.
    multimodal blocks ``[{"type": "text", "text": "..."}]``) need flattening.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _hf_normalize_messages(raw) -> list[dict]:
    """Normalize a HF row's conversation column into a chat-completions
    ``messages`` list. Drops empty messages; coerces unrecognized roles to
    ``user``."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for msg in raw:
        if isinstance(msg, str):
            text = msg
            role = "user"
        elif isinstance(msg, dict):
            role = str(msg.get("role") or "user").lower()
            text = _hf_content_to_str(msg.get("content"))
        else:
            continue
        if not text:
            continue
        if role not in _VALID_ROLES:
            role = "user"
        out.append({"role": role, "content": text})
    return out


def _iter_hf_rows(data_path: str) -> RowSource:
    """Yield ``(None, chat_completion_body)`` from a HF ``save_to_disk`` chat dataset.

    Picks the first non-empty among :data:`HF_MESSAGE_COLUMNS` as the source
    of messages, matching the convention used by
    ``data_processing/build_hf_prefix_trie.py``. Iteration is lazy
    (Arrow-backed), so memory stays bounded.
    """
    from datasets import Dataset, DatasetDict, load_from_disk

    dsd = load_from_disk(data_path)
    if isinstance(dsd, DatasetDict):
        # Match build_hf_prefix_trie's split selection.
        if "train" in dsd:
            ds = dsd["train"]
        else:
            ds = dsd[next(iter(dsd))]
    elif isinstance(dsd, Dataset):
        ds = dsd
    else:
        raise SystemExit(f"unsupported HF dataset object at {data_path!r}: {type(dsd)!r}")

    columns = set(ds.column_names)
    msg_col: str | None = None
    for k in HF_MESSAGE_COLUMNS:
        if k in columns:
            msg_col = k
            break
    if msg_col is None:
        raise SystemExit(
            f"HF dataset at {data_path!r} has none of {HF_MESSAGE_COLUMNS}; "
            f"available columns: {sorted(columns)}"
        )

    _log(f"hf dataset at {data_path!r}: {len(ds)} rows, using column {msg_col!r}")
    for row in ds:
        msgs = _hf_normalize_messages(row[msg_col])
        if not msgs:
            continue
        yield None, {"messages": msgs}


def _iter_mooncake_rows(data_path: str) -> RowSource:
    """Yield replayable request bodies from a body-included Mooncake JSONL trace.

    Rows without a parsed ``request`` object are skipped because body-free
    Mooncake traces are simulation-only. ``timestamp`` is carried through as a
    private field consumed by ``run_replay_async`` when ``arrival_mode`` is
    ``open-loop``.
    """
    path = data_path if os.path.isabs(data_path) else os.path.join("/data", data_path)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = row.get("request")
            if not isinstance(body, dict):
                raw = row.get("request_raw")
                if isinstance(raw, str):
                    try:
                        body = json.loads(raw)
                    except json.JSONDecodeError:
                        body = None
            if not isinstance(body, dict):
                continue
            body = dict(body)
            if row.get("request_id"):
                body["_gorgo_request_id"] = str(row["request_id"])
            if isinstance(row.get("timestamp"), int | float):
                body["_gorgo_scheduled_delay_ms"] = float(row["timestamp"])
            yield None, body


def _build_row_source(
    *,
    source: str,
    data_path: str | None,
    preset: str | None,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> tuple[RowSource, str]:
    """Resolve ``--source`` / ``--preset`` / ``--data-path`` into a concrete
    :data:`RowSource` plus the path it actually reads from. Raises
    :class:`SystemExit` for invalid combinations so the Modal CLI prints a
    clean error."""
    if source == SOURCE_PROD:
        if preset:
            raise SystemExit(f"--preset is only valid with --source hf (got {preset!r})")
        path = data_path or PROD_DEFAULT_PATH
        return _iter_prod_rows(path, start_dt=start_dt, end_dt=end_dt), path
    if source == SOURCE_HF:
        if preset is not None:
            if preset not in HF_PRESETS:
                raise SystemExit(
                    f"unknown --preset {preset!r}; expected one of {sorted(HF_PRESETS)}"
                )
            path = data_path or HF_PRESETS[preset]
        elif data_path:
            path = data_path
        else:
            raise SystemExit("--source hf requires either --preset {lmsys|wildchat} or --data-path")
        if start_dt is not None or end_dt is not None:
            _log("note: --start-time / --end-time ignored for --source hf (no per-row timestamps)")
        return _iter_hf_rows(path), path
    if source == SOURCE_MOONCAKE:
        if preset:
            raise SystemExit(f"--preset is only valid with --source hf (got {preset!r})")
        if not data_path:
            raise SystemExit("--source mooncake requires --data-path")
        path = data_path if os.path.isabs(data_path) else os.path.join("/data", data_path)
        if start_dt is not None or end_dt is not None:
            _log("note: --start-time / --end-time ignored for --source mooncake")
        return _iter_mooncake_rows(path), path
    raise SystemExit(f"unknown --source {source!r}; expected one of {list(SUPPORTED_SOURCES)}")


def _iter_bodies(
    rows: RowSource,
    *,
    offset: int,
    num_requests: int | None,
    model_override: str | None,
    stream_override: bool | None,
    max_tokens_override: int | None,
    max_input_tokens: int | None = None,
    filter_stats: dict | None = None,
) -> RowSource:
    """Layer offset / limit / per-request field overrides / stream_options
    injection on top of any :data:`RowSource`.

    Splitting source enumeration from this transformer step lets the production
    and HF readers stay simple generators -- all CLI knobs are honored
    uniformly here.

    ``max_input_tokens`` (when truthy) caps the approximate input token
    count per row using the same tiktoken-cl100k encoder as
    ``proxy/modal_proxy.py``. Rows whose count exceeds the cap are
    silently skipped *before* counting toward the offset/limit, so
    ``num_requests=N`` still yields ``N`` valid rows when the cap
    filters out the long tail. ``filter_stats`` (when provided) is
    populated in place with ``filtered`` / ``max_filtered_tokens``
    counters for end-of-run reporting.
    """
    skipped = 0
    yielded = 0
    for ts_dt, body in rows:
        if not isinstance(body, dict):
            continue
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue
        if model_override is not None:
            body["model"] = model_override
        if stream_override is not None:
            body["stream"] = stream_override
        if max_tokens_override is not None:
            body["max_tokens"] = max_tokens_override
            body.pop("max_completion_tokens", None)
        effective_max_tokens = (
            body.get("max_completion_tokens") or body.get("max_tokens") or max_tokens_override or 0
        )
        if max_input_tokens and max_input_tokens > 0:
            n_tokens = _approx_input_tokens(msgs)
            if n_tokens + effective_max_tokens > max_input_tokens:
                if filter_stats is not None:
                    filter_stats["filtered"] = filter_stats.get("filtered", 0) + 1
                    if n_tokens > filter_stats.get("max_filtered_tokens", 0):
                        filter_stats["max_filtered_tokens"] = n_tokens
                continue
        if skipped < offset:
            skipped += 1
            continue
        # On SSE requests, ask the upstream SGLang replica to emit a final
        # ``data: {... "usage": {...}}`` event by setting
        # ``stream_options.include_usage = True``. That gives us exact
        # prompt / completion token counts instead of having to approximate
        # ``completion_tokens`` by counting ``delta.content`` SSE events.
        if body.get("stream") is True:
            so = body.get("stream_options")
            if not isinstance(so, dict):
                so = {}
            so["include_usage"] = True
            body["stream_options"] = so
        yield ts_dt, body
        yielded += 1
        if num_requests is not None and yielded >= num_requests:
            return


def _resolve_input_token_cap(
    *,
    user_override: int | None,
    context_length: int | None,
    max_tokens: int | None,
) -> int | None:
    """Resolve the per-request input-token cap used by ``_iter_bodies``.

    Priority:
      * ``user_override > 0`` -> return as-is (operator knows best).
      * ``user_override < 0`` -> disable filtering entirely.
      * ``user_override == 0`` (the auto sentinel):
          - if ``context_length`` is known, return
            ``context_length - max_tokens - SAFETY_MARGIN`` so prompt +
            completion budget stays comfortably under the model's hard
            cap.
          - else return ``None`` (filter disabled; caller logs a
            warning so the operator knows to set the flag manually).
    """
    if user_override is not None:
        if user_override < 0:
            return None
        if user_override > 0:
            return user_override
    if context_length is None or context_length <= 0:
        return None
    reserve = (max_tokens or 0) + CONTEXT_LENGTH_SAFETY_MARGIN_TOKENS
    cap = context_length - reserve
    return cap if cap > 0 else None


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


async def _fetch_proxy_info(client) -> dict:
    """Snapshot the proxy's routing config + the served model, so each
    saved benchmark JSON records exactly what was being served when it
    ran. Best-effort: anything that fails is captured under ``errors``
    instead of bringing the run down.

    We pull policy + hyperparameters straight from ``GET /policy`` on the
    proxy; the replica list from ``GET /replicas``; and the model id from
    one of the replicas' ``GET /v1/models`` (the proxy doesn't proxy that
    OpenAI-compat route, but the workload is co-located in the same
    region as the replicas so a direct hop is fine).
    """
    info: dict = {"errors": {}}

    async def _get_json(url: str) -> dict | None:
        try:
            r = await client.get(url, timeout=10.0)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            info["errors"][url] = repr(e)
            return None

    policy = await _get_json("/policy")
    if policy is not None:
        info["policy"] = policy.get("policy")
        info["supported_policies"] = policy.get("supported")
        info["proxy_hyperparameters"] = policy.get("hyperparameters")

    reps = await _get_json("/replicas")
    if reps is not None:
        info["replicas"] = reps.get("replicas") or []
        info["replica_count"] = reps.get("count")

    replica_urls_local = info.get("replicas") or []
    if replica_urls_local:
        replica = replica_urls_local[0]
        models_doc = await _get_json(f"{replica}/v1/models")
        if models_doc is not None:
            models = [
                m.get("id")
                for m in (models_doc.get("data") or [])
                if isinstance(m, dict) and m.get("id")
            ]
            info["served_models"] = models
            info["served_model"] = models[0] if models else None
            info["model_source_replica"] = replica
        # SGLang's ``/get_server_info`` returns ``context_length: null``
        # when the operator didn't pass ``--context-length`` to
        # ``launch_server`` (in which case the model's tokenizer
        # ``model_max_length`` is used at runtime). Capture whatever the
        # server reports so the workload's input-token filter can
        # auto-size against the actual configured cap when one exists.
        server_info = await _get_json(f"{replica}/get_server_info")
        if server_info is not None:
            ctx = server_info.get("context_length")
            info["context_length"] = ctx if isinstance(ctx, int) and ctx > 0 else None
            info["max_total_num_tokens"] = server_info.get("max_total_num_tokens")
            info["max_prefill_tokens"] = server_info.get("max_prefill_tokens")

    if not info["errors"]:
        del info["errors"]
    return info


NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000


def _stats(xs: list[float]) -> dict:
    """avg/min/max/p50/p95/p99/n summary; zero-filled when ``xs`` is empty."""
    if not xs:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
    return {
        "avg": sum(xs) / len(xs),
        "min": min(xs),
        "max": max(xs),
        "p50": _percentile(xs, 0.50),
        "p95": _percentile(xs, 0.95),
        "p99": _percentile(xs, 0.99),
        "n": len(xs),
    }


class _NonStreamingResponse(Exception):
    """Upstream returned a non-SSE response. We rely on SSE chunk arrival
    timestamps for TTFT/ITL, so a non-streaming reply can't be measured
    correctly and is treated as a failed request."""


async def _send_one(
    client,
    body: dict,
    request_id: str | None = None,
    scheduled_delay_ms: float | None = None,
    sent_delay_ms: float | None = None,
) -> dict:
    """Send one chat-completions request and capture streaming-aware timings.

    Always expects an SSE response. Non-streaming replies cannot produce a
    meaningful TTFT/ITL pair under this measurement strategy and are rejected
    via :class:`_NonStreamingResponse` (recorded as a failed request).
    """
    # Capture the timer BEFORE opening the stream so TTFT includes
    # request-send and response-header latency. ``consume_sse_stream`` is
    # explicit about this contract.
    request_start_ns = time.perf_counter_ns()
    ttft_ns: int | None = None
    output_tokens = 0
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    try:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=body,
            headers={
                "accept-encoding": "identity",
                **({"x-gorgo-request-id": request_id} if request_id else {}),
            },
        ) as resp:
            is_sse = resp.headers.get("content-type", "").startswith("text/event-stream")
            if not is_sse:
                # Two cases here:
                #   (a) Upstream returned an error status (e.g. 400 from
                #       SGLang chat-template validation). The body is a
                #       small JSON error and we want both the real status
                #       code and a snippet of the body in ``error`` so
                #       downstream log lines / saved JSON reflect the
                #       actual failure mode.
                #   (b) Upstream returned 2xx but non-SSE. Unsupported by
                #       this measurement strategy -- keep ``status: 0``
                #       as the sentinel for "no usable HTTP status".
                body_bytes = await resp.aread()
                snippet = body_bytes[:512].decode("utf-8", errors="replace").strip()
                if not 200 <= resp.status_code < 300:
                    return {
                        "status": resp.status_code,
                        "ttft_ns": None,
                        "total_ns": time.perf_counter_ns() - request_start_ns,
                        "input_tokens": None,
                        "output_tokens": 0,
                        "is_sse": False,
                        "error": f"upstream_error: {snippet}" if snippet else "upstream_error",
                        "request_id": request_id,
                        "scheduled_delay_ms": scheduled_delay_ms,
                        "sent_delay_ms": sent_delay_ms,
                    }
                raise _NonStreamingResponse(
                    f"upstream returned content-type "
                    f"{resp.headers.get('content-type')!r}; "
                    "workload requires stream=True for accurate TTFT/ITL"
                )
            (
                ttft_ns,
                output_tokens,
                usage_prompt_tokens,
                usage_completion_tokens,
                _meta_info,
            ) = await consume_sse_stream(resp, request_start_ns=request_start_ns)
            final_output_tokens = (
                usage_completion_tokens if usage_completion_tokens is not None else output_tokens
            )
            return {
                "status": resp.status_code,
                "ttft_ns": ttft_ns,
                "total_ns": time.perf_counter_ns() - request_start_ns,
                "input_tokens": usage_prompt_tokens,
                "output_tokens": final_output_tokens,
                "is_sse": True,
                "error": None,
                "request_id": request_id,
                "scheduled_delay_ms": scheduled_delay_ms,
                "sent_delay_ms": sent_delay_ms,
            }
    except _NonStreamingResponse as e:
        return {
            "status": 0,
            "ttft_ns": None,
            "total_ns": time.perf_counter_ns() - request_start_ns,
            "input_tokens": usage_prompt_tokens,
            "output_tokens": (
                usage_completion_tokens if usage_completion_tokens is not None else output_tokens
            ),
            "is_sse": False,
            "error": f"non_streaming_response: {e}",
            "request_id": request_id,
            "scheduled_delay_ms": scheduled_delay_ms,
            "sent_delay_ms": sent_delay_ms,
        }
    except Exception as e:
        # Captures httpx.HTTPError plus anything raised while we were
        # parsing the SSE stream. Status 0 means "no upstream HTTP status
        # was committed"; the ``error`` field disambiguates the cause.
        return {
            "status": 0,
            "ttft_ns": ttft_ns,
            "total_ns": time.perf_counter_ns() - request_start_ns,
            "input_tokens": usage_prompt_tokens,
            "output_tokens": (
                usage_completion_tokens if usage_completion_tokens is not None else output_tokens
            ),
            "is_sse": False,
            "error": f"{type(e).__name__}: {e}",
            "request_id": request_id,
            "scheduled_delay_ms": scheduled_delay_ms,
            "sent_delay_ms": sent_delay_ms,
        }


def _print_summary(stats: dict, fail_breakdown: dict[int, int]) -> None:
    """Render the per-run summary lines from a fully-populated ``stats``
    dict. Kept separate from the dict construction so future output sinks
    (CSV, JSONL, dashboards) can consume the same source of truth."""
    elapsed = stats["elapsed_seconds"]
    sent = stats["sent"]
    ok = stats["ok"]
    fail = stats["fail"]
    success_rate = stats["success_rate"]
    ttft_s = stats["ttft_seconds"]
    total_s = stats["request_e2e_seconds"]
    itl_s = stats["itl_ms"]
    decode_s = stats["decode_tokens_per_second"]
    in_tok_s = stats["input_tokens"]
    out_tok_s = stats["output_tokens"]

    print()
    _log(f"done in {elapsed:.1f}s")
    _log(f"  sent={sent} ok={ok} fail={fail} success_rate={success_rate * 100:.1f}%")
    if fail_breakdown:
        _log(f"  failures by status: {dict(sorted(fail_breakdown.items()))}")
    _log(f"  request throughput={stats['request_throughput_rps']:.2f} req/s")
    _log(
        f"  token throughput   "
        f"input={stats['input_token_throughput']:,.1f} tok/s  "
        f"output={stats['output_token_throughput']:,.1f} tok/s  "
        f"total={stats['total_token_throughput']:,.1f} tok/s"
    )
    if ttft_s["n"]:
        _log(
            f"  TTFT (s)         avg={ttft_s['avg']:.3f} "
            f"p50={ttft_s['p50']:.3f} p95={ttft_s['p95']:.3f} p99={ttft_s['p99']:.3f} "
            f"(n={ttft_s['n']})"
        )
    if total_s["n"]:
        _log(
            f"  request E2E (s)  avg={total_s['avg']:.2f} "
            f"p50={total_s['p50']:.2f} p95={total_s['p95']:.2f} p99={total_s['p99']:.2f} "
            f"(n={total_s['n']})"
        )
    if itl_s["n"]:
        _log(
            f"  ITL (ms)         avg={itl_s['avg']:.1f} "
            f"p50={itl_s['p50']:.1f} p95={itl_s['p95']:.1f} p99={itl_s['p99']:.1f} "
            f"(n={itl_s['n']})"
        )
    if decode_s["n"]:
        _log(
            f"  decode (tok/s)   avg={decode_s['avg']:.1f} "
            f"p50={decode_s['p50']:.1f} p95={decode_s['p95']:.1f} p99={decode_s['p99']:.1f} "
            f"(n={decode_s['n']})"
        )
    if in_tok_s["n"]:
        _log(
            f"  input tokens     avg={in_tok_s['avg']:.0f} "
            f"p50={in_tok_s['p50']:.0f} p95={in_tok_s['p95']:.0f} p99={in_tok_s['p99']:.0f} "
            f"(n={in_tok_s['n']})"
        )
    if out_tok_s["n"]:
        _log(
            f"  output tokens    avg={out_tok_s['avg']:.0f} "
            f"p50={out_tok_s['p50']:.0f} p95={out_tok_s['p95']:.0f} p99={out_tok_s['p99']:.0f} "
            f"(n={out_tok_s['n']})"
        )


async def run_replay_async(
    proxy_url: str,
    *,
    source: str = SOURCE_PROD,
    preset: str | None = None,
    data_path: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    offset: int = 0,
    num_requests: int | None = None,
    concurrency: int = 16,
    model: str | None = DEFAULT_MODEL,
    stream: bool | None = None,
    max_tokens: int | None = None,
    max_input_tokens: int | None = None,
    output_path: str | None = None,
    save_per_request: bool = True,
    run_id: str | None = None,
    arrival_mode: str = "bounded",
    time_scale: float = 1.0,
) -> dict:
    """Replay chat-completions traffic against the GORGO proxy.

    Args:
        proxy_url: Base URL of the proxy (the ``modal.forward`` tunnel URL
            printed by ``modal run proxy/modal_proxy.py``). Trailing slashes
            are stripped.
        source: ``"prod"`` (default) replays the production ClickHouse export
            from ``/data``; ``"hf"`` replays a Hugging Face ``save_to_disk``
            chat dataset.
        preset: For ``source="hf"``, fills in a default ``data_path`` from
            :data:`HF_PRESETS` (currently ``lmsys`` and ``wildchat``). Mutually
            exclusive with passing a custom ``data_path`` only in the sense
            that an explicit ``data_path`` always wins.
        data_path: Override the dataset disk path. For ``prod`` defaults to
            ``/data``; for ``hf`` is required unless ``preset`` is set.
        start_time / end_time: Half-open ``[start, end)`` filter on the row
            ``timestamp`` column (ISO 8601, e.g. ``2026-04-01T12:00:00``).
            ``None`` on either side means unbounded. Only honored for
            ``source="prod"``; ignored (with a warning) for ``hf``.
        offset: Skip this many requests before sending.
        num_requests: Cap on requests sent (after ``offset``). ``None`` means
            consume the entire source.
        concurrency: Number of in-flight requests at the proxy.
        model: Replace each request's ``model`` field. Defaults to the
            served-model-name in ``engine/modal_sglang.py``. HF datasets
            don't carry a ``model`` field, so the override is what populates
            it. Pass ``None`` to leave the original alone.
        stream: Override the ``stream`` flag on every request. ``None`` =
            pass through (HF rows default to non-streaming, so passing
            ``True`` is recommended for accurate TTFT measurements).
        max_tokens: Override ``max_tokens``. ``None`` = pass through.
        max_input_tokens: Cap on the approximate input token count per
            request; rows above the cap are skipped before counting
            toward ``num_requests``. ``None`` (or ``0``) = auto-resolve
            from the SGLang replica's ``/get_server_info`` (using
            ``context_length - max_tokens - safety_margin``); negative
            disables the filter. Counts are computed via the same
            tiktoken-cl100k encoder the proxy uses, so they're
            directional rather than exact (apply your own headroom if
            tuning against multilingual corpora).
        output_path: Where to write the JSON results doc inside the
            ``GORGO-bench-results`` volume. Relative paths are resolved
            under ``/results``. ``None`` (default) -> auto-generated
            ``replay_<UTC-timestamp>.json``.
        save_per_request: Include the per-request rows (status, timings in
            ns, token counts) alongside the aggregate stats. Set to
            ``False`` for tiny output files when only the summary matters.
        arrival_mode: ``"bounded"`` preserves the historical concurrency
            behavior; ``"open-loop"`` honors Mooncake row timestamps as
            scheduled arrivals.
        time_scale: Additional multiplier applied to Mooncake timestamps at
            replay time.

    Returns:
        Summary dict with ``sent`` / ``ok`` / ``fail`` / ``elapsed_seconds``
        / ``throughput_rps`` / latency percentiles plus the resolved
        ``output_path`` of the saved JSON doc.
    """
    proxy_url = proxy_url.rstrip("/")
    if arrival_mode not in ("bounded", "open-loop"):
        raise SystemExit("--arrival-mode must be 'bounded' or 'open-loop'")
    if time_scale <= 0:
        raise SystemExit("--time-scale must be > 0")
    start_dt = _parse_iso(start_time)
    end_dt = _parse_iso(end_time)

    # Snapshot of the resolved invocation, embedded in the saved JSON so
    # each result file is self-describing (no need to cross-reference the
    # CLI that produced it).
    run_started_at = datetime.now(timezone.utc)
    run_id = run_id or f"replay-{run_started_at.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    config = {
        "run_id": run_id,
        "proxy_url": proxy_url,
        "source": source,
        "preset": preset,
        "data_path": data_path,
        "start_time": start_time,
        "end_time": end_time,
        "offset": offset,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "model": model,
        "stream": stream,
        "max_tokens": max_tokens,
        "max_input_tokens": max_input_tokens,
        "arrival_mode": arrival_mode,
        "time_scale": time_scale,
        "region": REGION,
        "run_started_at": run_started_at.isoformat().replace("+00:00", "Z"),
    }

    # Refresh the volume backing whichever source we're about to read.
    # Modal volumes are eventually consistent; ``reload()`` is a single
    # round-trip and cheap relative to the replay run. The other volumes
    # don't need refreshing because we only read from one per run.
    if source in (SOURCE_PROD, SOURCE_MOONCAKE):
        await completions_volume.reload.aio()
    elif source == SOURCE_HF:
        # Both LMSYS and WildChat datasets are read-only here; reload both
        # since presets may resolve to either volume.
        await lmsys_chat_1m_volume.reload.aio()
        await hf_datasets_volume.reload.aio()

    rows, resolved_path = _build_row_source(
        source=source,
        data_path=data_path,
        preset=preset,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    config["resolved_data_path"] = resolved_path

    range_desc = ""
    if source == SOURCE_PROD:
        range_desc = "".join(
            [
                f" from {start_time}" if start_time else "",
                f" until {end_time}" if end_time else "",
            ]
        )
    _log(
        f"source={source} path={resolved_path}{range_desc}"
        + (f" (preset={preset})" if preset else "")
    )
    _log(
        f"dispatching: proxy={proxy_url} concurrency={concurrency} "
        f"offset={offset} limit={num_requests if num_requests is not None else 'all'}"
    )

    timeout = httpx.Timeout(
        connect=15.0, read=None, write=30.0, pool=10.0
    )  # read=None so longer requests don't timeout
    limits = httpx.Limits(
        max_connections=concurrency
        * 2,  # double the maximum concurrency so requests will never queue for connections
        max_keepalive_connections=concurrency * 2,
        keepalive_expiry=None,
    )

    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 2)
    results: list[dict] = []
    sent = 0
    done = 0
    t_start = time.perf_counter()
    last_log = t_start

    async with httpx.AsyncClient(
        base_url=proxy_url,
        http2=True,
        timeout=timeout,
        limits=limits,
    ) as client:
        # Snapshot the currently policy info and print to console
        proxy_info = await _fetch_proxy_info(client)
        _log(
            f"  proxy: policy={proxy_info.get('policy')!r} "
            f"model={proxy_info.get('served_model')!r} "
            f"replicas={proxy_info.get('replica_count')}"
        )
        if proxy_info.get("proxy_hyperparameters"):
            _log(f"  proxy hyperparameters: {proxy_info['proxy_hyperparameters']}")

        # Resolve the input-token filter cap. ``user_override=0``
        # is the auto sentinel (matches the workload CLI default),
        # so a missing flag still triggers auto-detection.
        input_token_cap = _resolve_input_token_cap(
            user_override=max_input_tokens,
            context_length=proxy_info.get("context_length"),
            max_tokens=max_tokens,
        )
        if input_token_cap is not None:
            origin = (
                "operator override"
                if (max_input_tokens is not None and max_input_tokens > 0)
                else (
                    f"auto from context_length={proxy_info.get('context_length')} - "
                    f"max_tokens={max_tokens or 0} - margin={CONTEXT_LENGTH_SAFETY_MARGIN_TOKENS}"
                )
            )
            _log(
                f"  input-token filter: cap={input_token_cap} ({origin}); "
                f"rows above cap will be skipped before counting toward num_requests"
            )
        else:
            ctx_len = proxy_info.get("context_length")
            _log(
                f"  input-token filter: disabled "
                f"(context_length={ctx_len!r}, override={max_input_tokens!r}); "
                f"overlong rows will hit upstream as 4xx -- pass --max-input-tokens N to filter"
            )
        config["input_token_cap"] = input_token_cap
        config["context_length"] = proxy_info.get("context_length")

        progress_log: list[dict] = []
        filter_stats: dict = {"filtered": 0, "max_filtered_tokens": 0}

        async def worker() -> None:
            nonlocal done, last_log
            while True:
                item = await queue.get()
                if item is None:
                    return
                _, body, request_id, scheduled_delay_ms, sent_delay_ms = item
                res = await _send_one(
                    client,
                    body,
                    request_id=request_id,
                    scheduled_delay_ms=scheduled_delay_ms,
                    sent_delay_ms=sent_delay_ms,
                )
                results.append(res)
                done += 1
                # Emit a single-line failure record as soon as it
                # happens. Useful when the proxy returns 4xx/5xx
                # mid-run (e.g. SGLang chat-template validation
                # rejecting a malformed conversation): without this,
                # the only signal is a bumped ``fail`` counter on
                # the next 5-second progress tick.
                if not (200 <= res["status"] < 300):
                    snippet = (res.get("error") or "")[:160]
                    _log(
                        f"  request fail status={res['status']} "
                        f"input_tokens={res.get('input_tokens')} "
                        f'error="{snippet}"'
                    )
                now = time.perf_counter()
                if now - last_log >= 5.0:
                    elapsed = now - t_start
                    ok_n = sum(1 for r in results if 200 <= r["status"] < 300)
                    rate = done / elapsed if elapsed > 0 else 0.0
                    progress = {
                        "ts": _ts(),
                        "event": "progress",
                        "elapsed_seconds": round(elapsed, 3),
                        "sent": sent,
                        "done": done,
                        "ok": ok_n,
                        "fail": done - ok_n,
                        "rate_rps": round(rate, 2),
                    }
                    progress_log.append(progress)
                    # Emit as a single-line JSON record so log scrapers
                    # can grep ``"event": "progress"`` and parse with
                    # ``jq``; the ``ts`` field is the wall-clock time
                    # of the snapshot (paired with ``elapsed_seconds``
                    # which is monotonic-clock since dispatch start).
                    print(json.dumps(progress), flush=True)
                    last_log = now

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        try:
            # ``_iter_bodies`` wraps a per-source generator (parquet
            # ``iter_batches`` for production, Arrow-backed row iteration for
            # HF), so rows are pulled lazily; combined with ``queue``
            # (bounded at ``concurrency * 2``) the pipeline back-pressures
            # from the workers all the way down to the dataset reader.
            # Memory stays O(concurrency) regardless of dataset size.
            for ts, body in _iter_bodies(
                rows,
                offset=offset,
                num_requests=num_requests,
                model_override=model,
                stream_override=stream,
                max_tokens_override=max_tokens,
                max_input_tokens=input_token_cap,
                filter_stats=filter_stats,
            ):
                embedded_request_id = body.pop("_gorgo_request_id", None)
                raw_delay_ms = body.pop("_gorgo_scheduled_delay_ms", None)
                scheduled_delay_ms = (
                    float(raw_delay_ms) * time_scale
                    if raw_delay_ms is not None and arrival_mode == "open-loop"
                    else None
                )
                if scheduled_delay_ms is not None:
                    sleep_for = (t_start + scheduled_delay_ms / 1000.0) - time.perf_counter()
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                request_id = embedded_request_id or f"{run_id}-{sent:06d}"
                sent_delay_ms = (time.perf_counter() - t_start) * 1000.0
                await queue.put((ts, body, request_id, scheduled_delay_ms, sent_delay_ms))
                sent += 1
                # Yield periodically so embedded proxy runs don't monopolize
                # uvicorn's event loop while scanning/filtering large rows.
                if sent % max(1, concurrency) == 0:
                    await asyncio.sleep(0)
        finally:
            for _ in range(concurrency):
                await queue.put(None)
            await asyncio.gather(*workers)

    elapsed = max(time.perf_counter() - t_start, 1e-9)
    ok_results = [r for r in results if 200 <= r["status"] < 300]
    ok = len(ok_results)
    fail = len(results) - ok

    # All per-request timings come back in nanoseconds; convert to s/ms
    # at the boundary so display formatting stays cheap.
    ttfts = [r["ttft_ns"] / NS_PER_S for r in ok_results if r["ttft_ns"] is not None]
    totals = [r["total_ns"] / NS_PER_S for r in ok_results]
    input_tokens = [
        r["input_tokens"]
        for r in ok_results
        if r["input_tokens"] is not None and r["input_tokens"] > 0
    ]
    output_tokens = [r["output_tokens"] for r in ok_results if r["output_tokens"] > 0]

    # Inter-token latency and decode rate are only meaningful on SSE
    # responses with at least 2 emitted tokens (so we have a real decode
    # window between TTFT and end-of-stream).
    sse_decoded = [
        r
        for r in ok_results
        if r["is_sse"] and r["ttft_ns"] is not None and r["output_tokens"] >= 2
    ]
    itls_ms = [
        (r["total_ns"] - r["ttft_ns"]) / NS_PER_MS / (r["output_tokens"] - 1) for r in sse_decoded
    ]
    decode_rates = [
        (r["output_tokens"] - 1) * NS_PER_S / max(r["total_ns"] - r["ttft_ns"], 1)
        for r in sse_decoded
    ]

    # Aggregate token throughput is the headline number for inference
    # benchmarks: total tokens emitted (or consumed) by the server,
    # divided by wall-clock dispatch time. Uses ok-only token totals so
    # failed requests don't inflate or deflate the rate.
    total_input_tokens = sum(input_tokens)
    total_output_tokens = sum(output_tokens)
    input_throughput = total_input_tokens / elapsed
    output_throughput = total_output_tokens / elapsed
    total_throughput = (total_input_tokens + total_output_tokens) / elapsed

    # Per-status counts, useful for distinguishing "503 from queue
    # overflow" vs "0 from upstream connection drops".
    status_breakdown: dict[int, int] = {}
    for r in results:
        status_breakdown[r["status"]] = status_breakdown.get(r["status"], 0) + 1
    fail_breakdown = {s: c for s, c in status_breakdown.items() if not 200 <= s < 300}

    ttft_s = _stats(ttfts)
    total_s = _stats(totals)
    itl_s = _stats(itls_ms)
    decode_s = _stats(decode_rates)
    in_tok_s = _stats([float(x) for x in input_tokens])
    out_tok_s = _stats([float(x) for x in output_tokens])

    success_rate = (ok / len(results)) if results else 0.0

    stats = {
        "sent": sent,
        "ok": ok,
        "fail": fail,
        "success_rate": success_rate,
        "status_breakdown": status_breakdown,
        "elapsed_seconds": elapsed,
        "request_throughput_rps": len(results) / elapsed,
        "input_token_throughput": input_throughput,
        "output_token_throughput": output_throughput,
        "total_token_throughput": total_throughput,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "ttft_seconds": ttft_s,
        "request_e2e_seconds": total_s,
        "itl_ms": itl_s,
        "decode_tokens_per_second": decode_s,
        "input_tokens": in_tok_s,
        "output_tokens": out_tok_s,
    }
    if filter_stats.get("filtered"):
        stats["input_filtered"] = {
            "count": filter_stats["filtered"],
            "max_filtered_tokens": filter_stats["max_filtered_tokens"],
            "cap": input_token_cap,
        }
        _log(
            f"  input-token filter: skipped {filter_stats['filtered']} "
            f"row(s) over cap={input_token_cap} "
            f"(largest={filter_stats['max_filtered_tokens']} approx tokens)"
        )
    _print_summary(stats, fail_breakdown)
    raw_results = results
    config["proxy"] = proxy_info

    # Resolve where to drop the JSON: auto-name under /results when the
    # caller didn't pin a path, otherwise honor it (relative paths are
    # rooted at /results so callers don't have to know the mount point).
    if output_path is None:
        ts = run_started_at.strftime("%Y%m%d_%H%M%S")
        resolved_output_path = f"/results/replay_{ts}.json"
    else:
        resolved_output_path = (
            output_path if os.path.isabs(output_path) else os.path.join("/results", output_path)
        )
    os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)

    trace_status = None
    try:
        async with httpx.AsyncClient(
            base_url=proxy_url, timeout=httpx.Timeout(10.0)
        ) as trace_client:
            r = await trace_client.get("/trace/status")
            if r.status_code < 400:
                trace_status = r.json().get("trace")
    except Exception:
        trace_status = None

    output_doc: dict = {
        "config": config,
        "stats": stats,
        "progress": progress_log,
    }
    if trace_status is not None:
        output_doc["trace"] = trace_status
    if save_per_request:
        # Keep nanosecond ints as-is (JSON-friendly and lossless); analysis
        # code can divide by 1e9 / 1e6 at read time. ``error`` is None on
        # success, otherwise a short string identifying the failure reason.
        output_doc["requests"] = [
            {
                "status": r["status"],
                "ttft_ns": r["ttft_ns"],
                "total_ns": r["total_ns"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "is_sse": r["is_sse"],
                "error": r.get("error"),
                "request_id": r.get("request_id"),
                "scheduled_delay_ms": r.get("scheduled_delay_ms"),
                "sent_delay_ms": r.get("sent_delay_ms"),
            }
            for r in raw_results
        ]

    # Can be a sizeable file when ``save_per_request`` is on (one row per
    # replayed request); the volume is the right place for it.
    with open(resolved_output_path, "w") as f:
        json.dump(output_doc, f)
    bench_results_volume.commit()
    _log(f"  saved results to volume GORGO-bench-results at {resolved_output_path}")

    stats["output_path"] = resolved_output_path
    return stats


@app.function(
    image=image,
    region=REGION,
    timeout=24 * 60 * 60,
    volumes={
        "/data": completions_volume,
        "/lmsys": lmsys_chat_1m_volume,
        "/datasets": hf_datasets_volume,
        "/results": bench_results_volume,
    },
)
def replay(
    proxy_url: str,
    *,
    source: str = SOURCE_PROD,
    preset: str | None = None,
    data_path: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    offset: int = 0,
    num_requests: int | None = None,
    concurrency: int = 16,
    model: str | None = DEFAULT_MODEL,
    stream: bool | None = None,
    max_tokens: int | None = None,
    max_input_tokens: int | None = None,
    output_path: str | None = None,
    save_per_request: bool = True,
    run_id: str | None = None,
    arrival_mode: str = "bounded",
    time_scale: float = 1.0,
) -> dict:
    """Modal wrapper around :func:`run_replay_async`."""
    return asyncio.run(
        run_replay_async(
            proxy_url=proxy_url,
            source=source,
            preset=preset,
            data_path=data_path,
            start_time=start_time,
            end_time=end_time,
            offset=offset,
            num_requests=num_requests,
            concurrency=concurrency,
            model=model,
            stream=stream,
            max_tokens=max_tokens,
            max_input_tokens=max_input_tokens,
            output_path=output_path,
            save_per_request=save_per_request,
            run_id=run_id,
            arrival_mode=arrival_mode,
            time_scale=time_scale,
        )
    )


@app.local_entrypoint()
def main(
    proxy_url: str,
    source: str = SOURCE_PROD,
    preset: str = "",
    data_path: str = "",
    start_time: str = "",
    end_time: str = "",
    offset: int = 0,
    num_requests: int = 0,
    concurrency: int = 16,
    model: str = DEFAULT_MODEL,
    stream: str = "",
    max_tokens: int = 0,
    max_input_tokens: int = 0,
    run_id: str = "",
    arrival_mode: str = "bounded",
    time_scale: float = 1.0,
    output_path: str = "",
    save_per_request: bool = True,
):
    """CLI wrapper for ``replay``. Sentinel values map to ``None`` because
    Modal local_entrypoints don't accept ``Optional`` natively:

      empty string for preset / data_path / start_time / end_time /
        model / stream / output_path
      0 for num_requests / max_tokens
      0 for max_input_tokens (= auto-detect from /get_server_info);
        positive = explicit cap; negative = disable filter
    """
    stream_arg: bool | None
    s = stream.strip().lower()
    if s == "":
        stream_arg = None
    elif s in ("1", "true", "yes"):
        stream_arg = True
    elif s in ("0", "false", "no"):
        stream_arg = False
    else:
        raise SystemExit(f"invalid --stream={stream!r}; expected true/false")

    replay.remote(
        proxy_url=proxy_url,
        source=source,
        preset=preset or None,
        data_path=data_path or None,
        start_time=start_time or None,
        end_time=end_time or None,
        offset=offset,
        num_requests=num_requests or None,
        concurrency=concurrency,
        model=model or None,
        stream=stream_arg,
        max_tokens=max_tokens or None,
        max_input_tokens=max_input_tokens,
        run_id=run_id or None,
        arrival_mode=arrival_mode,
        time_scale=time_scale,
        output_path=output_path or None,
        save_per_request=save_per_request,
    )
