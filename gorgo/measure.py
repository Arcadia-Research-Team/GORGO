"""Measurement primitives shared by the GORGO proxy (``proxy/workload.py``,
``proxy/calibrate.py``, ``proxy/modal_proxy.py``) and any downstream router --
intentionally framework-free so online hyperparameter tuning can run wherever
real traffic flows.

The streaming helpers duck-type their ``client``/``resp`` arguments (anything
httpx-shaped works: ``resp.aiter_raw()``, ``client.get/post/stream``) so this
module has **no third-party imports** and stays usable from stdlib-only
installs of the ``gorgo`` package.

Each helper is a plain function / coroutine with no Modal- or SGLang-
specific plumbing. Collectively they implement the GORGO calibration
contract::

    score(u) = rtt_weight     * rtt_ms(u)
             + prefill_weight * prefill_rate(u) * (uncached + queued_tokens(u))

Both terms resolve to **milliseconds**.  ``prefill_rate`` has units of
**ms / token**; the calibrator measures the raw prefill rate in
seconds-per-token and multiplies by 1000 when emitting the
recommendation.  The dimensionless ``*_weight`` knobs default to 1.0
(physical magnitude) and are tuned separately.  (The decode rate is
still measured as a diagnostic but is no longer a routing parameter.)

Building blocks::

    consume_sse_stream(resp, *, request_start_ns)
        Drain an OpenAI-compatible SSE chat-completions response with
        chunk-arrival-precise timing. Returns ``(ttft_ns, output_tokens,
        prompt_tokens, completion_tokens)``. The caller MUST capture
        ``request_start_ns`` *before* awaiting ``client.stream(...)`` so
        TTFT includes request-send and response-header latency.

    ping_once(client, *, n=3)
        Median of N HTTP round-trips to ``/v1/models`` -- the cheapest
        universally-OpenAI-compatible endpoint. Used as a proxy for the
        irreducible network RTT subtracted from observed TTFT.

    measure_chat_completion(client, body, *, ping_rtt)
        One streaming chat completion + decomposition into ping/prefill/
        decode seconds and per-token rates. Returns ``None`` on transport,
        non-200, non-SSE, or missing-usage failures so the caller can skip
        and try the next prompt.

    flush_replica_cache(client)
        Best-effort ``POST /flush_cache`` (SGLang RadixAttention) so each
        calibration sample starts from a clean KV cache.

    compute_stats / percentile / ols_fit / recommend_rates
        Reduce a list of samples to summary stats and the recommended
        ``prefill_rate`` value.
"""

from __future__ import annotations

import json
import time
from typing import Any, Awaitable, Callable

NS_PER_S = 1_000_000_000


class NonStreamingResponse(Exception):
    """Upstream returned ``content-type`` other than ``text/event-stream``.
    SSE chunk-arrival timing is the only way to get a meaningful TTFT, so
    non-streaming replies are unusable for measurement."""


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


def compute_stats(xs: list[float]) -> dict:
    """Common-shape summary used by both workload and calibrate:
    ``n / mean / median / min / max / p50 / p95 / p99``."""
    if not xs:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    s = sorted(xs)
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "median": s[len(s) // 2],
        "min": s[0],
        "max": s[-1],
        "p50": percentile(xs, 0.50),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
    }


def ols_fit(xs: list[float], ys: list[float]) -> dict:
    """Plain OLS for ``y = a + b*x``. Returns ``{a, b, r2, n}``."""
    n = len(xs)
    if n < 2:
        return {"a": 0.0, "b": 0.0, "r2": 0.0, "n": n}
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return {"a": my, "b": 0.0, "r2": 0.0, "n": n}
    b = sxy / sxx
    a = my - b * mx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / syy if syy > 0 else 1.0
    return {"a": a, "b": b, "r2": r2, "n": n}


async def consume_sse_stream(
    resp: Any,
    *,
    request_start_ns: int,
    chunk_sink: Callable[[bytes], Awaitable[None]] | None = None,
    on_first_token: Callable[[], None] | None = None,
) -> tuple[int | None, int, int | None, int | None, dict | None]:
    """Drain an SSE chat-completions response, returning
    ``(ttft_ns, output_tokens, prompt_tokens, completion_tokens, meta_info)``.

    ``ttft_ns`` is the wire-arrival time -- relative to the caller-supplied
    ``request_start_ns`` -- of the chunk that delivered the first byte of
    the first ``choices[0].delta.content`` event. Capturing the start time
    in the caller (before ``client.stream(...)``) keeps TTFT inclusive of
    request-send and response-header latency.

    SGLang's leading role marker (``{"delta": {"role": "assistant"}}``) is
    correctly skipped because it has no ``content`` field. ``output_tokens``
    falls back to a count of ``delta.content`` events when the server
    doesn't honor ``stream_options.include_usage``.

    ``chunk_sink`` is called with each raw chunk as it arrives, before
    parsing, so the proxy can tee bytes to its downstream client without
    waiting for parse to complete -- this is what lets the proxy do
    on-the-fly tuning without delaying TTFT for live traffic.

    ``on_first_token`` is a cheap synchronous callback invoked exactly once,
    immediately when ``ttft_ns`` transitions from ``None`` to set (i.e. the
    first ``delta.content`` event arrives). The proxy uses it to release its
    queue+prefill load counters at first token rather than end-of-decode.
    It is guarded so an exception inside it can never break the parse loop.

    ``meta_info`` is the last per-request ``meta_info`` (or ``metadata``)
    object surfaced anywhere in the SSE payloads -- top-level on the event
    object or nested under ``choices[0]``. SGLang's chat stream may carry
    timing fields (``queue_time``, ``prefill_waiting_latency``,
    ``e2e_latency``, ``cached_tokens``, ``*_ts`` wall-clock timestamps,
    ...). ``None`` when the stream surfaces no such object.
    """
    ttft_ns: int | None = None
    output_tokens = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    meta_info: dict | None = None

    buffer = bytearray()
    # Wire-arrival time of the chunk that delivered the first byte
    # currently sitting in ``buffer`` (i.e. the first byte of the
    # in-progress event). Reset to the next chunk's timestamp every
    # time we drain a full event.
    buffer_first_chunk_ns: int | None = None

    async for chunk in resp.aiter_raw():
        chunk_ns = time.perf_counter_ns()
        if chunk_sink is not None and chunk:
            # Forward to the downstream client first so client-side TTFT
            # is unaffected by our parse loop. The await may yield, but
            # ``chunk_ns`` was captured before the yield so our own TTFT
            # reading still reflects wire arrival.
            await chunk_sink(chunk)
        if not buffer:
            buffer_first_chunk_ns = chunk_ns
        buffer.extend(chunk)
        while True:
            idx = buffer.find(b"\n\n")
            if idx < 0:
                break
            event_bytes = bytes(buffer[:idx])
            del buffer[: idx + 2]
            event_arrival_ns = buffer_first_chunk_ns
            buffer_first_chunk_ns = chunk_ns if buffer else None

            for line in event_bytes.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[len(b"data:") :].strip()
                if not payload or payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # Final usage event (sent when stream_options.include_usage
                # is set) carries accurate token counts; capture and keep
                # going since [DONE] may follow.
                usage = obj.get("usage")
                if isinstance(usage, dict):
                    pt = usage.get("prompt_tokens")
                    ct = usage.get("completion_tokens")
                    if isinstance(pt, int):
                        prompt_tokens = pt
                    if isinstance(ct, int):
                        completion_tokens = ct
                # Capture per-request meta_info wherever it shows up.
                # SGLang may surface it top-level on the event or nested
                # under choices[0]; keep the last-seen dict so the final
                # event's (most complete) timing block wins.
                top_meta = obj.get("meta_info") or obj.get("metadata")
                if isinstance(top_meta, dict):
                    meta_info = top_meta
                choices = obj.get("choices") or []
                if choices:
                    nested_meta = choices[0].get("meta_info") or choices[0].get("metadata")
                    if isinstance(nested_meta, dict):
                        meta_info = nested_meta
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    if ttft_ns is None and event_arrival_ns is not None:
                        ttft_ns = event_arrival_ns - request_start_ns
                        # First-token transition: release load counters now
                        # (when wired). Guarded so a buggy callback can't
                        # corrupt the parse loop or starve the stream.
                        if on_first_token is not None:
                            try:
                                on_first_token()
                            except Exception:
                                pass
                    # One ``delta.content`` event ~= one decoded token in
                    # SGLang's streaming output.
                    output_tokens += 1

    return ttft_ns, output_tokens, prompt_tokens, completion_tokens, meta_info


async def ping_once(
    client: Any,
    *,
    n: int = 3,
    path: str = "/v1/models",
) -> float:
    """Median of N HTTP RTTs (seconds) to ``path`` on the bound replica.

    We deliberately reuse the long-lived ``client`` so the RTT reflects the
    same TLS/HTTP/2 path the chat-completions request will traverse;
    raw ICMP would miss connection-setup overhead and is also frequently
    blocked at the Modal edge. ``/v1/models`` is the cheapest universally-
    OpenAI-compatible endpoint -- ``/health`` isn't standardized.
    """
    rtts: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        r = await client.get(path)
        r.raise_for_status()
        rtts.append((time.perf_counter_ns() - t0) / NS_PER_S)
    rtts.sort()
    return rtts[len(rtts) // 2]


async def flush_replica_cache(
    client: Any,
    *,
    timeout: float = 60.0,
) -> bool:
    """Best-effort ``POST /flush_cache`` on the bound SGLang replica.

    Returns whether the upstream responded with a 2xx. Used between
    calibration samples so each probe starts from a clean RadixAttention
    cache (otherwise consecutive prompts that share a prefix would let the
    second one skip prefill, biasing ``prefill_rate`` downward).
    """
    try:
        r = await client.post("/flush_cache", timeout=timeout)
        return r.is_success
    except Exception:
        return False


async def measure_chat_completion(
    client: Any,
    body: dict,
    *,
    ping_rtt: float,
) -> dict | None:
    """One streaming chat-completion + decomposition into ping / prefill /
    decode seconds and per-token rates.

    Capturing ``request_start_ns`` *before* opening the stream is what
    keeps TTFT honest: it includes the request-send and response-header
    latency that an inside-the-stream timer would miss. Returns ``None``
    on transport failures, non-200 status, non-SSE responses, or missing
    ``usage`` token counts so the caller can skip and try the next prompt.
    """
    request_start_ns = time.perf_counter_ns()
    try:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=body,
            headers={"accept-encoding": "identity"},
        ) as resp:
            if resp.status_code != 200:
                await resp.aread()
                return None
            ct = resp.headers.get("content-type", "")
            if not ct.startswith("text/event-stream"):
                await resp.aread()
                return None
            (
                ttft_ns,
                output_tokens,
                prompt_tokens,
                completion_tokens,
                _meta_info,
            ) = await consume_sse_stream(resp, request_start_ns=request_start_ns)
    except Exception:
        return None

    total_ns = time.perf_counter_ns() - request_start_ns
    if ttft_ns is None or prompt_tokens is None or prompt_tokens <= 0:
        return None
    final_completion_tokens = completion_tokens if completion_tokens is not None else output_tokens
    if final_completion_tokens <= 0:
        return None

    ttft_s = ttft_ns / NS_PER_S
    total_s = total_ns / NS_PER_S
    # ``ping_rtt`` represents the irreducible round-trip; clamp to 0 if a
    # transient blip pushed the ping above the observed prefill so we
    # never report negative per-token rates.
    prefill_s = max(ttft_s - ping_rtt, 0.0)
    decode_s = max(total_s - ttft_s, 0.0)

    return {
        "ping_seconds": ping_rtt,
        "ttft_seconds": ttft_s,
        "total_seconds": total_s,
        "prefill_seconds": prefill_s,
        "decode_seconds": decode_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": final_completion_tokens,
        "prefill_rate_seconds_per_token": prefill_s / prompt_tokens,
        "decode_rate_seconds_per_token": decode_s / final_completion_tokens,
    }


def recommend_rates(samples: list[dict]) -> dict:
    """Median-of-rates recommendation, pooled across replicas.

    Returns ``prefill_rate`` in **ms/tok** as a diagnostic.  The
    3-weight GORGO model does not use this value for routing — the ES
    absorbs hardware speed into ``prefill_weight`` directly.  Used by
    ``proxy/calibrate.py`` for offline rate fitting.
    """
    if not samples:
        return {"prefill_rate": 0.0}
    prefill_sorted = sorted(s["prefill_rate_seconds_per_token"] for s in samples)
    return {
        "prefill_rate": prefill_sorted[len(prefill_sorted) // 2] * 1000.0,
    }


def summarize_samples(samples: list[dict]) -> dict:
    """Aggregate a list of :func:`measure_chat_completion` samples into the
    full stats block embedded in the calibrate JSON output."""
    pings = [s["ping_seconds"] for s in samples]
    prefill_rates = [s["prefill_rate_seconds_per_token"] for s in samples]
    decode_rates = [s["decode_rate_seconds_per_token"] for s in samples]
    prompt_tokens = [float(s["prompt_tokens"]) for s in samples]
    prefill_secs = [s["prefill_seconds"] for s in samples]
    completion_tokens = [float(s["completion_tokens"]) for s in samples]
    decode_secs = [s["decode_seconds"] for s in samples]
    return {
        "ping_seconds": compute_stats(pings),
        "prefill_rate_seconds_per_token": compute_stats(prefill_rates),
        "decode_rate_seconds_per_token": compute_stats(decode_rates),
        # Linear-regression cross-checks: slope is a different (and often
        # cleaner) estimator of per-token cost than median-of-ratios.
        # Short prompts get less weight here because they contribute
        # mostly to the intercept rather than the slope.
        "prefill_ols_fit": ols_fit(prompt_tokens, prefill_secs),
        "decode_ols_fit": ols_fit(completion_tokens, decode_secs),
    }
