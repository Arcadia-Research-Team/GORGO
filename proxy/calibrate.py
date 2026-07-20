"""Calibrate the GORGO policy physical rate from a single idle replica.

Produces an *initial estimate* for ``prefill_rate`` -- the sole
hardware-rate parameter in ``policy/gorgo.py::route_gorgo``::

    score(u) = rtt_weight     * rtt_ms(u)
             + prefill_weight * prefill_rate(u) * (uncached + queued_tokens)

``prefill_rate`` has units of **ms / token** in the scoring function
(matching the ms-scale RTT term).  The calibrator measures the raw
prefill rate in seconds-per-token and
:func:`proxy.measure.recommend_rates` converts to ms/tok.  (The decode
rate is still measured and reported as a diagnostic, but it is no
longer a routing parameter -- the load/contention term was removed.)

Per-sample procedure
--------------------

1. (Optional) ``POST /flush_cache`` so the next probe starts from a clean
   RadixAttention KV cache. Without this, consecutive prompts that share a
   user-level prefix would let the second one skip prefill, biasing
   ``prefill_rate`` downward.
2. Ping the replica ``K`` times back-to-back; take the median RTT
   (filters single-ping jitter without being too costly).
3. Send one *streaming* chat completion with ``max_tokens=N`` (large),
   using a prompt sampled from the production dataset (long realistic
   prompts make the rate-per-token estimate stable -- with short
   prompts the fixed overheads dominate).
4. Parse the SSE stream with chunk-arrival-precise timing (shared with
   ``proxy/workload.py``) so TTFT reflects the wire-arrival of the first
   ``delta.content`` event, not the moment our parser happened to finish a
   line.
5. Pull ``prompt_tokens`` / ``completion_tokens`` from the final ``usage``
   event (we set ``stream_options.include_usage=True``).
6. Decompose::

       TTFT          = ping_rtt + prefill_seconds
       total_seconds = TTFT     + decode_seconds

   so

       prefill_rate  ~= (TTFT - ping_rtt)       / prompt_tokens
       decode_rate   ~= (total - TTFT)           / completion_tokens

The script reports per-sample rates plus aggregate statistics (median,
mean, min, max, p50/p95/p99). The recommended initial values are the
*medians* (robust to outliers); a linear-regression fit ``y = a + b*x``
is also reported so you can sanity-check that the slope (= prefill_rate) is
consistent with the median-of-rates and that the intercept (= fixed
per-request overhead) is small.

Sequential measurements only. The replica is expected to be idle during
the run -- concurrent load would smear all three terms (ping, prefill,
decode) in ways that defeat the decomposition. To enforce this, the
calibrator hits a replica *directly* by URL, bypassing the proxy.

Modular building blocks live in :mod:`proxy.measure` and are designed to
be reused by the proxy itself for online hyperparameter tuning as live
traffic flows -- this script is a thin Modal entrypoint over them.

Usage::

    modal run proxy/calibrate.py --replica-url https://replica.modal.host \\
        --start-time 2026-04-01T12:00:00 --num-samples 32

If you don't know the replica URL, point at the proxy and the
calibrator will pick the first replica it sees (auto-discovery via
``GET /replicas`` -- the proxy itself is not in the request path)::

    modal run proxy/calibrate.py --proxy-url https://proxy.modal.host
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone

import modal

from app import app, bench_results_volume, completions_volume
from gorgo.measure import (
    flush_replica_cache,
    measure_chat_completion,
    ping_once,
    recommend_rates,
    summarize_samples,
)
from proxy.workload import (
    DEFAULT_MODEL,
    _build_row_source,
    _iter_bodies,
    _parse_iso,
)

REGION = os.getenv("REGION", "us-east")

image = (
    modal.Image.debian_slim()
    .pip_install("httpx[http2]", "pyarrow", "datasets>=3.0")
    .add_local_python_source("app", "proxy", "gorgo")
)


async def _resolve_replica_url(
    *,
    replica_url: str | None,
    proxy_url: str | None,
) -> str:
    """Validate the ``--replica-url`` / ``--proxy-url`` choice and return
    the replica URL the calibrator should bind directly to. ``proxy_url``
    is only used as a discovery shortcut: ``GET /replicas`` -> first entry.
    The proxy itself is never in the request path during calibration."""
    import httpx

    if replica_url is None and proxy_url is None:
        raise SystemExit("either --replica-url or --proxy-url is required")
    if replica_url is not None and proxy_url is not None:
        raise SystemExit("specify only one of --replica-url / --proxy-url")
    if replica_url is not None:
        return replica_url.rstrip("/")
    proxy_url = proxy_url.rstrip("/")
    async with httpx.AsyncClient(timeout=15.0) as bootstrap:
        r = await bootstrap.get(f"{proxy_url}/replicas")
        r.raise_for_status()
        replicas = r.json().get("replicas") or []
    if not replicas:
        raise SystemExit(f"proxy {proxy_url!r} has no replicas registered")
    chosen = replicas[0].rstrip("/")
    print(f"[calibrate] using replica {chosen} (auto-discovered via {proxy_url})")
    return chosen


def _select_long_prompts(
    *,
    source: str,
    preset: str | None,
    data_path: str | None,
    start_dt: datetime | None,
    end_dt: datetime | None,
    offset: int,
    model: str | None,
    decode_max_tokens: int,
    min_prompt_chars: int,
):
    """Yield chat-completion bodies from ``proxy.workload`` sources, with
    ``stream=True`` and a fixed ``max_tokens`` so each probe produces a
    clean (TTFT, total) pair, and short prompts filtered out client-side.

    Char-count is a deliberately cheap stand-in for "long enough" -- the
    calibrator image doesn't ship a tokenizer, and 2000 chars is roughly
    500 tokens of English which is plenty for stable rate estimates.
    """
    rows, resolved_path = _build_row_source(
        source=source,
        data_path=data_path,
        preset=preset,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    print(f"[calibrate] prompt source={source} path={resolved_path}")

    bodies = _iter_bodies(
        rows,
        offset=offset,
        num_requests=None,
        model_override=model,
        stream_override=True,
        max_tokens_override=decode_max_tokens,
    )
    for ts, body in bodies:
        msgs = body.get("messages") or []
        total_chars = sum(len(m.get("content") or "") if isinstance(m, dict) else 0 for m in msgs)
        if total_chars < min_prompt_chars:
            continue
        yield ts, body


@app.function(
    image=image,
    region=REGION,
    timeout=24 * 60 * 60,
    volumes={
        "/data": completions_volume,
        "/results": bench_results_volume,
    },
)
def calibrate(
    *,
    replica_url: str | None = None,
    proxy_url: str | None = None,
    source: str = "prod",
    preset: str | None = None,
    data_path: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    offset: int = 0,
    num_samples: int = 32,
    warmup_samples: int = 4,
    decode_max_tokens: int = 256,
    min_prompt_chars: int = 2000,
    pings_per_sample: int = 3,
    flush_between_samples: bool = True,
    model: str | None = DEFAULT_MODEL,
    output_path: str | None = None,
) -> dict:
    """Estimate ``prefill_rate`` from a single replica (the decode rate
    is also measured as a diagnostic but is not a routing parameter).

    Args:
        replica_url: Direct URL of the SGLang replica to probe (bypasses
            the proxy). Mutually exclusive with ``proxy_url``.
        proxy_url: If ``replica_url`` is not given, the calibrator hits
            ``GET /replicas`` on this proxy and uses the first one. The
            proxy itself is never in the calibration request path.
        source / preset / data_path: Prompt sourcing -- see
            ``proxy/workload.py``. Defaults match workload (production from
            ``/data``).
        start_time / end_time / offset: Dataset slice for prompt sourcing.
            Honored only for sources that support time-range filtering
            (``prod``); ignored for HF sources.
        num_samples: Total number of measurement requests (after warmup).
        warmup_samples: Discard this many initial samples; first requests
            after a cold cache often show prefill artifacts.
        decode_max_tokens: ``max_tokens`` requested per probe. Larger
            values give a better decode-rate estimate but cost more
            wall-clock time. 256 is usually plenty.
        min_prompt_chars: Reject prompts shorter than this many chars.
            Heuristic stand-in for "long prompts" without tokenizing
            client-side; 2000 chars ~= 500 tokens.
        pings_per_sample: ``GET /v1/models`` probes per sample, median
            taken. 3 is enough to filter outliers without being slow.
        flush_between_samples: ``POST /flush_cache`` before each probe so
            consecutive prompts that share a prefix can't bias
            ``prefill_rate`` downward via cached KV. Disable only when
            measuring intentional prefix-cache speedups.
        model: Override the request body's ``model`` field. ``None``
            leaves the source-row value in place.
        output_path: Where to save the calibration JSON inside the
            ``GORGO-bench-results`` volume. ``None`` (default) ->
            ``calibrate_<UTC-timestamp>.json``.

    Returns:
        Dict with raw per-sample data + aggregate stats + recommended
        initial hyperparameter values, plus the resolved
        ``output_path``.
    """
    import httpx

    start_dt = _parse_iso(start_time)
    end_dt = _parse_iso(end_time)

    completions_volume.reload()

    run_started_at = datetime.now(timezone.utc)

    async def amain() -> dict:
        chosen_replica = await _resolve_replica_url(replica_url=replica_url, proxy_url=proxy_url)
        print(
            f"[calibrate] replica={chosen_replica} samples={num_samples} "
            f"warmup={warmup_samples} max_tokens={decode_max_tokens} "
            f"min_prompt_chars={min_prompt_chars} "
            f"flush_between_samples={flush_between_samples}"
        )

        # ``read=None`` so a long generation can't trip the timeout;
        # connect/write tight so dead replicas fail fast.
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        prompt_iter = _select_long_prompts(
            source=source,
            preset=preset,
            data_path=data_path,
            start_dt=start_dt,
            end_dt=end_dt,
            offset=offset,
            model=model,
            decode_max_tokens=decode_max_tokens,
            min_prompt_chars=min_prompt_chars,
        )

        samples: list[dict] = []
        warmup_seen = 0
        attempted = 0
        flush_failures = 0

        async with httpx.AsyncClient(
            base_url=chosen_replica, http2=True, timeout=timeout
        ) as client:
            from time import perf_counter

            t_run_start = perf_counter()
            for _ts, body in prompt_iter:
                if flush_between_samples:
                    if not await flush_replica_cache(client):
                        flush_failures += 1

                try:
                    ping_rtt = await ping_once(client, n=pings_per_sample)
                except httpx.HTTPError as e:
                    print(f"[calibrate]   ping failed: {e}; skipping")
                    continue

                attempted += 1
                sample = await measure_chat_completion(client, body, ping_rtt=ping_rtt)
                if sample is None:
                    continue

                if warmup_seen < warmup_samples:
                    warmup_seen += 1
                    print(
                        f"[calibrate]   warmup {warmup_seen}/{warmup_samples}: "
                        f"prompt_tokens={sample['prompt_tokens']} "
                        f"prefill={sample['prefill_seconds']:.3f}s "
                        f"decode={sample['decode_seconds']:.3f}s"
                    )
                    continue

                samples.append(sample)
                print(
                    f"[calibrate]   sample {len(samples)}/{num_samples}: "
                    f"pt={sample['prompt_tokens']} ct={sample['completion_tokens']} "
                    f"ping={sample['ping_seconds'] * 1000:.1f}ms "
                    f"prefill={sample['prefill_seconds']:.3f}s "
                    f"decode={sample['decode_seconds']:.3f}s "
                    f"prefill_rate={sample['prefill_rate_seconds_per_token'] * 1000:.3f}ms/tok "
                    f"decode_rate={sample['decode_rate_seconds_per_token'] * 1000:.3f}ms/tok",
                    flush=True,
                )

                if len(samples) >= num_samples:
                    break

            elapsed = perf_counter() - t_run_start

        print(
            f"[calibrate] collected {len(samples)} samples in {elapsed:.1f}s "
            f"(attempted {attempted}, flush_failures={flush_failures})"
        )
        return {
            "replica_url": chosen_replica,
            "samples": samples,
            "elapsed_seconds": elapsed,
            "attempted": attempted,
            "flush_failures": flush_failures,
        }

    inner = asyncio.run(amain())
    samples: list[dict] = inner["samples"]
    if not samples:
        raise SystemExit("no successful calibration samples collected")

    stats = summarize_samples(samples)
    stats["elapsed_seconds"] = inner["elapsed_seconds"]
    stats["attempted"] = inner["attempted"]
    stats["flush_failures"] = inner["flush_failures"]

    recommended = recommend_rates(samples)

    print()
    ping = stats["ping_seconds"]
    pre = stats["prefill_rate_seconds_per_token"]
    dec = stats["decode_rate_seconds_per_token"]
    pre_fit = stats["prefill_ols_fit"]
    dec_fit = stats["decode_ols_fit"]
    print(
        f"[calibrate] ping (s)            median={ping['median']:.4f} "
        f"p95={ping['p95']:.4f} (n={ping['n']})"
    )
    print(
        f"[calibrate] prefill rate (s/tok)  median={pre['median']:.6f} "
        f"mean={pre['mean']:.6f} p95={pre['p95']:.6f} "
        f"(slope={pre_fit['b']:.6f} intercept={pre_fit['a']:.4f}s "
        f"r2={pre_fit['r2']:.3f})"
    )
    print(
        f"[calibrate] decode rate  (s/tok)  median={dec['median']:.6f} "
        f"mean={dec['mean']:.6f} p95={dec['p95']:.6f} "
        f"(slope={dec_fit['b']:.6f} intercept={dec_fit['a']:.4f}s "
        f"r2={dec_fit['r2']:.3f})"
    )
    print(f"[calibrate] recommended initial rates (ms/tok): {recommended}")

    if output_path is None:
        ts = run_started_at.strftime("%Y%m%d_%H%M%S")
        resolved_output_path = f"/results/calibrate_{ts}.json"
    else:
        resolved_output_path = (
            output_path if os.path.isabs(output_path) else os.path.join("/results", output_path)
        )
    os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)

    doc = {
        "config": {
            "replica_url": inner["replica_url"],
            "proxy_url": proxy_url,
            "model": model,
            "source": source,
            "preset": preset,
            "data_path": data_path,
            "num_samples": num_samples,
            "warmup_samples": warmup_samples,
            "decode_max_tokens": decode_max_tokens,
            "min_prompt_chars": min_prompt_chars,
            "pings_per_sample": pings_per_sample,
            "flush_between_samples": flush_between_samples,
            "data_slice": {
                "start_time": start_time,
                "end_time": end_time,
                "offset": offset,
            },
            "region": REGION,
            "run_started_at": run_started_at.isoformat().replace("+00:00", "Z"),
        },
        "samples": samples,
        "stats": stats,
        "recommended_rates": recommended,
        "output_path": resolved_output_path,
    }
    with open(resolved_output_path, "w") as f:
        json.dump(doc, f)
    bench_results_volume.commit()
    print(f"[calibrate] saved results to volume GORGO-bench-results at {resolved_output_path}")

    return doc


@app.local_entrypoint()
def main(
    replica_url: str = "",
    proxy_url: str = "",
    source: str = "prod",
    preset: str = "",
    data_path: str = "",
    start_time: str = "",
    end_time: str = "",
    offset: int = 0,
    num_samples: int = 32,
    warmup_samples: int = 4,
    decode_max_tokens: int = 256,
    min_prompt_chars: int = 2000,
    pings_per_sample: int = 3,
    flush_between_samples: bool = True,
    model: str = DEFAULT_MODEL,
    output_path: str = "",
):
    """CLI wrapper for ``calibrate``.

    Pass exactly one of ``--replica-url`` / ``--proxy-url``. The
    ``--proxy-url`` form only uses the proxy as a service-discovery
    shortcut (``GET /replicas`` -> first entry); calibration requests go
    direct to the replica. Sentinel mapping matches the rest of the
    proxy/ scripts: empty string for str fields => ``None``.
    """
    calibrate.remote(
        replica_url=replica_url or None,
        proxy_url=proxy_url or None,
        source=source,
        preset=preset or None,
        data_path=data_path or None,
        start_time=start_time or None,
        end_time=end_time or None,
        offset=offset,
        num_samples=num_samples,
        warmup_samples=warmup_samples,
        decode_max_tokens=decode_max_tokens,
        min_prompt_chars=min_prompt_chars,
        pings_per_sample=pings_per_sample,
        flush_between_samples=flush_between_samples,
        model=model or None,
        output_path=output_path or None,
    )
