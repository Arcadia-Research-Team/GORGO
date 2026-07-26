"""Smoke-test the per-request engine timing log against a live replica.

Confirms, on real hardware, the three facts the TTFT-decomposition validation
depends on -- before any money goes into a multi-region fleet:

  1. A proxy-supplied ``rid`` on ``/v1/chat/completions`` becomes SGLang's
     internal request id, so engine records join to proxy trace rows.
  2. ``event: "request.finished"`` lines carry the scheduler timings
     (``queue_time``, ``forward_entry_time``, ``prefill_finished_time``) for
     *streaming chat* requests -- not just the native ``/generate`` path.
  3. ``queue_time`` is genuinely nonzero under concurrency. (An idle-replica
     probe reads 0 because there is no queueing to measure, which is what made
     an earlier look conclude these fields were unavailable.)

It also implements, in miniature, the join the offline analysis performs:

    Q = queue_time                                  (engine-local difference)
    P = prefill_finished_time - forward_entry_time   (engine-local difference)
    residual = TTFT_client - (Q + P)                 (network + API-server overhead)

Q and P are differences between timestamps taken on one host, so neither needs
cross-region clock alignment; the residual is what the RTT term must explain.

Prerequisite -- deploy an engine with the log enabled:

    REQUEST_TIMING_LOG=1 REGION=us-ashburn-1 GPU_TYPE=L40S N_GPUS=2 \
        modal deploy engine/modal_sglang.py

Then (``--registry-key`` defaults to the region, matching how the engine
registers itself in the ``GORGO-replicas`` Dict):

    modal run scripts/verify_engine_timing_log.py::main \
        --registry-key us-ashburn-1 --concurrency 8 --prompt-tokens 4000
"""

from __future__ import annotations

import modal

from app import app, bench_results_volume, replicas
from engine.modal_sglang import HF_REPO_ID, REQUEST_TIMING_LOG_ROOT

VERIFY_IMAGE = modal.Image.debian_slim().pip_install("httpx[http2]").add_local_python_source(
    "app", "engine", "scripts"
)

# Timing fields we require for the decomposition, and the ones that are
# expected to be absent in unified (non-disaggregated) mode.
REQUIRED_META_FIELDS = ("queue_time", "forward_entry_time", "prefill_finished_time")
OPTIONAL_META_FIELDS = (
    "request_received_ts",
    "api_server_dispatch_finish_ts",
    "response_sent_to_client_ts",
    "request_finished_ts",
    "inference_time",
    "cached_tokens",
    "prompt_tokens",
    "completion_tokens",
)
# Populated only when prefill_run_batch_start_time is set, which unified mode
# does not do -- absence here is expected and not a failure.
KNOWN_ABSENT_FIELDS = ("prefill_waiting_latency", "prefill_launch_latency")


@app.function(
    image=VERIFY_IMAGE,
    volumes={"/results": bench_results_volume},
    timeout=1800,
)
def verify(
    base_url: str,
    registry_key: str,
    concurrency: int,
    prompt_tokens: int,
    max_tokens: int,
    settle_seconds: float,
) -> dict:
    import asyncio
    import glob
    import json
    import os
    import time
    import uuid

    import httpx

    NS_PER_S = 1_000_000_000
    run_tag = uuid.uuid4().hex[:8]

    # A long, *unique-per-request* prompt: shared prefixes would let later
    # requests skip prefill and collapse the P we are trying to observe.
    def make_prompt(i: int) -> str:
        filler = f"seq{run_tag}n{i} " + " ".join(
            f"w{i}x{j}" for j in range(max(1, prompt_tokens))
        )
        return filler

    async def one(client: httpx.AsyncClient, i: int) -> dict:
        rid = f"verify-{run_tag}-{i}"
        body = {
            "model": HF_REPO_ID,
            "messages": [{"role": "user", "content": make_prompt(i)}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            # The join key under test: SGLang keeps a caller-supplied rid
            # (srt/entrypoints/openai/protocol.py) instead of generating one.
            "rid": rid,
        }
        started = time.perf_counter_ns()
        ttft_ns = None
        status = None
        error = None
        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=body,
                headers={"accept-encoding": "identity"},
                timeout=httpx.Timeout(connect=15.0, read=None, write=60.0, pool=15.0),
            ) as resp:
                status = resp.status_code
                if resp.status_code != 200:
                    error = (await resp.aread()).decode("utf-8", "replace")[:200]
                else:
                    async for raw in resp.aiter_lines():
                        line = raw.strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            obj = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if choices and (choices[0].get("delta") or {}).get("content"):
                            if ttft_ns is None:
                                ttft_ns = time.perf_counter_ns() - started
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        return {
            "rid": rid,
            "status": status,
            "ttft_seconds": (ttft_ns / NS_PER_S) if ttft_ns is not None else None,
            "error": error,
        }

    async def drive() -> list[dict]:
        async with httpx.AsyncClient(base_url=base_url, http2=True) as client:
            # All at once, so requests genuinely contend for the scheduler and
            # queue_time has something to report.
            return await asyncio.gather(*(one(client, i) for i in range(concurrency)))

    print(f"[verify] firing {concurrency} concurrent requests at {base_url}", flush=True)
    client_results = asyncio.run(drive())
    ok = [r for r in client_results if r["ttft_seconds"] is not None]
    print(f"[verify] {len(ok)}/{concurrency} got a first token", flush=True)
    for r in client_results:
        if r["error"]:
            print(f"[verify]   {r['rid']}: status={r['status']} error={r['error']}", flush=True)

    # Let the engine's periodic committer publish the log to the volume.
    print(f"[verify] waiting {settle_seconds}s for the engine to commit its log", flush=True)
    time.sleep(settle_seconds)

    safe_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in registry_key)
    log_dir = os.path.join(REQUEST_TIMING_LOG_ROOT, safe_key)
    bench_results_volume.reload()

    log_files = sorted(glob.glob(os.path.join(log_dir, "*.log*")))
    print(f"[verify] log dir {log_dir}: {len(log_files)} file(s) {log_files}", flush=True)

    # rid -> the request.finished record's meta_info
    finished: dict[str, dict] = {}
    events_seen: dict[str, int] = {}
    parse_errors = 0
    for path in log_files:
        try:
            with open(path, "r", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        continue
                    event = rec.get("event", "?")
                    events_seen[event] = events_seen.get(event, 0) + 1
                    if event != "request.finished":
                        continue
                    rid = rec.get("rid")
                    meta = ((rec.get("out") or {}).get("meta_info")) or {}
                    if isinstance(rid, str):
                        finished[rid] = meta
        except OSError as e:
            print(f"[verify] could not read {path}: {e}", flush=True)

    print(f"[verify] events in log: {events_seen} (parse errors: {parse_errors})", flush=True)

    rows = []
    for r in client_results:
        meta = finished.get(r["rid"])
        row = {**r, "matched": meta is not None}
        if meta is not None:
            q = meta.get("queue_time")
            fwd = meta.get("forward_entry_time")
            pf = meta.get("prefill_finished_time")
            recv = meta.get("request_received_ts")
            row["queue_time"] = q
            row["forward_entry_time"] = fwd
            row["prefill_finished_time"] = pf
            row["prompt_tokens"] = meta.get("prompt_tokens")
            if isinstance(fwd, (int, float)) and isinstance(pf, (int, float)):
                row["prefill_seconds"] = pf - fwd
            # Tokenize + dispatch: everything between the API server accepting
            # the request and the scheduler admitting it, minus the queue wait.
            # All three timestamps are engine-local, so this needs no clock
            # alignment -- and it is a stage the 3-term cost model does not
            # represent at all, so quantifying it bounds epsilon.
            if (
                isinstance(recv, (int, float))
                and isinstance(fwd, (int, float))
                and isinstance(q, (int, float))
            ):
                row["ingress_seconds"] = (fwd - recv) - q
            if (
                r["ttft_seconds"] is not None
                and isinstance(q, (int, float))
                and row.get("prefill_seconds") is not None
            ):
                # What the RTT term has to explain: client-observed TTFT less
                # every engine-local stage we can measure.
                row["residual_seconds"] = (
                    r["ttft_seconds"] - q - row["prefill_seconds"] - (row.get("ingress_seconds") or 0.0)
                )
            row["present_optional"] = sorted(k for k in OPTIONAL_META_FIELDS if k in meta)
            row["meta_keys"] = sorted(meta.keys())
        rows.append(row)

    matched = [r for r in rows if r["matched"]]
    with_all = [r for r in matched if all(r.get(f) is not None for f in REQUIRED_META_FIELDS)]
    nonzero_queue = [
        r for r in with_all if isinstance(r.get("queue_time"), (int, float)) and r["queue_time"] > 0
    ]

    print("", flush=True)
    print("=" * 78, flush=True)
    print("PER-REQUEST JOIN (client TTFT vs engine components)", flush=True)
    print("=" * 78, flush=True)
    print(
        f"{'rid':24} {'ptok':>6} {'TTFT':>8} {'ingress':>8} {'Q':>8} {'P':>8} {'resid':>8}",
        flush=True,
    )
    for r in rows:
        def fmt(v):
            return f"{v:8.3f}" if isinstance(v, (int, float)) else f"{'-':>8}"

        ptok = r.get("prompt_tokens")
        print(
            f"{r['rid'][:24]:24} {(ptok if isinstance(ptok, int) else '-'):>6} "
            f"{fmt(r.get('ttft_seconds'))} {fmt(r.get('ingress_seconds'))} "
            f"{fmt(r.get('queue_time'))} {fmt(r.get('prefill_seconds'))} "
            f"{fmt(r.get('residual_seconds'))}",
            flush=True,
        )

    verdict = {
        "requests_fired": concurrency,
        "requests_with_first_token": len(ok),
        "rid_join_matched": len(matched),
        "records_with_all_required_fields": len(with_all),
        "records_with_nonzero_queue_time": len(nonzero_queue),
        "log_files": log_files,
        "events_seen": events_seen,
        "known_absent_fields_present": sorted(
            {f for r in matched for f in KNOWN_ABSENT_FIELDS if r.get(f) is not None}
        ),
        "sample_meta_keys": (matched[0]["meta_keys"] if matched else []),
    }

    print("", flush=True)
    print("=" * 78, flush=True)
    print("VERDICT", flush=True)
    print("=" * 78, flush=True)
    for k, v in verdict.items():
        print(f"  {k}: {v}", flush=True)

    checks = {
        "rid join works": len(matched) == len(ok) and len(ok) > 0,
        "required timing fields present": len(with_all) == len(matched) and len(matched) > 0,
        "queue_time nonzero under load": len(nonzero_queue) > 0,
    }
    print("", flush=True)
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}", flush=True)
    verdict["checks"] = checks
    verdict["rows"] = rows
    return verdict


@app.local_entrypoint()
def main(
    registry_key: str = "us-ashburn-1",
    base_url: str = "",
    concurrency: int = 8,
    prompt_tokens: int = 4000,
    max_tokens: int = 16,
    settle_seconds: float = 25.0,
) -> None:
    """Resolve the replica URL from the registry (unless given) and verify."""
    url = base_url.strip().rstrip("/")
    if not url:
        url = (replicas.get(registry_key) or "").strip().rstrip("/")
        if not url:
            raise SystemExit(
                f"no URL registered for {registry_key!r} in GORGO-replicas; "
                "deploy the engine first (REQUEST_TIMING_LOG=1) or pass --base-url"
            )
        print(f"[verify] resolved {registry_key} -> {url}")
    verify.remote(
        base_url=url,
        registry_key=registry_key,
        concurrency=concurrency,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        settle_seconds=settle_seconds,
    )
