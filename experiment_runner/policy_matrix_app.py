"""Single-app policy matrix experiment controller.

This app avoids shelling out to ``modal run`` for engines/proxies. Region/GPU
settings are decorator-time in this Modal SDK, so we expose one engine function
per benchmark region and select the right one in the controller.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import modal

from app import (
    ENVIRONMENT_NAME,
    app,
    bench_results_volume,
    completions_volume,
    hf_datasets_volume,
    lmsys_chat_1m_volume,
    proxies,
    replicas,
)
from engine.modal_sglang import (
    CONTEXT_LENGTH,
    HF_REPO_ID,
    MODEL_REVISION,
    N_GPUS,
    PORT,
    SCALEDOWN_WINDOW_SECONDS,
    WAIT_READY_TIMEOUT,
    request_timing_log_args,
    sglang_image,
    start_timing_log_committer,
    stop_timing_log_committer,
    wait_ready,
)


PROXY_IMAGE = (
    modal.Image.debian_slim()
    .pip_install("httpx[http2]", "uvicorn", "transformers", "jinja2", "pyarrow", "datasets>=3.0")
    .add_local_python_source("app", "engine", "proxy", "gorgo")
)


def _serve_model(registry_key: str, tp_size: int | None = None) -> None:
    """Launch SGLang on this container.

    ``tp_size`` overrides the imported ``N_GPUS`` for the ``--tp`` arg.
    Set explicitly per-engine when the @app.function uses a different
    GPU count than the engine module's default (e.g. L40S:2 fleets).
    """
    os.environ["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "1"
    tp = tp_size if tp_size is not None else N_GPUS
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        HF_REPO_ID,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        HF_REPO_ID,
        "--host",
        "0.0.0.0",
        "--port",
        f"{PORT}",
        "--tp",
        f"{tp}",
        "--cuda-graph-max-bs",
        f"{10 * 2}",
        "--enable-metrics",
        "--decode-log-interval",
        "100",
        "--mem-fraction",
        "0.8",
        "--context-length",
        f"{CONTEXT_LENGTH}",
    ]
    # Per-request engine timing log (off unless REQUEST_TIMING_LOG is set in the
    # engine image env). Measurement only -- routing still reads just /metrics
    # and the proxy RTT probe. See engine/modal_sglang.py::REQUEST_TIMING_LOG.
    timing_args, timing_log_dir = request_timing_log_args(registry_key)
    cmd += timing_args
    with modal.forward(PORT) as tunnel:
        print(f"tunnel.url        = {tunnel.url}")
        print(f"tunnel.tls_socket = {tunnel.tls_socket}")
        process = subprocess.Popen(cmd)
        stop_committing, committer = start_timing_log_committer(timing_log_dir)
        try:
            wait_ready(process)
            replicas[registry_key] = tunnel.url
            print(replicas[registry_key])
            process.wait()
        finally:
            stop_timing_log_committer(stop_committing, committer)
            if replicas.get(registry_key) == tunnel.url:
                replicas[registry_key] = ""
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


# Shared by every H100 engine below. ``/results`` is the destination for the
# per-request engine timing log when REQUEST_TIMING_LOG is on; inert otherwise.
_H100_HF_VOLUME = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "Qwen3.5-35B-A3B-FP8-huggingface-cache",
        create_if_missing=True,
        environment_name=ENVIRONMENT_NAME,
    ),
    "/results": bench_results_volume,
}


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="CANADA-2",
    gpu="H100",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=8,
    retries=0,
    volumes=_H100_HF_VOLUME,
)
def engine_canada(registry_key: str) -> None:
    _serve_model(registry_key)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="sines-2",
    gpu="H100",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=8,
    retries=0,
    volumes=_H100_HF_VOLUME,
)
def engine_sines(registry_key: str) -> None:
    _serve_model(registry_key)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="us-west4",
    gpu="H100",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=8,
    retries=0,
    volumes=_H100_HF_VOLUME,
)
def engine_us_west4(registry_key: str) -> None:
    _serve_model(registry_key)


# ---- H100:1 engines (AZR regions for c=64 experiments) ----


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="centralus",
    gpu="H100",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=8,
    retries=0,
    volumes=_H100_HF_VOLUME,
)
def engine_centralus(registry_key: str) -> None:
    _serve_model(registry_key)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="northeurope",
    gpu="H100",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=8,
    retries=0,
    volumes=_H100_HF_VOLUME,
)
def engine_northeurope(registry_key: str) -> None:
    _serve_model(registry_key)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="malaysiawest",
    gpu="H100",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=8,
    retries=0,
    volumes=_H100_HF_VOLUME,
)
def engine_malaysiawest(registry_key: str) -> None:
    _serve_model(registry_key)


# ---- L40S:2 engines (tp=2; 35B FP8 weights ~35GB so 1xL40S is too tight) ----

_L40S_HF_VOLUME = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "Qwen3.5-35B-A3B-FP8-huggingface-cache",
        create_if_missing=True,
        environment_name=ENVIRONMENT_NAME,
    ),
    # Destination for the per-request engine timing log when
    # REQUEST_TIMING_LOG is on; inert otherwise.
    "/results": bench_results_volume,
}


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="ap-seoul-1",
    gpu="L40S:2",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=8,
    retries=0,
    volumes=_L40S_HF_VOLUME,
)
def engine_ap_seoul(registry_key: str) -> None:
    _serve_model(registry_key, tp_size=2)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="eu-frankfurt-1",
    gpu="L40S:2",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=8,
    retries=0,
    volumes=_L40S_HF_VOLUME,
)
def engine_eu_frankfurt(registry_key: str) -> None:
    _serve_model(registry_key, tp_size=2)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="us-ashburn-1",
    gpu="L40S:2",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=8,
    retries=0,
    volumes=_L40S_HF_VOLUME,
)
def engine_us_ashburn(registry_key: str) -> None:
    _serve_model(registry_key, tp_size=2)


# ---- L40S:2 engines pinned to coarse geographies ----
#
# Same GPU tier and tp as the three engines above, but constrained only to a
# continent rather than a single datacenter. L40S:2 in ap-seoul-1 /
# eu-frankfurt-1 / us-ashburn-1 is frequently unschedulable, and Modal's own
# scheduler hint is to relax the region requirement. Cross-continent placement
# is what the network term needs, and that survives: the exact datacenter does
# not matter for the cost model, only that the three replicas sit far apart and
# at different distances from the us-east proxy.
#
# Trade-off vs the pinned variants: placement (and therefore the precise RTT
# spread) is no longer reproducible run to run, so report the measured
# per-replica RTT rather than quoting the paper's 26-466 ms range.


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="ap",
    gpu="L40S:2",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=8,
    retries=0,
    volumes=_L40S_HF_VOLUME,
)
def engine_ap(registry_key: str) -> None:
    _serve_model(registry_key, tp_size=2)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="eu",
    gpu="L40S:2",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=8,
    retries=0,
    volumes=_L40S_HF_VOLUME,
)
def engine_eu(registry_key: str) -> None:
    _serve_model(registry_key, tp_size=2)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region="us",
    gpu="L40S:2",
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=8,
    retries=0,
    volumes=_L40S_HF_VOLUME,
)
def engine_us(registry_key: str) -> None:
    _serve_model(registry_key, tp_size=2)


@app.function(
    image=PROXY_IMAGE,
    region="us-east",
    timeout=24 * 60 * 60,
    max_containers=8,
    min_containers=1,
    retries=0,
    volumes={
        "/data": completions_volume,
        "/lmsys": lmsys_chat_1m_volume,
        "/datasets": hf_datasets_volume,
        "/results": bench_results_volume,
    },
)
def proxy_runner(registry_key: str) -> None:
    from proxy.modal_proxy import proxy

    proxy.local(registry_key=registry_key)


ENGINE_BY_REGION = {
    "CANADA-2": engine_canada,
    "sines-2": engine_sines,
    "us-west4": engine_us_west4,
    "ap-seoul-1": engine_ap_seoul,
    "eu-frankfurt-1": engine_eu_frankfurt,
    "us-ashburn-1": engine_us_ashburn,
    "centralus": engine_centralus,
    "northeurope": engine_northeurope,
    "malaysiawest": engine_malaysiawest,
    # Coarse geographies: same L40S:2 tier, far better schedulability.
    "ap": engine_ap,
    "eu": engine_eu,
    "us": engine_us,
}

# ---------- Fleet GPU configuration ----------
# Change these when switching GPU tiers for the entire fleet.
# All regions in the experiment use the same GPU type.
FLEET_GPU = "L40S:2"
FLEET_TP_SIZE = 2

GPU_BY_REGION = {
    "CANADA-2": "H100",
    "sines-2": "H100",
    "us-west4": "H100",
    "ap-seoul-1": FLEET_GPU,
    "eu-frankfurt-1": FLEET_GPU,
    "us-ashburn-1": FLEET_GPU,
    "centralus": "H100",
    "northeurope": "H100",
    "malaysiawest": "H100",
    "ap": FLEET_GPU,
    "eu": FLEET_GPU,
    "us": FLEET_GPU,
}

TERMINAL_WORKLOAD_STATUS = {"succeeded", "failed", "cancelled"}


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s).strip("_")


def _start_at_wall_time(spec: dict) -> str:
    explicit = spec.get("start_at_wall_time")
    if explicit:
        return explicit
    delay = float(spec.get("start_delay_seconds", 30.0))
    return (
        (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat().replace("+00:00", "Z")
    )


def _with_bodies_path(path: str) -> str:
    """Resolve the body-included variant of a trace path.

    Legacy sweep traces stored body-free and body-included variants in
    sibling directories (``<dir>/foo.jsonl`` + ``<dir>/with_bodies/foo.jsonl``).
    Newer bench traces are stored flat (no ``with_bodies/`` nesting) with
    bodies already included. This function handles both layouts:
      * path already under ``with_bodies/`` -> return as-is
      * path ends with ``.jsonl`` and is NOT under ``with_bodies/`` ->
        return as-is (assumed to be a flat bench trace with bodies inline)
      * otherwise -> inject ``with_bodies/`` (legacy compat)
    """
    p = Path(path)
    if p.parent.name == "with_bodies":
        return path
    if p.suffix == ".jsonl":
        return path
    return str(p.parent / "with_bodies" / p.name)


def _trace_stem(path: str) -> str:
    return Path(path).stem


def _unique_output_dir(base: str, experiment_id: str) -> str:
    """Return a collision-resistant result directory under ``base``.

    If ``base`` already appears to include the experiment id, still append a
    UTC timestamp so reruns do not overwrite previous manifests. This keeps
    volume layout predictable while making every controller invocation unique.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path(base)
    return str(root / f"{experiment_id}_{ts}")


def _label(policy: dict) -> str:
    return policy.get("label") or policy["name"]


def _homogeneity_config(spec: dict) -> dict:
    """Resolve the optional pre-flight replica homogeneity check config.

    Disabled by default to preserve existing behavior. When enabled, the
    controller hits each replica directly (bypassing the proxy/policy)
    with N small streaming chat-completions and records per-replica
    TTFT. This catches the Modal-cold-start variance that was the
    confound behind the spurious least-request advantage in
    ``moon_neurips_main_000_quick200`` -- one of three shared backends
    happened to be ~10x faster than the others, and the policy that
    accidentally herded onto it looked like a winner.

    Knobs:
      ``enabled``: master switch (default ``False``).
      ``warmup_requests_per_replica``: per-replica request count
        (default ``16``). First ``warmup_requests`` are discarded as
        warm-up; the remainder are kept for stats. Defaults sized so
        the post-warmup tail (12 samples by default) is large enough
        for a stable P50 -- earlier defaults of 6 total / 4 measured
        gave wide P50 confidence intervals and a meaningless P95.
      ``warmup_requests``: how many of the first per-replica requests
        to discard (default ``4``). Must be < ``warmup_requests_per_replica``.
        The engine's own ``wait_ready`` already issues one warmup chat
        before declaring ready, but the *first* probe here pays
        TCP/TLS handshake on the controller -> tunnel route, and the
        first batch at a given prompt shape pays one-time CUDA-graph
        capture; a discard of 4 covers both comfortably.
      ``max_tokens``: cap on output tokens per probe (default ``16``;
        small to keep the check fast since we only care about TTFT).
      ``prompt_size_tokens``: target prompt size in tokens for each
        probe (default ``0`` = tiny ``"ping"`` prompt, RTT-dominated).
        Set to a value close to your workload's typical input size
        (e.g. ``4000`` or ``8000``) to also catch *prefill-shape*
        heterogeneity -- a replica that's CUDA-graph-warm for tiny
        prompts but cold for large ones will pass a ``"ping"`` probe
        and still skew the workload. The probe content is filler
        text padded to roughly this size, with a unique per-probe
        counter prefix so SGLang's prefix cache misses on every probe
        (worst-case prefill, which is the heterogeneity signal we want).
      ``max_ttft_ratio``: per-pool max-P50 / min-P50 TTFT ratio above
        which ``on_violation`` fires. ``0`` disables the gate (still
        records stats in the manifest).
      ``on_violation``: ``"warn"`` (default; logs and continues) or
        ``"abort"`` (raises ``RuntimeError``). Use ``"abort"`` when
        you want a clean fail-fast for paper-grade comparisons. On
        abort the controller cancels the spawned engine + proxy
        FunctionCalls so GPUs are freed immediately rather than
        idling until each engine's ``scaledown_window`` expires.
      ``request_timeout_seconds``: per-probe upstream timeout.
    """
    cfg = dict(spec.get("replica_homogeneity_check") or {})
    out = {
        "enabled": bool(cfg.get("enabled", False)),
        "warmup_requests_per_replica": int(cfg.get("warmup_requests_per_replica", 16)),
        "warmup_requests": int(cfg.get("warmup_requests", 4)),
        "max_tokens": int(cfg.get("max_tokens", 16)),
        "prompt_size_tokens": int(cfg.get("prompt_size_tokens", 0)),
        "max_ttft_ratio": float(cfg.get("max_ttft_ratio", 0.0)),
        "on_violation": str(cfg.get("on_violation", "warn")).lower(),
        "request_timeout_seconds": float(cfg.get("request_timeout_seconds", 60.0)),
    }
    if out["on_violation"] not in ("warn", "abort"):
        raise ValueError(
            f"replica_homogeneity_check.on_violation must be 'warn' or 'abort', "
            f"got {out['on_violation']!r}"
        )
    if out["warmup_requests"] >= out["warmup_requests_per_replica"]:
        raise ValueError(
            "replica_homogeneity_check.warmup_requests must be strictly less than "
            "warmup_requests_per_replica (otherwise no measurement requests remain)"
        )
    if out["prompt_size_tokens"] < 0:
        raise ValueError(
            f"replica_homogeneity_check.prompt_size_tokens must be >= 0, "
            f"got {out['prompt_size_tokens']}"
        )
    return out


SUPPORTED_SCHEMA_VERSIONS = {"2.0"}


def _validate_spec(spec: dict) -> None:
    """Validate the experiment spec schema version and required fields.

    Raises ``ValueError`` on an unrecognized schema version so old code
    never silently misinterprets a newer spec format.
    """
    version = spec.get("schema_version")
    if version is None:
        return
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported spec schema_version={version!r}; "
            f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )


def _capture_environment(base_spec: dict, sweep_manifest: dict) -> dict:
    """Snapshot the runtime environment for reproducibility.

    Called at the start of the experiment. All fields are best-effort;
    failures (e.g. not in a git repo) produce ``None`` rather than
    crashing the experiment.
    """

    def _git(cmd: list[str]) -> str | None:
        try:
            return (
                subprocess.check_output(["git"] + cmd, stderr=subprocess.DEVNULL, timeout=10)
                .decode()
                .strip()
            )
        except Exception:
            return None

    fleet_regions = base_spec.get("fleet_regions") or []

    return {
        "gorgo_commit": _git(["rev-parse", "HEAD"]),
        "gorgo_dirty": bool(_git(["status", "--porcelain"])),
        "gorgo_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "python_version": sys.version,
        "modal_version": modal.__version__,
        "sglang_image_tag": sglang_image.tag if hasattr(sglang_image, "tag") else None,
        "model_repo_id": HF_REPO_ID,
        "model_revision": MODEL_REVISION,
        "context_length": CONTEXT_LENGTH,
        "tensor_parallel_size": N_GPUS,
        "environment_name": ENVIRONMENT_NAME,
        "fleet": {
            "gpu": FLEET_GPU,
            "tp_size": FLEET_TP_SIZE,
            "regions": fleet_regions,
            "replicas_per_policy": len(fleet_regions),
            "scaledown_window_seconds": SCALEDOWN_WINDOW_SECONDS,
        },
        "spec_hash": hashlib.sha256(json.dumps(base_spec, sort_keys=True).encode()).hexdigest(),
        "manifest_hash": hashlib.sha256(
            json.dumps(sweep_manifest, sort_keys=True).encode()
        ).hexdigest(),
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _validate_trace_integrity(sweep_manifest: dict) -> list[str]:
    """Check SHA-256 checksums declared in the manifest against actual trace files.

    The manifest points at trace JSONL files on a Modal volume (mounted at
    /data). Each entry can optionally include a ``"sha256"`` field. When
    present, this function reads the file and verifies the hash matches.

    Returns a list of violation strings. Empty list means all checks passed
    (or no checksums were declared — fully backwards compatible).
    """
    violations: list[str] = []
    for item in sweep_manifest.get("top") or []:
        result = item.get("result") or {}
        path = result.get("output_path")
        expected_sha = result.get("sha256")
        if not path or not expected_sha:
            continue
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            actual = h.hexdigest()
            if actual != expected_sha:
                violations.append(
                    f"trace {path}: expected sha256={expected_sha[:16]}..., got {actual[:16]}..."
                )
        except FileNotFoundError:
            pass
        except Exception as e:
            violations.append(f"trace {path}: integrity check error: {e}")
    return violations


# Approximate token count of one repetition of ``_PROBE_FILLER_BASE``
# under the gpt-4o tokenizer (used by the workload's input filter).
# Doesn't need to be exact -- ``prompt_size_tokens`` is a target, not a
# hard cap, and the per-probe counter prefix dominates only for tiny
# sizes. ``"the quick brown fox jumps over the lazy dog. "`` ≈ 10
# cl100k tokens; we round to 9 to slightly over-pad rather than under.
_PROBE_FILLER_BASE = "the quick brown fox jumps over the lazy dog. "
_PROBE_FILLER_TOKENS_PER_REP = 9


def _build_probe_prompt(prompt_size_tokens: int, probe_index: int) -> str:
    """Return a probe message ``content`` of approximately ``prompt_size_tokens``
    tokens, with a unique per-probe counter so SGLang's radix cache
    misses on every probe.

    ``prompt_size_tokens <= 0`` returns the original tiny ``"ping"``
    prompt (RTT-dominated; backwards-compatible default behavior).
    """
    if prompt_size_tokens <= 0:
        return "ping"
    # Unique prefix forces the prefix cache to miss across probes so we
    # measure the worst-case prefill cost at this prompt shape, which is
    # the level-shift signal we want. The prefix is also small enough
    # not to perturb the target token count for typical sizes (>=512).
    prefix = f"probe-{probe_index:06d}: "
    reps = max(1, prompt_size_tokens // _PROBE_FILLER_TOKENS_PER_REP)
    return prefix + (_PROBE_FILLER_BASE * reps).rstrip()


async def _measure_replica_ttft(
    client: httpx.AsyncClient,
    replica_url: str,
    *,
    total_requests: int,
    warmup_requests: int,
    max_tokens: int,
    request_timeout_seconds: float,
    prompt_size_tokens: int = 0,
) -> dict:
    """Send ``total_requests`` short streaming chats directly to a
    replica and return TTFT stats over the post-warmup tail.

    Sequential per-replica (concurrency=1) so warm-up effects show up
    cleanly in the discarded prefix instead of contaminating the
    measurement window. Hits ``{replica_url}/v1/chat/completions``
    directly -- no proxy, no policy, no metrics interaction -- so the
    measurement is independent of any routing logic under test.

    ``prompt_size_tokens`` controls the probe prompt shape (see
    :func:`_build_probe_prompt`); ``0`` keeps the original tiny
    ``"ping"`` prompt (RTT-dominated). Each probe gets a unique counter
    prefix so SGLang's radix cache misses, giving worst-case prefill
    latency at the chosen prompt shape -- which is what we want for
    detecting per-replica prefill heterogeneity.
    """
    headers = {"accept-encoding": "identity", "content-type": "application/json"}
    measurement_ttfts_ms: list[float] = []
    measurement_total_ms: list[float] = []
    error_count = 0
    last_error: str | None = None
    for i in range(total_requests):
        # Rebuild per-probe so each request has a unique cache-busting
        # prefix; for the tiny-ping default this is identical bytes
        # every iteration, which is fine.
        body = {
            "model": HF_REPO_ID,
            "messages": [{"role": "user", "content": _build_probe_prompt(prompt_size_tokens, i)}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        payload = json.dumps(body).encode()
        started_ns = time.perf_counter_ns()
        ttft_ns: int | None = None
        try:
            async with client.stream(
                "POST",
                f"{replica_url.rstrip('/')}/v1/chat/completions",
                content=payload,
                headers=headers,
                timeout=request_timeout_seconds,
            ) as resp:
                if resp.status_code != 200:
                    error_count += 1
                    last_error = f"status={resp.status_code}"
                    await resp.aread()
                    continue
                buffer = bytearray()
                buffer_first_chunk_ns: int | None = None
                async for chunk in resp.aiter_raw():
                    if ttft_ns is not None:
                        continue
                    chunk_ns = time.perf_counter_ns()
                    if not buffer:
                        buffer_first_chunk_ns = chunk_ns
                    buffer.extend(chunk)
                    while True:
                        idx = buffer.find(b"\n\n")
                        if idx < 0:
                            break
                        event = bytes(buffer[:idx])
                        del buffer[: idx + 2]
                        event_arrival_ns = buffer_first_chunk_ns
                        buffer_first_chunk_ns = chunk_ns if buffer else None
                        if ttft_ns is not None:
                            continue
                        for line in event.split(b"\n"):
                            if not line.startswith(b"data:"):
                                continue
                            data = line[len(b"data:") :].strip()
                            if not data or data == b"[DONE]":
                                continue
                            try:
                                obj = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            choices = obj.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            if delta.get("content"):
                                ttft_ns = (event_arrival_ns or chunk_ns) - started_ns
                                break
        except Exception as e:
            error_count += 1
            last_error = f"{type(e).__name__}: {e}"
            continue
        total_ns = time.perf_counter_ns() - started_ns
        if i < warmup_requests:
            continue
        if ttft_ns is None:
            error_count += 1
            last_error = "no content event received"
            continue
        measurement_ttfts_ms.append(ttft_ns / 1e6)
        measurement_total_ms.append(total_ns / 1e6)

    def _percentile(xs: list[float], p: float) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return s[k]

    return {
        "replica_url": replica_url,
        "requests_total": total_requests,
        "warmup_discarded": warmup_requests,
        "measured_n": len(measurement_ttfts_ms),
        "errors": error_count,
        "last_error": last_error,
        "ttft_ms_p50": _percentile(measurement_ttfts_ms, 0.50),
        "ttft_ms_p95": _percentile(measurement_ttfts_ms, 0.95),
        "ttft_ms_max": max(measurement_ttfts_ms) if measurement_ttfts_ms else None,
        "total_ms_p50": _percentile(measurement_total_ms, 0.50),
    }


async def _process_homogeneity_pool(
    client: httpx.AsyncClient, pool: dict, cfg: dict
) -> tuple[dict, list[str]]:
    """Measure TTFT for one pool's replicas in parallel and build the
    per-pool report + per-pool violation list.

    Returns ``(pool_report, violations)``. Replica probes within the
    pool fan out via ``asyncio.gather``; the caller fans pools out the
    same way so the whole gate runs in roughly one pool's worth of
    wall time regardless of pool count.
    """
    label = pool["label"]
    replica_urls = pool["replica_urls"]
    per_replica = await asyncio.gather(
        *[
            _measure_replica_ttft(
                client,
                url,
                total_requests=cfg["warmup_requests_per_replica"],
                warmup_requests=cfg["warmup_requests"],
                max_tokens=cfg["max_tokens"],
                request_timeout_seconds=cfg["request_timeout_seconds"],
                prompt_size_tokens=cfg["prompt_size_tokens"],
            )
            for url in replica_urls
        ]
    )
    valid_p50s = [r["ttft_ms_p50"] for r in per_replica if r["ttft_ms_p50"] is not None]
    unmeasurable_urls = [r["replica_url"] for r in per_replica if r["ttft_ms_p50"] is None]
    ratio: float | None = None
    if len(valid_p50s) >= 2:
        lo = min(valid_p50s)
        hi = max(valid_p50s)
        if lo > 0:
            ratio = hi / lo
    measured_ns = sorted({r["measured_n"] for r in per_replica})
    uneven_sample = len(measured_ns) > 1
    pool_report = {
        "label": label,
        "policy": pool["policy"],
        "replica_count": len(replica_urls),
        "replicas": per_replica,
        "ttft_ms_p50_min": min(valid_p50s) if valid_p50s else None,
        "ttft_ms_p50_max": max(valid_p50s) if valid_p50s else None,
        "max_ttft_ratio_observed": ratio,
        "uneven_sample_sizes": uneven_sample,
        "unmeasurable_replicas": unmeasurable_urls,
    }
    print(
        f"[homogeneity] pool={label!r} "
        f"per_replica_p50_ms="
        f"{[round(r['ttft_ms_p50']) if r['ttft_ms_p50'] else None for r in per_replica]} "
        f"ratio={ratio if ratio is not None else 'n/a'}",
        flush=True,
    )
    if uneven_sample:
        # Per-replica error counts diverged, so the ratio is
        # comparing an unequal number of samples per replica.
        # Surface it so the operator knows the gate verdict
        # rests on noisier-than-expected stats.
        print(
            f"[homogeneity][warn] pool {label!r} uneven measurement counts "
            f"per replica: {measured_ns} (some probes failed)",
            flush=True,
        )
    pool_violations: list[str] = []
    # Unmeasurable replicas (every probe failed -> ttft_ms_p50 is None)
    # always count as a violation. Without this check, a totally-dead
    # replica produces a "homogeneous" 2-replica ratio that can pass
    # the gate -- the operator would think a 3-replica pool is healthy
    # when it's silently down to 2.
    if unmeasurable_urls:
        pool_violations.append(
            f"pool {label!r} has {len(unmeasurable_urls)} unmeasurable "
            f"replica(s) (every probe failed): {unmeasurable_urls}"
        )
    if cfg["max_ttft_ratio"] > 0 and ratio is not None and ratio > cfg["max_ttft_ratio"]:
        pool_violations.append(
            f"pool {label!r} ratio={ratio:.2f} > max_ttft_ratio={cfg['max_ttft_ratio']}"
        )
    return pool_report, pool_violations


async def _check_replica_homogeneity(
    fleet_manifest: dict,
    cfg: dict,
    *,
    cleanup_callback: Callable[[], Awaitable[None]] | None = None,
) -> dict:
    """Pre-flight per-pool TTFT measurement and homogeneity gate.

    Hits each replica directly so the result is independent of the
    routing policy that will later run against the same pool. Returns
    a manifest the caller can stash inside ``fleet_manifest``; raises
    ``RuntimeError`` when ``on_violation == "abort"`` and any pool's
    P50 TTFT spread exceeds ``max_ttft_ratio``, or when any replica
    is fully unmeasurable.

    What this catches: level-shift heterogeneity -- one replica is
    routinely slower than its siblings (cold/wrong-tier instance,
    noisy neighbor, model still loading after ``wait_ready`` returned).
    This was the confound behind the spurious least-request advantage
    in ``moon_neurips_main_000_quick200``, where one of three shared
    backends was ~10x faster than the others. Also catches
    silently-dead replicas via the ``unmeasurable_replicas`` violation.

    What this does NOT catch (default config): prefill-shape
    heterogeneity. The default probe uses a tiny ``"ping"`` prompt
    (TTFT dominated by RTT + minimal prefill), while real Mooncake-
    replay workloads carry up to 24k input tokens (TTFT dominated by
    full-prompt prefill). To extend the gate to large-prompt prefill,
    set ``prompt_size_tokens`` to a value close to your workload's
    typical input size.

    On abort: ``cleanup_callback`` (if provided) is awaited before the
    ``RuntimeError`` is raised. The matrix controller passes a callback
    that cancels the spawned engine + proxy ``FunctionCall`` futures so
    GPUs are freed immediately rather than idling until each engine's
    ``scaledown_window`` expires.
    """
    if not cfg["enabled"]:
        return {"enabled": False}
    n_replicas = sum(len(p["replica_urls"]) for p in fleet_manifest["fleets"])
    n_pools = len(fleet_manifest["fleets"])
    print(
        f"[homogeneity] checking {n_replicas} replicas across {n_pools} pools "
        f"(probes={cfg['warmup_requests_per_replica']}, "
        f"warmup_discarded={cfg['warmup_requests']}, "
        f"prompt_size_tokens={cfg['prompt_size_tokens']})",
        flush=True,
    )
    started = time.time()
    async with httpx.AsyncClient() as client:
        # Fan out across pools so total wall time ~ one pool's worth
        # rather than n_pools-times-one. Within a pool, replicas already
        # fan out via _process_homogeneity_pool's inner gather.
        per_pool = await asyncio.gather(
            *[_process_homogeneity_pool(client, pool, cfg) for pool in fleet_manifest["fleets"]]
        )
    pools_report = [r for r, _ in per_pool]
    violations: list[str] = []
    for _, pool_violations in per_pool:
        violations.extend(pool_violations)

    elapsed = time.time() - started
    print(
        f"[homogeneity] done in {elapsed:.1f}s "
        f"violations={len(violations)} action={cfg['on_violation']!r}",
        flush=True,
    )
    report = {
        "enabled": True,
        "config": cfg,
        "elapsed_seconds": round(elapsed, 2),
        "pools": pools_report,
        "violations": violations,
    }
    if violations:
        if cfg["on_violation"] == "abort":
            if cleanup_callback is not None:
                print(
                    f"[homogeneity][abort] cancelling spawned engine + proxy "
                    f"FunctionCalls to free GPUs immediately",
                    flush=True,
                )
                try:
                    await cleanup_callback()
                except Exception as e:
                    print(
                        f"[homogeneity][abort][warn] cleanup_callback raised "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
            else:
                print(
                    f"[homogeneity][abort] {n_pools} pools spawned; "
                    "no cleanup_callback provided -- engine fleet will idle "
                    "until each engine_* function's scaledown_window expires",
                    flush=True,
                )
            raise RuntimeError(
                "replica homogeneity check failed: "
                + "; ".join(violations)
                + ". Set replica_homogeneity_check.on_violation='warn' to ignore."
            )
        for v in violations:
            print(f"[homogeneity][warn] {v}", flush=True)
    return report


async def _cancel_spawns(calls: list, kind: str = "spawn") -> None:
    """Best-effort cancel a list of Modal ``FunctionCall`` futures.

    Logs but doesn't raise on per-call cancel failures so we always
    iterate the full list -- a partial failure on one engine
    shouldn't keep us from cancelling the rest. Used by the homogeneity
    abort path to free GPUs immediately instead of waiting for each
    engine's ``scaledown_window`` to expire.
    """
    cancelled = 0
    for call in calls:
        try:
            await call.cancel.aio()
            cancelled += 1
        except Exception as e:
            print(
                f"[cancel-spawns][warn] failed to cancel {kind}: {type(e).__name__}: {e}",
                flush=True,
            )
    print(
        f"[cancel-spawns] cancelled {cancelled}/{len(calls)} {kind} FunctionCall(s)",
        flush=True,
    )


async def _wait_for_keys(dct, keys: list[str], timeout_s: float) -> dict[str, str]:
    deadline = time.time() + timeout_s
    pending = set(keys)
    found: dict[str, str] = {}
    while pending and time.time() < deadline:
        for key in list(pending):
            value = await dct.get.aio(key)
            if isinstance(value, str) and value:
                found[key] = value
                pending.remove(key)
                print(f"[ready] {key} -> {value}", flush=True)
        if pending:
            await asyncio.sleep(5)
    if pending:
        raise TimeoutError(f"timed out waiting for {sorted(pending)}")
    return found


async def _post_json(
    url: str,
    path: str,
    payload: dict | list,
    *,
    retries: int = 3,
    backoff_base: float = 2.0,
) -> dict:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(url.rstrip("/") + path, json=payload)
                r.raise_for_status()
                return r.json()
        except (httpx.TransportError, OSError) as e:
            # httpx.TransportError covers ConnectError/ConnectTimeout/ReadTimeout
            # *and* RemoteProtocolError ("Server disconnected without sending a
            # response"), which a proxy throws while it is still cold-starting:
            # it accepts the TCP connection but drops it before replying. These
            # are all transient during fleet bring-up, so retry rather than
            # letting one kill the whole phase.
            last_err = e
            if attempt < retries:
                wait = backoff_base**attempt
                print(
                    f"[retry] POST {path} to {url[-40:]} failed "
                    f"({type(e).__name__}), retry {attempt + 1}/{retries} "
                    f"in {wait:.0f}s",
                    flush=True,
                )
                await asyncio.sleep(wait)
            else:
                raise
    raise last_err  # unreachable but keeps type checker happy


async def _get_json(
    url: str,
    path: str,
    *,
    retries: int = 3,
    backoff_base: float = 2.0,
) -> dict:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.get(url.rstrip("/") + path)
                r.raise_for_status()
                return r.json()
        except (httpx.TransportError, OSError) as e:
            # See _post_json: TransportError also covers RemoteProtocolError,
            # the cold-start "server disconnected" race during fleet bring-up.
            last_err = e
            if attempt < retries:
                wait = backoff_base**attempt
                print(
                    f"[retry] GET {path} from {url[-40:]} failed "
                    f"({type(e).__name__}), retry {attempt + 1}/{retries} "
                    f"in {wait:.0f}s",
                    flush=True,
                )
                await asyncio.sleep(wait)
            else:
                raise
    raise last_err


async def _wait_for_proxy_metrics(
    url: str,
    *,
    deadline_seconds: float = 10 * 60,
    poll_interval_seconds: float = 1.0,
) -> dict:
    started = time.time()
    deadline = started + deadline_seconds
    last_state: dict | None = None

    while time.time() < deadline:
        replicas_doc = await _get_json(url, "/replicas")
        replica_count = len(replicas_doc.get("replicas") or [])
        metrics_doc = await _get_json(url, "/replica_metrics")
        metrics = metrics_doc.get("metrics") or {}
        errors = metrics_doc.get("errors") or {}
        last_state = {
            "replica_count": replica_count,
            "metrics_count": len(metrics),
            "errors": errors,
            "last_refresh_age_seconds": metrics_doc.get("last_refresh_age_seconds"),
        }
        if replica_count > 0 and len(metrics) == replica_count and not errors:
            print(
                f"[metrics-ready] {url} replicas={replica_count} "
                f"waited={time.time() - started:.1f}s",
                flush=True,
            )
            return last_state
        await asyncio.sleep(poll_interval_seconds)

    raise TimeoutError(f"timed out waiting for proxy metrics at {url}: {last_state}")


def _workload_payload(
    *,
    spec: dict,
    run_id: str,
    trace_path: str,
    output_path: str,
    start_at_wall_time: str | None = None,
) -> dict:
    payload = {
        "data_path": trace_path,
        "run_id": run_id,
        "concurrency": int(spec.get("concurrency", 16)),
        "model": spec.get("model", ""),
        "stream": spec.get("stream", True),
        "max_tokens": int(spec.get("max_tokens", 0)),
        "max_input_tokens": int(spec.get("max_input_tokens", 0)),
        "arrival_mode": spec.get("arrival_mode", "open-loop"),
        "time_scale": float(spec.get("time_scale", 1.0)),
        "output_path": output_path,
        "save_per_request": bool(spec.get("save_per_request", True)),
    }
    num = int(spec.get("num_requests", 0))
    if num > 0:
        payload["num_requests"] = num
    if start_at_wall_time:
        payload["start_at_wall_time"] = start_at_wall_time
    return payload


async def _wait_for_workload(
    url: str,
    poll_interval: float,
    timeout_seconds: float | None = None,
) -> dict:
    deadline = time.time() + timeout_seconds if timeout_seconds else None
    while True:
        status_doc = await _get_json(url, "/workload/status")
        workload = status_doc.get("workload") or {}
        if workload.get("status") in TERMINAL_WORKLOAD_STATUS:
            return workload
        if deadline and time.time() > deadline:
            raise TimeoutError(f"workload on {url} did not finish within {timeout_seconds:.0f}s")
        await asyncio.sleep(poll_interval)


def _extract_learned_weights(matrix: dict) -> dict | None:
    """Extract the learned gorgo weights from a tuning matrix result.

    Walks the per-policy results looking for a gorgo policy with
    ``auto_tune.hyperparameters.defaults``. Prefers ``online-es`` (hillclimb)
    over ``fit`` (autotune) when both are present.
    """
    fallback: dict | None = None
    for trace_result in matrix.get("results") or []:
        manifest = trace_result.get("manifest") or {}
        for r in manifest.get("results") or []:
            if not isinstance(r, dict) or r.get("error"):
                continue
            at = r.get("auto_tune")
            if not at:
                continue
            hp = at.get("hyperparameters") or {}
            defaults = hp.get("defaults")
            if not defaults or not isinstance(defaults, dict):
                continue
            config = at.get("config") or {}
            mode = config.get("mode", "")
            if mode == "online-es":
                return defaults
            if fallback is None:
                fallback = defaults
    return fallback


def _extract_calibrated_rates(matrix: dict) -> dict | None:
    """Extract pooled live-calibrated physical rates from a calibration
    matrix result. Walks per-policy results for the first gorgo policy
    carrying a ``calibrated_rates`` block with a non-null rate.
    """
    for trace_result in matrix.get("results") or []:
        manifest = trace_result.get("manifest") or {}
        for r in manifest.get("results") or []:
            if not isinstance(r, dict) or r.get("error"):
                continue
            cr = r.get("calibrated_rates")
            if not isinstance(cr, dict):
                continue
            if cr.get("prefill_rate") is not None or cr.get("queue_rate") is not None:
                return cr
    return None


def _auto_tune_payload(policy_spec: dict, *, global_seed: int | None = None) -> dict | None:
    auto = policy_spec.get("auto_tune")
    if not auto:
        return None
    payload: dict = {
        "enabled": True,
        "window_size": int(auto.get("window_size", 200)),
        "hop_size": int(auto.get("hop_size", 50)),
        "apply": bool(auto.get("apply", True)),
    }
    # Forward the full auto-tune knob set so spec-driven `mode` /
    # `objective_metric` (and any future fields) reach the proxy. The
    # earlier short list silently dropped these, which made
    # ``gorgo-online-es`` fall back to the default ``fit`` mode.
    if "mode" in auto:
        payload["mode"] = str(auto["mode"])
    if "objective_metric" in auto:
        payload["objective_metric"] = str(auto["objective_metric"])
    # Propagate seed for reproducible online-ES perturbations.
    # Per-policy auto_tune.seed overrides the global spec seed.
    seed = auto.get("seed", global_seed)
    if seed is not None:
        payload["seed"] = int(seed)
    # Forward custom hyperparam ranges so the proxy uses spec-driven
    # bounds instead of its hardcoded defaults.
    if "hyperparam_ranges" in auto:
        payload["hyperparam_ranges"] = auto["hyperparam_ranges"]
    return payload


async def _respawn_proxy(policy_spec: dict, timeout_s: float = 120.0) -> str:
    """Respawn a dead proxy container and return the new URL.

    Spawns a fresh ``proxy_runner``, waits for it to register under the
    same key, then reconfigures replicas so the new proxy knows its
    backends.  Returns the new tunnel URL.
    """
    key = policy_spec["_proxy_key"]
    label = policy_spec.get("label") or policy_spec.get("name")
    replica_urls = policy_spec["_replica_urls"]
    replica_entries = policy_spec.get("_replica_entries")

    print(f"[respawn] spawning new proxy for {label!r} (key={key})", flush=True)
    await proxy_runner.spawn.aio(key)
    new_urls = await _wait_for_keys(proxies, [key], timeout_s)
    new_url = new_urls[key]
    print(f"[respawn] new proxy URL: {new_url}", flush=True)

    await _post_json(new_url, "/replicas", {"replicas": replica_entries or replica_urls})
    return new_url


async def _run_one_policy(global_spec: dict, policy_spec: dict) -> dict:
    name = policy_spec["name"]
    label = policy_spec.get("label") or name
    url = policy_spec["proxy_url"].rstrip("/")
    run_id = f"{global_spec.get('run_id', 'policy_matrix')}_{_slug(label)}"
    exp_id = global_spec.get("experiment_id", "")
    trace_id = policy_spec.get("trace_id") or (f"{exp_id}/{run_id}" if exp_id else run_id)
    default_template = (
        "/results/workload_runs/{experiment_id}/{run_id}.json"
        if exp_id
        else "/results/workload_runs/{run_id}.json"
    )
    output_path = global_spec.get("output_path_template", default_template).format(
        run_id=run_id,
        policy=_slug(name),
        experiment_id=exp_id,
    )
    try:
        await _post_json(url, "/policy", {"policy": name})
    except (httpx.ConnectError, httpx.ConnectTimeout, OSError) as e:
        if "_proxy_key" not in policy_spec or "_replica_urls" not in policy_spec:
            raise
        print(
            f"[respawn] proxy for {label!r} unreachable after retries "
            f"({type(e).__name__}), respawning",
            flush=True,
        )
        url = await _respawn_proxy(policy_spec)
        policy_spec["proxy_url"] = url
        await _post_json(url, "/policy", {"policy": name})
    if policy_spec.get("hyperparameters"):
        await _post_json(url, "/hyperparameters", policy_spec["hyperparameters"])
    await _post_json(url, "/flush", {})
    global_seed = global_spec.get("seed")
    auto_tune_config = _auto_tune_payload(policy_spec, global_seed=global_seed)
    if auto_tune_config:
        await _post_json(url, "/tune", auto_tune_config)
    metrics_ready = await _wait_for_proxy_metrics(
        url,
        deadline_seconds=float(global_spec.get("metrics_ready_timeout_seconds", 10 * 60)),
        poll_interval_seconds=float(global_spec.get("metrics_ready_poll_interval_seconds", 1.0)),
    )
    await _post_json(
        url,
        "/trace/start",
        {
            "trace_id": trace_id,
            "sample_metrics": True,
            "sample_requests": True,
            "max_events": int(global_spec.get("max_trace_events", 200_000)),
        },
    )
    await _post_json(
        url,
        "/workload/start",
        _workload_payload(
            spec=global_spec,
            run_id=run_id,
            trace_path=global_spec["trace_path"],
            output_path=output_path,
            start_at_wall_time=global_spec["start_at_wall_time"],
        ),
    )
    started = time.time()
    max_minutes = global_spec.get("max_experiment_minutes")
    workload_timeout = max_minutes * 60.0 if max_minutes else None
    workload = await _wait_for_workload(
        url,
        float(global_spec.get("poll_interval_seconds", 5.0)),
        timeout_seconds=workload_timeout,
    )
    final_hyperparameters = None
    calibrated_rates = None
    if auto_tune_config:
        # Harvest live-calibrated physical rates before disabling the tuner
        # so a 'calibrate' run surfaces its pooled prefill_rate/queue_rate.
        if auto_tune_config.get("mode") == "calibrate":
            try:
                cr = await _get_json(url, "/calibrated_rates")
                calibrated_rates = (cr or {}).get("calibrated_rates")
            except Exception as e:
                print(f"[calib][warn] failed to read /calibrated_rates: {e}", flush=True)
        await _post_json(url, "/tune", {"enabled": False})
        hps = await _get_json(url, "/hyperparameters")
        final_hyperparameters = hps.get("hyperparameters")
    await _post_json(url, "/trace/stop", {})
    trace_doc = await _post_json(url, "/trace/save", {})
    # Surface the trace's fallback summary at the top level of the
    # per-policy result so an analyst spotting an anomalous result
    # doesn't have to crack the trace to see whether random-fallback
    # rows contaminated the per-policy aggregate. The proxy computes it
    # over its in-memory buffer in /trace/save (see
    # ``_compute_fallback_summary`` in proxy/modal_proxy.py); a non-zero
    # ``fallback_rate`` is a yellow flag for the run's validity.
    fallback_summary = (trace_doc or {}).get("fallback_summary") or {}
    if fallback_summary:
        rate = fallback_summary.get("fallback_rate", 0.0) or 0.0
        if rate > 0:
            print(
                f"[fallback] policy={label!r} fallback_rate={rate:.1%} "
                f"({fallback_summary.get('fallback_count')}/"
                f"{fallback_summary.get('total_requests')}) "
                f"by_effective_policy={fallback_summary.get('by_effective_policy')}",
                flush=True,
            )
    return {
        "policy": name,
        "label": label,
        "proxy_url": url,
        "run_id": run_id,
        "trace_id": trace_id,
        "auto_tune": (
            {"config": auto_tune_config, "hyperparameters": final_hyperparameters}
            if auto_tune_config
            else None
        ),
        "calibrated_rates": calibrated_rates,
        "workload": workload,
        "trace": trace_doc,
        "fallback_summary": fallback_summary,
        "metrics_ready": metrics_ready,
        "elapsed_seconds": time.time() - started,
    }


async def _run_policy_matrix(base_spec: dict) -> dict:
    spec = dict(base_spec)
    spec["start_at_wall_time"] = _start_at_wall_time(spec)
    results = await asyncio.gather(
        *[_run_one_policy(spec, p) for p in spec["policies"]],
        return_exceptions=True,
    )
    normalized = []
    for policy, result in zip(spec["policies"], results):
        if isinstance(result, Exception):
            normalized.append(
                {
                    "policy": policy.get("name"),
                    "label": policy.get("label") or policy.get("name"),
                    "proxy_url": policy.get("proxy_url"),
                    "error": f"{type(result).__name__}: {result}",
                }
            )
        else:
            normalized.append(result)
    return {
        "run_id": spec.get("run_id"),
        "trace_path": spec.get("trace_path"),
        "start_at_wall_time": spec.get("start_at_wall_time"),
        "policies": [p.get("label") or p.get("name") for p in spec["policies"]],
        "results": normalized,
    }


async def _run_sweep_matrix(
    *,
    base_spec: dict,
    sweep_manifest: dict,
    start_index: int,
    top_k: int,
    output_dir: Path,
) -> dict:
    top = sweep_manifest.get("top") or []
    selected = top[start_index : start_index + top_k]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for local_idx, item in enumerate(selected):
        idx = start_index + local_idx
        result = item.get("result") or {}
        trace_path = result.get("output_path")
        if not trace_path:
            results.append({"ok": False, "index": idx, "error": "missing result.output_path"})
            continue
        body_trace = _with_bodies_path(trace_path)
        trace_name = _trace_stem(trace_path)
        spec = deepcopy(base_spec)
        base_run_id = base_spec.get("run_id", "policy_matrix")
        spec["trace_path"] = body_trace
        spec["run_id"] = f"{base_run_id}_{idx:03d}_{trace_name}"
        try:
            manifest = await _run_policy_matrix(spec)
            matrix_path = output_dir / f"{spec['run_id']}.json"
            matrix_path.write_text(json.dumps(manifest, indent=2))
            results.append(
                {
                    "ok": True,
                    "index": idx,
                    "trace_path": body_trace,
                    "source_trace": trace_path,
                    "matrix_manifest_path": str(matrix_path),
                    "manifest": manifest,
                }
            )
        except Exception as e:
            results.append(
                {
                    "ok": False,
                    "index": idx,
                    "trace_path": body_trace,
                    "source_trace": trace_path,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    aggregate = {
        "base_run_id": base_spec.get("run_id"),
        "sweep_manifest_path": sweep_manifest.get("_path"),
        "start_index": start_index,
        "top_k": top_k,
        "results": results,
    }
    aggregate_path = output_dir / f"{base_spec.get('run_id', 'policy_matrix')}_sweep_matrix.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2))
    aggregate["aggregate_manifest_path"] = str(aggregate_path)
    return aggregate


@app.function(
    image=modal.Image.debian_slim()
    .pip_install("httpx")
    .add_local_python_source("app", "engine", "proxy", "scripts", "gorgo"),
    volumes={"/results": bench_results_volume},
    timeout=24 * 60 * 60,
)
def run_policy_matrix_experiment(
    base_spec: dict,
    sweep_manifest: dict,
    experiment_id: str = "neurips_h100_matrix",
    start_index: int = 1,
    top_k: int = 1,
    output_dir: str = "/results/policy_matrix_sweep/moon_neurips_one_trace",
    engine_timeout_s: float = 24 * 60 * 60,
    proxy_timeout_s: float = 10 * 60,
    environment: dict | None = None,
) -> dict:
    return asyncio.run(
        _run_policy_matrix_experiment(
            base_spec=base_spec,
            sweep_manifest=sweep_manifest,
            experiment_id=experiment_id,
            start_index=start_index,
            top_k=top_k,
            output_dir=output_dir,
            engine_timeout_s=engine_timeout_s,
            proxy_timeout_s=proxy_timeout_s,
            environment=environment,
        )
    )


async def _run_policy_matrix_experiment(
    *,
    base_spec: dict,
    sweep_manifest: dict,
    experiment_id: str,
    start_index: int,
    top_k: int,
    output_dir: str,
    engine_timeout_s: float,
    proxy_timeout_s: float,
    environment: dict | None = None,
) -> dict:
    # Validate trace integrity if SHA-256 checksums are declared in the manifest.
    # Runs inside the Modal container where the volume is mounted, so trace
    # files at /data/... are accessible.
    integrity_violations = _validate_trace_integrity(sweep_manifest)
    if integrity_violations:
        for v in integrity_violations:
            print(f"[integrity][ERROR] {v}", flush=True)
        raise RuntimeError("trace integrity check failed: " + "; ".join(integrity_violations))

    experiment_started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    base_spec["experiment_id"] = experiment_id
    policies = base_spec["policies"]
    regions = base_spec.get("fleet_regions") or ["CANADA-2", "sines-2", "us-west4"]
    output_dir = _unique_output_dir(output_dir, experiment_id)

    engine_calls = []
    engine_keys = []
    proxy_calls = []
    proxy_keys = []

    async def _cleanup_all_containers():
        """Cancel all spawned engine and proxy containers."""
        print("[cleanup] cancelling all spawned containers...", flush=True)
        await _cancel_spawns(proxy_calls, kind="proxy")
        await _cancel_spawns(engine_calls, kind="engine")
        print("[cleanup] done", flush=True)

    try:
        return await _run_policy_matrix_experiment_inner(
            base_spec=base_spec,
            sweep_manifest=sweep_manifest,
            experiment_id=experiment_id,
            start_index=start_index,
            top_k=top_k,
            output_dir=output_dir,
            engine_timeout_s=engine_timeout_s,
            proxy_timeout_s=proxy_timeout_s,
            environment=environment,
            engine_calls=engine_calls,
            engine_keys=engine_keys,
            proxy_calls=proxy_calls,
            proxy_keys=proxy_keys,
        )
    finally:
        await _cleanup_all_containers()


async def _run_policy_matrix_experiment_inner(
    *,
    base_spec: dict,
    sweep_manifest: dict,
    experiment_id: str,
    start_index: int,
    top_k: int,
    output_dir: str,
    engine_timeout_s: float,
    proxy_timeout_s: float,
    environment: dict | None,
    engine_calls: list,
    engine_keys: list,
    proxy_calls: list,
    proxy_keys: list,
) -> dict:
    experiment_started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    run_instance_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    fleet_key_prefix = f"{experiment_id}-{run_instance_id}"
    policies = base_spec["policies"]
    regions = base_spec.get("fleet_regions") or ["CANADA-2", "sines-2", "us-west4"]

    for policy in policies:
        label = _label(policy)
        for region in regions:
            fn = ENGINE_BY_REGION[region]
            key = f"{fleet_key_prefix}-{label}-{region}"
            engine_keys.append(key)
            engine_calls.append(await fn.spawn.aio(key))

    replica_urls = await _wait_for_keys(replicas, engine_keys, engine_timeout_s)

    for policy in policies:
        label = _label(policy)
        key = f"{fleet_key_prefix}-{label}"
        proxy_keys.append(key)
        proxy_calls.append(await proxy_runner.spawn.aio(key))

    proxy_urls = await _wait_for_keys(proxies, proxy_keys, proxy_timeout_s)

    launched_spec = deepcopy(base_spec)
    launched_spec["policies"] = []
    fleet_manifest = {
        "experiment_id": experiment_id,
        "run_instance_id": run_instance_id,
        "fleet_key_prefix": fleet_key_prefix,
        "regions": regions,
        "fleets": [],
    }

    for policy in policies:
        label = _label(policy)
        policy_replica_keys = [f"{fleet_key_prefix}-{label}-{region}" for region in regions]
        policy_replica_urls = [replica_urls[k] for k in policy_replica_keys]
        policy_replica_entries = [
            {
                "url": u,
                "replica_key": k,
                "replica_region": r,
            }
            for k, r, u in zip(policy_replica_keys, regions, policy_replica_urls)
        ]
        proxy_key = f"{fleet_key_prefix}-{label}"
        proxy_url = proxy_urls[proxy_key]
        await _post_json(proxy_url, "/replicas", {"replicas": policy_replica_entries})
        await _post_json(proxy_url, "/policy", {"policy": policy["name"]})
        if policy.get("hyperparameters"):
            await _post_json(proxy_url, "/hyperparameters", policy["hyperparameters"])
        p = deepcopy(policy)
        p["proxy_url"] = proxy_url
        p["_proxy_key"] = proxy_key
        p["_replica_urls"] = policy_replica_urls
        p["_replica_entries"] = policy_replica_entries
        launched_spec["policies"].append(p)
        fleet_manifest["fleets"].append(
            {
                "label": label,
                "policy": policy["name"],
                "proxy_key": proxy_key,
                "proxy_url": proxy_url,
                "replica_keys": policy_replica_keys,
                "replica_urls": policy_replica_urls,
            }
        )

    homogeneity_cfg = _homogeneity_config(base_spec)

    async def _abort_cleanup() -> None:
        await _cancel_spawns(proxy_calls, kind="proxy")
        await _cancel_spawns(engine_calls, kind="engine")

    homogeneity_report = await _check_replica_homogeneity(
        fleet_manifest,
        homogeneity_cfg,
        cleanup_callback=_abort_cleanup,
    )
    fleet_manifest["homogeneity_check"] = homogeneity_report

    # Write run manifest once the fleet is fully provisioned and validated.
    # Updated again after workloads complete with result paths and timing.
    fleet_ready_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    run_manifest_path = Path(output_dir) / "run_manifest.json"
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": "2.0",
        "status": "running",
        "output_dir": output_dir,
        "environment": environment,
        "timing": {
            "started_utc": experiment_started_utc,
            "fleet_ready_utc": fleet_ready_utc,
            "completed_utc": None,
        },
        "spec": base_spec,
        "manifest": sweep_manifest,
        "fleet": fleet_manifest,
        "results": None,
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2))
    bench_results_volume.commit()
    print(f"[manifest] run_manifest.json written (status=running)", flush=True)

    matrix = await _run_sweep_matrix(
        base_spec=launched_spec,
        sweep_manifest=sweep_manifest,
        start_index=start_index,
        top_k=top_k,
        output_dir=Path(output_dir),
    )

    # --- Tune → Eval chaining ---
    # If eval spec(s) + manifest(s) were provided, extract the learned gorgo
    # weights from the tuning results, reconfigure the gorgo proxy with
    # static weights, and run each eval on the same fleet sequentially.
    eval_chain = base_spec.get("_eval_chain") or []
    if not eval_chain:
        eval_spec_raw = base_spec.get("_eval_spec")
        eval_manifest_raw = base_spec.get("_eval_manifest")
        if eval_spec_raw and eval_manifest_raw:
            eval_chain = [{"spec": eval_spec_raw, "manifest": eval_manifest_raw}]

    if eval_chain:
        learned = _extract_learned_weights(matrix)
        if learned:
            print(f"[tune→eval] learned weights: {learned}", flush=True)
            run_manifest["evals"] = []

            for eval_idx, eval_entry in enumerate(eval_chain):
                eval_spec_raw = eval_entry["spec"]
                eval_manifest_raw = eval_entry["manifest"]
                eval_label = eval_manifest_raw.get("_note", f"eval-{eval_idx}")

                eval_spec = deepcopy(eval_spec_raw)
                for p in eval_spec.get("policies", []):
                    if p.get("name") == "gorgo":
                        p["hyperparameters"] = learned
                        print(
                            f"[tune→eval][{eval_idx}] patched gorgo policy "
                            f"{p.get('label')!r} with learned weights",
                            flush=True,
                        )

                eval_suffix = f"_eval{eval_idx}" if len(eval_chain) > 1 else "_eval"
                eval_output_dir = _unique_output_dir(
                    output_dir.rsplit("/", 1)[0] + "/" + eval_spec.get("run_id", "eval"),
                    experiment_id + eval_suffix,
                )

                # Reconfigure fleet for eval: disable auto_tune, set static
                # weights, flush workload state on each proxy.
                for fleet_entry in fleet_manifest["fleets"]:
                    proxy_url = fleet_entry["proxy_url"]
                    policy_name = fleet_entry["policy"]
                    if policy_name == "gorgo":
                        await _post_json(proxy_url, "/tune", {"enabled": False})
                        await _post_json(proxy_url, "/hyperparameters", learned)
                    await _post_json(proxy_url, "/flush", {})

                eval_launched = deepcopy(eval_spec)
                eval_launched["experiment_id"] = experiment_id + eval_suffix
                eval_launched["policies"] = []
                for p in eval_spec.get("policies", []):
                    label = _label(p)
                    matching = [f for f in fleet_manifest["fleets"] if f["label"] == label]
                    if matching:
                        ep = deepcopy(p)
                        ep["proxy_url"] = matching[0]["proxy_url"]
                        eval_launched["policies"].append(ep)
                    else:
                        print(
                            f"[tune→eval][{eval_idx}][warn] no fleet entry for "
                            f"eval policy {label!r}, skipping",
                            flush=True,
                        )

                print(
                    f"[tune→eval][{eval_idx}] starting eval with "
                    f"{len(eval_launched['policies'])} policies on {eval_label}",
                    flush=True,
                )
                eval_matrix = await _run_sweep_matrix(
                    base_spec=eval_launched,
                    sweep_manifest=eval_manifest_raw,
                    start_index=start_index,
                    top_k=top_k,
                    output_dir=Path(eval_output_dir),
                )
                run_manifest["evals"].append(
                    {
                        "index": eval_idx,
                        "label": eval_label,
                        "learned_weights": learned,
                        "eval_output_dir": eval_output_dir,
                        "eval_spec": eval_spec_raw,
                        "eval_manifest": eval_manifest_raw,
                        "eval_results": eval_matrix,
                    }
                )

            # Backward compat: copy first eval to "eval" key
            if run_manifest["evals"]:
                run_manifest["eval"] = run_manifest["evals"][0]
        else:
            print("[tune→eval][warn] no learned weights found, skipping eval", flush=True)

    # Always extract and record learned weights if available, even
    # without eval chaining.  The sequencer uses this to pass weights
    # between phases.
    if "learned_weights" not in run_manifest:
        lw = _extract_learned_weights(matrix)
        if lw:
            run_manifest["learned_weights"] = lw

    # Surface live-calibrated physical rates so the sequencer can pass them
    # from a calibration phase into the tuning + eval phases.
    cr = _extract_calibrated_rates(matrix)
    if cr:
        run_manifest["calibrated_rates"] = cr

    # Update the manifest with completion status and result paths.
    experiment_completed_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    run_manifest["status"] = "completed"
    run_manifest["timing"]["completed_utc"] = experiment_completed_utc
    run_manifest["results"] = {
        "sweep_matrix_path": matrix.get("aggregate_manifest_path"),
        "policies": [
            {
                "label": r.get("label"),
                "workload_output_path": (r.get("workload") or {})
                .get("config", {})
                .get("output_path"),
                "trace_id": r.get("trace_id"),
            }
            for result in matrix.get("results") or []
            for r in (result.get("manifest", {}).get("results") or [])
            if isinstance(r, dict) and not r.get("error")
        ],
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2))
    bench_results_volume.commit()
    print(f"[manifest] run_manifest.json updated (status=completed)", flush=True)

    return run_manifest


@app.local_entrypoint()
def main(
    base_spec_path: str = "specs/policy_matrix_neurips_main.json",
    sweep_manifest_path: str = "specs/manifest.json",
    experiment_id: str = "neurips_h100_matrix",
    start_index: int = 1,
    top_k: int = 1,
    output_dir: str = "/results/policy_matrix_sweep/moon_neurips_one_trace",
    eval_spec_path: str = "",
    eval_manifest_path: str = "",
):
    base_spec = json.loads(Path(base_spec_path).read_text())
    sweep_manifest = json.loads(Path(sweep_manifest_path).read_text())

    if eval_spec_path and eval_manifest_path:
        eval_specs = [s.strip() for s in eval_spec_path.split(",") if s.strip()]
        eval_manifests = [m.strip() for m in eval_manifest_path.split(",") if m.strip()]
        if len(eval_specs) == 1 and len(eval_manifests) > 1:
            eval_specs = eval_specs * len(eval_manifests)
        if len(eval_specs) != len(eval_manifests):
            raise ValueError(
                f"eval_spec_path has {len(eval_specs)} entries but "
                f"eval_manifest_path has {len(eval_manifests)}"
            )
        eval_chain = [
            {"spec": json.loads(Path(s).read_text()), "manifest": json.loads(Path(m).read_text())}
            for s, m in zip(eval_specs, eval_manifests)
        ]
        base_spec["_eval_chain"] = eval_chain
        base_spec["_eval_spec"] = eval_chain[0]["spec"]
        base_spec["_eval_manifest"] = eval_chain[0]["manifest"]
        print(
            f"[tune→eval] will chain {len(eval_chain)} eval(s) after tuning: "
            f"specs={eval_specs} manifests={eval_manifests}",
            flush=True,
        )

    _validate_spec(base_spec)

    environment = _capture_environment(base_spec, sweep_manifest)
    print(f"[env] commit={environment['gorgo_commit']}", flush=True)
    if environment.get("gorgo_dirty"):
        print("[env][warn] working tree has uncommitted changes", flush=True)

    # Use .spawn() (not .remote()): for a detached run the experiment must
    # outlive this local client. .remote()/.map() are coupled to the caller's
    # connection and Modal may cancel them when the local process disconnects.
    # .spawn() launches in the background and returns a handle immediately.
    call = run_policy_matrix_experiment.spawn(
        base_spec,
        sweep_manifest,
        experiment_id=experiment_id,
        start_index=start_index,
        top_k=top_k,
        output_dir=output_dir,
        environment=environment,
    )
    print(
        f"[policy-matrix] spawned run_policy_matrix_experiment call_id={call.object_id} "
        f"experiment_id={experiment_id}",
        flush=True,
    )
    print(
        "[policy-matrix] detached: experiment runs independently of this client. "
        "Track via the Modal app dashboard or `modal app logs`. Results land under "
        f"{output_dir} on the bench-results volume.",
        flush=True,
    )
