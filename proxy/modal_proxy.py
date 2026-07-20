import asyncio
import json
import logging
import os
import random
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Callable

# Force ``logging.Formatter`` to format ``%(asctime)s`` in UTC so the
# ``Z`` suffix in the uvicorn log config below is honest regardless of
# the container's tzdata. Touches the base class so it propagates to
# uvicorn's ``DefaultFormatter`` / ``AccessFormatter`` subclasses too.
logging.Formatter.converter = time.gmtime


def _log(message: str) -> None:
    """Emit a ``[proxy]`` log line prefixed with an ISO 8601 UTC
    timestamp (millisecond precision). Wraps ``print`` so it composes
    with Modal's stdout streaming; ``flush=True`` keeps lines visible
    on long-running uvicorn workers where stdout would otherwise
    line-buffer."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    print(f"{ts} [proxy] {message}", flush=True)


# uvicorn ships a sensible LOGGING_CONFIG but its formatters have no
# timestamps -- fine for short request/response loops, but we run the
# proxy for hours and need to correlate access lines with the
# ``_log()`` output above (replica syncs, hyperparameter applies, etc.)
# and with the workload-side logs. Same datefmt as ``_log`` (sans
# milliseconds, which uvicorn's logging.Formatter doesn't support
# natively without a custom Formatter class) so visual scan order
# matches across streams.
_UVICORN_LOG_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(asctime)sZ %(levelprefix)s %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(asctime)sZ %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}

import httpx
import modal
from transformers import AutoTokenizer

from app import (
    app,
    bench_results_volume,
    completions_volume,
    hf_datasets_volume,
    lmsys_chat_1m_volume,
    proxies,
    replicas,
)
from gorgo.measure import (
    NS_PER_S,
    consume_sse_stream,
    summarize_samples,
)
from gorgo.policy import (
    POLICY_REGISTRY,
    ROUTING_POLICIES,
    PolicyDef,
    ReplicaSnapshot,
    RouteContext,
    normalize_policy,
)
from gorgo.policy.gorgo import (
    ALLOWED_HYPERPARAM_KEYS,
    DEFAULT_GORGO_HYPERPARAMETERS,
    make_default_store,
    merge_update,
    prune_per_target,
    validate_update,
)
from gorgo.radix_trie import RadixTrie
from gorgo.tuner import (
    DEFAULT_TUNE_APPLY,
    DEFAULT_TUNE_HOP_SIZE,
    DEFAULT_TUNE_WINDOW_SIZE,
    HYPERPARAM_RANGES,
    ONLINE_SCORE_FUNCTIONS,
    SUPPORTED_AUTO_TUNE_MODES,
    GaussianESTuner,
    accumulate_calibration,
    calibrated_rates_payload,
    new_calibration_state,
    validated_ranges,
)

# Match engine/modal_sglang.py so the proxy lands in the same datacenter as
# its replicas (and as the workload generator in proxy/workload.py).
REGION = os.getenv("REGION", "us-east")

DEFAULT_POLICY = "random"
METRICS_REFRESH_INTERVAL_SECONDS = float(os.getenv("METRICS_REFRESH_INTERVAL_SECONDS", 30.0))
METRICS_FETCH_TIMEOUT_SECONDS = 2.0
# SGLang may wait until idle; allow a generous read window for POST /flush_cache.
FLUSH_UPSTREAM_TIMEOUT_SECONDS = 120.0

# Known engine regions used across single-proxy and matrix runs. We use this
# to derive ``replica_region`` from ``replicas`` registry keys when possible.
KNOWN_REPLICA_REGIONS: tuple[str, ...] = (
    "ap-seoul-1",
    "eu-frankfurt-1",
    "us-ashburn-1",
    "CANADA-2",
    "sines-2",
    "us-west4",
    "centralus",
    "northeurope",
    "malaysiawest",
    "us-east",
)


def _infer_replica_region_from_key(registry_key: str) -> str | None:
    """Best-effort region extraction from a replica registry key.

    Keys are either plain regions (e.g. ``ap-seoul-1``) or experiment-scoped
    names ending in a region suffix (e.g. ``<prefix>-<policy>-ap-seoul-1``).
    """
    key = (registry_key or "").strip()
    if not key:
        return None
    if key in KNOWN_REPLICA_REGIONS:
        return key
    for region in sorted(KNOWN_REPLICA_REGIONS, key=len, reverse=True):
        if key.endswith(f"-{region}"):
            return region
    return None


SCORE_FUNCTIONS: dict[str, Callable[[dict], float]] = {
    "output_throughput": lambda s: float(s["output_token_throughput"]),
    "total_throughput": lambda s: float(s["total_token_throughput"]),
    "request_throughput": lambda s: float(s["request_throughput_rps"]),
    "neg_p95_ttft": lambda s: -float(s["ttft_seconds"]["p95"]),
    "neg_p99_ttft": lambda s: -float(s["ttft_seconds"]["p99"]),
    "neg_p95_e2e": lambda s: -float(s["request_e2e_seconds"]["p95"]),
    "neg_avg_itl": lambda s: -float(s["itl_ms"]["avg"]),
}


# The online tuning metric functions (``ONLINE_SCORE_FUNCTIONS``), the
# ``GaussianESTuner``, and the calibration accumulator now live in
# ``gorgo.tuner`` (imported above) so any router embedding the gorgo
# package can tune on live traffic. ``SCORE_FUNCTIONS`` above stays
# here: it scores workload-level stats dicts from ``proxy.workload_core``
# and must never share a metric name with divergent semantics.


def build_tuner(
    *,
    initial_params: dict[str, float],
    ranges: dict[str, tuple[float, float]],
    max_steps: int,
    relative_tolerance: float,
    sigma: float,
    sigma_min: float,
    seed: int | None,
) -> GaussianESTuner:
    return GaussianESTuner(
        initial_params=initial_params,
        ranges=ranges,
        sigma=sigma,
        sigma_min=sigma_min,
        tol=relative_tolerance,
        max_steps=max_steps,
        seed=seed,
    )


def build_summary(
    *,
    run_started_at: datetime,
    proxy_url: str,
    workload_kwargs: dict,
    metric: str,
    tuner: GaussianESTuner,
    baseline_score: float | None,
    history: list[dict],
    active_policy: str | None,
    ranges: dict[str, tuple[float, float]],
    finished_at: datetime | None,
    output_path: str,
) -> dict:
    return {
        "started_at": run_started_at.isoformat().replace("+00:00", "Z"),
        "finished_at": finished_at.isoformat().replace("+00:00", "Z") if finished_at else None,
        "proxy_url": proxy_url,
        "active_policy": active_policy,
        "metric": metric,
        "workload": workload_kwargs,
        "tuning": {
            "algorithm": tuner.name,
            "ranges": {k: list(v) for k, v in ranges.items()},
            "max_steps": tuner.max_steps,
            "relative_tolerance": tuner.tol,
            "trials_run": len(history),
            "current_state": dict(tuner.state),
        },
        "best_params": dict(tuner.best_params),
        "best_score": tuner.best_score,
        "baseline_score": baseline_score,
        "improvement_over_baseline": (
            (tuner.best_score - baseline_score) / abs(baseline_score)
            if baseline_score not in (None, 0)
            else None
        ),
        "history": history,
        "output_path": output_path,
    }


# On-the-fly tuning. The proxy keeps a bounded ring buffer of per-request
# samples (TTFT / total / token counts / per-token rates); when the
# auto-tuner is enabled (``POST /tune``) it sliding-window-recomputes
# ``gorgo`` hyperparameters from this buffer via
# ``proxy.measure.recommend_rates`` -- the same primitive
# ``proxy/calibrate.py`` uses for one-shot calibration.
#
# The auto-tuner is a stateful toggle, *not* a one-shot endpoint: once
# enabled it keeps recomputing every ``hop_size`` new samples until
# explicitly disabled (``POST /tune {"enabled": false}``). Subsequent
# POSTs while it's running atomically reconfigure the live tuner.
MAX_REQUEST_SAMPLES = 1000

# Headers that are connection-local and must not be forwarded verbatim when
# we proxy upstream responses back to the client. Re-emitting these would
# either confuse uvicorn (e.g. it wants to manage ``transfer-encoding`` and
# ``content-length`` itself when we stream chunks) or leak upstream-specific
# keep-alive semantics to the client. We ask the upstream not to compress
# via ``accept-encoding: identity`` so ``content-encoding`` is also safe to
# drop unconditionally.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "content-encoding",
        "content-length",
    }
)

_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-35B-A3B-FP8",
            revision="0b2752837483aa34b3db6e83e151b150c0e00e49",
            trust_remote_code=False,
        )
    return _TOKENIZER


def _message_text(content) -> str:
    """Extract the text payload from an OpenAI chat-completions ``content`` field.

    ``content`` is either a string, or a list of content blocks
    (``{"type": "text", "text": "..."}``, ``{"type": "image_url", ...}``, ...).
    Non-text blocks are dropped -- they don't contribute to the input token
    count we care about for routing decisions.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "") or ""
                    if text:
                        parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def tokenize_input(messages: list[dict]) -> list[int]:
    """Tokenize an OpenAI chat-completions ``messages`` payload into a flat
    list of token ids using the model's own tokenizer.

    Uses ``apply_chat_template`` with ``add_generation_prompt=True``
    so the token count and IDs match SGLang's server-side tokenization
    exactly. Returns an empty list if the template fails (the request
    would be rejected by SGLang for the same reason).
    """
    if not isinstance(messages, list) or not messages:
        return []
    tok = _get_tokenizer()
    try:
        rendered = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return tok.encode(rendered)
    except Exception:
        return []


def _parse_metrics_text(text: str) -> dict[str, float]:
    """Parse a Prometheus exposition snippet into ``{metric_name: value}``,
    dropping label suffixes and skipping non-numeric or commented lines."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        value = parts[1]
        if (
            not value.replace(".", "", 1)
            .replace("e+", "", 1)
            .replace("e-", "", 1)
            .lstrip("-")
            .isdigit()
        ):
            continue
        out[parts[0].split("{")[0]] = float(value)
    return out


@app.function(
    image=modal.Image.debian_slim()
    .pip_install("httpx[http2]", "uvicorn", "transformers", "jinja2", "pyarrow", "datasets>=3.0")
    .add_local_python_source("app", "proxy", "policy", "utils"),
    region=REGION,
    timeout=(24 * 60 * 60),
    volumes={
        "/data": completions_volume,
        "/lmsys": lmsys_chat_1m_volume,
        "/datasets": hf_datasets_volume,
        "/results": bench_results_volume,
    },
)
def proxy(registry_key: str = ""):
    import time

    import httpx
    import uvicorn

    replica_urls: list[str] = []
    # Stable per-replica identity metadata keyed by URL. ``replica_key`` is the
    # registration key (when known); ``replica_region`` is the actual engine
    # region for that URL.
    replica_url_meta: dict[str, dict[str, str | None]] = {}

    # Routing state. Kept in a dict so the asgi_app closure can mutate it in
    # place from the /policy handler; uvicorn is single-process / single-loop
    # so we don't need a lock (all writes happen on the asyncio thread).
    #
    # ``upstream_client`` is a long-lived httpx.AsyncClient shared across all
    # handlers (metrics refresh + chat completions). Keeping it alive means
    # TCP + TLS handshakes to each replica happen once and keepalive
    # connections are reused for every subsequent request, which is a big
    # TTFT win on cold proxies. Created in lifespan.startup so it binds to
    # the right event loop; closed in lifespan.shutdown.
    state: dict = {
        "policy": DEFAULT_POLICY,
        # Structured GORGO hyperparameter store (see policy/gorgo.py).
        # ``defaults`` applies to every replica; ``per_target`` keys
        # specific replicas to override individual scalars. The auto-
        # tuner writes per-target overrides automatically once enough
        # live samples accumulate per replica; offline tools (calibrate,
        # tuning.py) typically write only ``defaults``.
        "hyperparameters": make_default_store(),
        # A/B switch for the in-flight load-signal semantics. When False
        # (default), the queue+prefill load counters
        # (``endpoints_queued_tokens`` / ``endpoints_queued_uncached_tokens``
        # / ``endpoints_inflight_requests``) are released at end-of-decode
        # in the request's ``finally`` (all-stages load, the original
        # behavior). When True, they are released at first token so the
        # signal represents queue+prefill load only. Toggled at runtime via
        # ``POST /config`` and stamped onto each request trace event so a
        # calibration run can tell which semantics produced it.
        "load_release_at_ttft": False,
        # Live physical-rate calibration accumulator (regression, not ratios).
        # We fit ``ttft_ms ~ intercept_r + P*uncached + Q*queued`` by ordinary
        # least squares using ONLY proxy-measured TTFT and the proxy's own
        # ``queued_tokens_at_dispatch`` -- no engine meta_info. P (prefill_rate,
        # ms/uncached-tok) and Q (queue_rate, ms/queued-tok) are shared physical
        # constants; the per-replica intercept soaks up RTT + fixed overhead so
        # it doesn't leak into P (a raw TTFT/uncached ratio explodes because it
        # mis-attributes queue+RTT to prefill). We keep online sufficient
        # statistics (normal-equation aggregates) so the fit is O(1) per request
        # and solved on demand in ``_calibrated_rates_payload``. Read via
        # GET /calibrated_rates.
        "calibration": new_calibration_state(),
        "upstream_client": None,
        "metrics_task": None,
        # Total samples appended over the proxy lifetime. Doesn't
        # saturate at ``MAX_REQUEST_SAMPLES`` (unlike ``len(samples)``)
        # so it can be diffed across snapshots to count arrivals.
        "total_samples_appended": 0,
        # Live auto-tuner config. Reconfigured atomically by ``POST
        # /tune``; consumed by ``_record_request_sample`` after every
        # successful sample append. ``enabled=False`` means the
        # background recompute is dormant and the rest of the dict is
        # frozen at its last value.
        "auto_tune": {
            "enabled": False,
            "window_size": DEFAULT_TUNE_WINDOW_SIZE,
            "hop_size": DEFAULT_TUNE_HOP_SIZE,
            "apply": DEFAULT_TUNE_APPLY,
            # Counter zeroed on enable / on apply, incremented per
            # sample. Triggers a recompute when it crosses ``hop_size``.
            "samples_since_last_apply": 0,
            # Diagnostics surfaced by GET /tune.
            "applied_count": 0,
            "last_applied_at_monotonic": None,
            "last_recommendation": None,
            "enabled_at_monotonic": None,
            # Tuning mode:
            #   "online-es" -- treat the dimensionless weights as knobs and use
            #                  Gaussian-(1+1)-ES to minimize the configured
            #                  ``objective_metric`` over the rolling window.
            #   "calibrate" -- accumulate the physical-rate regression (see
            #                  ``_accumulate_calibration``); never writes weights.
            "mode": "online-es",
            "objective_metric": "neg_p95_ttft",
            "online_tuner": None,
            "online_state": None,
            # Currently-applied candidate; ``None`` until the first ES
            # apply. Lifecycle:
            #   1. ES proposes -> applied to ``hyperparameters.defaults``,
            #      counter zeroed, ``pending_candidate`` set.
            #   2. Wait for ``window_size`` new samples to land.
            #   3. Score the window, report to ES, ES updates incumbent.
            #   4. Propose next candidate -> back to (1).
            "pending_candidate": None,
            "pending_started_at_count": 0,
            "last_score": None,
        },
        "batch_tuning": {
            "running": False,
            "status": "idle",
            "task": None,
            "run_id": None,
            "started_at": None,
            "finished_at": None,
            "config": None,
            "trial": None,
            "history": [],
            "best_params": None,
            "best_score": None,
            "baseline_score": None,
            "output_path": None,
            "error": None,
            "stop_requested": False,
        },
        "workload_run": {
            "running": False,
            "status": "idle",
            "phase": "idle",
            "task": None,
            "run_id": None,
            "started_at": None,
            "finished_at": None,
            "config": None,
            "stats": None,
            "output_path": None,
            "error": None,
            "stop_requested": False,
        },
        "trace": {
            "enabled": False,
            "trace_id": None,
            "sample_metrics": True,
            "sample_requests": True,
            "max_events": 200_000,
            "started_at": None,
            "stopped_at": None,
            "dropped_metrics": 0,
            "dropped_requests": 0,
            "dropped_tune": 0,
            "saved_paths": None,
        },
    }
    endpoints_queued_tokens: dict[str, int] = {url: 0 for url in replica_urls}
    # Cache-aware load counter: increments by the request's uncached tokens on
    # the chosen replica at dispatch time, decrements by the same amount on
    # completion. Used by the 2D GORGO variant.
    endpoints_queued_uncached_tokens: dict[str, int] = {url: 0 for url in replica_urls}
    # Per-target proxy-side in-flight request counter; bumped on every
    # dispatch, decremented in finally on every completion / error path.
    # Mirrors ``endpoints_queued_tokens`` but counts requests rather than
    # tokens. Read by ``route_least_request`` (see policy/lb_aibrix.py)
    # so the policy's score remains correct between SGLang metrics
    # scrapes -- otherwise the score is frozen for a full
    # ``METRICS_REFRESH_INTERVAL_SECONDS`` and every request in the
    # window herds onto the snapshot's minimum.
    endpoints_inflight_requests: dict[str, int] = {url: 0 for url in replica_urls}

    # Bounded ring buffer of per-request samples produced by the
    # SSE-tee in ``_handle_chat_completions``. Each entry has the same
    # shape as ``proxy.measure.measure_chat_completion`` returns, plus
    # ``target`` (chosen replica) and ``recorded_at_monotonic`` so /tune
    # consumers can do their own time-windowing if desired.
    samples: deque[dict] = deque(maxlen=MAX_REQUEST_SAMPLES)
    metrics_trace_events: deque[dict] = deque(maxlen=state["trace"]["max_events"])
    request_trace_events: deque[dict] = deque(maxlen=state["trace"]["max_events"])
    tune_trace_events: deque[dict] = deque(maxlen=state["trace"]["max_events"])

    # Live radix trie of every prompt we've forwarded. Each node along a
    # sequence's insertion path is tagged with the replica URL that received
    # that prompt, so the trie doubles as an approximate "which replica
    # currently has this prefix cached in KV" index. Updated inline in the
    # /v1/chat/completions handler after the upstream request is dispatched.
    #
    # Note: currently unbounded; a long-running proxy will accumulate state
    # until the container is recycled. That's fine for the evaluation setup;
    # a production deployment would need LRU/TTL eviction.
    radix_trie = RadixTrie()

    # Default timeout used by the shared client. ``read=None`` is critical for
    # chat completions: httpx treats ``read`` as "seconds between bytes", so a
    # generation that thinks for a while before emitting its first token would
    # otherwise trip the timeout. Metrics calls pass their own shorter
    # override via the per-request ``timeout=`` kwarg.
    default_upstream_timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
    # Keep enough warm connections around to saturate concurrent load without
    # re-handshaking. ``keepalive_expiry=None`` means "never expire idle
    # connections"; Modal will eventually tear down the container anyway.
    upstream_limits = httpx.Limits(
        max_connections=200, max_keepalive_connections=100, keepalive_expiry=None
    )

    # Live mirror of each replica's /metrics output. A background task refreshes
    # this every ``METRICS_REFRESH_INTERVAL_SECONDS``; policy functions read
    # snapshots of it per request instead of fetching synchronously.
    live_metrics: dict[str, ReplicaSnapshot] = {}
    metrics_meta: dict = {
        "last_refresh_monotonic": 0.0,
        "last_refresh_wall_ts": None,
        "refresh_seq": 0,
        "last_refresh_errors": {},  # url -> str
    }
    # EWMA-smoothed pure-network RTT per replica, refreshed by ``_probe_rtt``
    # alongside the metrics scrape. Kept outside ``ReplicaSnapshot`` so the
    # smoothed value survives across snapshots even when a single probe fails.
    network_rtt_ewma: dict[str, float] = {}
    NETWORK_RTT_EWMA_ALPHA = 0.3
    NETWORK_RTT_PROBE_TIMEOUT_SECONDS = 2.0

    def _registry_from_items(items) -> dict[str, str]:
        registry: dict[str, str] = {}
        for key, value in items:
            if not isinstance(key, str):
                key = str(key)
            registry[key] = value if isinstance(value, str) else ""
        return registry

    def _active_urls_from_registry(
        registry: dict[str, str],
    ) -> tuple[list[str], dict[str, dict[str, str | None]]]:
        seen: set[str] = set()
        normalized: list[str] = []
        url_meta: dict[str, dict[str, str | None]] = {}
        for key, url in registry.items():
            url = url.strip().rstrip("/")
            if not url or not (url.startswith("http://") or url.startswith("https://")):
                continue
            if url in seen:
                continue
            seen.add(url)
            normalized.append(url)
            url_meta[url] = {
                "replica_key": key,
                "replica_region": _infer_replica_region_from_key(key),
            }
        return normalized, url_meta

    def _replace_replica_urls(
        normalized: list[str],
        *,
        source: str,
        url_metadata: dict[str, dict[str, str | None]] | None = None,
    ) -> tuple[list[str], list[str]]:
        old = set(replica_urls)
        new = set(normalized)
        added = sorted(new - old)
        removed = sorted(old - new)
        changed_urls = bool(added or removed or list(replica_urls) != normalized)

        if changed_urls:
            replica_urls.clear()
            replica_urls.extend(normalized)
            for url in added:
                endpoints_queued_tokens[url] = 0
                endpoints_queued_uncached_tokens[url] = 0
                endpoints_inflight_requests[url] = 0
            for url in removed:
                endpoints_queued_tokens.pop(url, None)
                endpoints_queued_uncached_tokens.pop(url, None)
                endpoints_inflight_requests.pop(url, None)
                live_metrics.pop(url, None)
                metrics_meta["last_refresh_errors"].pop(url, None)
                replica_url_meta.pop(url, None)
            prune_per_target(state["hyperparameters"], set(replica_urls))
            _log(
                f"replicas synced from {source}: "
                f"+{len(added)} -{len(removed)} (total={len(replica_urls)})"
            )

        incoming_meta = url_metadata or {}
        for url in normalized:
            prev = dict(replica_url_meta.get(url) or {})
            cur = incoming_meta.get(url) or {}
            # Preserve old identity when no new value is provided.
            key = cur.get("replica_key") or prev.get("replica_key")
            region = cur.get("replica_region") or prev.get("replica_region")
            replica_url_meta[url] = {"replica_key": key, "replica_region": region}
        return added, removed

    def _sync_replicas_from_modal_dict() -> tuple[dict[str, str], list[str], list[str]]:
        registry = _registry_from_items(replicas.items())
        urls, url_meta = _active_urls_from_registry(registry)
        added, removed = _replace_replica_urls(
            urls,
            source="modal dict",
            url_metadata=url_meta,
        )
        return registry, added, removed

    async def _read_registry_async() -> dict[str, str]:
        """Read the global ``replicas`` modal Dict without mutating local state.

        Used by ``GET /replicas`` so callers can observe the registry
        contents for debugging without triggering a sync that would
        clobber a controller-set ``replica_urls`` (matrix mode pins its
        per-policy 3-replica list via POST /replicas; a concurrent GET
        that re-synced from the global Dict would replace the pin with
        whatever the last writer left there).
        """
        items = []
        async for item in replicas.items.aio():
            items.append(item)
        return _registry_from_items(items)

    async def _sync_replicas_from_modal_dict_async() -> tuple[dict[str, str], list[str], list[str]]:
        registry = await _read_registry_async()
        urls, url_meta = _active_urls_from_registry(registry)
        added, removed = _replace_replica_urls(
            urls,
            source="modal dict",
            url_metadata=url_meta,
        )
        return registry, added, removed

    def _sync_replicas_from_manual_urls(
        normalized: list[str],
        *,
        url_metadata: dict[str, dict[str, str | None]] | None = None,
    ) -> tuple[list[str], list[str]]:
        return _replace_replica_urls(
            normalized,
            source="/replicas",
            url_metadata=url_metadata,
        )

    # Experiment proxies are configured explicitly by the controller with
    # POST /replicas. Do not ingest every URL in the global replica registry on
    # startup, otherwise a one-fleet smoke proxy can temporarily see unrelated
    # replicas before the controller narrows it back down.
    if not registry_key:
        _sync_replicas_from_modal_dict()

    # ---------- ASGI helpers ----------

    async def _read_json_body(receive) -> dict | list:
        """Drain the ASGI request body and parse it as JSON.

        Combines body assembly with JSON decode so handler call sites stay
        a single ``try / except`` instead of two. Returns ``{}`` for an
        empty body. Raises ``json.JSONDecodeError`` on malformed input
        (handlers convert that to a 400).
        """
        chunks: list[bytes] = []
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                continue
            chunks.append(msg.get("body", b"") or b"")
            if not msg.get("more_body"):
                break
        body = b"".join(chunks)
        return json.loads(body.decode()) if body else {}

    async def _send_json(send, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _now_wall_ts() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _trace_append(kind: str, event: dict) -> None:
        tr = state["trace"]
        if not tr["enabled"]:
            return
        if kind == "metrics":
            if not tr["sample_metrics"]:
                return
            target = metrics_trace_events
            dropped_key = "dropped_metrics"
        elif kind == "request":
            if not tr["sample_requests"]:
                return
            target = request_trace_events
            dropped_key = "dropped_requests"
        elif kind == "tune":
            target = tune_trace_events
            dropped_key = "dropped_tune"
        else:
            return
        if target.maxlen is not None and len(target) >= target.maxlen:
            tr[dropped_key] += 1
        target.append(event)

    def _trace_status_payload() -> dict:
        tr = state["trace"]
        first_ts = None
        last_ts = None
        for buf in (metrics_trace_events, request_trace_events, tune_trace_events):
            if not buf:
                continue
            b_first = buf[0].get("wall_ts")
            b_last = buf[-1].get("wall_ts")
            first_ts = b_first if first_ts is None else min(first_ts, b_first)
            last_ts = b_last if last_ts is None else max(last_ts, b_last)
        return {
            **tr,
            "metrics_events": len(metrics_trace_events),
            "request_events": len(request_trace_events),
            "tune_events": len(tune_trace_events),
            "first_event_ts": first_ts,
            "last_event_ts": last_ts,
        }

    def _write_jsonl(path: str, rows: deque[dict]) -> None:
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")

    def _compute_fallback_summary() -> dict:
        """Walk the in-memory request trace buffer and tally how often the
        configured policy actually selected the target vs how often
        ``_select_endpoint`` (or a policy-internal precondition failure)
        fell back to ``random.choice``.

        Returns a JSON-serializable dict shaped like::

          {
            "total_requests": int,
            "fallback_count": int,
            "fallback_rate": float,         # 0.0 - 1.0
            "by_effective_policy": {        # only fallback rows
              "random-fallback:missing-metrics": int,
              "random-fallback:internal:least-request:empty-candidates": int,
              ...
            }
          }

        Counts only rows where ``effective_policy != policy`` -- i.e.
        rows whose configured-policy attribution is misleading. The
        ``"single-replica"`` short-circuit is a degenerate case (no
        choice to make) and is *not* counted as a fallback even though
        ``effective_policy != policy``; it's surfaced separately under
        ``single_replica_count``.
        """
        total = 0
        fallback = 0
        single_replica = 0
        by_eff: dict[str, int] = {}
        for row in request_trace_events:
            total += 1
            policy = row.get("policy")
            eff = row.get("effective_policy")
            if eff is None or eff == policy:
                continue
            if eff == "single-replica":
                single_replica += 1
                continue
            fallback += 1
            by_eff[eff] = by_eff.get(eff, 0) + 1
        return {
            "total_requests": total,
            "fallback_count": fallback,
            "fallback_rate": (fallback / total) if total else 0.0,
            "single_replica_count": single_replica,
            "by_effective_policy": by_eff,
        }

    def _compute_workload_stats_from_trace() -> dict:
        """Compute TTFT/E2E stats from the in-memory request trace buffer.

        Called on workload cancellation or failure so the manifest gets
        partial stats instead of ``null``. Mirrors the summary shape
        that ``run_replay_async`` produces on success, minus fields
        that require the workload's internal bookkeeping (throughput,
        progress log, per-request saved JSON).
        """
        ok_rows = [
            r for r in request_trace_events if r.get("kind") == "request" and r.get("status") == 200
        ]
        fail_rows = [
            r
            for r in request_trace_events
            if r.get("kind") == "request" and r.get("status") and r["status"] != 200
        ]
        n = len(ok_rows) + len(fail_rows)
        ttfts = [r["ttft_ns"] / NS_PER_S for r in ok_rows if r.get("ttft_ns")]
        e2es = [r["total_ns"] / NS_PER_S for r in ok_rows if r.get("total_ns")]

        def _pct(xs, p):
            if not xs:
                return 0.0
            s = sorted(xs)
            return s[max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))]

        def _stat_block(xs):
            if not xs:
                return {
                    "avg": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "n": 0,
                }
            return {
                "avg": sum(xs) / len(xs),
                "min": min(xs),
                "max": max(xs),
                "p50": _pct(xs, 0.50),
                "p95": _pct(xs, 0.95),
                "p99": _pct(xs, 0.99),
                "n": len(xs),
            }

        return {
            "partial": True,
            "n": n,
            "ok": len(ok_rows),
            "fail": len(fail_rows),
            "success_rate": len(ok_rows) / n if n else 0.0,
            "ttft_seconds": _stat_block(ttfts),
            "request_e2e_seconds": _stat_block(e2es),
        }

    def _save_trace_to_volume() -> dict:
        tr = state["trace"]
        trace_id = tr["trace_id"] or f"trace-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        out_dir = os.path.join("/results", "proxy_traces", trace_id)
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = os.path.join(out_dir, "metrics.jsonl")
        requests_path = os.path.join(out_dir, "requests.jsonl")
        tune_path = os.path.join(out_dir, "tune.jsonl")
        manifest_path = os.path.join(out_dir, "manifest.json")
        _write_jsonl(metrics_path, metrics_trace_events)
        _write_jsonl(requests_path, request_trace_events)
        _write_jsonl(tune_path, tune_trace_events)
        fallback_summary = _compute_fallback_summary()
        manifest = {
            "trace": _trace_status_payload(),
            "metrics_path": metrics_path,
            "requests_path": requests_path,
            "tune_path": tune_path,
            "fallback_summary": fallback_summary,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        bench_results_volume.commit()
        paths = {
            "metrics_path": metrics_path,
            "requests_path": requests_path,
            "tune_path": tune_path,
            "manifest_path": manifest_path,
        }
        tr["saved_paths"] = paths
        tr["fallback_summary"] = fallback_summary
        return paths

    # ---------- Upstream client + metrics refresh ----------

    def _new_upstream_client() -> httpx.AsyncClient:
        # ``http2=True`` negotiates HTTP/2 via ALPN on HTTPS replicas; if the
        # upstream (SGLang-on-uvicorn) only speaks HTTP/1.1, httpx transparently
        # falls back. Either way we keep the keep-alive pool benefit.
        return httpx.AsyncClient(
            http2=True,
            timeout=default_upstream_timeout,
            limits=upstream_limits,
        )

    async def _probe_rtt(client: httpx.AsyncClient, url: str) -> float | None:
        """Lightweight RTT probe to a replica's base URL.

        Issues a ``GET /`` and times the round-trip. SGLang typically
        404s the root path (no router for ``/``), which is fine here --
        we want pure HTTP RTT, not a useful response. Returns ``None``
        on failure so the caller skips the EWMA update for this tick.

        Reuses the upstream ``client``'s connection pool, so steady-state
        we measure data RTT (no TCP/TLS handshake), matching what real
        chat-completions requests pay.
        """
        t0 = time.monotonic()
        try:
            await client.get(url + "/", timeout=NETWORK_RTT_PROBE_TIMEOUT_SECONDS)
        except Exception:
            return None
        return time.monotonic() - t0

    async def _refresh_one(client: httpx.AsyncClient, url: str, seq: int) -> None:
        """Scrape one replica's /metrics + probe its RTT into ``live_metrics[url]``."""
        t0 = time.monotonic()
        wall_ts = _now_wall_ts()
        # Run the metrics scrape and the lightweight RTT probe concurrently.
        # The probe is independent of /metrics handler load (which is what
        # makes ``snap.latency`` a noisy stand-in for pure network RTT).
        scrape_task = asyncio.create_task(
            client.get(f"{url}/metrics", timeout=METRICS_FETCH_TIMEOUT_SECONDS)
        )
        probe_task = asyncio.create_task(_probe_rtt(client, url))
        try:
            resp = await scrape_task
            resp.raise_for_status()
        except Exception as e:
            # Cancel the in-flight probe defensively if the scrape failed
            # before it completed (we still record whatever the probe got).
            try:
                rtt = await probe_task
            except Exception:
                rtt = None
            if rtt is not None:
                prev = network_rtt_ewma.get(url)
                network_rtt_ewma[url] = (
                    rtt
                    if prev is None
                    else NETWORK_RTT_EWMA_ALPHA * rtt + (1 - NETWORK_RTT_EWMA_ALPHA) * prev
                )
            latency = time.monotonic() - t0
            metrics_meta["last_refresh_errors"][url] = repr(e)
            replica_meta = replica_url_meta.get(url) or {}
            replica_region = replica_meta.get("replica_region")
            _trace_append(
                "metrics",
                {
                    "kind": "metrics",
                    "trace_id": state["trace"]["trace_id"],
                    "seq": seq,
                    "wall_ts": wall_ts,
                    "monotonic_s": t0,
                    "replica_url": url,
                    "replica_key": replica_meta.get("replica_key"),
                    "replica_region": replica_region,
                    "proxy_region": REGION,
                    "region": replica_region or REGION,
                    "scrape_latency_seconds": latency,
                    "network_rtt_seconds": network_rtt_ewma.get(url),
                    "ok": False,
                    "num_running_reqs": None,
                    "num_queue_reqs": None,
                    "num_used_tokens": None,
                    "utilization": None,
                    "gen_throughput": None,
                    "error": repr(e),
                },
            )
            return
        latency = time.monotonic() - t0
        try:
            rtt = await probe_task
        except Exception:
            rtt = None
        if rtt is not None:
            prev = network_rtt_ewma.get(url)
            network_rtt_ewma[url] = (
                rtt
                if prev is None
                else NETWORK_RTT_EWMA_ALPHA * rtt + (1 - NETWORK_RTT_EWMA_ALPHA) * prev
            )
        parsed = _parse_metrics_text(resp.text)
        snap = ReplicaSnapshot(
            num_running_reqs=int(parsed.get("sglang:num_running_reqs", 0)),
            num_queue_reqs=int(parsed.get("sglang:num_queue_reqs", 0)),
            num_used_tokens=int(parsed.get("sglang:num_used_tokens", 0)),
            latency=latency,
            network_rtt=network_rtt_ewma.get(url, 0.0),
            gen_throughput=float(parsed.get("sglang:gen_throughput", 0.0)),
            utilization=float(parsed.get("sglang:utilization", 0.0)),
        )
        live_metrics[url] = snap
        metrics_meta["last_refresh_errors"].pop(url, None)
        replica_meta = replica_url_meta.get(url) or {}
        replica_region = replica_meta.get("replica_region")
        _trace_append(
            "metrics",
            {
                "kind": "metrics",
                "trace_id": state["trace"]["trace_id"],
                "seq": seq,
                "wall_ts": wall_ts,
                "monotonic_s": t0,
                "replica_url": url,
                "replica_key": replica_meta.get("replica_key"),
                "replica_region": replica_region,
                "proxy_region": REGION,
                "region": replica_region or REGION,
                "scrape_latency_seconds": latency,
                "network_rtt_seconds": snap.network_rtt,
                "ok": True,
                "num_running_reqs": snap.num_running_reqs,
                "num_queue_reqs": snap.num_queue_reqs,
                "num_used_tokens": snap.num_used_tokens,
                "utilization": snap.utilization,
                "gen_throughput": snap.gen_throughput,
                "error": None,
            },
        )

    async def _refresh_all(client: httpx.AsyncClient | None) -> None:
        """One pass: refresh every registered replica in parallel."""
        # Auto-discover from the global Dict only when this proxy isn't
        # under explicit controller control. Matrix-mode proxies
        # (``registry_key`` set) pin their 3-replica list via POST
        # /replicas; auto-discovering would re-read every engine in the
        # global Dict and re-introduce the cross-policy clobber. Same
        # gate as the synchronous startup prime at module load time.
        if not replica_urls and not registry_key:
            await _sync_replicas_from_modal_dict_async()
        if client is None or not replica_urls:
            return
        seq = metrics_meta["refresh_seq"] + 1
        await asyncio.gather(
            *[_refresh_one(client, url, seq) for url in replica_urls],
            return_exceptions=True,
        )
        metrics_meta["refresh_seq"] = seq
        metrics_meta["last_refresh_monotonic"] = time.monotonic()
        metrics_meta["last_refresh_wall_ts"] = _now_wall_ts()

    async def _metrics_refresh_loop() -> None:
        """Background task: refresh every ``METRICS_REFRESH_INTERVAL_SECONDS``.
        Cancelled in lifespan.shutdown. Per-iteration exceptions are
        logged so a single transient failure doesn't kill the loop."""
        try:
            while True:
                try:
                    await _refresh_all(state["upstream_client"])
                except Exception as e:
                    _log(f"metrics refresh iteration failed: {e}")
                await asyncio.sleep(METRICS_REFRESH_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            pass

    async def _flush_upstream_replica(client: httpx.AsyncClient, base_url: str) -> tuple[str, dict]:
        url = f"{base_url.rstrip('/')}/flush_cache"
        try:
            resp = await client.post(
                url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=FLUSH_UPSTREAM_TIMEOUT_SECONDS,
                    write=30.0,
                    pool=10.0,
                ),
            )
            out = {"ok": resp.is_success, "status_code": resp.status_code}
            if not resp.is_success:
                body = (resp.text or "")[:512]
                if body:
                    out["body_preview"] = body
            return base_url, out
        except Exception as e:
            return base_url, {"ok": False, "error": repr(e)}

    # ---------- Routing ----------

    def _select_endpoint(token_ids: list[int]) -> tuple[str, str, str, dict[str, float] | None]:
        """Pick an upstream URL using the policy registry in :mod:`policy`.

        Returns ``(target, configured, effective_policy, scores)``.

        * ``configured`` is the policy name read from ``state["policy"]``
          atomically at the start of the call. The caller writes this
          (rather than re-reading state) into the trace event's
          ``policy`` field so a concurrent ``POST /policy`` mid-request
          can't produce a row whose ``policy`` and ``effective_policy``
          disagree just from a race.
        * ``effective_policy`` is what actually selected ``target``:
          ``configured`` when the policy function ran successfully,
          ``"single-replica"`` when there was no choice to make, or
          ``"random-fallback:<reason>"`` when this function bypassed the
          policy and picked a random replica. Distinct values let
          post-hoc trace analysis filter fallback rows out of per-policy
          aggregates -- otherwise a brief metrics-missing window
          silently biases the comparison toward random's distribution.

        For policies that don't need ``/metrics`` data (random, pd*,
        simple-session-affinity) we skip the snapshot entirely so the
        proxy can keep routing during a metrics-refresh outage. For
        metrics-using policies we filter ``live_metrics`` to
        currently-registered replicas; if any replica has no snapshot
        yet (cold start, /metrics timeout) we fall back to random
        rather than passing a partial view to the policy.

        Policy-level fallbacks are also captured: when a policy fn
        signals an internal random fallback by returning
        ``RouteDecision(target, fallback_reason="...")`` (see
        :class:`policy.base.RouteDecision`), this function emits
        ``"random-fallback:internal:<configured>:<reason>"`` instead
        of ``configured``. Reasons are short kebab-case strings
        defined by the policy module (see ``policy/lb_aibrix.py`` for
        the vocabulary).
        """
        configured = state["policy"]
        if not replica_urls:
            raise ValueError("no replicas configured")
        if len(replica_urls) == 1:
            return replica_urls[0], configured, "single-replica", None

        pdef: PolicyDef = POLICY_REGISTRY[normalize_policy(configured)]

        if pdef.needs_metrics:
            # Filter to replicas with a live snapshot. Both this code and the
            # background refresh run on the same asyncio thread so there's
            # no race; the dict comprehension just drops missing entries.
            metrics = {u: live_metrics[u] for u in replica_urls if u in live_metrics}
            if len(metrics) < len(replica_urls):
                missing = len(replica_urls) - len(metrics)
                _log(
                    f"live metrics missing for {missing} replica(s); "
                    f"falling back to random for this request"
                )
                return (
                    random.choice(replica_urls),
                    configured,
                    "random-fallback:missing-metrics",
                    None,
                )
        else:
            metrics = {}

        decision = pdef.fn(
            RouteContext(
                replica_urls=replica_urls,
                metrics=metrics,
                endpoints_queued_tokens=endpoints_queued_tokens,
                endpoints_queued_uncached_tokens=endpoints_queued_uncached_tokens,
                endpoints_inflight_requests=endpoints_inflight_requests,
                radix_trie=radix_trie,
                token_ids=token_ids,
                request_tokens=len(token_ids),
                hyperparameters=state["hyperparameters"],
            )
        )
        if decision.fallback_reason is not None:
            return (
                decision.target,
                configured,
                f"random-fallback:internal:{configured}:{decision.fallback_reason}",
                decision.scores,
            )
        return decision.target, configured, configured, decision.scores

    # ---------- JSON route handlers ----------
    #
    # Each handler returns ``(status, payload)``; the dispatcher reads
    # the body, calls the handler, and serializes the response. Body
    # parsing / 400-on-invalid-json is handled by ``_dispatch_json`` so
    # individual handlers can focus on validation and side effects.
    #
    # All handlers are nested closures because they read / mutate state
    # captured in this ``proxy()`` invocation -- ``replicas``,
    # ``live_metrics``, ``samples``, ``radix_trie``,
    # ``endpoints_queued_tokens``, ``state``. Lifting them to module
    # scope would require threading a ``ProxyContext`` through every
    # call. The naming convention matches the other request-shaped
    # closures already living here (``_handle_chat_completions``,
    # ``_handle_lifespan``).

    def _policy_payload(*, include_supported: bool) -> dict:
        policy = state["policy"]
        uses_hyperparameters = normalize_policy(policy) == "gorgo"
        payload = {
            "policy": policy,
            "uses_hyperparameters": uses_hyperparameters,
            "hyperparameters": state["hyperparameters"] if uses_hyperparameters else None,
        }
        if include_supported:
            payload["supported"] = sorted(ROUTING_POLICIES)
        return payload

    async def _handle_get_policy(_data) -> tuple[int, dict]:
        return 200, _policy_payload(include_supported=True)

    async def _handle_post_policy(data) -> tuple[int, dict]:
        raw = (data or {}).get("policy") or (data or {}).get("name")
        if not isinstance(raw, str):
            return 400, {"error": "body must include string policy or name"}
        name = normalize_policy(raw)
        if name not in ROUTING_POLICIES:
            return 400, {
                "error": f"unknown policy {raw!r}",
                "supported": sorted(ROUTING_POLICIES),
            }
        state["policy"] = name
        _log(f"routing policy set to {name!r}")
        return 200, _policy_payload(include_supported=False)

    def _config_payload() -> dict:
        """Runtime config knobs that can be flipped without a redeploy.
        Currently just the load-signal A/B switch; shaped as a dict so
        more boolean/scalar knobs can be added without changing callers."""
        return {
            "load_release_at_ttft": bool(state.get("load_release_at_ttft")),
        }

    async def _handle_get_config(_data) -> tuple[int, dict]:
        return 200, {"config": _config_payload()}

    async def _handle_post_config(data) -> tuple[int, dict]:
        if not isinstance(data, dict):
            return 400, {"error": "body must be a JSON object"}
        if "load_release_at_ttft" in data:
            val = _parse_optional_bool(data, "load_release_at_ttft")
            if val is None:
                return 400, {"error": "load_release_at_ttft must be true/false"}
            state["load_release_at_ttft"] = val
            _log(f"config load_release_at_ttft set to {val}")
        return 200, {"config": _config_payload()}

    def _accumulate_calibration(
        *,
        target: str,
        uncached_at_dispatch: int,
        queued_at_dispatch: int,
        ttft_ms: float | None,
    ) -> None:
        """Fold one successful request into the online regression accumulator
        (``gorgo.tuner.accumulate_calibration``). Uses only the proxy-measured
        TTFT and the proxy's own dispatch counters -- no engine ``meta_info``
        (its prefill timings are null and ``queue_time`` is 0 in unified mode
        on this build)."""
        accumulate_calibration(
            state["calibration"],
            target=target,
            uncached_at_dispatch=uncached_at_dispatch,
            queued_at_dispatch=queued_at_dispatch,
            ttft_ms=ttft_ms,
        )

    def _calibrated_rates_payload() -> dict:
        """Solve the accumulated regression for the fleet-shared P/Q rates
        the sequencer patches into tuning + eval."""
        return calibrated_rates_payload(state["calibration"])

    async def _handle_get_calibrated_rates(_data) -> tuple[int, dict]:
        return 200, {"calibrated_rates": _calibrated_rates_payload()}

    async def _handle_get_replicas(_data) -> tuple[int, dict]:
        # Read-only: never mutate ``replica_urls`` from a GET. Used by
        # controllers (e.g. experiments/policy_matrix_app.py) to poll
        # readiness; if this re-synced from the global modal Dict it
        # would race against concurrent POST /replicas calls from sibling
        # proxies and clobber the per-policy pin -- the bug that made
        # all 9 matrix proxies converge on whichever 3 URLs the last
        # POST landed.
        registry = await _read_registry_async()
        return 200, {
            "replicas": list(replica_urls),
            "count": len(replica_urls),
            "replica_metadata": {u: replica_url_meta.get(u, {}) for u in replica_urls},
            "registry": registry,
        }

    async def _handle_post_replicas(data) -> tuple[int, dict]:
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("replicas") or data.get("endpoints")
        else:
            raw = None
        if not isinstance(raw, list):
            return 400, {
                "error": (
                    "body must be a JSON array of endpoint URLs/objects "
                    'or an object like {"replicas": [...]}'
                )
            }

        seen: set[str] = set()
        normalized: list[str] = []
        invalid: list[str] = []
        url_metadata: dict[str, dict[str, str | None]] = {}
        for entry in raw:
            replica_key: str | None = None
            replica_region: str | None = None
            if isinstance(entry, str):
                u = entry
            elif isinstance(entry, dict):
                u = (
                    entry.get("url")
                    or entry.get("replica_url")
                    or entry.get("endpoint")
                    or entry.get("target")
                )
                replica_key = entry.get("replica_key") or entry.get("key")
                replica_region = entry.get("replica_region") or entry.get("region")
                if replica_key is not None and not isinstance(replica_key, str):
                    invalid.append(str(entry))
                    continue
                if replica_region is not None and not isinstance(replica_region, str):
                    invalid.append(str(entry))
                    continue
            else:
                invalid.append(str(entry))
                continue
            if not isinstance(u, str):
                invalid.append(str(entry))
                continue
            u = u.strip().rstrip("/")
            if not u:
                continue
            if not (u.startswith("http://") or u.startswith("https://")):
                invalid.append(u)
                continue
            if u not in seen:
                seen.add(u)
                normalized.append(u)
            if replica_key or replica_region:
                url_metadata[u] = {
                    "replica_key": replica_key or None,
                    "replica_region": replica_region or None,
                }
        if invalid:
            return 400, {
                "error": "all endpoints must start with http:// or https://",
                "invalid": invalid,
            }

        # Update local state only. Do NOT mutate the global ``replicas``
        # modal Dict here: that Dict is shared across every proxy in the
        # environment, and the matrix experiment runs N proxies in
        # parallel each calling POST /replicas with their own per-policy
        # URLs. Writing back to the shared Dict caused each call to
        # clobber the previous one, and any sibling proxy that
        # subsequently re-synced (via the now-fixed GET /replicas)
        # would adopt whichever URLs the last writer left -- collapsing
        # all N policies onto the same 3 backends.
        added, removed = _sync_replicas_from_manual_urls(normalized, url_metadata=url_metadata)
        registry = await _read_registry_async()

        _log(f"replicas updated: +{len(added)} -{len(removed)} (total={len(replica_urls)})")
        return 200, {
            "replicas": list(replica_urls),
            "count": len(replica_urls),
            "replica_metadata": {u: replica_url_meta.get(u, {}) for u in replica_urls},
            "registry": registry,
            "added": added,
            "removed": removed,
        }

    async def _handle_get_trie(_data) -> tuple[int, dict]:
        # Summary stats only -- the full trie is too large to serialize on
        # every request. ``coverage`` counts how many nodes are tagged with
        # each replica URL: a useful sanity check that routing is producing
        # the prefix-sharing shape we expect.
        coverage: dict[str, int] = {url: 0 for url in replica_urls}
        stack = [radix_trie.root]
        tagged_nodes = 0
        while stack:
            node = stack.pop()
            if node.replica_endpoints:
                tagged_nodes += 1
                for url in node.replica_endpoints:
                    coverage[url] = coverage.get(url, 0) + 1
            stack.extend(node.children.values())
        return 200, {
            "num_sequences": radix_trie.num_sequences,
            "total_tokens_inserted": radix_trie.total_tokens_inserted,
            "unique_token_count": radix_trie.unique_token_count(),
            "node_count": radix_trie.node_count(),
            "tagged_node_count": tagged_nodes,
            "replica_coverage": coverage,
        }

    async def _handle_get_replica_metrics(_data) -> tuple[int, dict]:
        now = time.monotonic()
        last = metrics_meta["last_refresh_monotonic"]
        return 200, {
            "refresh_interval_seconds": METRICS_REFRESH_INTERVAL_SECONDS,
            "last_refresh_age_seconds": (now - last) if last else None,
            "errors": metrics_meta["last_refresh_errors"],
            "metrics": {
                url: {
                    "replica_key": (replica_url_meta.get(url) or {}).get("replica_key"),
                    "replica_region": (replica_url_meta.get(url) or {}).get("replica_region"),
                    "proxy_region": REGION,
                    "num_running_reqs": m.num_running_reqs,
                    "num_queue_reqs": m.num_queue_reqs,
                    "num_used_tokens": m.num_used_tokens,
                    "latency_seconds": m.latency,
                    "network_rtt_seconds": m.network_rtt or None,
                    "gen_throughput": m.gen_throughput,
                    "utilization": m.utilization,
                }
                for url, m in live_metrics.items()
            },
            "endpoints_queued_tokens": endpoints_queued_tokens,
            "endpoints_queued_uncached_tokens": endpoints_queued_uncached_tokens,
            "endpoints_inflight_requests": endpoints_inflight_requests,
        }

    async def _handle_get_hyperparameters(_data) -> tuple[int, dict]:
        return 200, {
            "hyperparameters": state["hyperparameters"],
            "allowed_keys": sorted(ALLOWED_HYPERPARAM_KEYS),
            "defaults": dict(DEFAULT_GORGO_HYPERPARAMETERS),
        }

    async def _handle_write_hyperparameters(data, *, replace: bool) -> tuple[int, dict]:
        """Validate + apply a hyperparameter update against the
        structured store (see :mod:`policy.gorgo`).

        Two body shapes are accepted, both validated by
        ``policy.gorgo.validate_update``:

        * **Flat** -- ``{"prefill_rate": X, "rtt_weight": Y}`` (or any
          subset of allowed keys) updates ``defaults`` only. This is
          what ``proxy/calibrate.py`` POSTs for the rate and the tuner
          POSTs for weights.
        * **Structured** -- ``{"defaults": {...}, "per_target": {url:
          {...}}}`` lets callers write per-replica overrides.

        ``replace=True`` (PUT) resets the store to factory defaults
        before applying the update; ``replace=False`` (POST/PATCH)
        layers the update on top of the existing store.
        """
        update, err = validate_update(data, known_targets=set(replica_urls))
        if err is not None:
            return 400, {
                "error": err,
                "allowed_keys": sorted(ALLOWED_HYPERPARAM_KEYS),
            }
        state["hyperparameters"] = merge_update(
            state["hyperparameters"], update or {}, replace=replace
        )
        _log(f"hyperparameters updated: {state['hyperparameters']}")
        return 200, {"hyperparameters": state["hyperparameters"]}

    async def _handle_post_hyperparameters(data) -> tuple[int, dict]:
        return await _handle_write_hyperparameters(data, replace=False)

    async def _handle_put_hyperparameters(data) -> tuple[int, dict]:
        return await _handle_write_hyperparameters(data, replace=True)

    async def _handle_post_flush(_data) -> tuple[int, dict]:
        radix_trie.clear()
        # Drop tuning samples too: post-flush per-token rates will look
        # different (cold KV cache, fresh queue depths) so mixing them
        # with pre-flush samples would bias the next auto-tune
        # recompute. The auto-tuner config (enabled / window / hop /
        # apply) is intentionally preserved -- only the buffered
        # samples + the per-window counter are reset.
        samples.clear()
        state["total_samples_appended"] = 0
        state["auto_tune"]["samples_since_last_apply"] = 0
        client = state["upstream_client"]
        replica_results: dict[str, dict] = {}
        if client is None:
            for u in replica_urls:
                replica_results[u] = {
                    "ok": False,
                    "error": "upstream client not yet initialized",
                }
        elif replica_urls:
            pairs = await asyncio.gather(
                *[_flush_upstream_replica(client, u) for u in replica_urls],
            )
            replica_results = dict(pairs)
        return 200, {"radix_trie_cleared": True, "replicas": replica_results}

    def _auto_tune_status() -> dict:
        """Snapshot of the live auto-tuner config + diagnostics.
        Shared by ``GET /tune``, ``GET /samples``, and the response of
        every ``POST /tune`` so callers always see what they just
        configured."""
        at = state["auto_tune"]
        tuner = at.get("online_tuner")
        tuner_state = None
        if tuner is not None:
            try:
                tuner_state = {
                    "name": tuner.name,
                    "best_score": tuner.best_score,
                    "best_params": tuner.best_params,
                    "evaluated_after_baseline": tuner.evaluated_after_baseline,
                    **tuner.state,
                }
            except Exception:
                tuner_state = None
        return {
            "enabled": at["enabled"],
            "window_size": at["window_size"],
            "hop_size": at["hop_size"],
            "apply": at["apply"],
            "mode": at.get("mode", "online-es"),
            "objective_metric": at.get("objective_metric", "neg_p95_ttft"),
            "online_tuner_state": tuner_state,
            "pending_candidate": at.get("pending_candidate"),
            "last_score": at.get("last_score"),
            "buffered_samples": len(samples),
            "samples_since_last_apply": at["samples_since_last_apply"],
            "samples_until_next_apply": (
                max(0, at["hop_size"] - at["samples_since_last_apply"]) if at["enabled"] else None
            ),
            "applied_count": at["applied_count"],
            "last_applied_at_monotonic": at["last_applied_at_monotonic"],
            "last_recommendation": at["last_recommendation"],
            "enabled_at_monotonic": at["enabled_at_monotonic"],
            "current_policy": state["policy"],
            "current_hyperparameters": state["hyperparameters"],
        }

    async def _handle_get_samples(_data) -> tuple[int, dict]:
        """Visibility into the tuning sample buffer. Returns the most
        recent samples (capped to keep the response small) plus the
        live auto-tuner status."""
        recent = list(samples)[-50:]
        return 200, {
            "buffered_samples": len(samples),
            "max_buffer_size": samples.maxlen,
            "total_samples_appended": state["total_samples_appended"],
            "auto_tune": _auto_tune_status(),
            "recent": recent,
        }

    async def _handle_get_tune(_data) -> tuple[int, dict]:
        """Live auto-tuner status (no side effects)."""
        return 200, {"auto_tune": _auto_tune_status()}

    async def _handle_post_tune(data) -> tuple[int, dict]:
        """Toggle / reconfigure the on-the-fly auto-tuner.

        The auto-tuner is *stateful*: once enabled it keeps
        recomputing rates (``fit`` mode) or tuning weights
        (``online-es`` mode) every ``hop_size`` new samples (after
        the first ``window_size`` samples have buffered to fill the
        window) until disabled.
        Each ``POST /tune`` atomically merges the body into the live
        config; only the keys present in the body are touched.

        Body (all optional):
          * ``enabled``:     bool. When omitted, defaults to ``True``
            so a bare ``POST /tune {}`` turns the tuner on with the
            current config. Pass ``{"enabled": false}`` to disable.
          * ``window_size``: int. Trailing-sample window fed to
            ``recommend_rates``.
          * ``hop_size``:    int (>0). Recompute every N new samples.
            A small ``hop_size`` reacts faster to load shifts; a
            larger one is more stable.
          * ``apply``:       bool. When ``False`` the recommender
            still runs (and ``last_recommendation`` is updated) but
            ``state["hyperparameters"]`` is not mutated -- useful for
            shadow-mode comparison against the current settings.

        Enabling requires ``policy == 'gorgo'`` (the only policy that
        consumes the tuned scalars). Disabling works regardless of
        policy so a misconfigured tuner can always be turned off.
        """
        if not isinstance(data, dict):
            return 400, {"error": "body must be a JSON object"}

        at = state["auto_tune"]

        # Validate first; only mutate after every requested key passes.
        # Otherwise a bad ``hop_size`` after a good ``window_size``
        # would leave the tuner half-configured.
        new_window = at["window_size"]
        new_hop = at["hop_size"]
        new_apply = at["apply"]
        new_mode = at.get("mode", "online-es")
        new_metric = at.get("objective_metric", "neg_p95_ttft")

        if "window_size" in data:
            try:
                new_window = int(data["window_size"])
            except (TypeError, ValueError):
                return 400, {"error": "window_size must be an integer"}
            if new_window <= 0:
                return 400, {"error": "window_size must be positive"}
        if "hop_size" in data:
            try:
                new_hop = int(data["hop_size"])
            except (TypeError, ValueError):
                return 400, {"error": "hop_size must be an integer"}
            if new_hop <= 0:
                return 400, {"error": "hop_size must be > 0"}
        if "apply" in data:
            new_apply = bool(data["apply"])
        if "mode" in data:
            mv = data["mode"]
            if not isinstance(mv, str) or mv not in SUPPORTED_AUTO_TUNE_MODES:
                return 400, {
                    "error": f"mode must be one of {sorted(SUPPORTED_AUTO_TUNE_MODES)}",
                }
            new_mode = mv
        if "objective_metric" in data:
            mv = data["objective_metric"]
            if not isinstance(mv, str) or mv not in ONLINE_SCORE_FUNCTIONS:
                return 400, {
                    "error": f"objective_metric must be one of {sorted(ONLINE_SCORE_FUNCTIONS)}",
                }
            new_metric = mv

        # Default to enabling so a bare POST /tune {} turns it on. To
        # leave the toggle alone (e.g. just adjust window_size while
        # already running) callers can pass the current value back, but
        # in practice the explicit-default keeps the common case terse.
        new_enabled = bool(data.get("enabled", True))

        if new_enabled and normalize_policy(state["policy"]) not in {"gorgo", "gorgo-2d"}:
            return 400, {
                "error": (
                    "auto-tuning can only be enabled when the active policy is "
                    "'gorgo' or 'gorgo-2d'"
                ),
                "current_policy": state["policy"],
            }

        was_enabled = at["enabled"]
        was_mode = at.get("mode", "online-es")
        at["window_size"] = new_window
        at["hop_size"] = new_hop
        at["apply"] = new_apply
        at["enabled"] = new_enabled
        at["mode"] = new_mode
        at["objective_metric"] = new_metric

        # Lifecycle for the online-ES tuner instance:
        #   - fresh enable into online-es      -> create tuner from current
        #     defaults, reset pending state
        #   - reconfiguration within online-es -> keep tuner, only reset
        #     pending if window_size changed (the prior pending window is
        #     no longer the right size)
        if new_enabled and new_mode == "online-es":
            seed_defaults = (
                state["hyperparameters"].get("defaults") or DEFAULT_GORGO_HYPERPARAMETERS
            )
            # Allow callers to override hyperparam ranges via POST body.
            custom_ranges = data.get("hyperparam_ranges")
            active_ranges = (
                validated_ranges(
                    {k: tuple(v) for k, v in custom_ranges.items()},
                    merge_defaults=False,
                )
                if custom_ranges
                else HYPERPARAM_RANGES
            )
            at["hyperparam_ranges"] = active_ranges
            seed = {k: float(seed_defaults.get(k, v)) for k, v in active_ranges.items()}
            need_new_tuner = (
                at.get("online_tuner") is None or was_mode != "online-es" or not was_enabled
            )
            if need_new_tuner:
                at["online_tuner"] = GaussianESTuner(
                    initial_params=seed,
                    ranges=active_ranges,
                    sigma=0.5,
                    sigma_min=0.05,
                    max_steps=10_000,
                )
                at["pending_candidate"] = None
                at["pending_started_at_count"] = state["total_samples_appended"]
                at["last_score"] = None

        if new_enabled and not was_enabled:
            # Fresh enable: zero the per-window counter so the first
            # recompute is measured from this moment, not from stale
            # samples that landed while the tuner was off.
            at["samples_since_last_apply"] = 0
            at["enabled_at_monotonic"] = time.monotonic()
            _log(
                f"auto-tune ENABLED mode={new_mode} window={new_window} "
                f"hop={new_hop} apply={new_apply} metric={new_metric}"
            )
        elif not new_enabled and was_enabled:
            _log("auto-tune DISABLED")
        elif new_enabled:
            # Reconfigured while running -- keep the existing counter
            # so we don't reset the "samples until next apply" clock
            # on every adjustment.
            _log(
                f"auto-tune RECONFIGURED mode={new_mode} window={new_window} "
                f"hop={new_hop} apply={new_apply} metric={new_metric}"
            )

        # Best-effort summary of the current trailing window if it's
        # already large enough; gives the caller something useful to
        # see immediately even before the next recompute fires.
        preview: dict | None = None
        if samples:
            window = list(samples)[-new_window:]
            preview = {
                "window_size_used": len(window),
                "stats": summarize_samples(window),
            }

        return 200, {
            "auto_tune": _auto_tune_status(),
            "preview": preview,
        }

    async def _handle_post_trace_start(data) -> tuple[int, dict]:
        nonlocal metrics_trace_events, request_trace_events, tune_trace_events
        if not isinstance(data, dict):
            return 400, {"error": "body must be a JSON object"}
        tr = state["trace"]
        trace_id = data.get("trace_id")
        if trace_id is not None and not isinstance(trace_id, str):
            return 400, {"error": "trace_id must be a string"}
        try:
            max_events = int(data.get("max_events", tr["max_events"]))
        except (TypeError, ValueError):
            return 400, {"error": "max_events must be an integer"}
        if max_events <= 0:
            return 400, {"error": "max_events must be positive"}

        append = bool(data.get("append", False))
        if not append or max_events != tr["max_events"]:
            metrics_trace_events = deque(maxlen=max_events)
            request_trace_events = deque(maxlen=max_events)
            tune_trace_events = deque(maxlen=max_events)

        tr.update(
            {
                "enabled": True,
                "trace_id": trace_id
                or f"trace-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",
                "sample_metrics": bool(data.get("sample_metrics", True)),
                "sample_requests": bool(data.get("sample_requests", True)),
                "max_events": max_events,
                "started_at": _now_wall_ts(),
                "stopped_at": None,
                "saved_paths": None,
            }
        )
        if not append:
            tr["dropped_metrics"] = 0
            tr["dropped_requests"] = 0
            tr["dropped_tune"] = 0
        _log(f"trace started id={tr['trace_id']} max_events={max_events}")
        return 200, {"trace": _trace_status_payload()}

    async def _handle_get_trace_status(_data) -> tuple[int, dict]:
        return 200, {"trace": _trace_status_payload()}

    async def _handle_post_trace_stop(_data) -> tuple[int, dict]:
        state["trace"]["enabled"] = False
        state["trace"]["stopped_at"] = _now_wall_ts()
        _log(f"trace stopped id={state['trace']['trace_id']}")
        return 200, {"trace": _trace_status_payload()}

    async def _handle_post_trace_save(_data) -> tuple[int, dict]:
        paths = _save_trace_to_volume()
        return 200, {
            "trace": _trace_status_payload(),
            "paths": paths,
            "fallback_summary": state["trace"].get("fallback_summary"),
        }

    def _batch_tuning_public_state() -> dict:
        bt = state["batch_tuning"]
        payload = {k: v for k, v in bt.items() if k != "task"}
        payload["history"] = list(bt.get("history") or [])
        return payload

    def _parse_int(data: dict, key: str, default: int) -> int:
        try:
            return int(data.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer")

    def _parse_float(data: dict, key: str, default: float) -> float:
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number")

    def _parse_optional_str(data: dict, key: str, default: str = "") -> str:
        value = data.get(key, default)
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value

    def _parse_optional_bool(data: dict, key: str):
        value = data.get(key, None)
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off"):
                return False
        raise ValueError(f"{key} must be true/false/null")

    def _normalize_batch_tuning_config(data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        metric = _parse_optional_str(data, "metric", "output_throughput")
        if metric not in SCORE_FUNCTIONS:
            raise ValueError(f"unknown metric {metric!r}; choices: {sorted(SCORE_FUNCTIONS)}")
        algorithm = _parse_optional_str(data, "algorithm", "gaussian-es")
        ranges = validated_ranges(
            {
                "prefill_weight": (
                    _parse_float(data, "prefill_weight_min", 1e-5),
                    _parse_float(data, "prefill_weight_max", 5.0),
                ),
                "load_weight": (
                    _parse_float(data, "load_weight_min", 1e-5),
                    _parse_float(data, "load_weight_max", 5.0),
                ),
                "rtt_weight": (
                    _parse_float(data, "rtt_weight_min", 1e-5),
                    _parse_float(data, "rtt_weight_max", 50.0),
                ),
            }
        )
        return {
            "source": _parse_optional_str(data, "source", "prod") or "prod",
            "data_path": _parse_optional_str(data, "data_path", ""),
            "arrival_mode": _parse_optional_str(data, "arrival_mode", "bounded") or "bounded",
            "time_scale": _parse_float(data, "time_scale", 1.0),
            "start_time": _parse_optional_str(data, "start_time"),
            "end_time": _parse_optional_str(data, "end_time"),
            "offset": _parse_int(data, "offset", 0),
            "num_requests": (_parse_int(data, "num_requests", 0) or None),
            "concurrency": _parse_int(data, "concurrency", 16),
            "model": _parse_optional_str(data, "model", ""),
            "stream": _parse_optional_bool(data, "stream"),
            "max_tokens": (_parse_int(data, "max_tokens", 0) or None),
            "max_input_tokens": _parse_int(data, "max_input_tokens", 0),
            "metric": metric,
            "max_steps": _parse_int(data, "max_steps", 16),
            "sigma": _parse_float(data, "sigma", 0.5),
            "sigma_min": _parse_float(data, "sigma_min", 0.02),
            "seed": (_parse_int(data, "seed", -1)),
            "relative_tolerance": _parse_float(data, "relative_tolerance", 0.005),
            "output_dir": _parse_optional_str(data, "output_dir", ""),
            "ranges": ranges,
        }

    async def _run_batch_tuning(run_id: str, config: dict) -> None:
        from proxy.workload_core import DEFAULT_MODEL, run_replay_async

        bt = state["batch_tuning"]
        run_started_at = datetime.now(timezone.utc)
        out_dir = (
            config["output_dir"]
            if config["output_dir"] and os.path.isabs(config["output_dir"])
            else os.path.join(
                "/results",
                config["output_dir"] or f"tune_{run_started_at.strftime('%Y%m%d_%H%M%S')}",
            )
        )
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, "summary.json")
        bt["output_path"] = summary_path

        workload_kwargs = {
            "source": config["source"],
            "data_path": config["data_path"] or None,
            "start_time": config["start_time"] or None,
            "end_time": config["end_time"] or None,
            "offset": config["offset"],
            "num_requests": config["num_requests"],
            "concurrency": config["concurrency"],
            "model": config["model"] or DEFAULT_MODEL,
            "stream": config["stream"],
            "max_tokens": config["max_tokens"],
            "max_input_tokens": config["max_input_tokens"],
            "arrival_mode": config["arrival_mode"],
            "time_scale": config["time_scale"],
            "save_per_request": False,
        }
        metric = config["metric"]
        score_fn = SCORE_FUNCTIONS[metric]
        active_policy = state["policy"]
        current_hp = dict((state["hyperparameters"] or {}).get("defaults") or {})
        tuner = build_tuner(
            initial_params=current_hp,
            ranges=config["ranges"],
            max_steps=config["max_steps"],
            relative_tolerance=config["relative_tolerance"],
            sigma=config["sigma"],
            sigma_min=config["sigma_min"],
            seed=None if config["seed"] < 0 else config["seed"],
        )
        history: list[dict] = []
        baseline_score: float | None = None

        def persist_summary(finished_at: datetime | None = None) -> None:
            summary = build_summary(
                run_started_at=run_started_at,
                proxy_url="http://127.0.0.1:8000",
                workload_kwargs=workload_kwargs,
                metric=metric,
                tuner=tuner,
                baseline_score=baseline_score,
                history=history,
                active_policy=active_policy,
                ranges=config["ranges"],
                finished_at=finished_at,
                output_path=summary_path,
            )
            summary["trace"] = _trace_status_payload()
            tmp_path = f"{summary_path}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(summary, f, indent=2)
            os.replace(tmp_path, summary_path)
            bench_results_volume.commit()

        try:
            _log(
                f"batch tuning {run_id} started algorithm={tuner.name} "
                f"metric={metric} max_steps={config['max_steps']}"
            )
            trial_idx = 0
            while True:
                candidate = tuner.propose()
                if candidate is None:
                    break
                if bt.get("stop_requested"):
                    raise asyncio.CancelledError()
                is_baseline = tuner.best_score is None
                bt["trial"] = trial_idx
                _log(f"batch tuning {run_id} trial {trial_idx}: evaluating {candidate}")

                status, payload = await _handle_post_flush({})
                if status >= 400:
                    raise RuntimeError(f"flush: {payload}")
                status, payload = await _handle_post_hyperparameters(candidate)
                if status >= 400:
                    raise RuntimeError(f"hyperparameters: {payload}")

                trial_output = os.path.join(out_dir, f"replay_trial_{trial_idx:03d}.json")
                stats = await run_replay_async(
                    proxy_url="http://127.0.0.1:8000",
                    output_path=trial_output,
                    run_id=f"{run_id}-trial-{trial_idx:03d}",
                    **workload_kwargs,
                )
                score = score_fn(stats)
                accepted = tuner.report(candidate, score)
                tuner_state = dict(tuner.state)
                if is_baseline:
                    baseline_score = score
                    _log(f"batch tuning {run_id} trial {trial_idx}: baseline score={score:.4f}")
                else:
                    tag = "ACCEPTED" if accepted else "rejected"
                    _log(
                        f"batch tuning {run_id} trial {trial_idx}: "
                        f"score={score:.4f} {tag} incumbent={tuner.best_score:.4f}"
                    )
                history.append(
                    {
                        "trial": trial_idx,
                        "kind": "baseline" if is_baseline else "trial",
                        "params": candidate,
                        "score": score,
                        "incumbent_score": tuner.best_score,
                        "incumbent_params": dict(tuner.best_params),
                        "tuner_state": tuner_state,
                        "accepted": accepted,
                        "metric": metric,
                        "stats": stats,
                    }
                )
                bt["history"] = history
                bt["best_params"] = dict(tuner.best_params)
                bt["best_score"] = tuner.best_score
                bt["baseline_score"] = baseline_score
                persist_summary()
                trial_idx += 1

            if tuner.best_params:
                status, payload = await _handle_post_hyperparameters(tuner.best_params)
                if status >= 400:
                    raise RuntimeError(f"final hyperparameters: {payload}")
                status, payload = await _handle_post_flush({})
                if status >= 400:
                    raise RuntimeError(f"final flush: {payload}")

            finished_at = datetime.now(timezone.utc)
            persist_summary(finished_at=finished_at)
            bt["status"] = "succeeded"
            bt["finished_at"] = finished_at.isoformat().replace("+00:00", "Z")
            _log(f"batch tuning {run_id} succeeded output={summary_path}")
        except asyncio.CancelledError:
            finished_at = datetime.now(timezone.utc)
            bt["status"] = "cancelled"
            bt["finished_at"] = finished_at.isoformat().replace("+00:00", "Z")
            bt["error"] = "cancelled"
            try:
                persist_summary(finished_at=finished_at)
            except Exception as e:
                _log(f"batch tuning {run_id} summary persist after cancel failed: {e}")
            _log(f"batch tuning {run_id} cancelled")
        except Exception as e:
            finished_at = datetime.now(timezone.utc)
            bt["status"] = "failed"
            bt["finished_at"] = finished_at.isoformat().replace("+00:00", "Z")
            bt["error"] = f"{type(e).__name__}: {e}"
            _log(f"batch tuning {run_id} failed: {bt['error']}")
        finally:
            bt["running"] = False
            bt["task"] = None
            bt["stop_requested"] = False

    async def _handle_post_tuning_start(data) -> tuple[int, dict]:
        if not isinstance(data, dict):
            return 400, {"error": "body must be a JSON object"}
        bt = state["batch_tuning"]
        task = bt.get("task")
        if task is not None and not task.done():
            return 409, {
                "error": "batch tuning already running",
                "batch_tuning": _batch_tuning_public_state(),
            }
        if normalize_policy(state["policy"]) != "gorgo":
            return 400, {
                "error": "batch tuning requires active policy 'gorgo'",
                "current_policy": state["policy"],
            }
        if state["auto_tune"]["enabled"]:
            return 409, {"error": "disable live auto-tune before starting batch tuning"}
        if not replica_urls:
            return 400, {"error": "no replicas registered"}
        try:
            config = _normalize_batch_tuning_config(data)
        except ValueError as e:
            return 400, {"error": str(e)}

        run_id = (
            f"tune-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        bt.update(
            {
                "running": True,
                "status": "running",
                "run_id": run_id,
                "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "finished_at": None,
                "config": config,
                "trial": None,
                "history": [],
                "best_params": None,
                "best_score": None,
                "baseline_score": None,
                "output_path": None,
                "error": None,
                "stop_requested": False,
            }
        )
        task = asyncio.create_task(_run_batch_tuning(run_id, config))
        bt["task"] = task
        return 202, {
            "run_id": run_id,
            "status": "running",
            "status_url": "/tuning/status",
            "batch_tuning": _batch_tuning_public_state(),
        }

    async def _handle_get_tuning_status(_data) -> tuple[int, dict]:
        return 200, {"batch_tuning": _batch_tuning_public_state()}

    async def _handle_post_tuning_stop(_data) -> tuple[int, dict]:
        bt = state["batch_tuning"]
        task = bt.get("task")
        if task is None or task.done():
            return 200, {"batch_tuning": _batch_tuning_public_state()}
        bt["stop_requested"] = True
        task.cancel()
        return 202, {"batch_tuning": _batch_tuning_public_state()}

    def _workload_public_state() -> dict:
        wr = state["workload_run"]
        return {k: v for k, v in wr.items() if k != "task"}

    def _normalize_workload_config(data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        data_path = data.get("data_path") or data.get("trace_path")
        if not isinstance(data_path, str) or not data_path:
            raise ValueError("data_path or trace_path is required")
        return {
            "data_path": data_path,
            "run_id": _parse_optional_str(data, "run_id", ""),
            "concurrency": _parse_int(data, "concurrency", 16),
            "model": _parse_optional_str(data, "model", ""),
            "stream": _parse_optional_bool(data, "stream"),
            "max_tokens": (_parse_int(data, "max_tokens", 0) or None),
            "max_input_tokens": _parse_int(data, "max_input_tokens", 0),
            "num_requests": (_parse_int(data, "num_requests", 0) or None),
            "arrival_mode": _parse_optional_str(data, "arrival_mode", "open-loop"),
            "time_scale": _parse_float(data, "time_scale", 1.0),
            "output_path": _parse_optional_str(data, "output_path", ""),
            "save_per_request": bool(data.get("save_per_request", True)),
            "start_at_wall_time": _parse_optional_str(data, "start_at_wall_time", ""),
        }

    async def _run_workload(run_id: str, config: dict) -> None:
        from proxy.workload_core import DEFAULT_MODEL, run_replay_async

        wr = state["workload_run"]
        try:
            start_at = config.get("start_at_wall_time")
            if start_at:
                target = datetime.fromisoformat(start_at.replace("Z", "+00:00"))
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                delay = (target.astimezone(timezone.utc) - now).total_seconds()
                if delay > 0:
                    wr["status"] = "scheduled"
                    wr["phase"] = "waiting-start-at-wall-time"
                    _log(f"workload {run_id} scheduled for {start_at} ({delay:.3f}s)")
                    await asyncio.sleep(delay)
                    wr["status"] = "running"
                    wr["phase"] = "starting"
            _log(f"workload {run_id} started path={config['data_path']}")
            output_path = config["output_path"] or f"/results/workload_runs/{run_id}.json"
            wr["phase"] = "replay-running"
            stats = await run_replay_async(
                proxy_url="http://127.0.0.1:8000",
                source="mooncake",
                data_path=config["data_path"],
                concurrency=config["concurrency"],
                model=config["model"] or DEFAULT_MODEL,
                stream=config["stream"],
                max_tokens=config["max_tokens"],
                max_input_tokens=config["max_input_tokens"],
                num_requests=config.get("num_requests"),
                output_path=output_path,
                save_per_request=config["save_per_request"],
                run_id=run_id,
                arrival_mode=config["arrival_mode"],
                time_scale=config["time_scale"],
            )
            finished_at = datetime.now(timezone.utc)
            wr["status"] = "succeeded"
            wr["phase"] = "done"
            wr["finished_at"] = finished_at.isoformat().replace("+00:00", "Z")
            wr["stats"] = stats
            wr["output_path"] = stats.get("output_path")
            _log(f"workload {run_id} succeeded output={wr['output_path']}")
        except asyncio.CancelledError:
            finished_at = datetime.now(timezone.utc)
            wr["status"] = "cancelled"
            wr["phase"] = "cancelled"
            wr["finished_at"] = finished_at.isoformat().replace("+00:00", "Z")
            wr["error"] = "cancelled"
            wr["stats"] = _compute_workload_stats_from_trace()
            _log(
                f"workload {run_id} cancelled (partial stats from {wr['stats'].get('n', 0)} trace events)"
            )
        except Exception as e:
            finished_at = datetime.now(timezone.utc)
            wr["status"] = "failed"
            wr["phase"] = "failed"
            wr["finished_at"] = finished_at.isoformat().replace("+00:00", "Z")
            wr["error"] = f"{type(e).__name__}: {e}"
            wr["stats"] = _compute_workload_stats_from_trace()
            _log(
                f"workload {run_id} failed: {wr['error']} (partial stats from {wr['stats'].get('n', 0)} trace events)"
            )
        finally:
            wr["running"] = False
            wr["task"] = None
            wr["stop_requested"] = False

    async def _handle_post_workload_start(data) -> tuple[int, dict]:
        wr = state["workload_run"]
        task = wr.get("task")
        if task is not None and not task.done():
            return 409, {"error": "workload already running", "workload": _workload_public_state()}
        try:
            config = _normalize_workload_config(data)
        except ValueError as e:
            return 400, {"error": str(e)}
        run_id = (
            config["run_id"]
            or f"workload-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        wr.update(
            {
                "running": True,
                "status": "running",
                "phase": "starting",
                "run_id": run_id,
                "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "finished_at": None,
                "config": config,
                "stats": None,
                "output_path": None,
                "error": None,
                "stop_requested": False,
            }
        )
        task = asyncio.create_task(_run_workload(run_id, config))
        wr["task"] = task
        return 202, {"run_id": run_id, "status": "running", "workload": _workload_public_state()}

    async def _handle_get_workload_status(_data) -> tuple[int, dict]:
        return 200, {"workload": _workload_public_state()}

    async def _handle_post_workload_stop(_data) -> tuple[int, dict]:
        wr = state["workload_run"]
        task = wr.get("task")
        if task is None or task.done():
            return 200, {"workload": _workload_public_state()}
        wr["stop_requested"] = True
        task.cancel()
        return 202, {"workload": _workload_public_state()}

    # JSON route table. The dispatcher distinguishes 405 (method exists
    # for path but not this verb) from 404 (path unknown) by membership
    # checks, so adding a new (method, path) entry here is the only edit
    # required to expose a new JSON endpoint.
    json_routes: dict[tuple[str, str], object] = {
        ("GET", "/policy"): _handle_get_policy,
        ("POST", "/policy"): _handle_post_policy,
        ("GET", "/config"): _handle_get_config,
        ("POST", "/config"): _handle_post_config,
        ("GET", "/calibrated_rates"): _handle_get_calibrated_rates,
        ("GET", "/replicas"): _handle_get_replicas,
        ("POST", "/replicas"): _handle_post_replicas,
        ("GET", "/trie"): _handle_get_trie,
        ("GET", "/replica_metrics"): _handle_get_replica_metrics,
        ("GET", "/hyperparameters"): _handle_get_hyperparameters,
        ("POST", "/hyperparameters"): _handle_post_hyperparameters,
        ("PATCH", "/hyperparameters"): _handle_post_hyperparameters,
        ("PUT", "/hyperparameters"): _handle_put_hyperparameters,
        ("POST", "/flush"): _handle_post_flush,
        ("GET", "/samples"): _handle_get_samples,
        ("GET", "/tune"): _handle_get_tune,
        ("POST", "/tune"): _handle_post_tune,
        ("POST", "/trace/start"): _handle_post_trace_start,
        ("GET", "/trace/status"): _handle_get_trace_status,
        ("POST", "/trace/stop"): _handle_post_trace_stop,
        ("POST", "/trace/save"): _handle_post_trace_save,
        ("POST", "/tuning/start"): _handle_post_tuning_start,
        ("GET", "/tuning/status"): _handle_get_tuning_status,
        ("POST", "/tuning/stop"): _handle_post_tuning_stop,
        ("POST", "/workload/start"): _handle_post_workload_start,
        ("GET", "/workload/status"): _handle_get_workload_status,
        ("POST", "/workload/stop"): _handle_post_workload_stop,
    }

    async def _dispatch_json(method: str, path: str, receive, send) -> bool:
        handler = json_routes.get((method, path))
        if handler is None:
            if any(p == path for (_, p) in json_routes):
                await _send_json(send, 405, {"error": "method not allowed"})
                return True
            return False  # let caller emit 404

        # Body-bearing methods may include a JSON payload; GET callers
        # pass an empty dict so handlers don't have to special-case.
        if method in ("POST", "PUT", "PATCH"):
            try:
                data = await _read_json_body(receive)
            except json.JSONDecodeError:
                await _send_json(send, 400, {"error": "invalid JSON body"})
                return True
        else:
            data = {}

        status, payload = await handler(data)
        await _send_json(send, status, payload)
        return True

    # ---------- Chat completions (streaming passthrough + tuning tap) ----------

    def _record_request_sample(
        *,
        target: str,
        ttft_ns: int | None,
        total_ns: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens_at_dispatch: int = 0,
        queued_tokens_at_dispatch: int = 0,
        queued_uncached_tokens_at_dispatch: int = 0,
        inflight_requests_at_dispatch: int = 0,
    ) -> None:
        """Append a per-request sample shaped like
        :func:`proxy.measure.measure_chat_completion`'s output. Skipped
        silently when token counts are missing (calibrate-style safeguard
        against corrupted per-token rates).

        Raw per-token rates are stored in **seconds per token** (the
        natural measurement unit).  Conversion to ms/tok happens inside
        :func:`proxy.measure.recommend_rates` so the scoring
        function's RTT-in-ms convention is matched without baking unit
        choices into the sample buffer.

        ``ping_seconds`` is the irreducible RTT subtracted from TTFT
        before fitting the prefill rate. Source order:
          1. ``snap.network_rtt`` (EWMA-smoothed dedicated probe of the
             replica's base URL; preferred -- isolates pure network RTT
             from SGLang's /metrics handler load).
          2. ``snap.latency`` (scrape RTT; fallback when the probe
             hasn't completed a successful round-trip yet, e.g. cold
             start). Inflates under load, hence only a fallback.
          3. ``0.0`` (no snapshot at all -- prefill rate becomes a
             slight overestimate rather than negative).
        Same role ``ping_once`` plays in ``proxy/calibrate.py``.

        ``cached_tokens_at_dispatch`` is the radix-trie's cached prefix
        length on ``target`` measured *at routing decision time*. The
        per-token prefill rate is fitted against ``prompt_tokens -
        cached_tokens_at_dispatch`` so it represents the cost per
        *uncached* token -- which is what ``policy.gorgo``'s cost model
        multiplies by. Without this correction the fitted rate is
        amortized over cache hits and the cost model under-weighs the
        prefill term in cache-heavy workloads (Option A fix).
        """
        if (
            ttft_ns is None
            or prompt_tokens is None
            or prompt_tokens <= 0
            or completion_tokens is None
            or completion_tokens <= 0
        ):
            return

        snap = live_metrics.get(target)
        if snap is None:
            ping_rtt_s = 0.0
        elif snap.network_rtt > 0.0:
            ping_rtt_s = snap.network_rtt
        else:
            ping_rtt_s = snap.latency

        ttft_s = ttft_ns / NS_PER_S
        total_s = total_ns / NS_PER_S
        prefill_s = max(ttft_s - ping_rtt_s, 0.0)
        decode_s = max(total_s - ttft_s, 0.0)

        # Clamp cached_tokens_at_dispatch into [0, prompt_tokens) so the
        # divisor is at least 1 token -- a fully-cached prompt still has
        # *some* prefill work (the next token must be scheduled) and we
        # don't want to divide by zero when block boundaries align.
        uncached_tokens = max(1, prompt_tokens - max(0, cached_tokens_at_dispatch))

        samples.append(
            {
                "ping_seconds": ping_rtt_s,
                "ttft_seconds": ttft_s,
                "total_seconds": total_s,
                "prefill_seconds": prefill_s,
                "decode_seconds": decode_s,
                "prompt_tokens": prompt_tokens,
                "cached_tokens_at_dispatch": cached_tokens_at_dispatch,
                "queued_tokens_at_dispatch": queued_tokens_at_dispatch,
                "queued_uncached_tokens_at_dispatch": queued_uncached_tokens_at_dispatch,
                "inflight_requests_at_dispatch": inflight_requests_at_dispatch,
                "uncached_tokens": uncached_tokens,
                "completion_tokens": completion_tokens,
                "prefill_rate_seconds_per_token": prefill_s / uncached_tokens,
                "decode_rate_seconds_per_token": decode_s / completion_tokens,
                "target": target,
                "recorded_at_monotonic": time.monotonic(),
            }
        )
        state["total_samples_appended"] += 1

        # Auto-tune hook: only fire on the same code path that produced
        # the sample so the recompute is naturally serialized with
        # appends (single asyncio thread, no lock needed). The fast path
        # is a single bool check when the tuner is disabled.
        at = state["auto_tune"]
        if not at["enabled"]:
            return
        at["samples_since_last_apply"] += 1
        if at["samples_since_last_apply"] < at["hop_size"]:
            return
        if len(samples) < at["window_size"]:
            # Don't fire until the buffer holds at least one full window;
            # earlier recomputes would over-weight whichever short prefix
            # of the run happens to have landed first.
            return
        if normalize_policy(state["policy"]) not in {"gorgo", "gorgo-2d"}:
            # Auto-tune only writes gorgo hyperparameters (rates or
            # weights); under any other policy the writes are inert. Skip
            # and keep the counter pinned so a switch back to gorgo
            # immediately resumes recomputing on the next sample.
            at["samples_since_last_apply"] = at["hop_size"]
            return

        window = list(samples)[-at["window_size"] :]
        mode = at.get("mode", "online-es")

        if mode == "calibrate":
            # Calibration accumulates physical rates per-request from the
            # /generate meta_info (see ``_accumulate_calibration``); the
            # windowed tuner is a deliberate no-op here and never writes
            # weights (apply is effectively forced off).
            return

        if mode == "online-es":
            # ----- Option B: empirical hyperparameter search -----
            # Treat (rtt_weight, prefill_weight) as dimensionless knobs
            # and use Gaussian (1+1)-ES to minimize the configured
            # objective metric over the rolling window.
            #
            metric = at.get("objective_metric", "neg_p95_ttft")
            score_fn = ONLINE_SCORE_FUNCTIONS.get(metric)
            tuner: GaussianESTuner | None = at.get("online_tuner")
            if score_fn is None or tuner is None:
                # Defensive: shouldn't happen because POST /tune
                # validates these fields, but if state is corrupted
                # don't crash the request loop.
                at["samples_since_last_apply"] = 0
                return
            score = float(score_fn(window))
            pending = at.get("pending_candidate")
            if pending is not None:
                accepted = tuner.report(pending, score)
            else:
                accepted = tuner.report(dict(tuner.best_params), score)
            at["last_score"] = score

            proposal = tuner.propose()
            if proposal is None:
                if at["apply"]:
                    state["hyperparameters"] = merge_update(
                        state["hyperparameters"],
                        {"defaults": dict(tuner.best_params)},
                        replace=False,
                    )
                at["enabled"] = False
                at["pending_candidate"] = None
                _trace_append(
                    "tune",
                    {
                        "kind": "tune",
                        "mode": "online-es",
                        "wall_ts": _now_wall_ts(),
                        "monotonic_s": time.monotonic(),
                        "step": at["applied_count"],
                        "total_samples": state["total_samples_appended"],
                        "window_size": len(window),
                        "converged": True,
                        "accepted": accepted,
                        "candidate": pending,
                        "score": score,
                        "best_score": tuner.best_score,
                        "best_params": dict(tuner.best_params),
                        "proposal": None,
                        "sigma": tuner.sigma,
                        "success_rate": (
                            sum(tuner._recent) / len(tuner._recent) if tuner._recent else None
                        ),
                        "objective_metric": metric,
                        "rechenberg": {
                            "recent_outcomes": list(tuner._recent),
                            "success_window": tuner.success_window,
                            "target_rate": tuner.target_rate,
                            "sigma_decay": tuner.sigma_decay,
                            "sigma_min": tuner.sigma_min,
                            "evaluated_after_baseline": tuner.evaluated_after_baseline,
                            "max_steps": tuner.max_steps,
                            "tol": tuner.tol,
                        },
                    },
                )
                _log(
                    f"auto-tune online-es CONVERGED best={tuner.best_params} "
                    f"score={tuner.best_score}"
                )
                return

            if at["apply"]:
                state["hyperparameters"] = merge_update(
                    state["hyperparameters"],
                    {"defaults": dict(proposal)},
                    replace=False,
                )
            at["pending_candidate"] = dict(proposal)
            at["pending_started_at_count"] = state["total_samples_appended"]
            at["applied_count"] += 1
            at["last_applied_at_monotonic"] = time.monotonic()
            at["last_recommendation"] = {
                "defaults": dict(proposal),
                "per_target": {},
            }
            at["samples_since_last_apply"] = 0
            _trace_append(
                "tune",
                {
                    "kind": "tune",
                    "mode": "online-es",
                    "wall_ts": _now_wall_ts(),
                    "monotonic_s": time.monotonic(),
                    "step": at["applied_count"],
                    "total_samples": state["total_samples_appended"],
                    "window_size": len(window),
                    "converged": False,
                    "accepted": accepted,
                    "candidate": pending,
                    "score": score,
                    "best_score": tuner.best_score,
                    "best_params": dict(tuner.best_params),
                    "proposal": dict(proposal),
                    "sigma": tuner.sigma,
                    "success_rate": (
                        sum(tuner._recent) / len(tuner._recent) if tuner._recent else None
                    ),
                    "objective_metric": metric,
                    "rechenberg": {
                        "recent_outcomes": list(tuner._recent),
                        "success_window": tuner.success_window,
                        "target_rate": tuner.target_rate,
                        "sigma_decay": tuner.sigma_decay,
                        "sigma_min": tuner.sigma_min,
                        "evaluated_after_baseline": tuner.evaluated_after_baseline,
                        "max_steps": tuner.max_steps,
                        "tol": tuner.tol,
                    },
                },
            )
            _log(
                f"auto-tune online-es #{at['applied_count']} "
                f"window={len(window)} score={score:.4f} "
                f"best_score={tuner.best_score} "
                f"sigma={tuner.sigma:.4f} "
                f"proposal={proposal} (apply={at['apply']})"
            )
            return

        # Only "online-es" and "calibrate" are supported. The legacy
        # live-autotune "fit" mode (median-of-rates per-target fitting) was
        # removed in favor of the calibrate -> tune -> eval pipeline; any
        # other mode is a no-op.
        return

    async def _handle_chat_completions(scope, receive, send) -> None:
        headers = {
            k.decode("latin1").lower(): v.decode("latin1") for k, v in (scope.get("headers") or [])
        }
        request_id = headers.get("x-gorgo-request-id") or f"proxy-{uuid.uuid4().hex}"
        raw_trace_row = headers.get("x-gorgo-trace-row-index")
        trace_row_index = int(raw_trace_row) if raw_trace_row else None
        raw_slip = headers.get("x-gorgo-scheduling-slip-ms")
        scheduling_slip_ms = float(raw_slip) if raw_slip else None
        source_row_id = headers.get("x-gorgo-source-row-id")
        source_row_hash = headers.get("x-gorgo-source-row-hash")
        try:
            data = await _read_json_body(receive)
        except json.JSONDecodeError:
            await _send_json(send, 400, {"error": "invalid JSON body"})
            return
        if not isinstance(data, dict):
            await _send_json(send, 400, {"error": "body must be a JSON object"})
            return

        messages = data.get("messages", [])
        if not isinstance(messages, list):
            await _send_json(send, 400, {"error": "'messages' must be a list"})
            return

        token_ids = tokenize_input(messages)
        request_tokens = len(token_ids)
        decision_monotonic = time.monotonic()
        decision_wall_ts = _now_wall_ts()
        metrics_seq_at_decision = metrics_meta.get("refresh_seq", 0)
        last_refresh = metrics_meta.get("last_refresh_monotonic") or 0.0
        metrics_age_seconds = (decision_monotonic - last_refresh) if last_refresh else None
        # Per-replica cached-prefix lookup. When tracing requests we need the
        # full per-replica view for the candidate snapshot; otherwise we'll
        # do a cheaper single-target lookup after the routing decision below.
        # Either way ``cached_for_target`` ends up populated and is fed into
        # ``_record_request_sample`` so the auto-tuner fits prefill rate
        # against *uncached* tokens (Option A).
        if token_ids and state["trace"]["enabled"] and state["trace"]["sample_requests"]:
            cached_by_replica = radix_trie.cached_prefix_lengths(token_ids, replica_urls)
        else:
            cached_by_replica = {}
        candidate_snapshot = None
        if state["trace"]["enabled"] and state["trace"]["sample_requests"]:
            candidate_snapshot = {}
            for u in replica_urls:
                snap = live_metrics.get(u)
                replica_meta = replica_url_meta.get(u) or {}
                candidate_snapshot[u] = {
                    "replica_key": replica_meta.get("replica_key"),
                    "replica_region": replica_meta.get("replica_region"),
                    "latency_seconds": snap.latency if snap else None,
                    "num_running_reqs": snap.num_running_reqs if snap else None,
                    "num_queue_reqs": snap.num_queue_reqs if snap else None,
                    "num_used_tokens": snap.num_used_tokens if snap else None,
                    "utilization": snap.utilization if snap else None,
                    "gen_throughput": snap.gen_throughput if snap else None,
                    "queued_tokens": endpoints_queued_tokens.get(u, 0),
                    "queued_uncached_tokens": endpoints_queued_uncached_tokens.get(u, 0),
                    "inflight_requests": endpoints_inflight_requests.get(u, 0),
                    "cached_prefix_tokens": cached_by_replica.get(u, 0),
                }

        candidate_scores = None
        try:
            target, configured_policy, effective_policy, candidate_scores = _select_endpoint(
                token_ids
            )
        except Exception as e:
            configured_policy = state["policy"]
            _log(f"policy {configured_policy!r} failed ({e}); falling back to random")
            if not replica_urls:
                await _send_json(send, 503, {"error": "no replicas registered"})
                return
            target = random.choice(replica_urls)
            effective_policy = f"random-fallback:exception:{configured_policy}:{type(e).__name__}"

        # Resolve cached-prefix length for the chosen target. Uses the
        # batched lookup if we already paid for it (trace path), otherwise
        # the cheaper single-target form.
        if cached_by_replica:
            cached_for_target = cached_by_replica.get(target, 0)
        elif token_ids:
            cached_for_target = radix_trie.cached_prefix_length(token_ids, target)
        else:
            cached_for_target = 0

        at = state.get("auto_tune") or {}
        # Snapshot the load-signal A/B flag once at dispatch so the whole
        # request (first-token release callback, finally fallback, and the
        # trace record) agrees on a single value even if /config flips it
        # mid-stream.
        release_at_ttft = bool(state.get("load_release_at_ttft"))
        request_trace_event = {
            "kind": "request",
            "trace_id": state["trace"]["trace_id"],
            "load_release_at_ttft": release_at_ttft,
            "request_id": request_id,
            "trace_row_index": trace_row_index,
            "source_row_id": source_row_id,
            "source_row_hash": source_row_hash,
            "wall_ts": decision_wall_ts,
            "monotonic_s": decision_monotonic,
            "policy": configured_policy,
            "effective_policy": effective_policy,
            "target": target,
            "target_replica_key": (replica_url_meta.get(target) or {}).get("replica_key"),
            "target_replica_region": (replica_url_meta.get(target) or {}).get("replica_region"),
            "request_tokens": request_tokens,
            "cached_tokens_at_dispatch": cached_for_target,
            "metrics_seq_at_decision": metrics_seq_at_decision,
            "metrics_age_seconds": metrics_age_seconds,
            "candidate_snapshot": candidate_snapshot,
            "candidate_scores": candidate_scores,
            "hyperparameters_at_decision": dict(state["hyperparameters"].get("defaults") or {}),
            "tune_step_at_decision": at.get("applied_count"),
            "scheduling_slip_ms": scheduling_slip_ms,
            "status": None,
            "ttft_ns": None,
            "total_ns": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "meta_info": None,
            "error": None,
        }

        client = state["upstream_client"]
        if client is None:
            # Race between startup and the first inbound request; defensive.
            # No counters incremented yet; nothing to roll back.
            await _send_json(send, 503, {"error": "upstream client not yet initialized"})
            request_trace_event.update(
                {
                    "status": 503,
                    "total_ns": 0,
                    "error": "upstream client not yet initialized",
                }
            )
            _trace_append("request", request_trace_event)
            return

        # Serialize the original parsed JSON exactly once for the upstream
        # request. We forward bytes directly instead of letting httpx do
        # its own json= serialization to avoid a second round-trip through
        # the encoder. ``accept-encoding: identity`` tells the upstream
        # not to compress so we don't have to juggle ``content-encoding``
        # on the way out to the client.
        # Upstream is always the OpenAI chat endpoint. (Calibration previously
        # forwarded to SGLang's native ``/generate`` to harvest per-request
        # ``meta_info`` scheduler timings, but those are null in unified mode on
        # this build, so calibration now regresses on proxy-measured TTFT +
        # ``queued_tokens`` and runs on the same chat path as tuning / eval /
        # production -- keeping the calibrated TTFT relationship consistent.)
        upstream_headers = {
            "accept-encoding": "identity",
            "content-type": "application/json",
        }
        if data.get("stream") is True:
            stream_options = data.get("stream_options")
            if not isinstance(stream_options, dict):
                stream_options = {}
                data["stream_options"] = stream_options
            stream_options.setdefault("include_usage", True)
        upstream_path = "/v1/chat/completions"
        upstream_body = json.dumps(data).encode()
        headers_sent = False
        upstream_status = None
        ttft_ns = None
        total_ns = None
        prompt_tokens = None
        completion_tokens = None
        output_tokens = 0
        # Bump the per-target load counters immediately before the
        # try/finally so there's no leak window if request-prep above
        # raises (e.g. ``json.dumps`` on a non-serializable body). The
        # matching decrement goes through ``_release_counters`` below,
        # which fires exactly once -- at first token when the A/B flag is
        # on, otherwise in the finally. These counters are read by routing
        # policies on the next request, so an unmatched increment would
        # persistently bias future decisions against this target.
        queued_tokens_at_dispatch = endpoints_queued_tokens.get(target, 0)
        endpoints_queued_tokens[target] = queued_tokens_at_dispatch + request_tokens
        queued_uncached_tokens_at_dispatch = endpoints_queued_uncached_tokens.get(target, 0)
        uncached_tokens_at_dispatch = max(0, request_tokens - cached_for_target)
        endpoints_queued_uncached_tokens[target] = (
            queued_uncached_tokens_at_dispatch + uncached_tokens_at_dispatch
        )
        inflight_requests_at_dispatch = endpoints_inflight_requests.get(target, 0)
        endpoints_inflight_requests[target] = inflight_requests_at_dispatch + 1

        # Single exactly-once release of the queue+prefill load counters.
        # Two callers may invoke it: the first-token callback (only when
        # ``release_at_ttft`` is on) and the ``finally`` block (always).
        # ``counter_released`` makes the second call a no-op so every path
        # -- first-token, end-of-decode, errors before first token, non-SSE
        # / stream=False passthrough, 4xx/5xx, and empty/zero-token
        # generations -- decrements each counter exactly once. The
        # ``if target in <dict>`` guards mirror the original finally so a
        # replica removed mid-request via POST /replicas isn't re-created.
        counter_released = False

        def _release_counters() -> None:
            nonlocal counter_released
            if counter_released:
                return
            counter_released = True
            if target in endpoints_queued_tokens:
                endpoints_queued_tokens[target] = max(
                    0, endpoints_queued_tokens[target] - request_tokens
                )
            if target in endpoints_queued_uncached_tokens:
                endpoints_queued_uncached_tokens[target] = max(
                    0,
                    endpoints_queued_uncached_tokens[target] - uncached_tokens_at_dispatch,
                )
            if target in endpoints_inflight_requests:
                endpoints_inflight_requests[target] = max(
                    0, endpoints_inflight_requests[target] - 1
                )

        # Captured before client.stream(...) so TTFT measured by the SSE
        # tee includes request-send + response-headers latency, matching
        # the contract in proxy/measure.py::consume_sse_stream.
        request_start_ns = time.perf_counter_ns()
        try:
            async with client.stream(
                "POST",
                f"{target}{upstream_path}",
                content=upstream_body,
                headers=upstream_headers,
            ) as upstream:
                upstream_status = upstream.status_code
                # At this point the request body has been sent upstream
                # and the response headers have come back, so the replica
                # has ingested the prompt and (at minimum) started
                # prefill. Record the prefix on the trie so concurrent
                # requests arriving during our streaming phase can see
                # that ``target`` now caches this prefix.
                #
                # Only tag successful dispatches -- a 4xx/5xx likely
                # means the replica never got far enough to cache
                # anything meaningful.
                if token_ids and 200 <= upstream.status_code < 300:
                    try:
                        radix_trie.insert(token_ids, endpoint=target)
                    except Exception as e:
                        # Trie bookkeeping must never break forwarding.
                        _log(f"radix trie insert failed: {e}")

                response_headers = [
                    (k.lower().encode(), v.encode())
                    for k, v in upstream.headers.items()
                    if k.lower() not in _HOP_BY_HOP_HEADERS
                ]
                await send(
                    {
                        "type": "http.response.start",
                        "status": upstream.status_code,
                        "headers": response_headers,
                    }
                )
                headers_sent = True

                is_sse = upstream.headers.get("content-type", "").startswith("text/event-stream")

                async def _sink(chunk: bytes) -> None:
                    """Forward an upstream byte chunk to the ASGI client.
                    Used as the ``chunk_sink`` for ``consume_sse_stream``
                    so passthrough latency is unaffected by the parse loop."""
                    await send(
                        {
                            "type": "http.response.body",
                            "body": chunk,
                            "more_body": True,
                        }
                    )

                if is_sse and 200 <= upstream.status_code < 300:
                    # Tee path: parse SSE for tuning samples while the raw bytes
                    # flow straight through to the client. ``on_first_token``
                    # releases the load counters at TTFT only when the A/B flag
                    # is on; otherwise release stays in ``finally``
                    # (end-of-decode, original behavior).
                    (
                        ttft_ns,
                        output_tokens,
                        prompt_tokens,
                        completion_tokens,
                        meta_info,
                    ) = await consume_sse_stream(
                        upstream,
                        request_start_ns=request_start_ns,
                        chunk_sink=_sink,
                        on_first_token=_release_counters if release_at_ttft else None,
                    )
                    request_trace_event["meta_info"] = meta_info
                    # Feed the live regression accumulator (only while a
                    # calibrate run is active, so tuning/eval traffic doesn't
                    # pollute the fit). Uses proxy-measured TTFT + the proxy's
                    # queued_tokens_at_dispatch -- no engine meta_info.
                    if (state.get("auto_tune") or {}).get("mode") == "calibrate":
                        _accumulate_calibration(
                            target=target,
                            uncached_at_dispatch=uncached_tokens_at_dispatch,
                            queued_at_dispatch=queued_tokens_at_dispatch,
                            ttft_ms=(ttft_ns / 1e6) if ttft_ns is not None else None,
                        )
                    total_ns = time.perf_counter_ns() - request_start_ns
                    _record_request_sample(
                        target=target,
                        ttft_ns=ttft_ns,
                        total_ns=total_ns,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=(
                            completion_tokens if completion_tokens is not None else output_tokens
                        ),
                        cached_tokens_at_dispatch=cached_for_target,
                        queued_tokens_at_dispatch=queued_tokens_at_dispatch,
                        queued_uncached_tokens_at_dispatch=queued_uncached_tokens_at_dispatch,
                        inflight_requests_at_dispatch=inflight_requests_at_dispatch,
                    )
                else:
                    # Plain passthrough for non-SSE bodies (e.g. error
                    # JSON, or a caller that asked for stream=False).
                    async for chunk in upstream.aiter_raw():
                        if chunk:
                            await _sink(chunk)
                    total_ns = time.perf_counter_ns() - request_start_ns
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            request_trace_event["error"] = f"{type(e).__name__}: {e}"
            if not headers_sent:
                await _send_json(send, 502, {"error": f"upstream replica unreachable: {e}"})
                upstream_status = 502
        except httpx.HTTPError as e:
            request_trace_event["error"] = f"{type(e).__name__}: {e}"
            if headers_sent:
                # Already committed to a status; best we can do is close
                # the body cleanly so the client sees a truncated stream.
                _log(f"upstream stream error mid-response: {e}")
                try:
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
                except Exception:
                    pass
            else:
                await _send_json(send, 502, {"error": f"upstream stream error: {e}"})
                upstream_status = 502
        finally:
            request_trace_event.update(
                {
                    "status": upstream_status,
                    "ttft_ns": ttft_ns,
                    "total_ns": total_ns
                    if total_ns is not None
                    else time.perf_counter_ns() - request_start_ns,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": (
                        completion_tokens if completion_tokens is not None else output_tokens
                    ),
                }
            )
            _trace_append("request", request_trace_event)
            # Exactly-once release of the load counters. When the A/B flag
            # is on and the first ``delta.content`` event already fired
            # ``_release_counters`` this is a no-op (``counter_released`` is
            # set); otherwise (flag off, or no token ever emitted -- errors
            # before first token, non-SSE / stream=False passthrough,
            # 4xx/5xx, empty generations) this is where the release
            # happens. The ``if target in <dict>`` guards live inside the
            # helper so a replica removed mid-request via POST /replicas
            # isn't leaked back into the counters.
            _release_counters()

    # ---------- Lifespan ----------

    async def _handle_lifespan(receive, send) -> None:
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                state["upstream_client"] = _new_upstream_client()
                # Prime synchronously so the first inbound request doesn't
                # see empty live_metrics and fall back to random.
                try:
                    await _refresh_all(state["upstream_client"])
                except Exception as e:
                    _log(f"initial metrics refresh failed: {e}")
                state["metrics_task"] = asyncio.create_task(_metrics_refresh_loop())
                _log(
                    f"metrics refresh loop started "
                    f"(interval={METRICS_REFRESH_INTERVAL_SECONDS}s, "
                    f"{len(replica_urls)} replicas); "
                    f"upstream client: http2=True, "
                    f"max_connections={upstream_limits.max_connections}, "
                    f"max_keepalive_connections={upstream_limits.max_keepalive_connections}"
                )
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                batch_task = state["batch_tuning"].get("task")
                if batch_task is not None:
                    state["batch_tuning"]["stop_requested"] = True
                    batch_task.cancel()
                    try:
                        await batch_task
                    except (asyncio.CancelledError, Exception):
                        pass
                workload_task = state["workload_run"].get("task")
                if workload_task is not None:
                    state["workload_run"]["stop_requested"] = True
                    workload_task.cancel()
                    try:
                        await workload_task
                    except (asyncio.CancelledError, Exception):
                        pass
                task = state.get("metrics_task")
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                client = state.get("upstream_client")
                if client is not None:
                    try:
                        await client.aclose()
                    except Exception as e:
                        _log(f"upstream client close failed: {e}")
                    state["upstream_client"] = None
                await send({"type": "lifespan.shutdown.complete"})
                return

    # ---------- ASGI entry point ----------

    async def asgi_app(scope, receive, send):
        if scope["type"] == "lifespan":
            await _handle_lifespan(receive, send)
            return
        if scope["type"] != "http":
            return

        path = scope["path"]
        method = scope["method"]

        if path == "/v1/chat/completions" and method == "POST":
            await _handle_chat_completions(scope, receive, send)
            return

        if await _dispatch_json(method, path, receive, send):
            return

        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"Not found"})

    with modal.forward(8000) as tunnel:
        _log(f"proxy listening at {tunnel.url}")
        if registry_key:
            proxies[registry_key] = tunnel.url
            _log(f"proxy registered in modal dict key={registry_key!r}")
        # TODO: replace uvicorn with a faster reverse-proxy (e.g. nginx, envoy, or Rust-based)
        uvicorn.run(
            asgi_app,
            host="0.0.0.0",
            port=8000,
            log_config=_UVICORN_LOG_CONFIG,
        )
        if registry_key and proxies.get(registry_key) == tunnel.url:
            proxies[registry_key] = ""
