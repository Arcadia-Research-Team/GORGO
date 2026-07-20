"""GORGO routing policy and per-target hyperparameter store.

The GORGO policy scores each replica by a linear cost model and picks
the minimum.  Three terms, three weights, all learned by the ES tuner::

    score(u) = rtt_weight      * rtt_ms(u)
             + prefill_weight  * uncached_tokens(u)
             + load_weight     * queued_tokens(u)

Each term captures a distinct cost source:

* **Network** — ``rtt_ms``, a directly measured round-trip time.
  Scaled by ``rtt_weight`` (units: dimensionless, but interpretable
  as "ms of score per ms of RTT").

* **Own prefill** — ``uncached_tokens``, the prompt tokens not in
  the replica's KV cache.  Scaled by ``prefill_weight`` (units:
  implicitly ms/tok — absorbs the hardware prefill speed).

* **Queue load** — ``queued_tokens``, the prompt tokens already
  dispatched to the replica but not yet completed.  Scaled by
  ``load_weight`` (units: implicitly ms/tok — absorbs the
  load-dependent queue drain rate).  Decoupled from
  ``prefill_weight`` so the tuner can independently balance
  "prefer cached replicas" vs "avoid loaded replicas".

All three weights are searched by the ``online-es`` tuner.  There are
no physical rates (``prefill_rate``, ``queue_rate``) — the ES absorbs
hardware speed into the weights directly, avoiding the complexity of
idle calibration and load-dependent rate fitting.

``rtt_ms`` is the probe RTT converted from seconds to milliseconds
(``rtt * 1000``).  The raw probe value comes from
``snap.network_rtt`` (EWMA-smoothed lightweight probe populated by
``proxy/modal_proxy.py``), falling back to ``snap.latency`` (the
``/metrics`` scrape RTT) when no probe has completed yet.

This module also owns the *hyperparameter store* shape::

    {
        "defaults":   {"rtt_weight": ..., "prefill_weight": ...,
                       "load_weight": ...},
        "per_target": {}
    }

``per_target`` is retained for forward compatibility (e.g. mixed-GPU
fleets) but is empty under the 3-weight model — all weights are
fleet-wide in ``defaults``.
"""

from __future__ import annotations

from typing import Any

from gorgo.policy.base import PolicyDef, RouteContext, RouteDecision, route_random

# ---------------------------------------------------------------------------
# Hyperparameter schema
# ---------------------------------------------------------------------------

DEFAULT_GORGO_HYPERPARAMETERS: dict[str, float] = {
    "rtt_weight": 1.0,
    "prefill_weight": 1.0,
    "load_weight": 1.0,
    "queue_weight": 1.0,
    # ms-normalized 2D model: physical rates calibrated live (see the
    # calibration phase) so every term resolves to milliseconds. Defaults of
    # 1.0 reproduce the prior token-unit behavior until rates are patched in.
    "prefill_rate": 1.0,
    "queue_rate": 1.0,
}

ALLOWED_HYPERPARAM_KEYS: frozenset[str] = frozenset(DEFAULT_GORGO_HYPERPARAMETERS)


def make_default_store() -> dict[str, Any]:
    """Fresh hyperparameter store with no per-target overrides."""
    return {
        "defaults": dict(DEFAULT_GORGO_HYPERPARAMETERS),
        "per_target": {},
    }


def effective_hyperparameters(store: dict[str, Any], target: str) -> dict[str, float]:
    """Resolve the *effective* hyperparameters for ``target`` by
    layering its per-target overrides on top of the global defaults.
    """
    defaults = store.get("defaults") or {}
    per_target = store.get("per_target") or {}
    override = per_target.get(target) or {}
    merged = dict(DEFAULT_GORGO_HYPERPARAMETERS)
    merged.update(defaults)
    merged.update(override)
    return merged


def validate_update(
    data: Any,
    *,
    known_targets: set[str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a body bound for ``POST /hyperparameters`` and return
    a normalized update dict (or ``(None, error)``).

    Two body shapes are accepted:

    1. **Flat** -- ``{"rtt_weight": X, "prefill_weight": Y}`` (or any
       subset of allowed keys). Written to ``defaults`` only.

    2. **Structured** -- ``{"defaults": {...}, "per_target": {url:
       {...}}}``. Either branch is optional.
    """
    if not isinstance(data, dict):
        return None, "body must be a JSON object"

    has_structured_keys = "defaults" in data or "per_target" in data
    flat_keys = {k for k in data if k not in {"defaults", "per_target"}}

    if has_structured_keys and flat_keys:
        return None, (
            "body cannot mix flat hyperparameter keys with 'defaults'/'per_target'; pick one shape"
        )

    out: dict[str, Any] = {"defaults": {}, "per_target": {}}

    def _coerce_block(block: Any, where: str) -> tuple[dict[str, float] | None, str | None]:
        if not isinstance(block, dict):
            return None, f"{where} must be a JSON object"
        unknown = sorted(k for k in block if k not in ALLOWED_HYPERPARAM_KEYS)
        if unknown:
            return None, f"unknown hyperparameter(s) under {where}: {unknown}"
        try:
            return {k: float(v) for k, v in block.items()}, None
        except (TypeError, ValueError):
            return None, f"hyperparameter values under {where} must be numeric"

    if has_structured_keys:
        if "defaults" in data:
            block, err = _coerce_block(data["defaults"], "defaults")
            if err:
                return None, err
            out["defaults"] = block or {}
        if "per_target" in data:
            pt = data["per_target"]
            if not isinstance(pt, dict):
                return None, "'per_target' must be a JSON object keyed by replica URL"
            for url, block in pt.items():
                if not isinstance(url, str) or not url:
                    return None, "per_target keys must be non-empty replica URL strings"
                if known_targets is not None and url not in known_targets:
                    return None, (f"per_target URL {url!r} is not a currently-registered replica")
                block, err = _coerce_block(block, f"per_target[{url!r}]")
                if err:
                    return None, err
                out["per_target"][url] = block or {}
    else:
        block, err = _coerce_block(data, "body")
        if err:
            return None, err
        out["defaults"] = block or {}

    return out, None


def merge_update(
    store: dict[str, Any],
    update: dict[str, Any],
    *,
    replace: bool,
) -> dict[str, Any]:
    """Apply a normalized ``update`` to ``store`` and return the new
    store.

    * ``replace=False`` (POST/PATCH) -- key-level merge.
    * ``replace=True`` (PUT) -- reset to factory defaults, then apply.
    """
    if replace:
        out: dict[str, Any] = {
            "defaults": dict(DEFAULT_GORGO_HYPERPARAMETERS),
            "per_target": {},
        }
    else:
        out = {
            "defaults": dict(store.get("defaults") or DEFAULT_GORGO_HYPERPARAMETERS),
            "per_target": {
                url: dict(block) for url, block in (store.get("per_target") or {}).items()
            },
        }

    out["defaults"].update(update.get("defaults") or {})
    for url, block in (update.get("per_target") or {}).items():
        merged_block = dict(out["per_target"].get(url, {}))
        merged_block.update(block)
        out["per_target"][url] = merged_block
    return out


def prune_per_target(store: dict[str, Any], known_targets: set[str]) -> dict[str, Any]:
    """Drop per-target entries whose URL is no longer a registered replica."""
    pt = store.get("per_target") or {}
    for url in list(pt):
        if url not in known_targets:
            pt.pop(url, None)
    return store


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_gorgo(ctx: RouteContext) -> RouteDecision:
    """Score each replica and pick the minimum.

    ``score(u) = rtt_weight * rtt_ms + prefill_weight * uncached + load_weight * queued``
    """
    if not ctx.replica_urls:
        raise ValueError("no replicas")

    endpoints_cached_tokens = (
        ctx.radix_trie.cached_prefix_lengths(ctx.token_ids, ctx.replica_urls)
        if ctx.token_ids
        else {u: 0 for u in ctx.replica_urls}
    )
    scores: dict[str, float] = {}
    for u in ctx.replica_urls:
        snap = ctx.metrics.get(u)
        if snap is None:
            continue
        eff = effective_hyperparameters(ctx.hyperparameters, u)
        cached = endpoints_cached_tokens.get(u, 0)
        uncached = max(0, ctx.request_tokens - cached)
        queued = ctx.endpoints_queued_tokens.get(u, 0)

        rtt = snap.network_rtt if snap.network_rtt > 0.0 else snap.latency
        rtt_cost = eff["rtt_weight"] * (rtt * 1000.0)
        prefill_cost = eff["prefill_weight"] * uncached
        load_cost = eff["load_weight"] * queued

        scores[u] = rtt_cost + prefill_cost + load_cost
    if not scores:
        return RouteDecision(route_random(ctx.replica_urls).target, "empty-candidates", None)
    return RouteDecision(min(scores, key=scores.get), None, scores)


def route_gorgo_2d(ctx: RouteContext) -> RouteDecision:
    """Score each replica with the ms-normalized 2D GORGO model.

    ``score(u) = rtt_weight * rtt_ms
               + prefill_rate * uncached
               + queue_rate * queue_weight * queued``

    ``prefill_rate`` (ms / uncached token) and ``queue_rate`` (ms / queued
    token) are physical constants calibrated live from SGLang ``meta_info``
    so all three terms resolve to milliseconds; the dimensionless
    ``rtt_weight`` / ``queue_weight`` knobs (~1.0) are what the ES tunes on
    top. With the default rates of 1.0 this reduces to the prior
    token-unit model (``rtt_weight*rtt + uncached + queue_weight*queued``).

    The own-prefill term stays cache-aware: ``uncached`` counts only the
    current request's tokens that miss the replica's KV cache. The load
    term uses ``queued`` -- the raw count of all prompt tokens already
    dispatched to the replica and not yet completed (released at first token
    when the ``load_release_at_ttft`` proxy flag is on), *not* discounted by
    cache hits, so a queue full of cache-hit work is still seen as load.
    """
    if not ctx.replica_urls:
        raise ValueError("no replicas")

    endpoints_cached_tokens = (
        ctx.radix_trie.cached_prefix_lengths(ctx.token_ids, ctx.replica_urls)
        if ctx.token_ids
        else {u: 0 for u in ctx.replica_urls}
    )
    scores: dict[str, float] = {}
    for u in ctx.replica_urls:
        snap = ctx.metrics.get(u)
        if snap is None:
            continue
        eff = effective_hyperparameters(ctx.hyperparameters, u)
        cached = endpoints_cached_tokens.get(u, 0)
        uncached = max(0, ctx.request_tokens - cached)
        queued = ctx.endpoints_queued_tokens.get(u, 0)

        rtt = snap.network_rtt if snap.network_rtt > 0.0 else snap.latency
        rtt_cost = eff["rtt_weight"] * (rtt * 1000.0)
        prefill_cost = eff["prefill_rate"] * uncached
        queue_cost = eff["queue_rate"] * eff["queue_weight"] * queued

        scores[u] = rtt_cost + prefill_cost + queue_cost
    if not scores:
        return RouteDecision(route_random(ctx.replica_urls).target, "empty-candidates", None)
    return RouteDecision(min(scores, key=scores.get), None, scores)


# ---------------------------------------------------------------------------
# Registry export
# ---------------------------------------------------------------------------

GORGO_POLICIES: list[PolicyDef] = [
    PolicyDef("gorgo", needs_metrics=True, fn=route_gorgo),
    PolicyDef("gorgo-2d", needs_metrics=True, fn=route_gorgo_2d),
]
