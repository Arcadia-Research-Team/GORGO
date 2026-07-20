"""Aibrix-derived load-balancing policies.

Names mirror Aibrix's ``pkg/plugins/gateway/algorithms`` (vLLM /
aibrix), adapted to the metrics this proxy scrapes from SGLang
``/metrics``. Policies that Aibrix backs with K8s cache / Redis /
SLO queues are approximated; see individual docstrings.

This module deliberately does not own the registry, the route
context, or any non-aibrix policy. The shared infrastructure lives
in :mod:`policy.base`; the GORGO policy lives in :mod:`policy.gorgo`.
Each ``route_*`` function below is exported via the
:data:`AIBRIX_POLICIES` list at the bottom, which
:mod:`policy.base` composes into the final ``POLICY_REGISTRY``.

Every ``route_*`` function returns a :class:`RouteDecision`. Most
return ``RouteDecision(target=url)`` with no fallback reason. Policies
with internal preconditions that can't be met (e.g. ``power-of-two``
needs >= 2 candidates, ``simple-session-affinity`` needs token ids) set
``fallback_reason`` to a short kebab-case string when they bail out to
``random.choice`` -- this is propagated by ``proxy/modal_proxy.py``
into the trace's ``effective_policy`` field as
``"random-fallback:internal:<policy-name>:<reason>"`` so post-hoc
analysis can tell apart "policy fired" from "policy bailed out and
rolled dice."
"""

from __future__ import annotations

import random
import statistics
from typing import TYPE_CHECKING, Callable

from gorgo.policy.base import PolicyDef, ReplicaSnapshot, RouteDecision, route_random

if TYPE_CHECKING:
    from gorgo.radix_trie import RadixTrie


def _tie_break_min(candidates: list[str], score: Callable[[str], float]) -> str:
    best = min(score(u) for u in candidates)
    tied = [u for u in candidates if score(u) == best]
    return random.choice(tied)


def _tie_break_max(candidates: list[str], score: Callable[[str], float]) -> str:
    best = max(score(u) for u in candidates)
    tied = [u for u in candidates if score(u) == best]
    return random.choice(tied)


def route_power_of_two(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
) -> RouteDecision:
    """Aibrix ``power-of-two``: two random choices, pick lower load.

    Load = ``num_used_tokens + queued_prompt_tokens`` (GORGO's existing signal).
    """
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return RouteDecision(random.choice(replica_urls), "empty-candidates")
    if len(candidates) < 2:
        # One candidate: trivially the answer, but the policy's
        # power-of-two scoring path didn't run, so flag it.
        return RouteDecision(candidates[0], "insufficient-candidates")
    a, b = random.sample(candidates, 2)
    la = metrics[a].num_used_tokens + endpoints_queued_tokens.get(a, 0)
    lb = metrics[b].num_used_tokens + endpoints_queued_tokens.get(b, 0)
    return RouteDecision(a if la <= lb else b)


def route_least_request(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_inflight_requests: dict[str, int],
) -> RouteDecision:
    """``least-request``: minimize in-flight request count per replica.

    Scores by ``max(snap.num_running_reqs, inflight[u])`` rather than
    SGLang's metric alone:

    * ``snap.num_running_reqs`` reflects what SGLang's scheduler sees
      *as of the last metrics scrape*, including this proxy's traffic
      and any other clients of the same replica. Frozen between scrapes
      (default 30s on this proxy), so without a corrective term the
      policy herds all between-scrape requests onto whichever replica
      was the snapshot's minimum.
    * ``endpoints_inflight_requests[u]`` is this proxy's own
      dispatched-but-not-completed counter, updated synchronously on
      every dispatch and completion. Fresh, but blind to other clients
      in shared-pool deployments.

    ``max`` of the two is the honest non-double-counting estimate:
    fresh metrics dominate when they're caught up; the local counter
    takes over during a stale window. In steady state with a single
    proxy on the replica they converge.
    """
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return RouteDecision(random.choice(replica_urls), "empty-candidates")
    return RouteDecision(
        _tie_break_min(
            candidates,
            lambda u: float(
                max(metrics[u].num_running_reqs, endpoints_inflight_requests.get(u, 0))
            ),
        )
    )


def route_least_load(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
) -> RouteDecision:
    """``least-load``: minimize running + queue + proxy queued tokens + used KV tokens."""
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return RouteDecision(random.choice(replica_urls), "empty-candidates")
    return RouteDecision(
        _tie_break_min(
            candidates,
            lambda u: metrics[u].combined_load(endpoints_queued_tokens.get(u, 0)),
        )
    )


def route_least_kv_cache(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> RouteDecision:
    """``least-kv-cache``: minimize ``sglang:num_used_tokens``."""
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return RouteDecision(random.choice(replica_urls), "empty-candidates")
    return RouteDecision(_tie_break_min(candidates, lambda u: float(metrics[u].num_used_tokens)))


def route_least_gpu_cache(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> RouteDecision:
    """``least-gpu-cache``: same as KV for homogeneous SGLang (single pool)."""
    return route_least_kv_cache(replica_urls, metrics)


def route_least_latency(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> RouteDecision:
    """``least-latency``: minimize last /metrics scrape RTT (proxy-side latency)."""
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return RouteDecision(random.choice(replica_urls), "empty-candidates")
    return RouteDecision(_tie_break_min(candidates, lambda u: metrics[u].latency))


def route_least_utilization(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> RouteDecision:
    """``least-utilization``: minimize ``sglang:utilization``."""
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return RouteDecision(random.choice(replica_urls), "empty-candidates")
    return RouteDecision(_tie_break_min(candidates, lambda u: metrics[u].utilization))


def route_least_busy_time(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> RouteDecision:
    """``least-busy-time``: Aibrix uses GPU busy ratio; we proxy with utilization."""
    return route_least_utilization(replica_urls, metrics)


def route_throughput(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> RouteDecision:
    """``throughput``: prefer highest ``sglang:gen_throughput`` (tokens/s)."""
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return RouteDecision(random.choice(replica_urls), "empty-candidates")
    return RouteDecision(_tie_break_max(candidates, lambda u: metrics[u].gen_throughput))


def route_pack_load(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
) -> RouteDecision:
    """``pack-load``: maximize load among replicas still under a soft cap (pack work).

    Aibrix uses pull-mode utilization + cap; we maximize ``combined_load`` capped
    at ``cap = median_load + 2*MAD`` when MAD>0 else +inf.
    """
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return RouteDecision(random.choice(replica_urls), "empty-candidates")
    loads = [metrics[u].combined_load(endpoints_queued_tokens.get(u, 0)) for u in candidates]
    med = statistics.median(loads)
    if len(loads) >= 2:
        sorted_loads = sorted(loads)
        deviations = [abs(x - med) for x in sorted_loads]
        mad = statistics.median(deviations) if deviations else 0.0
    else:
        mad = 0.0
    cap = med + 2.0 * mad + 1e-6
    under = [
        u for u in candidates if metrics[u].combined_load(endpoints_queued_tokens.get(u, 0)) <= cap
    ]
    pool = under if under else candidates
    return RouteDecision(
        _tie_break_max(
            pool,
            lambda u: metrics[u].combined_load(endpoints_queued_tokens.get(u, 0)),
        )
    )


def route_queue_router(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
) -> RouteDecision:
    """``queue-router``: Aibrix wraps a queue + backend; we route to min queue depth."""
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return RouteDecision(random.choice(replica_urls), "empty-candidates")
    return RouteDecision(_tie_break_min(candidates, lambda u: float(metrics[u].num_queue_reqs)))


def route_prefix_cache(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
    endpoints_inflight_requests: dict[str, int],
    radix_trie: RadixTrie,
    token_ids: list[int],
    *,
    imbalance_abs: int = 8,
    std_factor: float = 2.0,
) -> RouteDecision:
    """Aibrix-style prefix cache routing (see algorithms README).

    If running-request imbalance > ``imbalance_abs``, use least-request.
    Else among replicas with best radix prefix match, pick running < mean + factor*std;
    if none qualify, least-request among matches; if no matches, least-request global.

    Delegations to ``route_least_request`` / ``route_least_load`` propagate
    those policies' fallback reasons unchanged.
    """
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return RouteDecision(random.choice(replica_urls), "empty-candidates")

    running = [metrics[u].num_running_reqs for u in candidates]
    if running and max(running) - min(running) > imbalance_abs:
        return route_least_request(candidates, metrics, endpoints_inflight_requests)

    if not token_ids:
        return route_least_load(candidates, metrics, endpoints_queued_tokens)

    cached = radix_trie.cached_prefix_lengths(token_ids, candidates)
    best_cached = max(cached.values()) if cached else 0
    if best_cached <= 0:
        return route_least_request(candidates, metrics, endpoints_inflight_requests)

    match_urls = [u for u in candidates if cached.get(u, 0) == best_cached]
    mean_r = statistics.mean(metrics[u].num_running_reqs for u in candidates)
    if len(candidates) >= 2:
        std_r = statistics.stdev(metrics[u].num_running_reqs for u in candidates)
    else:
        std_r = 0.0
    threshold = mean_r + std_factor * std_r

    qualified = [u for u in match_urls if float(metrics[u].num_running_reqs) <= threshold]
    pool = qualified if qualified else match_urls
    return RouteDecision(_tie_break_min(pool, lambda u: float(metrics[u].num_running_reqs)))


def route_simple_session_affinity(
    replica_urls: list[str],
    token_ids: list[int],
) -> RouteDecision:
    """Sticky routing from prompt token hash (no client IP in proxy)."""
    if not token_ids:
        return RouteDecision(random.choice(replica_urls), "missing-token-ids")
    h = hash(tuple(token_ids[:256]))
    return RouteDecision(replica_urls[h % len(replica_urls)])


def route_vtc_basic(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
) -> RouteDecision:
    """``vtc-basic`` needs per-client fairness; without user id we blend load + util."""
    candidates = [u for u in replica_urls if u in metrics]
    if not candidates:
        return RouteDecision(random.choice(replica_urls), "empty-candidates")
    return RouteDecision(
        _tie_break_min(
            candidates,
            lambda u: (
                metrics[u].combined_load(endpoints_queued_tokens.get(u, 0))
                + 10.0 * metrics[u].utilization
            ),
        )
    )


def route_fallback(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_inflight_requests: dict[str, int],
) -> RouteDecision:
    """Aibrix fallback defaults to least-request."""
    return route_least_request(replica_urls, metrics, endpoints_inflight_requests)


def route_slo_family(
    replica_urls: list[str],
    metrics: dict[str, ReplicaSnapshot],
    endpoints_queued_tokens: dict[str, int],
    variant: str,
) -> RouteDecision:
    """SLO routers in Aibrix use queues; we map variants to load heuristics."""
    if variant == "slo-pack-load":
        return route_pack_load(replica_urls, metrics, endpoints_queued_tokens)
    if variant in ("slo-least-load", "slo-least-load-pulling", "slo"):
        return route_least_load(replica_urls, metrics, endpoints_queued_tokens)
    return route_least_load(replica_urls, metrics, endpoints_queued_tokens)


def route_pd_stub(replica_urls: list[str]) -> RouteDecision:
    """Prefill/decode split not modeled; random."""
    return route_random(replica_urls)


# ---------------------------------------------------------------------------
# Registry export
# ---------------------------------------------------------------------------

AIBRIX_POLICIES: list[PolicyDef] = [
    PolicyDef(
        "power-of-two",
        True,
        lambda c: route_power_of_two(c.replica_urls, c.metrics, c.endpoints_queued_tokens),
    ),
    PolicyDef(
        "least-request",
        True,
        lambda c: route_least_request(c.replica_urls, c.metrics, c.endpoints_inflight_requests),
    ),
    PolicyDef(
        "least-load",
        True,
        lambda c: route_least_load(c.replica_urls, c.metrics, c.endpoints_queued_tokens),
    ),
    PolicyDef("least-kv-cache", True, lambda c: route_least_kv_cache(c.replica_urls, c.metrics)),
    PolicyDef("least-gpu-cache", True, lambda c: route_least_gpu_cache(c.replica_urls, c.metrics)),
    PolicyDef("least-latency", True, lambda c: route_least_latency(c.replica_urls, c.metrics)),
    PolicyDef(
        "least-utilization", True, lambda c: route_least_utilization(c.replica_urls, c.metrics)
    ),
    PolicyDef("least-busy-time", True, lambda c: route_least_busy_time(c.replica_urls, c.metrics)),
    PolicyDef("throughput", True, lambda c: route_throughput(c.replica_urls, c.metrics)),
    PolicyDef(
        "pack-load",
        True,
        lambda c: route_pack_load(c.replica_urls, c.metrics, c.endpoints_queued_tokens),
    ),
    PolicyDef(
        "prefix-cache",
        True,
        lambda c: route_prefix_cache(
            c.replica_urls,
            c.metrics,
            c.endpoints_queued_tokens,
            c.endpoints_inflight_requests,
            c.radix_trie,
            c.token_ids,
        ),
    ),
    PolicyDef("queue-router", True, lambda c: route_queue_router(c.replica_urls, c.metrics)),
    PolicyDef(
        "simple-session-affinity",
        False,
        lambda c: route_simple_session_affinity(c.replica_urls, c.token_ids),
    ),
    PolicyDef(
        "vtc",
        True,
        lambda c: route_vtc_basic(c.replica_urls, c.metrics, c.endpoints_queued_tokens),
    ),
    PolicyDef(
        "vtc-basic",
        True,
        lambda c: route_vtc_basic(c.replica_urls, c.metrics, c.endpoints_queued_tokens),
    ),
    PolicyDef(
        "slo",
        True,
        lambda c: route_slo_family(c.replica_urls, c.metrics, c.endpoints_queued_tokens, "slo"),
    ),
    PolicyDef(
        "slo-pack-load",
        True,
        lambda c: route_slo_family(
            c.replica_urls, c.metrics, c.endpoints_queued_tokens, "slo-pack-load"
        ),
    ),
    PolicyDef(
        "slo-least-load",
        True,
        lambda c: route_slo_family(
            c.replica_urls, c.metrics, c.endpoints_queued_tokens, "slo-least-load"
        ),
    ),
    PolicyDef(
        "slo-least-load-pulling",
        True,
        lambda c: route_slo_family(
            c.replica_urls, c.metrics, c.endpoints_queued_tokens, "slo-least-load-pulling"
        ),
    ),
    PolicyDef(
        "fallback",
        True,
        lambda c: route_fallback(c.replica_urls, c.metrics, c.endpoints_inflight_requests),
    ),
    # Explicitly not supported without PD split / Redis tracker -> stubbed to random.
    PolicyDef("pd", False, lambda c: route_pd_stub(c.replica_urls)),
    PolicyDef("pd-disaggregation", False, lambda c: route_pd_stub(c.replica_urls)),
]
