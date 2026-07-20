"""GORGO -- prefix-cache-, load-, and network-aware routing for LLM
replica fleets, with online weight tuning.

The package core is stdlib-only. It provides the pieces a router embeds:

* :mod:`gorgo.policy`     -- the policy registry (``gorgo``, ``gorgo-2d``,
  ``session-affinity``, ``random``, and the aibrix-derived baselines),
  the :class:`RouteContext` -> :class:`RouteDecision` contract, and
  :func:`register_policy` for downstream extensions.
* :mod:`gorgo.radix_trie` -- the prefix->replica KV-cache index.
* :mod:`gorgo.tuner`      -- online (1+1)-ES weight tuning and the
  physical-rate OLS calibration.
* :mod:`gorgo.measure`    -- SSE stream timing and summary statistics.

The caller owns transport and replica lifecycle: maintain the queued/
inflight counters and a :class:`RadixTrie` (``insert`` on dispatch),
build a :class:`RouteContext` per request, and dispatch to
``get_policy(name).fn(ctx).target``. The reference integration is the
GORGO proxy (``proxy/modal_proxy.py`` in the GORGO repository).
"""

from gorgo.policy.base import (
    PolicyDef,
    ReplicaSnapshot,
    RouteContext,
    RouteDecision,
    get_policy,
    normalize_policy,
    register_policy,
    route_random,
    route_session_affinity,
)
from gorgo.policy.gorgo import (
    ALLOWED_HYPERPARAM_KEYS,
    DEFAULT_GORGO_HYPERPARAMETERS,
    effective_hyperparameters,
    make_default_store,
    merge_update,
    prune_per_target,
    route_gorgo,
    route_gorgo_2d,
    validate_update,
)
from gorgo.radix_trie import RadixTrie
from gorgo.tuner import (
    HYPERPARAM_RANGES,
    ONLINE_SCORE_FUNCTIONS,
    Calibration,
    GaussianESTuner,
    OnlineTuner,
    validated_ranges,
)


def __getattr__(name: str):
    """Forward the lazily-built registry attributes so ``gorgo.
    POLICY_REGISTRY`` is always the *current* registry -- an eager
    import here would snapshot the dict and go stale after a later
    :func:`register_policy` call invalidates the cache."""
    if name in {"POLICY_REGISTRY", "ROUTING_POLICIES"}:
        from gorgo.policy import base

        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALLOWED_HYPERPARAM_KEYS",
    "Calibration",
    "DEFAULT_GORGO_HYPERPARAMETERS",
    "GaussianESTuner",
    "HYPERPARAM_RANGES",
    "ONLINE_SCORE_FUNCTIONS",
    "OnlineTuner",
    "POLICY_REGISTRY",
    "PolicyDef",
    "RadixTrie",
    "ReplicaSnapshot",
    "RouteContext",
    "RouteDecision",
    "effective_hyperparameters",
    "get_policy",
    "make_default_store",
    "merge_update",
    "normalize_policy",
    "prune_per_target",
    "register_policy",
    "route_gorgo",
    "route_gorgo_2d",
    "route_random",
    "route_session_affinity",
    "validate_update",
    "validated_ranges",
]
