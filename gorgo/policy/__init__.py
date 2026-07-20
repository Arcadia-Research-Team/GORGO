"""Routing policies for GORGO.

Public surface re-exported from this package's submodules:

* :mod:`gorgo.policy.base`      -- shared dataclasses, lazy registry, dispatch
* :mod:`gorgo.policy.lb_aibrix` -- aibrix-derived ``route_*`` policies
* :mod:`gorgo.policy.gorgo`     -- the GORGO policy and its per-target
                                   hyperparameter store

Most callers want one of:

    from gorgo.policy import POLICY_REGISTRY, RouteContext, normalize_policy
    from gorgo.policy.gorgo import make_default_store, effective_hyperparameters

Symbols specific to a single policy (gorgo's hyperparameter store
helpers, aibrix's individual ``route_*`` functions) are deliberately
*not* re-exported here -- they're imported directly from the
submodule that owns them so each call site documents which policy
family it's reaching into.
"""

from gorgo.policy.base import (
    PolicyDef,
    ReplicaSnapshot,
    RouteContext,
    RouteDecision,
    get_policy,
    normalize_policy,
    register_policy,
    route,
    route_random,
    route_session_affinity,
)


def __getattr__(name: str):
    """Forward lazy attributes from :mod:`gorgo.policy.base` so callers
    can write ``from gorgo.policy import POLICY_REGISTRY`` without
    triggering the registry build at package-import time."""
    if name in {"POLICY_REGISTRY", "ROUTING_POLICIES"}:
        from gorgo.policy import base

        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PolicyDef",
    "POLICY_REGISTRY",
    "ROUTING_POLICIES",
    "ReplicaSnapshot",
    "RouteContext",
    "RouteDecision",
    "get_policy",
    "normalize_policy",
    "register_policy",
    "route",
    "route_random",
    "route_session_affinity",
]
