import pytest

from gorgo.policy import base
from gorgo.policy.base import (
    PolicyDef,
    RouteContext,
    get_policy,
    normalize_policy,
    register_policy,
    route_session_affinity,
)
from gorgo.radix_trie import RadixTrie


def make_ctx(urls, *, affinity_key=None, token_ids=None):
    return RouteContext(
        replica_urls=list(urls),
        metrics={},
        endpoints_queued_tokens={},
        endpoints_queued_uncached_tokens={},
        endpoints_inflight_requests={},
        radix_trie=RadixTrie(),
        token_ids=token_ids or [],
        request_tokens=len(token_ids or []),
        hyperparameters={},
        affinity_key=affinity_key,
    )


def test_registry_contains_core_policies():
    registry = base.POLICY_REGISTRY
    for name in ("random", "session-affinity", "gorgo", "gorgo-2d", "least-request"):
        assert name in registry
    assert not get_policy("session-affinity").needs_metrics


def test_normalize_policy():
    assert normalize_policy("Power_Of_Two ") == "power-of-two"
    with pytest.raises(ValueError):
        get_policy("definitely-not-a-policy")


def test_register_policy_extends_and_rejects_collisions():
    marker = PolicyDef("test-external", False, lambda c: None)
    register_policy(marker)
    try:
        assert "test-external" in base.POLICY_REGISTRY
        with pytest.raises(ValueError):
            register_policy(PolicyDef("test-external", False, lambda c: None))
        with pytest.raises(ValueError):
            register_policy(PolicyDef("random", False, lambda c: None))
    finally:
        base._EXTERNAL_POLICIES.clear()
        base._POLICY_REGISTRY_CACHE = None


def test_session_affinity_is_deterministic_and_sticky():
    urls = [f"http://r{i}" for i in range(5)]
    d1 = route_session_affinity(make_ctx(urls, affinity_key="sess-42"))
    d2 = route_session_affinity(make_ctx(urls, affinity_key="sess-42"))
    assert d1.target == d2.target
    assert d1.fallback_reason is None
    # Different sessions spread across replicas (statistically certain
    # with 200 sessions over 5 replicas under any reasonable hash).
    targets = {
        route_session_affinity(make_ctx(urls, affinity_key=f"s{i}")).target for i in range(200)
    }
    assert len(targets) == len(urls)


def test_session_affinity_churn_stability():
    urls = [f"http://r{i}" for i in range(5)]
    keys = [f"sess-{i}" for i in range(300)]
    before = {k: route_session_affinity(make_ctx(urls, affinity_key=k)).target for k in keys}
    removed = "http://r2"
    shrunk = [u for u in urls if u != removed]
    after = {k: route_session_affinity(make_ctx(shrunk, affinity_key=k)).target for k in keys}
    # Rendezvous hashing: only sessions pinned to the removed replica move.
    for k in keys:
        if before[k] != removed:
            assert after[k] == before[k]
    # And they return home when the replica comes back.
    restored = {k: route_session_affinity(make_ctx(urls, affinity_key=k)).target for k in keys}
    assert restored == before


def test_session_affinity_missing_key_falls_back_random():
    d = route_session_affinity(make_ctx(["http://a", "http://b"], affinity_key=None))
    assert d.fallback_reason == "missing-affinity-key"
    assert d.target in ("http://a", "http://b")
