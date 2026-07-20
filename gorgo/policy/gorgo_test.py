from gorgo.policy.base import ReplicaSnapshot, RouteContext
from gorgo.policy.gorgo import (
    make_default_store,
    merge_update,
    prune_per_target,
    route_gorgo,
    route_gorgo_2d,
    validate_update,
)
from gorgo.radix_trie import RadixTrie


def snap(rtt: float) -> ReplicaSnapshot:
    return ReplicaSnapshot(
        num_running_reqs=0, num_queue_reqs=0, num_used_tokens=0, latency=rtt, network_rtt=rtt
    )


def make_ctx(
    *,
    metrics,
    queued=None,
    trie=None,
    token_ids=None,
    store=None,
):
    urls = list(metrics)
    token_ids = token_ids or []
    return RouteContext(
        replica_urls=urls,
        metrics=metrics,
        endpoints_queued_tokens=queued or {},
        endpoints_queued_uncached_tokens={},
        endpoints_inflight_requests={},
        radix_trie=trie or RadixTrie(),
        token_ids=token_ids,
        request_tokens=len(token_ids),
        hyperparameters=store or make_default_store(),
    )


def test_route_gorgo_prefers_low_rtt_all_else_equal():
    ctx = make_ctx(metrics={"http://near": snap(0.010), "http://far": snap(0.200)})
    d = route_gorgo(ctx)
    assert d.target == "http://near"
    assert d.scores is not None and set(d.scores) == {"http://near", "http://far"}


def test_route_gorgo_prefers_cache_hits():
    trie = RadixTrie()
    prompt = list(range(1000))
    trie.insert(prompt, endpoint="http://cached")
    ctx = make_ctx(
        metrics={"http://cached": snap(0.05), "http://cold": snap(0.05)},
        trie=trie,
        token_ids=prompt,
    )
    assert route_gorgo(ctx).target == "http://cached"


def test_route_gorgo_avoids_loaded_replica():
    ctx = make_ctx(
        metrics={"http://busy": snap(0.05), "http://idle": snap(0.05)},
        queued={"http://busy": 50_000, "http://idle": 0},
    )
    assert route_gorgo(ctx).target == "http://idle"


def test_route_gorgo_weights_shift_the_decision():
    # Heavy rtt_weight makes a 10ms RTT edge beat a full cache hit.
    trie = RadixTrie()
    prompt = list(range(100))
    trie.insert(prompt, endpoint="http://far-cached")
    store = merge_update(make_default_store(), {"defaults": {"rtt_weight": 100.0}}, replace=False)
    ctx = make_ctx(
        metrics={"http://far-cached": snap(0.200), "http://near-cold": snap(0.010)},
        trie=trie,
        token_ids=prompt,
        store=store,
    )
    assert route_gorgo(ctx).target == "http://near-cold"


def test_route_gorgo_empty_metrics_falls_back_random():
    ctx = make_ctx(metrics={})
    ctx = RouteContext(
        replica_urls=["http://a"],
        metrics={},
        endpoints_queued_tokens={},
        endpoints_queued_uncached_tokens={},
        endpoints_inflight_requests={},
        radix_trie=RadixTrie(),
        token_ids=[],
        request_tokens=0,
        hyperparameters=make_default_store(),
    )
    d = route_gorgo(ctx)
    assert d.fallback_reason == "empty-candidates"


def test_route_gorgo_2d_uses_calibrated_rates():
    # queue_rate=0 makes load free; the busy-but-cached replica wins.
    trie = RadixTrie()
    prompt = list(range(500))
    trie.insert(prompt, endpoint="http://busy")
    store = merge_update(
        make_default_store(),
        {"defaults": {"prefill_rate": 1.0, "queue_rate": 0.0}},
        replace=False,
    )
    ctx = make_ctx(
        metrics={"http://busy": snap(0.05), "http://idle": snap(0.05)},
        queued={"http://busy": 10_000, "http://idle": 0},
        trie=trie,
        token_ids=prompt,
        store=store,
    )
    assert route_gorgo_2d(ctx).target == "http://busy"


def test_validate_and_merge_update():
    upd, err = validate_update({"rtt_weight": 2})
    assert err is None and upd["defaults"] == {"rtt_weight": 2.0}
    _, err = validate_update({"nope": 1})
    assert err is not None
    _, err = validate_update({"defaults": {"rtt_weight": 1}, "rtt_weight": 1})
    assert err is not None  # mixed shapes rejected

    store = merge_update(make_default_store(), upd, replace=False)
    assert store["defaults"]["rtt_weight"] == 2.0
    reset = merge_update(store, {"defaults": {}, "per_target": {}}, replace=True)
    assert reset["defaults"]["rtt_weight"] == 1.0


def test_prune_per_target():
    store = make_default_store()
    store["per_target"] = {"http://gone": {"rtt_weight": 3.0}, "http://live": {}}
    prune_per_target(store, {"http://live"})
    assert list(store["per_target"]) == ["http://live"]
