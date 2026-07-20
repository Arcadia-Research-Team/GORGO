import random

import pytest

from gorgo.policy.gorgo import make_default_store
from gorgo.tuner import (
    Calibration,
    GaussianESTuner,
    OnlineTuner,
    validated_ranges,
)

RANGES = {"rtt_weight": (1e-5, 5.0), "prefill_weight": (1e-5, 5.0)}


def test_validated_ranges():
    merged = validated_ranges({"load_weight": (0.1, 2.0)})
    assert set(merged) == {"rtt_weight", "prefill_weight", "load_weight"}
    with pytest.raises(ValueError):
        validated_ranges({"x": (0.0, 1.0)})  # lo must be > 0 (log-space)
    with pytest.raises(ValueError):
        validated_ranges({"x": (2.0, 1.0)})


def test_es_tuner_baseline_then_search():
    tuner = GaussianESTuner({"rtt_weight": 1.0, "prefill_weight": 1.0}, RANGES, seed=7)
    # First propose returns the incumbent (baseline evaluation).
    baseline = tuner.propose()
    assert baseline == tuner.best_params
    assert tuner.report(baseline, score=-1.0)  # first report sets the baseline
    cand = tuner.propose()
    assert cand is not None and cand != {}
    for k, v in cand.items():
        lo, hi = RANGES[k]
        assert lo <= v <= hi
    # A clearly better candidate is accepted and becomes the incumbent.
    assert tuner.report(cand, score=-0.5)
    assert tuner.best_params == cand and tuner.best_score == -0.5


def test_es_tuner_one_fifth_rule_shrinks_sigma_on_failures():
    tuner = GaussianESTuner({"rtt_weight": 1.0}, {"rtt_weight": (1e-5, 5.0)}, seed=1)
    tuner.report(tuner.propose(), score=0.0)
    sigma0 = tuner.sigma
    for _ in range(tuner.success_window):
        cand = tuner.propose()
        assert cand is not None
        tuner.report(cand, score=-1.0)  # every candidate is worse
    assert tuner.sigma < sigma0


def test_es_tuner_stops_after_max_steps():
    tuner = GaussianESTuner({"rtt_weight": 1.0}, {"rtt_weight": (1e-5, 5.0)}, max_steps=3, seed=2)
    tuner.report(tuner.propose(), score=0.0)
    for _ in range(3):
        cand = tuner.propose()
        assert cand is not None
        tuner.report(cand, score=-1.0)
    assert tuner.propose() is None


def test_calibration_recovers_known_rates():
    # Synthesize ttft = intercept_r + P*uncached + Q*queued exactly and
    # check the OLS solve recovers P and Q.
    P, Q = 0.35, 0.02
    intercepts = {"http://a": 12.0, "http://b": 30.0}
    cal = Calibration()
    rng = random.Random(3)
    for _ in range(200):
        target = rng.choice(list(intercepts))
        u = rng.randint(1, 4000)
        q = rng.randint(0, 60_000)
        ttft = intercepts[target] + P * u + Q * q
        cal.add(target=target, uncached_at_dispatch=u, queued_at_dispatch=q, ttft_ms=ttft)
    rates = cal.rates()
    assert rates["prefill_rate"] == pytest.approx(P, rel=1e-6)
    assert rates["queue_rate"] == pytest.approx(Q, rel=1e-6)
    per = rates["diagnostics"]["per_replica_intercept_ms"]
    assert per["http://a"] == pytest.approx(12.0, rel=1e-6)
    assert not rates["diagnostics"]["warnings"]


def test_calibration_insufficient_samples():
    cal = Calibration()
    cal.add(target="http://a", uncached_at_dispatch=10, queued_at_dispatch=0, ttft_ms=5.0)
    rates = cal.rates()
    assert rates["prefill_rate"] is None
    assert "insufficient samples" in rates["diagnostics"]["warnings"]
    cal.add(target="http://a", uncached_at_dispatch=10, queued_at_dispatch=0, ttft_ms=None)
    assert cal.state["skipped"] == 1


def sample(ttft: float, total: float | None = None) -> dict:
    return {"ttft_seconds": ttft, "total_seconds": total if total is not None else ttft + 1.0}


def test_online_tuner_configure_validation():
    t = OnlineTuner()
    assert t.configure({"window_size": 0}, active_policy="gorgo") is not None
    assert t.configure({"hop_size": -1}, active_policy="gorgo") is not None
    assert t.configure({"mode": "bogus"}, active_policy="gorgo") is not None
    assert t.configure({"objective_metric": "bogus"}, active_policy="gorgo") is not None
    # Enabling under a non-gorgo policy is rejected; disabling is fine.
    assert t.configure({}, active_policy="random") is not None
    assert t.configure({"enabled": False}, active_policy="random") is None
    assert not t.enabled
    err = t.configure(
        {"window_size": 4, "hop_size": 2, "objective_metric": "neg_p95_e2e"},
        active_policy="gorgo",
        current_defaults={"rtt_weight": 1.0, "prefill_weight": 1.0},
    )
    assert err is None and t.enabled and t.online_tuner is not None


def test_online_tuner_hop_window_gating_and_apply():
    t = OnlineTuner()
    assert (
        t.configure(
            {"window_size": 4, "hop_size": 2},
            active_policy="gorgo",
            current_defaults={"rtt_weight": 1.0, "prefill_weight": 1.0},
        )
        is None
    )
    store = make_default_store()
    samples: list[dict] = []
    updates = 0
    for i in range(12):
        samples.append(sample(0.1 + 0.001 * i))
        new = t.on_sample(samples, store, policy="gorgo")
        if new is not None:
            store = new
            updates += 1
    # window=4, hop=2 over 12 samples -> several recomputes fired and the
    # proposed weights landed in the store's defaults.
    assert updates >= 2
    assert t.applied_count >= 2
    assert t.pending_candidate is not None
    assert set(t.pending_candidate) == {"rtt_weight", "prefill_weight"}
    assert store["defaults"]["rtt_weight"] == pytest.approx(
        t.pending_candidate["rtt_weight"]
    )


def test_online_tuner_ignores_non_gorgo_policy_and_disabled():
    t = OnlineTuner()
    store = make_default_store()
    assert t.on_sample([sample(0.1)] * 10, store, policy="gorgo") is None  # disabled
    t.configure(
        {"window_size": 2, "hop_size": 1},
        active_policy="gorgo",
        current_defaults={"rtt_weight": 1.0, "prefill_weight": 1.0},
    )
    # Non-gorgo policy: never writes, but stays primed to resume.
    assert t.on_sample([sample(0.1)] * 10, store, policy="least-request") is None
    assert t.samples_since_last_apply == t.hop_size


def test_online_tuner_status_shape():
    t = OnlineTuner()
    st = t.status(buffered_samples=5)
    assert st["enabled"] is False and st["buffered_samples"] == 5
    assert st["samples_until_next_apply"] is None
