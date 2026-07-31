"""Validate GORGO's TTFT cost model against measured per-component latency.

Answers the reviewer's question directly: for each term of the cost model --
network RTT, queueing delay, prefill compute -- how close is the *predicted*
component to the *measured* one, and where does the model's accuracy break down?

Inputs
------
1. Proxy trace  ``<results>/proxy_traces/<run>/requests.jsonl``
   One row per routed request: the features the policy scored (per-candidate
   ``cached_prefix_tokens`` / ``queued_tokens`` / ``network_rtt_seconds``), the
   weights in force, and proxy-measured TTFT.
2. Engine logs  ``<results>/engine_req_logs/<replica_key>/*.log``
   One JSON object per request from SGLang's own request logger, carrying
   ``queue_time``, ``forward_entry_time``, ``prefill_finished_time``.

Joined on ``rid`` (the proxy sends its request id as SGLang's ``rid``).

Decomposition
-------------
Measured, all engine-local differences except TTFT (so no cross-region clock
alignment is needed anywhere):

    Q_meas       = queue_time                              = forward_entry - wait_queue_entry
    P_meas       = prefill_finished_time - forward_entry_time
    ingress_meas = (forward_entry_time - request_received_ts) - queue_time
    resid_meas   = TTFT_proxy - ingress - Q - P

``ingress`` (tokenize + dispatch) and ``resid`` (network round-trip + response
framing) are reported because the cost model has **no term** for ingress and
only an RTT term for resid, so together they bound the epsilon in Eq. 1.

Predicted, from the same features the router saw:

    RTT_pred = rtt_weight   * rtt_ms
    P_pred   = prefill_rate * uncached_tokens
    Q_pred   = queue_rate   * queue_weight * queued_tokens

Two parameterizations are reported side by side, because they answer different
questions:

  * ``deployed``  -- the weights actually in force during the run. The ES
    minimizes p95 TTFT under an argmin, so only *ratios* between terms matter
    and the overall scale is unidentifiable. Absolute ms error here is expected
    to be large and is not by itself a defect of the decomposition.
  * ``physical``  -- ``prefill_rate`` and ``queue_rate`` re-fit from the
    measured components by least squares (weights = 1). This is the honest test
    of whether the *structure* of the decomposition holds.

Ranking quality is deliberately not reported here: with one dispatch per
request the latency of the replicas *not* chosen is unobserved, so ordering
cannot be scored without the shadow-probe run.

Usage
-----
    modal volume get --env=alessio-dev --force GORGO-bench-results \
        /proxy_traces results/
    modal volume get --env=alessio-dev --force GORGO-bench-results \
        /engine_req_logs results/

    python scripts/analyze_cost_model.py --results-dir results \
        --run-prefix costmodel_apr6_ts2_v1 --out results/analysis/cost_model
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

MS_PER_S = 1000.0

# A restored timestamp that was never set arrives as a small clock-offset
# artifact rather than exactly 0.0 (see engine/sglang_timing_patch.py), so
# absolute engine timestamps are only trusted above this epoch threshold.
MIN_PLAUSIBLE_EPOCH_S = 1_000_000_000.0
# Durations outside this range indicate a field that was not populated for that
# request rather than a real measurement.
MAX_PLAUSIBLE_DURATION_S = 3600.0


# ---------------------------------------------------------------------------
# Small dependency-free statistics
# ---------------------------------------------------------------------------


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[idx]


def median(xs: list[float]) -> float:
    return percentile(xs, 0.5)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return pearson(_ranks(xs), _ranks(ys))


def ols_through_origin(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope for ``y = b*x``.

    No intercept on purpose: a physical rate is ms per token, and a request
    with zero uncached tokens (or an empty queue) must cost zero on that term.
    """
    sxx = sum(x * x for x in xs)
    if sxx <= 0:
        return float("nan")
    return sum(x * y for x, y in zip(xs, ys)) / sxx


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """``y = a + b*x``; returns ``(a, b, r2)``."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    syy = sum((y - my) ** 2 for y in ys)
    if syy <= 0:
        return a, b, 1.0
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, 1.0 - resid / syy


def multi_ols(
    predictors: list[list[float]], ys: list[float], intercept: bool = True
) -> tuple[list[float], float]:
    """Least squares for ``y = c0 + c1*x1 + ... + ck*xk``.

    Returns ``(coefficients_with_intercept_first, r2)``. Reuses the tiny
    dependency-free solver the online calibrator already relies on. With
    ``intercept=False`` the returned list still leads with a 0.0 placeholder so
    callers can index coefficients uniformly.
    """
    from gorgo.tuner import solve_spd

    n = len(ys)
    k = len(predictors)
    if not intercept:
        design = [[predictors[j][i] for j in range(k)] for i in range(n)]
        dim = k
        a = [[0.0] * dim for _ in range(dim)]
        b = [0.0] * dim
        for i in range(n):
            row = design[i]
            for p in range(dim):
                b[p] += row[p] * ys[i]
                for q in range(dim):
                    a[p][q] += row[p] * row[q]
        coef = solve_spd(a, b)
        if coef is None:
            return [float("nan")] * (dim + 1), float("nan")
        my = sum(ys) / n
        syy = sum((y - my) ** 2 for y in ys)
        resid = sum((ys[i] - sum(c * v for c, v in zip(coef, design[i]))) ** 2 for i in range(n))
        return [0.0] + list(coef), (1.0 if syy <= 0 else 1.0 - resid / syy)

    design = [[1.0] + [predictors[j][i] for j in range(k)] for i in range(n)]
    dim = k + 1
    a = [[0.0] * dim for _ in range(dim)]
    b = [0.0] * dim
    for i in range(n):
        row = design[i]
        for p in range(dim):
            b[p] += row[p] * ys[i]
            for q in range(dim):
                a[p][q] += row[p] * row[q]
    coef = solve_spd(a, b)
    if coef is None:
        return [float("nan")] * dim, float("nan")
    my = sum(ys) / n
    syy = sum((y - my) ** 2 for y in ys)
    if syy <= 0:
        return coef, 1.0
    resid = sum((ys[i] - sum(c * v for c, v in zip(coef, design[i]))) ** 2 for i in range(n))
    return coef, 1.0 - resid / syy


def error_stats(pred: list[float], meas: list[float]) -> dict:
    """Median / p95 absolute and relative error plus correlations, in ms."""
    if not pred:
        return {"n": 0}
    abs_err = [abs(p - m) for p, m in zip(pred, meas)]
    rel_err = [abs(p - m) / m for p, m in zip(pred, meas) if m > 0]
    signed = [p - m for p, m in zip(pred, meas)]
    return {
        "n": len(pred),
        "median_abs_error_ms": median(abs_err),
        "p95_abs_error_ms": percentile(abs_err, 0.95),
        "median_rel_error": median(rel_err) if rel_err else float("nan"),
        "p95_rel_error": percentile(rel_err, 0.95) if rel_err else float("nan"),
        "median_signed_error_ms": median(signed),
        "pearson_r": pearson(pred, meas),
        "spearman_rho": spearman(pred, meas),
        "median_predicted_ms": median(pred),
        "median_measured_ms": median(meas),
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_engine_logs(results_dir: str) -> dict[tuple[str, str], dict]:
    """``(replica_key, rid) -> meta_info`` from every engine request log.

    Keyed by replica too: the same rid legitimately appears on several replicas
    once shadow probes are in play, and a bare rid key would collide.
    """
    out: dict[tuple[str, str], dict] = {}
    root = os.path.join(results_dir, "engine_req_logs")
    if not os.path.isdir(root):
        raise SystemExit(
            f"no engine logs under {root}; pull them with\n"
            "  modal volume get --force GORGO-bench-results /engine_req_logs results/"
        )
    for replica_dir in sorted(glob.glob(os.path.join(root, "*"))):
        replica_key = os.path.basename(replica_dir)
        for path in sorted(glob.glob(os.path.join(replica_dir, "*.log*"))):
            with open(path, "r", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event") != "request.finished":
                        continue
                    rid = rec.get("rid")
                    meta = ((rec.get("out") or {}).get("meta_info")) or {}
                    if isinstance(rid, str) and meta:
                        out[(replica_key, rid)] = meta
    return out


def load_proxy_requests(results_dir: str, run_prefix: str) -> list[dict]:
    """Successful streaming request rows from every matching proxy trace."""
    rows: list[dict] = []
    # The controller nests traces as proxy_traces/<experiment>/<trace>/ while
    # standalone proxy runs write proxy_traces/<trace>/ directly, so search
    # recursively and filter on the prefix appearing anywhere in the path.
    root = os.path.join(results_dir, "proxy_traces")
    paths = sorted(
        p
        for p in glob.glob(os.path.join(root, "**", "requests.jsonl"), recursive=True)
        if run_prefix in os.path.relpath(p, root)
    )
    if not paths:
        raise SystemExit(
            f"no requests.jsonl under {root} matching {run_prefix!r}; pull traces with\n"
            "  modal volume get --force GORGO-bench-results /proxy_traces results/"
        )
    for path in paths:
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") != "request":
                    continue
                rows.append(rec)
    print(f"[load] {len(rows)} request rows from {len(paths)} trace file(s)")
    return rows


def load_tune_trajectory(results_dir: str, run_prefix: str) -> list[dict]:
    """ES steps from ``tune.jsonl``: the weight path the hillclimb actually walked.

    Empty for a frozen run, which is the expected state for a held-out eval.
    """
    root = os.path.join(results_dir, "proxy_traces")
    steps: list[dict] = []
    for path in sorted(
        p
        for p in glob.glob(os.path.join(root, "**", "tune.jsonl"), recursive=True)
        if run_prefix in os.path.relpath(p, root)
    ):
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") == "tune":
                    steps.append(rec)
    steps.sort(key=lambda s: (s.get("total_samples") or 0, s.get("step") or 0))
    return steps


def _duration(value) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if value < 0 or value > MAX_PLAUSIBLE_DURATION_S:
        return None
    return float(value)


def _epoch(value) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if value < MIN_PLAUSIBLE_EPOCH_S:
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Join + decomposition
# ---------------------------------------------------------------------------


def build_records(rows: list[dict], engine: dict[tuple[str, str], dict]) -> tuple[list[dict], dict]:
    """Join proxy rows to engine timings and decompose each request."""
    records: list[dict] = []
    skipped: dict[str, int] = defaultdict(int)

    for row in rows:
        if row.get("status") != 200 or row.get("ttft_ns") is None:
            skipped["not_a_successful_streamed_request"] += 1
            continue
        # Fallback rows were routed randomly, not by the cost model, so their
        # scores do not represent a policy decision.
        eff = row.get("effective_policy") or ""
        if eff.startswith("random-fallback"):
            skipped["policy_fallback"] += 1
            continue

        rid = row.get("request_id")
        replica_key = row.get("target_replica_key")
        meta = engine.get((replica_key, rid))
        if meta is None:
            skipped["no_engine_record"] += 1
            continue

        q_s = _duration(meta.get("queue_time"))
        fwd = _epoch(meta.get("forward_entry_time"))
        pfin = _epoch(meta.get("prefill_finished_time"))
        recv = _epoch(meta.get("request_received_ts"))
        if q_s is None or fwd is None or pfin is None:
            skipped["engine_timings_incomplete"] += 1
            continue
        p_s = pfin - fwd
        if p_s < 0 or p_s > MAX_PLAUSIBLE_DURATION_S:
            skipped["implausible_prefill"] += 1
            continue

        ttft_ms = row["ttft_ns"] / 1e6
        q_ms = q_s * MS_PER_S
        p_ms = p_s * MS_PER_S
        ingress_ms = None
        if recv is not None:
            ing = (fwd - recv) - q_s
            if ing is not None and -1.0 < ing < MAX_PLAUSIBLE_DURATION_S:
                ingress_ms = max(0.0, ing) * MS_PER_S

        target = row.get("target")
        snap = (row.get("candidate_snapshot") or {}).get(target) or {}
        rtt_s = snap.get("network_rtt_seconds")
        if not isinstance(rtt_s, (int, float)) or rtt_s <= 0:
            # Older traces predate per-candidate RTT; fall back to the scrape
            # latency the policy would have used in that case.
            rtt_s = snap.get("latency_seconds")
        rtt_ms = float(rtt_s) * MS_PER_S if isinstance(rtt_s, (int, float)) else None

        prompt_tokens = row.get("prompt_tokens") or row.get("request_tokens") or 0
        cached = row.get("cached_tokens_at_dispatch") or 0
        uncached = max(0, prompt_tokens - cached)
        queued = snap.get("queued_tokens")
        if not isinstance(queued, int):
            skipped["no_queued_tokens_feature"] += 1
            continue

        resid_ms = ttft_ms - q_ms - p_ms - (ingress_ms or 0.0)
        hp = row.get("hyperparameters_at_decision") or {}
        records.append(
            {
                "rid": rid,
                "replica_key": replica_key,
                "region": row.get("target_replica_region"),
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached,
                "uncached_tokens": uncached,
                "queued_tokens": queued,
                "rtt_ms": rtt_ms,
                "ttft_ms": ttft_ms,
                "q_meas_ms": q_ms,
                "p_meas_ms": p_ms,
                "ingress_meas_ms": ingress_ms,
                "resid_meas_ms": resid_ms,
                "hp": hp,
            }
        )

    return records, dict(skipped)


_PARAM_DEFAULTS = (
    ("rtt_weight", 1.0),
    ("prefill_rate", 1.0),
    ("queue_rate", 1.0),
    ("queue_weight", 1.0),
)


def with_defaults(params: dict | None) -> dict:
    out = dict(params or {})
    for key, default in _PARAM_DEFAULTS:
        out.setdefault(key, default)
    return out


def predict(rec: dict, params: dict | None = None) -> dict:
    """Cost-model prediction for one request.

    ``params=None`` means "use the weights that were actually in force when this
    request was routed" (``hyperparameters_at_decision``). That matters during a
    tuning window, where the ES rewrites the weights every ``hop_size`` samples:
    scoring the whole run with one weight vector would attribute every request
    to whichever vector happened to be live first.
    """
    p = with_defaults(rec.get("hp") if params is None else params)
    rtt_pred = p["rtt_weight"] * (rec["rtt_ms"] or 0.0)
    p_pred = p["prefill_rate"] * rec["uncached_tokens"]
    q_pred = p["queue_rate"] * p["queue_weight"] * rec["queued_tokens"]
    return {
        "rtt_pred_ms": rtt_pred,
        "p_pred_ms": p_pred,
        "q_pred_ms": q_pred,
        "ttft_pred_ms": rtt_pred + p_pred + q_pred,
    }


def weight_census(records: list[dict]) -> dict:
    """Distinct ``(rtt_weight, queue_weight)`` vectors actually used, with counts.

    A held-out eval must show exactly one. More than one means the tuner was
    still live -- the failure mode the ``gorgo-2d`` guard fix in
    ``experiment_runner/policy_matrix_app.py`` exists to prevent.
    """
    counts: dict[tuple, int] = defaultdict(int)
    for r in records:
        p = with_defaults(r.get("hp"))
        counts[
            (
                round(float(p["rtt_weight"]), 6),
                round(float(p["queue_weight"]), 6),
                round(float(p["prefill_rate"]), 6),
                round(float(p["queue_rate"]), 6),
            )
        ] += 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return {
        "n_distinct": len(ordered),
        "frozen": len(ordered) == 1,
        "vectors": [
            {
                "rtt_weight": k[0],
                "queue_weight": k[1],
                "prefill_rate": k[2],
                "queue_rate": k[3],
                "n_requests": n,
            }
            for k, n in ordered
        ],
    }


def fit_physical_rates(records: list[dict]) -> dict:
    """Fit ms/token rates directly against the *measured* components.

    This is the estimate the cost model would use if its terms were meant to be
    read as milliseconds, and the yardstick the tuned weights are compared to.
    """
    p_rate = ols_through_origin(
        [r["uncached_tokens"] for r in records], [r["p_meas_ms"] for r in records]
    )
    q_rate = ols_through_origin(
        [r["queued_tokens"] for r in records], [r["q_meas_ms"] for r in records]
    )
    return {
        "rtt_weight": 1.0,
        "prefill_rate": p_rate,
        "queue_rate": q_rate,
        "queue_weight": 1.0,
    }


def bucketize(records: list[dict], key: str, edges: list[float]) -> list[tuple[str, list[dict]]]:
    buckets: list[tuple[str, list[dict]]] = []
    lo = 0.0
    for edge in edges:
        label = f"{int(lo)}-{int(edge)}"
        buckets.append((label, [r for r in records if lo <= r[key] < edge]))
        lo = edge
    buckets.append((f">={int(lo)}", [r for r in records if r[key] >= lo]))
    return [(label, rs) for label, rs in buckets if rs]


# ---------------------------------------------------------------------------
# Scale-free tests
# ---------------------------------------------------------------------------

# The ES box from specs/c64/tuning/policy_matrix_c64_tuning_p95ttft_2d.json,
# which paper v3 tunes inside. Reported against the weights that would actually
# reproduce the measured composition, to show whether the calibrated point is
# even reachable by the search.
PAPER_ES_BOX = {"rtt_weight": (0.05, 2.0), "queue_weight": (0.05, 0.5)}
WIDE_BOX = {"rtt_weight": (1e-3, 200.0), "queue_weight": (1e-6, 2.0)}

_TERMS = ("network", "prefill", "queue")


def _shares(vals: dict[str, float]) -> dict[str, float]:
    total = sum(vals.values())
    if total <= 0:
        return {k: float("nan") for k in vals}
    return {k: v / total for k, v in vals.items()}


def report_composition(records: list[dict]) -> dict:
    """Compare the *composition* of predicted vs measured TTFT, scale-free.

    The deployed score has no units -- ``argmin`` is invariant to scale, so the
    absolute prediction is not a well-posed thing to check. What *is* identifiable
    is how the score divides itself between terms. This compares that against how
    measured TTFT actually divides itself, with **zero fitted parameters**.

    ``ingress`` is excluded from the denominator and reported separately: the cost
    model has no term for it, so including it would charge the model for a stage
    it never claimed to explain.
    """
    print()
    print("=" * 78)
    print("COMPOSITION: predicted vs measured share of TTFT (zero fitted parameters)")
    print("=" * 78)

    pred_rows, meas_rows = [], []
    for r in records:
        p = predict(r)  # per-request deployed weights
        pv = {"network": p["rtt_pred_ms"], "prefill": p["p_pred_ms"], "queue": p["q_pred_ms"]}
        mv = {
            "network": max(0.0, r["resid_meas_ms"]),
            "prefill": max(0.0, r["p_meas_ms"]),
            "queue": max(0.0, r["q_meas_ms"]),
        }
        if sum(pv.values()) <= 0 or sum(mv.values()) <= 0:
            continue
        pred_rows.append(_shares(pv))
        meas_rows.append(_shares(mv))
        r["_pred_terms"], r["_meas_terms"] = pv, mv

    if not pred_rows:
        print("  no scorable requests")
        return {}

    scored = [r for r in records if "_pred_terms" in r]
    agg_pred = _shares({t: sum(r["_pred_terms"][t] for r in scored) for t in _TERMS})
    agg_meas = _shares({t: sum(r["_meas_terms"][t] for r in scored) for t in _TERMS})
    ing_total = sum(r["ingress_meas_ms"] or 0.0 for r in scored)
    meas_total = sum(sum(r["_meas_terms"].values()) for r in scored)

    print("  aggregate share of modeled TTFT:")
    for t in _TERMS:
        print(
            f"    {t:9} predicted {agg_pred[t] * 100:5.1f}%   measured {agg_meas[t] * 100:5.1f}%"
            f"   delta {(agg_pred[t] - agg_meas[t]) * 100:+6.1f} pp"
        )
    print(
        f"    [unmodeled ingress = "
        f"{100.0 * ing_total / (meas_total + ing_total):.1f}% of full TTFT]"
    )

    print("  per-request share error (percentage points):")
    print(
        f"    {'term':9} {'pred p50':>9} {'meas p50':>9} {'|d| p50':>8} {'|d| p95':>8} {'rho':>7}"
    )
    per_term = {}
    for t in _TERMS:
        pv = [row[t] for row in pred_rows]
        mv = [row[t] for row in meas_rows]
        err = [abs(a - b) * 100 for a, b in zip(pv, mv)]
        per_term[t] = {
            "pred_share_p50": median(pv),
            "meas_share_p50": median(mv),
            "abs_share_err_pp_p50": median(err),
            "abs_share_err_pp_p95": percentile(err, 0.95),
            "spearman_rho": spearman(pv, mv),
            "agg_pred_share": agg_pred[t],
            "agg_meas_share": agg_meas[t],
        }
        print(
            f"    {t:9} {median(pv) * 100:8.1f}% {median(mv) * 100:8.1f}% "
            f"{median(err):8.1f} {percentile(err, 0.95):8.1f} {spearman(pv, mv):7.3f}"
        )

    tv = [0.5 * sum(abs(a[t] - b[t]) for t in _TERMS) for a, b in zip(pred_rows, meas_rows)]
    print(
        f"    total-variation distance between compositions: p50 {median(tv):.3f}  "
        f"p95 {percentile(tv, 0.95):.3f}   (0 = identical, 1 = disjoint)"
    )

    # Dominant-term agreement, against the baseline you would get if the two
    # were statistically independent -- otherwise a skewed marginal makes a
    # useless predictor look informative.
    dom_p = [max(_TERMS, key=lambda t: row[t]) for row in pred_rows]
    dom_m = [max(_TERMS, key=lambda t: row[t]) for row in meas_rows]
    agree = sum(1 for a, b in zip(dom_p, dom_m) if a == b) / len(dom_p)
    fp = {t: dom_p.count(t) / len(dom_p) for t in _TERMS}
    fm = {t: dom_m.count(t) / len(dom_m) for t in _TERMS}
    baseline = sum(fp[t] * fm[t] for t in _TERMS)
    print(
        f"  dominant-term agreement: {agree * 100:.1f}%  "
        f"(independence baseline {baseline * 100:.1f}%)"
    )
    print(f"    predicted { {t: round(fp[t], 3) for t in _TERMS} }")
    print(f"    measured  { {t: round(fm[t], 3) for t in _TERMS} }")

    return {
        "per_term": per_term,
        "tv_distance_p50": median(tv),
        "tv_distance_p95": percentile(tv, 0.95),
        "dominant_term_agreement": agree,
        "dominant_term_independence_baseline": baseline,
        "predicted_dominant_freq": fp,
        "measured_dominant_freq": fm,
        "unmodeled_ingress_share": ing_total / (meas_total + ing_total),
    }


def report_form_ceiling(records: list[dict]) -> dict:
    """Best R^2 any weights could achieve for this functional form.

    Fits ``TTFT ~ a*rtt + b*uncached + c*queued`` directly against measured TTFT
    with full hindsight. No online tuner can beat this, so it upper-bounds what
    retuning -- on any hardware, in any search box -- could ever deliver. The
    ``true components`` row is a sanity check: it must be ~1.0, since the
    decomposition is exact by construction.
    """
    print()
    print("=" * 78)
    print("CEILING: best R^2 achievable by ANY weights in this functional form")
    print("=" * 78)
    have = [r for r in records if isinstance(r.get("rtt_ms"), (int, float))]
    if len(have) < 10:
        print("  too few records with RTT")
        return {}
    ys = [r["ttft_ms"] for r in have]
    rtts = [r["rtt_ms"] for r in have]
    unc = [float(r["uncached_tokens"]) for r in have]
    qd = [float(r["queued_tokens"]) for r in have]
    ptok = [float(r["prompt_tokens"]) for r in have]

    out = {}
    for label, preds, names, icept in (
        (
            "cost-model form, through origin",
            [rtts, unc, qd],
            ["rtt_ms", "uncached", "queued"],
            False,
        ),
        ("cost-model form + intercept", [rtts, unc, qd], ["rtt_ms", "uncached", "queued"], True),
        (
            "+ prompt_tokens (models ingress)",
            [rtts, unc, qd, ptok],
            ["rtt_ms", "uncached", "queued", "prompt_tokens"],
            True,
        ),
    ):
        coef, r2 = multi_ols(preds, ys, intercept=icept)
        terms = "  ".join(f"{nm}={c:+.5f}" for nm, c in zip(names, coef[1:]))
        print(f"  {label:34} R^2={r2:6.3f}  intercept={coef[0]:7.1f}  {terms}")
        out[label] = {"r2": r2, "intercept_ms": coef[0], "coefficients": dict(zip(names, coef[1:]))}

    comps = [
        [r["resid_meas_ms"] for r in have],
        [r["p_meas_ms"] for r in have],
        [r["q_meas_ms"] for r in have],
        [r["ingress_meas_ms"] or 0.0 for r in have],
    ]
    _, r2_true = multi_ols(comps, ys, intercept=False)
    print(f"  {'true measured components (sanity)':34} R^2={r2_true:6.3f}  (must be ~1.000)")
    out["true_components_r2"] = r2_true
    print(
        "  The gap between these rows and 1.0 is the form's own error, not the tuner's.\n"
        "  Retuning cannot cross it."
    )
    return out


def report_box_reachability(records: list[dict], chosen: dict | None = None) -> dict:
    """Can any weights inside the ES search box reproduce the measured composition?

    Grid-searches ``(rtt_weight, queue_weight)`` for the minimum total-variation
    distance between the aggregate predicted and measured compositions, inside the
    paper's box and inside a deliberately wide one. If the wide-box optimum lies
    outside the paper's box, the search itself -- not the model form and not the
    hardware -- is what prevents calibration.
    """
    print()
    print("=" * 78)
    print("REACHABILITY: can the ES box reproduce the measured composition?")
    print("=" * 78)
    have = [r for r in records if isinstance(r.get("rtt_ms"), (int, float))]
    if len(have) < 10:
        print("  too few records with RTT")
        return {}
    sum_rtt = sum(r["rtt_ms"] for r in have)
    sum_unc = sum(float(r["uncached_tokens"]) for r in have)
    sum_qd = sum(float(r["queued_tokens"]) for r in have)
    tgt = _shares(
        {
            "network": sum(max(0.0, r["resid_meas_ms"]) for r in have),
            "prefill": sum(max(0.0, r["p_meas_ms"]) for r in have),
            "queue": sum(max(0.0, r["q_meas_ms"]) for r in have),
        }
    )
    print(
        f"  measured composition: network {tgt['network'] * 100:.1f}% / "
        f"prefill {tgt['prefill'] * 100:.1f}% / queue {tgt['queue'] * 100:.1f}%"
    )

    def tv_at(rw: float, qw: float) -> float:
        c = _shares({"network": rw * sum_rtt, "prefill": sum_unc, "queue": qw * sum_qd})
        return 0.5 * sum(abs(c[t] - tgt[t]) for t in _TERMS)

    out: dict = {"measured_composition": tgt}
    for name, box in (("paper ES box", PAPER_ES_BOX), ("wide box", WIDE_BOX)):
        rlo, rhi = box["rtt_weight"]
        qlo, qhi = box["queue_weight"]
        n = 300
        grid_r = [
            math.exp(math.log(rlo) + (math.log(rhi) - math.log(rlo)) * i / (n - 1))
            for i in range(n)
        ]
        grid_q = [
            math.exp(math.log(qlo) + (math.log(qhi) - math.log(qlo)) * i / (n - 1))
            for i in range(n)
        ]
        best = min(((tv_at(rw, qw), rw, qw) for rw in grid_r for qw in grid_q))
        tv_best, rw, qw = best
        c = _shares({"network": rw * sum_rtt, "prefill": sum_unc, "queue": qw * sum_qd})
        print(
            f"  {name:13} best TV {tv_best:.3f} at rtt_weight={rw:.4g} queue_weight={qw:.4g}"
            f"  -> network {c['network'] * 100:5.1f}% / prefill {c['prefill'] * 100:5.1f}%"
            f" / queue {c['queue'] * 100:5.1f}%"
        )
        out[name] = {"tv": tv_best, "rtt_weight": rw, "queue_weight": qw, "composition": c}

    # Closed form: the weights that match the composition exactly, unbounded.
    rw_star = (tgt["network"] / tgt["prefill"]) * sum_unc / sum_rtt if sum_rtt > 0 else float("nan")
    qw_star = (tgt["queue"] / tgt["prefill"]) * sum_unc / sum_qd if sum_qd > 0 else float("nan")
    out["exact_match_unbounded"] = {"rtt_weight": rw_star, "queue_weight": qw_star}
    rlo, rhi = PAPER_ES_BOX["rtt_weight"]
    qlo, qhi = PAPER_ES_BOX["queue_weight"]
    inside = (rlo <= rw_star <= rhi) and (qlo <= qw_star <= qhi)
    out["exact_match_inside_paper_box"] = inside
    print(f"  exact-match weights (unbounded): rtt_weight={rw_star:.4g} queue_weight={qw_star:.4g}")
    print(
        f"  -> {'INSIDE' if inside else 'OUTSIDE'} the ES box (rtt {rlo}-{rhi}, queue {qlo}-{qhi})"
    )

    # Whether the box *excludes* the calibrated point is a different question
    # from whether the box is what stopped the optimizer. If the tuner settled
    # strictly inside the box, it had room left in the calibrated direction and
    # declined to use it -- so the binding constraint is the objective, not the
    # bounds. Saying "the box binds" without this check overclaims.
    if chosen and isinstance(chosen.get("rtt_weight"), (int, float)):
        crw, cqw = float(chosen["rtt_weight"]), float(chosen["queue_weight"])
        edge_tol = 0.02
        at_edge = (
            abs(crw - rhi) / rhi < edge_tol
            or abs(crw - rlo) / rlo < edge_tol
            or abs(cqw - qhi) / qhi < edge_tol
            or abs(cqw - qlo) / qlo < edge_tol
        )
        out["chosen_weights"] = {"rtt_weight": crw, "queue_weight": cqw}
        out["chosen_at_box_edge"] = at_edge
        headroom = (rw_star - crw) / (rhi - crw) if rhi > crw else float("nan")
        print(
            f"  tuner settled at rtt_weight={crw:.4g} queue_weight={cqw:.4g} -- "
            f"{'ON the box edge' if at_edge else 'STRICTLY INSIDE the box'}"
        )
        if not at_edge and not inside:
            print(
                f"     It had {rhi / crw:.2f}x more rtt_weight available inside the box and did\n"
                f"     not use it, so the bounds are NOT what prevents calibration -- the\n"
                f"     p95 objective and the calibrated composition are different optima."
            )
        elif at_edge and not inside:
            print(
                "     It ran to the boundary in the calibrated direction, so the bounds are\n"
                "     plausibly binding; widening the box would test that directly."
            )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(records: list[dict], skipped: dict, out_prefix: str) -> dict:
    census = weight_census(records)
    # The modal weight vector, for the headline "deployed" numbers. Per-request
    # predictions still use each request's own weights (predict(rec, None)).
    modal = census["vectors"][0] if census["vectors"] else {}
    deployed = with_defaults(
        {
            k: modal[k]
            for k in ("rtt_weight", "queue_weight", "prefill_rate", "queue_rate")
            if k in modal
        }
    )
    physical = fit_physical_rates(records)

    summary: dict = {
        "n_joined": len(records),
        "skipped": skipped,
        "weight_census": census,
        "deployed_params": deployed,
        "physical_params": physical,
        "components": {},
        "by_prompt_length": {},
        "by_load": {},
    }

    print()
    print("=" * 78)
    print("COVERAGE")
    print("=" * 78)
    print(f"  joined requests: {len(records)}")
    for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  skipped ({reason}): {n}")

    print()
    print("=" * 78)
    print("WEIGHTS ACTUALLY IN FORCE")
    print("=" * 78)
    print(f"  distinct weight vectors: {census['n_distinct']}")
    for v in census["vectors"][:8]:
        print(
            f"    rtt_weight={v['rtt_weight']:<10.5f} queue_weight={v['queue_weight']:<10.5f} "
            f"prefill_rate={v['prefill_rate']:<8.4f} queue_rate={v['queue_rate']:<8.4f} "
            f"n={v['n_requests']}"
        )
    if census["n_distinct"] > 8:
        print(f"    ... and {census['n_distinct'] - 8} more")
    if census["frozen"]:
        print("  -> FROZEN: one weight vector for the whole run (a valid held-out eval).")
    else:
        print(
            "  -> LIVE TUNING: weights changed during this run. Per-request predictions\n"
            "     below use each request's own weights. If this was meant to be a\n"
            "     held-out eval, the tuner was not disabled and the run is invalid."
        )

    print()
    print("=" * 78)
    print("MEASURED DECOMPOSITION (ms)")
    print("=" * 78)
    stage_names = [
        ("ttft_ms", "TTFT (proxy)"),
        ("resid_meas_ms", "network+framing residual"),
        ("ingress_meas_ms", "ingress (tokenize+dispatch)"),
        ("q_meas_ms", "Q  queueing"),
        ("p_meas_ms", "P  prefill"),
    ]
    print(f"  {'stage':32} {'p50':>10} {'p95':>10} {'share of TTFT p50':>20}")
    ttft_p50 = median([r["ttft_ms"] for r in records]) if records else float("nan")
    for key, label in stage_names:
        vals = [r[key] for r in records if isinstance(r.get(key), (int, float))]
        if not vals:
            continue
        p50 = median(vals)
        share = "" if key == "ttft_ms" else f"{100.0 * p50 / ttft_p50:18.1f}%"
        print(f"  {label:32} {p50:10.1f} {percentile(vals, 0.95):10.1f} {share:>20}")
        summary["components"].setdefault("measured", {})[key] = {
            "p50_ms": p50,
            "p95_ms": percentile(vals, 0.95),
        }

    # Physical rates implied by the measurements, next to what was deployed.
    print()
    print("=" * 78)
    print("PHYSICAL RATES: measured vs deployed")
    print("=" * 78)
    print(f"  prefill_rate  measured {physical['prefill_rate']:.5f} ms/uncached-token")
    print(f"                deployed {deployed['prefill_rate']:.5f} (x weight 1.0)")
    print(f"  queue_rate    measured {physical['queue_rate']:.5f} ms/queued-token")
    print(
        f"                deployed {deployed['queue_rate']:.5f} "
        f"x queue_weight {deployed['queue_weight']:.5f} "
        f"= {deployed['queue_rate'] * deployed['queue_weight']:.5f}"
    )
    print(f"  rtt_weight    deployed {deployed['rtt_weight']:.5f} (1.0 = unbiased RTT)")

    # Per-component accuracy under both parameterizations. ``deployed`` passes
    # params=None so each request is scored with its own hyperparameters_at_decision.
    for label, params in (("deployed", None), ("physical", physical)):
        print()
        print("=" * 78)
        print(f"PER-COMPONENT ACCURACY -- {label} parameters")
        print("=" * 78)
        preds = [predict(r, params) for r in records]
        comp = {}
        for pred_key, meas_key, name in (
            ("rtt_pred_ms", "resid_meas_ms", "RTT vs network+framing"),
            ("q_pred_ms", "q_meas_ms", "Q   queueing"),
            ("p_pred_ms", "p_meas_ms", "P   prefill"),
            ("ttft_pred_ms", "ttft_ms", "TTFT total"),
        ):
            pairs = [
                (p[pred_key], r[meas_key])
                for p, r in zip(preds, records)
                if isinstance(r.get(meas_key), (int, float))
            ]
            if not pairs:
                continue
            stats = error_stats([a for a, _ in pairs], [b for _, b in pairs])
            comp[name] = stats
            print(
                f"  {name:26} n={stats['n']:5d}  "
                f"pred_p50={stats['median_predicted_ms']:8.1f}  "
                f"meas_p50={stats['median_measured_ms']:8.1f}  "
                f"|err|_p50={stats['median_abs_error_ms']:8.1f}  "
                f"|err|_p95={stats['p95_abs_error_ms']:9.1f}  "
                f"rel_p50={stats['median_rel_error']:6.2f}  "
                f"r={stats['pearson_r']:5.2f}  rho={stats['spearman_rho']:5.2f}"
            )
        summary["components"][label] = comp

        # Scale-free check: does the score track TTFT after one affine map?
        a, b, r2 = ols([p["ttft_pred_ms"] for p in preds], [r["ttft_ms"] for r in records])
        print(
            f"  affine fit TTFT_meas = {a:.1f} + {b:.4f} * score   R^2={r2:.3f}  "
            "(shape-only test; absorbs the unidentifiable score scale)"
        )
        summary["components"][label]["affine_fit"] = {"a": a, "b": b, "r2": r2}

    summary["composition"] = report_composition(records)
    summary["form_ceiling"] = report_form_ceiling(records)
    summary["box_reachability"] = report_box_reachability(records, chosen=deployed)

    # Calibration by request length and by load.
    for bucket_label, key, edges, store in (
        (
            "PROMPT LENGTH (uncached tokens)",
            "uncached_tokens",
            [1000, 4000, 8000, 16000],
            "by_prompt_length",
        ),
        (
            "LOAD (queued tokens at dispatch)",
            "queued_tokens",
            [1000, 10000, 50000, 150000],
            "by_load",
        ),
    ):
        print()
        print("=" * 78)
        print(f"CALIBRATION BY {bucket_label} -- physical parameters")
        print("=" * 78)
        print(
            f"  {'bucket':>16} {'n':>6} {'P_meas':>9} {'P_pred':>9} "
            f"{'Q_meas':>9} {'Q_pred':>9} {'resid':>9} {'RTT_pred':>9}"
        )
        for label, rs in bucketize(records, key, edges):
            preds = [predict(r, physical) for r in rs]
            row = {
                "n": len(rs),
                "p_meas_p50": median([r["p_meas_ms"] for r in rs]),
                "p_pred_p50": median([p["p_pred_ms"] for p in preds]),
                "q_meas_p50": median([r["q_meas_ms"] for r in rs]),
                "q_pred_p50": median([p["q_pred_ms"] for p in preds]),
                "resid_p50": median([r["resid_meas_ms"] for r in rs]),
                "rtt_pred_p50": median([p["rtt_pred_ms"] for p in preds]),
            }
            summary[store][label] = row
            print(
                f"  {label:>16} {row['n']:6d} {row['p_meas_p50']:9.1f} {row['p_pred_p50']:9.1f} "
                f"{row['q_meas_p50']:9.1f} {row['q_pred_p50']:9.1f} "
                f"{row['resid_p50']:9.1f} {row['rtt_pred_p50']:9.1f}"
            )

    # What is actually in the residual? The cost model explains it with an RTT
    # term alone. If prompt size matters, the network term is mis-specified: a
    # long prompt must be *uploaded* to the replica, so the cost is a
    # bandwidth-delay product, not a fixed round-trip.
    print()
    print("=" * 78)
    print("WHAT EXPLAINS THE network+framing RESIDUAL?")
    print("=" * 78)
    have_rtt = [r for r in records if isinstance(r.get("rtt_ms"), (int, float))]
    if have_rtt:
        ys = [r["resid_meas_ms"] for r in have_rtt]
        rtts = [r["rtt_ms"] for r in have_rtt]
        ptoks = [float(r["prompt_tokens"]) for r in have_rtt]
        inter = [p * t / 1000.0 for p, t in zip(ptoks, rtts)]
        models = [
            ("RTT only", [rtts], ["rtt_ms"]),
            ("prompt tokens only", [ptoks], ["prompt_tokens"]),
            ("RTT + prompt tokens", [rtts, ptoks], ["rtt_ms", "prompt_tokens"]),
            (
                "RTT + tokens + RTT*tokens/1e3",
                [rtts, ptoks, inter],
                ["rtt_ms", "prompt_tokens", "rtt_x_tokens"],
            ),
        ]
        summary["residual_models"] = {}
        for label, preds, names in models:
            coef, r2 = multi_ols(preds, ys)
            terms = "  ".join(f"{nm}={c:+.5f}" for nm, c in zip(names, coef[1:]))
            print(f"  {label:32} R^2={r2:5.3f}   intercept={coef[0]:7.1f}  {terms}")
            summary["residual_models"][label] = {
                "r2": r2,
                "intercept_ms": coef[0],
                "coefficients": dict(zip(names, coef[1:])),
            }
        print(
            "  A large jump from 'RTT only' to a model including prompt tokens means the\n"
            "  network term should scale with bytes shipped, not just round-trip time --\n"
            "  which matters most for exactly the long-prompt, cross-region case."
        )

        print()
        print(
            f"  {'region':>14} {'n':>6} {'rtt_p50':>9} {'resid_p50':>10} {'resid_p95':>10} {'ptok_p50':>9}"
        )
        by_region: dict[str, list[dict]] = defaultdict(list)
        for r in have_rtt:
            by_region[str(r.get("region"))].append(r)
        for region, rs in sorted(by_region.items()):
            row = {
                "n": len(rs),
                "rtt_p50_ms": median([r["rtt_ms"] for r in rs]),
                "resid_p50_ms": median([r["resid_meas_ms"] for r in rs]),
                "resid_p95_ms": percentile([r["resid_meas_ms"] for r in rs], 0.95),
                "prompt_tokens_p50": median([float(r["prompt_tokens"]) for r in rs]),
            }
            summary.setdefault("by_region", {})[region] = row
            print(
                f"  {region[:14]:>14} {row['n']:6d} {row['rtt_p50_ms']:9.1f} "
                f"{row['resid_p50_ms']:10.1f} {row['resid_p95_ms']:10.1f} "
                f"{row['prompt_tokens_p50']:9.0f}"
            )

    # Is the residual load-dependent? If so, the RTT term cannot explain it and
    # the ES must compensate through the queue weight.
    q_vals = [r["q_meas_ms"] for r in records]
    resid_vals = [r["resid_meas_ms"] for r in records]
    a, b, r2 = ols(q_vals, resid_vals)
    print()
    print("=" * 78)
    print("IS THE RESIDUAL LOAD-DEPENDENT?")
    print("=" * 78)
    print(
        f"  resid_ms = {a:.1f} + {b:.4f} * Q_ms    R^2={r2:.3f}  r={pearson(q_vals, resid_vals):.3f}"
    )
    print(
        "  A slope near zero would mean the residual is pure network RTT; a positive\n"
        "  slope means part of TTFT grows with load outside both the queue and prefill\n"
        "  terms, so the fitted weights must absorb it."
    )
    summary["residual_vs_queue"] = {"intercept_ms": a, "slope": b, "r2": r2}

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    json_path = f"{out_prefix}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"[out] wrote {json_path}")

    csv_path = f"{out_prefix}_per_request.csv"
    fields = [
        "rid",
        "replica_key",
        "region",
        "prompt_tokens",
        "cached_tokens",
        "uncached_tokens",
        "queued_tokens",
        "rtt_ms",
        "ttft_ms",
        "q_meas_ms",
        "p_meas_ms",
        "ingress_meas_ms",
        "resid_meas_ms",
    ]
    # Physical-rate predictions, the weights in force at decision time, the
    # deployed-weight term values, and both share vectors -- everything needed
    # to redo the composition test offline without rerunning this script.
    extra = [
        "rtt_pred_ms",
        "q_pred_ms",
        "p_pred_ms",
        "ttft_pred_ms",
        "hp_rtt_weight",
        "hp_queue_weight",
        "hp_prefill_rate",
        "hp_queue_rate",
        "term_network",
        "term_prefill",
        "term_queue",
        "pred_share_network",
        "pred_share_prefill",
        "pred_share_queue",
        "meas_share_network",
        "meas_share_prefill",
        "meas_share_queue",
    ]
    with open(csv_path, "w") as f:
        f.write(",".join(fields + extra) + "\n")
        for r in records:
            p = predict(r, physical)
            hp = with_defaults(r.get("hp"))
            pv = r.get("_pred_terms") or {}
            mv = r.get("_meas_terms") or {}
            ps, ms = _shares(pv) if pv else {}, _shares(mv) if mv else {}
            vals = [r.get(k) for k in fields] + [
                p["rtt_pred_ms"],
                p["q_pred_ms"],
                p["p_pred_ms"],
                p["ttft_pred_ms"],
                hp["rtt_weight"],
                hp["queue_weight"],
                hp["prefill_rate"],
                hp["queue_rate"],
                pv.get("network"),
                pv.get("prefill"),
                pv.get("queue"),
                ps.get("network"),
                ps.get("prefill"),
                ps.get("queue"),
                ms.get("network"),
                ms.get("prefill"),
                ms.get("queue"),
            ]
            f.write(",".join("" if v is None else str(v) for v in vals) + "\n")
    print(f"[out] wrote {csv_path}")
    return summary


def plot_tune_trajectory(steps: list[dict], summary: dict, out_prefix: str) -> None:
    """The ES weight path, against the search box and the calibrated point.

    Answers "where did the tuner actually go, and was the calibrated point even
    available to it?" in one figure.
    """
    if not steps:
        print("[plot] no tune.jsonl steps (frozen run); skipping trajectory figure")
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping figures")
        return

    best = [s.get("best_params") or {} for s in steps]
    rw = [b.get("rtt_weight") for b in best]
    qw = [b.get("queue_weight") for b in best]
    keep = [
        i
        for i in range(len(rw))
        if isinstance(rw[i], (int, float)) and isinstance(qw[i], (int, float))
    ]
    if not keep:
        print("[plot] tune steps carry no weights; skipping trajectory figure")
        return
    rw = [rw[i] for i in keep]
    qw = [qw[i] for i in keep]
    scores = [steps[i].get("best_score") for i in keep]
    sigmas = [steps[i].get("sigma") for i in keep]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    rlo, rhi = PAPER_ES_BOX["rtt_weight"]
    qlo, qhi = PAPER_ES_BOX["queue_weight"]
    ax.add_patch(
        plt.Rectangle(
            (rlo, qlo),
            rhi - rlo,
            qhi - qlo,
            fill=False,
            ls="--",
            ec="tab:gray",
            lw=1.5,
            label="ES search box",
        )
    )
    ax.plot(rw, qw, "-o", ms=3.5, lw=1, color="tab:blue", alpha=0.8, label="ES best-so-far")
    ax.plot(rw[0], qw[0], "s", ms=9, color="tab:green", label=f"start ({rw[0]:.3g}, {qw[0]:.3g})")
    ax.plot(
        rw[-1], qw[-1], "*", ms=16, color="tab:red", label=f"final ({rw[-1]:.3g}, {qw[-1]:.3g})"
    )

    reach = summary.get("box_reachability") or {}
    exact = reach.get("exact_match_unbounded") or {}
    if isinstance(exact.get("rtt_weight"), (int, float)):
        ax.plot(
            exact["rtt_weight"],
            exact["queue_weight"],
            "P",
            ms=13,
            color="tab:purple",
            label=f"composition-calibrated ({exact['rtt_weight']:.3g}, {exact['queue_weight']:.3g})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rtt_weight")
    ax.set_ylabel("queue_weight")
    ax.set_title("ES weight trajectory vs. search box\nand the composition-calibrated point")
    ax.legend(fontsize=7.5, loc="best")

    ax2.plot(
        range(len(scores)),
        scores,
        "-o",
        ms=3.5,
        color="tab:blue",
        label="best score (neg p95 TTFT, s)",
    )
    ax2.set_xlabel("ES step")
    ax2.set_ylabel("best score")
    ax2.legend(fontsize=8, loc="lower right")
    ax3 = ax2.twinx()
    ax3.plot(range(len(sigmas)), sigmas, "-", lw=1, color="tab:orange", alpha=0.7)
    ax3.set_ylabel("sigma (orange)", color="tab:orange")
    ax2.set_title(f"Convergence over {len(scores)} ES steps")

    fig.tight_layout()
    path = f"{out_prefix}_tune_trajectory.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[out] wrote {path}")


def plot(records: list[dict], summary: dict, out_prefix: str) -> None:
    """Predicted-vs-measured scatter per component, plus the measured stack."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping figures")
        return

    physical = summary["physical_params"]
    preds = [predict(r, physical) for r in records]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    panels = [
        ("p_pred_ms", "p_meas_ms", "Prefill P", "tab:blue"),
        ("q_pred_ms", "q_meas_ms", "Queueing Q", "tab:orange"),
        ("rtt_pred_ms", "resid_meas_ms", "RTT vs network+framing", "tab:green"),
        ("ttft_pred_ms", "ttft_ms", "Total TTFT", "tab:red"),
    ]
    for ax, (pk, mk, title, color) in zip(axes, panels):
        xs = [p[pk] for p, r in zip(preds, records) if isinstance(r.get(mk), (int, float))]
        ys = [r[mk] for r in records if isinstance(r.get(mk), (int, float))]
        if not xs:
            continue
        ax.scatter(xs, ys, s=6, alpha=0.25, color=color, edgecolors="none")
        hi = max(max(xs), max(ys))
        ax.plot([0, hi], [0, hi], "k--", lw=1, label="y = x")
        ax.set_xlabel(f"predicted {title} (ms)")
        ax.set_ylabel(f"measured {title} (ms)")
        ax.set_title(f"{title}\nr={pearson(xs, ys):.2f}  rho={spearman(xs, ys):.2f}")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "GORGO cost model: predicted vs measured TTFT components (physical rates)",
        y=1.02,
    )
    fig.tight_layout()
    scatter_path = f"{out_prefix}_scatter.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"[out] wrote {scatter_path}")

    # Measured stack ordered by TTFT: shows which stage dominates, and where.
    fig2, ax = plt.subplots(figsize=(9, 4.6))
    order = sorted(range(len(records)), key=lambda i: records[i]["ttft_ms"])
    xs = list(range(len(order)))
    stack_keys = [
        ("p_meas_ms", "prefill P", "tab:blue"),
        ("q_meas_ms", "queueing Q", "tab:orange"),
        ("ingress_meas_ms", "ingress (tokenize+dispatch)", "tab:purple"),
        ("resid_meas_ms", "network+framing", "tab:green"),
    ]
    bottom = [0.0] * len(order)
    for key, label, color in stack_keys:
        vals = [max(0.0, records[i].get(key) or 0.0) for i in order]
        ax.fill_between(xs, bottom, [b + v for b, v in zip(bottom, vals)], label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xlabel("requests, sorted by measured TTFT")
    ax.set_ylabel("ms")
    ax.set_title("Measured TTFT decomposition")
    ax.legend(loc="upper left", fontsize=8)
    fig2.tight_layout()
    stack_path = f"{out_prefix}_stack.png"
    fig2.savefig(stack_path, dpi=150, bbox_inches="tight")
    print(f"[out] wrote {stack_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument(
        "--run-prefix",
        default="",
        help="substring of the proxy_traces subdirectory to analyze (e.g. the experiment id)",
    )
    ap.add_argument("--out", default="results/analysis/cost_model")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    engine = load_engine_logs(args.results_dir)
    print(f"[load] {len(engine)} engine request records")
    rows = load_proxy_requests(args.results_dir, args.run_prefix)
    records, skipped = build_records(rows, engine)
    if not records:
        raise SystemExit(f"no requests joined; skip reasons: {skipped}")
    steps = load_tune_trajectory(args.results_dir, args.run_prefix)
    print(f"[load] {len(steps)} ES tuning steps")
    summary = report(records, skipped, args.out)
    summary["tune_steps"] = len(steps)
    if steps:
        last = steps[-1]
        summary["tune_final"] = {
            "step": last.get("step"),
            "best_params": last.get("best_params"),
            "best_score": last.get("best_score"),
            "sigma": last.get("sigma"),
            "converged": last.get("converged"),
        }
        print(
            f"[tune] {len(steps)} steps, final best_params={last.get('best_params')} "
            f"best_score={last.get('best_score')} converged={last.get('converged')}"
        )
        with open(f"{args.out}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    if not args.no_plots:
        plot(records, summary, args.out)
        plot_tune_trajectory(steps, summary, args.out)


if __name__ == "__main__":
    main()
