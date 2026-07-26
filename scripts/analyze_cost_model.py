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


def multi_ols(predictors: list[list[float]], ys: list[float]) -> tuple[list[float], float]:
    """Least squares for ``y = c0 + c1*x1 + ... + ck*xk``.

    Returns ``(coefficients_with_intercept_first, r2)``. Reuses the tiny
    dependency-free solver the online calibrator already relies on.
    """
    from gorgo.tuner import solve_spd

    n = len(ys)
    k = len(predictors)
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
    resid = sum(
        (ys[i] - sum(c * v for c, v in zip(coef, design[i]))) ** 2 for i in range(n)
    )
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


def predict(rec: dict, params: dict) -> dict:
    """Cost-model prediction for one request under ``params``."""
    rtt_pred = params["rtt_weight"] * (rec["rtt_ms"] or 0.0)
    p_pred = params["prefill_rate"] * rec["uncached_tokens"]
    q_pred = params["queue_rate"] * params["queue_weight"] * rec["queued_tokens"]
    return {
        "rtt_pred_ms": rtt_pred,
        "p_pred_ms": p_pred,
        "q_pred_ms": q_pred,
        "ttft_pred_ms": rtt_pred + p_pred + q_pred,
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
# Reporting
# ---------------------------------------------------------------------------


def report(records: list[dict], skipped: dict, out_prefix: str) -> dict:
    deployed = dict(records[0]["hp"]) if records else {}
    for key, default in (
        ("rtt_weight", 1.0),
        ("prefill_rate", 1.0),
        ("queue_rate", 1.0),
        ("queue_weight", 1.0),
    ):
        deployed.setdefault(key, default)
    physical = fit_physical_rates(records)

    summary: dict = {
        "n_joined": len(records),
        "skipped": skipped,
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

    # Per-component accuracy under both parameterizations.
    for label, params in (("deployed", deployed), ("physical", physical)):
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
        a, b, r2 = ols(
            [p["ttft_pred_ms"] for p in preds], [r["ttft_ms"] for r in records]
        )
        print(
            f"  affine fit TTFT_meas = {a:.1f} + {b:.4f} * score   R^2={r2:.3f}  "
            "(shape-only test; absorbs the unidentifiable score scale)"
        )
        summary["components"][label]["affine_fit"] = {"a": a, "b": b, "r2": r2}

    # Calibration by request length and by load.
    for bucket_label, key, edges, store in (
        ("PROMPT LENGTH (uncached tokens)", "uncached_tokens", [1000, 4000, 8000, 16000], "by_prompt_length"),
        ("LOAD (queued tokens at dispatch)", "queued_tokens", [1000, 10000, 50000, 150000], "by_load"),
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
        print(f"  {'region':>14} {'n':>6} {'rtt_p50':>9} {'resid_p50':>10} {'resid_p95':>10} {'ptok_p50':>9}")
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
    print(f"  resid_ms = {a:.1f} + {b:.4f} * Q_ms    R^2={r2:.3f}  r={pearson(q_vals, resid_vals):.3f}")
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
        "rid", "replica_key", "region", "prompt_tokens", "cached_tokens", "uncached_tokens",
        "queued_tokens", "rtt_ms", "ttft_ms", "q_meas_ms", "p_meas_ms",
        "ingress_meas_ms", "resid_meas_ms",
    ]
    with open(csv_path, "w") as f:
        f.write(",".join(fields + ["rtt_pred_ms", "q_pred_ms", "p_pred_ms", "ttft_pred_ms"]) + "\n")
        for r in records:
            p = predict(r, physical)
            vals = [r.get(k) for k in fields] + [
                p["rtt_pred_ms"], p["q_pred_ms"], p["p_pred_ms"], p["ttft_pred_ms"]
            ]
            f.write(",".join("" if v is None else str(v) for v in vals) + "\n")
    print(f"[out] wrote {csv_path}")
    return summary


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
    summary = report(records, skipped, args.out)
    if not args.no_plots:
        plot(records, summary, args.out)


if __name__ == "__main__":
    main()
