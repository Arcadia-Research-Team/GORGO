# Cost-model rank validation: raw results (Jul 31, 2026)

Offline analysis for Reviewer MZTU's follow-up ("why is the relative ranking
preserved despite large absolute prediction errors?"). No new runs — everything
below is recomputed from existing proxy traces and engine logs on the
`GORGO-bench-results` volume (alessio-dev).

## Method

Every proxy trace row logs a `candidate_snapshot`: RTT, queued tokens, and
cached-prefix tokens for ALL replicas, on every arm including `random`.
Because the random policy dispatches independently of any score, the measured
TTFT of a random dispatch is an unbiased sample of that replica's
counterfactual latency. So, per request on the random arm:

1. Recompute the GORGO 2D score offline for each candidate:
   `score = w_rtt * rtt_ms + uncached_tokens + w_queue * queued_tokens`
2. Rank the 3 candidates by score (rank 1 = predicted best).
3. Bucket the request's measured TTFT by the predicted rank of the replica
   random happened to pick.

If the model's ordering is informative, TTFT should be monotone in rank —
regardless of absolute calibration.

Scripts: `/tmp/gorgo_rank/rank_analysis_wc.py` (rank/gap analysis),
`/tmp/gorgo_rank/per_replica_fit.py` (engine-log rate fits),
`/tmp/gorgo_cm/reproduce_rebuttal.py` (rebuttal error-rate reproduction).

## Data sources

| Label | Run (volume path under proxy_traces/) | Window | ts | Weights |
|---|---|---|---|---|
| WildChat ts2, random arm | `costmodel_wildchat_2d_ts2_v1/..._random` | window2 | 2.0 | scored offline with w_rtt=2.0, w_queue=0.104 (learned on window1) |
| WildChat ts2, gorgo arm | `costmodel_wildchat_2d_ts2_v1/..._gorgo-static-p95-2d-wildchat` | window2 | 2.0 | deployed w_rtt=2.0, w_queue=0.104 (paired comparison, §5) |
| Apr5-tune / Apr6-eval chain | `costmodel_tune_apr5_eval_apr6_h100_v1_eval0` + its engine logs | Apr 6 15:05-15:35 | 2.0 | learned w_rtt=1.13, w_queue=0.242 (rebuttal reproduction, §6) |

WildChat runs: H100:1 fleet in us-west4 / CANADA-2 / sines-2, c=64 open-loop —
the rebuttal rerun documented in `research/reports/wildchat_2d_rerun.md`, and
the run behind the paper's Table `tab:wildchat`. The chain run: H100:1 AZR
fleet in centralus / northeurope / malaysiawest, the run behind the rebuttal's
per-component error rates.

## 1. Measured TTFT by predicted rank of the dispatched replica

WildChat window2, ts=2.0 (20,421 usable; TTFT ms, E2E s):

| rank | n | ttft mean | p50 | p95 | p99 | e2e p50 | e2e p95 |
|---|---|---|---|---|---|---|---|
| 1 | 6,823 | 235 | 193 | 455 | 695 | 1.24 | 2.24 |
| 2 | 6,856 | 310 | 241 | 572 | 843 | 1.39 | 2.44 |
| 3 | 6,742 | 427 | 321 | 764 | 2,142 | 1.47 | 2.83 |

Spearman rho (predicted score of dispatched replica vs measured TTFT): **0.611**
Top-2 predicted score gap: p50 = 228 units, p95 = 721 units.

## 2. TTFT by predicted-score gap (dispatched replica vs predicted best)

Addresses "similar predicted costs": mis-ranking is only possible where the
gap is small, and there the measured outcomes are close too.
TTFT mean / p50 / p95 (ms); gap in score units (with w_rtt=2.0, 100 units
= 50 ms of RTT or ~960 queued tokens):

| rank1 | gap<100 | gap 100-400 | gap>400 |
|---|---|---|---|
| 235 / 193 / 455 | 283 / 225 / 528 | 298 / 232 / 545 | 452 / 336 / 795 |

Dispatches within 100 units of the predicted best cost +20% mean TTFT vs
rank-1; gaps >400 units cost +92%. Where the ordering could plausibly be
wrong, the regret is bounded by the gap.

## 3. Rank-region correspondence: the ranking is dynamic, not a static RTT order

Probe RTT per region (proxy in us-east, ms): CANADA-2 p50=55 (p95=545),
us-west4 p50=151 (p95=228), sines-2 p50=244 (p95=418).

How often each region occupies each predicted rank (% of requests):

| rank | CANADA-2 | us-west4 | sines-2 |
|---|---|---|---|
| 1 | 67.0% | 24.4% | 8.6% |
| 2 | 16.0% | 47.3% | 36.6% |
| 3 | 17.0% | 28.3% | 54.7% |

The modal ordering follows median RTT (CANADA-2 < us-west4 < sines-2), but a
pure-RTT score would make this table 100% diagonal. It is not: CANADA-2 is
demoted off rank 1 on a third of requests, and the demotions are correct —
measured TTFT of random dispatches to the SAME region by its predicted rank
at dispatch time (n / mean / p50 / p95 ms):

| region | when rank 1 | when rank 2 | when rank 3 |
|---|---|---|---|
| CANADA-2 | 4,542 / 207 / 141 / 382 | 1,085 / 292 / 212 / 507 | 1,146 / 729 / 242 / 3,074 |
| us-west4 | 1,680 / 281 / 215 / 480 | 3,290 / 300 / 269 / 533 | 1,895 / 342 / 328 / 583 |
| sines-2 | 601 / 320 / 239 / 588 | 2,481 / 332 / 239 / 637 | 3,701 / 378 / 338 / 752 |

CANADA-2 ranked 3rd is 3.5x slower on mean TTFT (and 8x on p95) than
CANADA-2 ranked 1st: the queue term and the RTT probe's EWMA (note the
545 ms p95 spikes) catch transient congestion on the nearest replica. The
ordering carries per-request dynamic signal beyond the inter-region RTT
constant.

## 4. Per-replica component homogeneity (engine logs, WildChat ts=2.0 random arm)

Measured prefill rate fit through origin (prefill duration vs uncached tokens,
requests with >=256 uncached tokens), plus scheduler queue_time:

| replica | n | fit n | prefill slope (ms/uncached tok) | queue_time p50 / p95 (ms) |
|---|---|---|---|---|
| us-west4 | 6,469 | 3,287 | 0.0834 | 0.3 / 0.6 |
| CANADA-2 | 6,431 | 3,123 | 0.0690 | 0.3 / 0.5 |
| sines-2 | 6,377 | 3,214 | 0.0694 | 0.3 / 0.5 |

CANADA-2 and sines-2 agree within 1%; us-west4 (the spot node) is ~20%
slower. So the prefill-scale bias reported in the rebuttal (−42.2%) is mostly
a fleet-wide constant — common-mode under the argmin — with one
replica-differential outlier that the queue term ends up compensating for.
The RTT term is a direct per-replica measurement (EWMA probe), not a model
output. Only replica-differential error can reorder candidates.

## 5. Paired per-request comparison: gorgo arm vs random arm

Both arms replay the same request at the same time on identical (separate)
fleets, and `request_id` is deterministic per trace row, so requests pair
across arms (20,332 pairs, 99.6% join). Paired TTFT difference,
gorgo − random (ms; negative = gorgo faster):

| subset | n | mean | p50 | gorgo wins |
|---|---|---|---|---|
| all requests | 20,332 | −67.4 | −27.1 | 60.4% |
| same region chosen (control) | 6,877 | −46.2 | **+2.6** | 47.0% |
| different region chosen | 13,455 | −78.3 | −59.3 | 67.3% |

The same-choice subset is a built-in placebo control: median paired
difference ~0 (+2.6 ms) means fleet-level confounds between the arms are
negligible, so the disagreement-subset gap is attributable to the routing
decisions. (The same-region mean is −46 ms — the arms' diverged queue states
show up in the tail; the median is the control statistic.)

Disagreements stratified by the gorgo-model rank of random's chosen replica
(scored in the random arm's own state):

| random's choice was predicted... | n | mean | p50 | gorgo wins |
|---|---|---|---|---|
| rank 1 | 3,386 | +30.5 | +54.5 | 40.0% |
| rank 2 | 4,973 | −63.6 | −62.3 | 72.9% |
| rank 3 | 5,096 | −164.9 | −88.4 | 79.9% |

Dose-response: the worse the model rates the alternative, the larger gorgo's
measured advantage. The rank-1 row is the honest self-check — when random
landed on the model's preferred replica, random won modestly. A noise-driven
ranking would show flat rows.

Caveat: the arms' cache/queue states diverge as a consequence of their
policies, so pairs measure the full policy effect (decision + induced state),
which is the deployment-relevant quantity.

## 6. Reproducing the rebuttal's per-component error rates

The rebuttal quoted: mean TTFT error −54.5%; components −86.6% (RTT),
−42.2% (prefill), +28.0% (queueing). Reproduced from the
`costmodel_tune_apr5_eval_apr6_h100_v1` eval0 run (6,864 proxy-trace rows
joined to engine timing logs on rid; deployed weights w_rtt=1.13,
w_queue=0.242):

The procedure was a SINGLE global scalar — least-squares through origin
mapping total score to measured TTFT (k=0.0700) — applied to each component
mean. Not per-component fits:

| quantity | reproduction (single global k) | rebuttal quoted |
|---|---|---|
| mean TTFT error | −55.1% | −54.5% |
| network RTT (vs raw probe RTT) | −92.1% | −86.6% |
| prefill (vs engine prefill time) | −49.0% | −42.2% |
| queueing (vs engine queue_time) | −70.0% | +28.0% |
| queueing (vs prefill_waiting_latency) | +1.4% | +28.0% |

Total, RTT, and prefill match the single-scalar procedure; per-component
least-squares fits (the alternative hypothesis) give −165%/−79%/−78% and
cannot produce a positive queue error at all. The exact +28.0% queue figure
was not reproduced — it likely used a different measured-queue definition or
request subset — but its sign only arises under the shared-scale accounting.

Implications:

- With one scalar fit to the total, the scaled components are forced to
  roughly sum to measured TTFT, so the mixed signs are coupled by
  construction. They are not three independent component-accuracy
  measurements; they measure how the tuned weight RATIOS deviate from
  physical ms-per-unit ratios — the operating point the ES chose while
  optimizing measured p95 TTFT, not accumulated prediction error.
- Decomposition wart: the engine-log network residual
  (TTFT − ingress − queue − prefill) is NEGATIVE on average (−78 ms),
  because with chunked prefill the first token can stream before
  `prefill_finished_time`. The rebuttal's "measured RTT" was therefore the
  raw probe RTT, and the additive decomposition should not be leaned on as
  ground truth per-component.

## 7. Why differential component errors do not (automatically) break the argmin

The reviewer's concern is technically valid: per-component scale errors are
NOT an order-preserving transform. If the score is effectively
`b_rtt*RTT + b_p*P + b_q*Q` with different b_c, two replicas with different
cost composition (near-but-busy vs far-but-idle) can flip order vs true
TTFT. "We route on relative scores, therefore ranking survives" does not
follow — it is an empirical property that had to be demonstrated. The
resolution:

1. The quoted error signs are coupled by the single-scalar accounting (§6),
   so they overstate inter-component inconsistency.
2. The component scales ARE the tuned weights: the tilt of the decision
   boundary away from physical ratios is what the ES optimizes on measured
   p95 TTFT. Deviating from physics can be routing-correct (queued tokens
   also predict decode contention beyond their literal queueing-time
   contribution; cf. the w_queue=0 reward-hacking appendix).
3. A fixed tilt flips only pairs near the decision boundary, and the
   gap-binned analysis (§2) shows regret there is bounded and small; large
   gaps are reliably ordered.
4. The rank validation (§1), rank-region crosstab (§3), and paired
   comparison (§5) demonstrate the induced ordering is empirically correct.

### Worked flip example (WildChat weights, measured physical rates)

Ground truth `T = RTT + 0.07*U + 0.005*Q` (ms; measured H100 rates) vs
deployed score `S = 2.0*RTT + U + 0.104*Q`. Exchange rates: ground truth
prices 1 ms of RTT at ~14.3 uncached tokens; the score prices it at 2.0 —
a ~7x under-pricing of RTT relative to prefill.

Near-but-uncached vs far-but-cached:

| replica | RTT (ms) | uncached U | S (score) | T (ms, truth) |
|---|---|---|---|---|
| A: CANADA-2, cache miss | 55 | 840 | 950 | 113.8 |
| B: sines-2, prefix cached | 245 | 0 | 490 | 245.0 |

The score picks B (gap 460 units); A is actually 131 ms faster. General
condition at dRTT=190 ms: flips occur for token asymmetry dU in
(2.0*190, 14.3*190) = (380, ~2,714) tokens. WildChat prompts (~840 tok) sit
inside this window — the original WildChat failure was this same geometry
with w_rtt effectively ~0.002 (units bug), which made the window enormous.
ART-Chat with paper weights under-prices RTT ~129x, but its cache
asymmetries (~15k tokens) exceed the window's upper bound and near-ties fall
below it, so flips concentrate in a middle band the workload rarely visits.

Empirical frequency/cost of the tilt (random arm, deployed vs
physically-calibrated argmin):

- The two models choose the same replica on 73.9% of requests.
- On disagreements, random dispatches landing on the deployed argmin
  measured 291 ms mean / 229 p50 TTFT vs 315 / 232 for the physical argmin —
  the deployed tilt costs nothing where it could have mattered (slightly
  better on the mean).

## 8. Derived headline numbers

- Measured TTFT is monotone in predicted rank at every reported percentile.
- Rank-3 vs rank-1 mean TTFT penalty: +82%.
- Spearman rho (predicted score vs measured TTFT): 0.611 — despite the cache
  term contributing nothing on WildChat (~5% reuse): the ordering comes from
  RTT + queue alone.
- Median top-2 score gap is 228 units, i.e. typical decisions are not close
  calls even on a low-signal workload — inter-region RTT separation alone
  (100 ms diff = 200 units at w_rtt=2.0) separates the candidates.
- The ranking is not a static nearest-region order: CANADA-2 (nearest) is
  demoted off rank 1 on 33% of requests, and dispatches to it while ranked
  3rd measured 3.5x the mean TTFT of dispatches while ranked 1st.
- Paired vs random on the same requests: gorgo wins 60.4% overall (median
  −27 ms), 79.9% when random picked the model's rank-3 replica (median
  −88 ms), with a ~0 median difference on the same-choice control subset.
- The rebuttal's per-component error rates come from one global scalar
  applied to all terms; their mixed signs are coupled by construction and
  are not independent component accuracies.

## What existing data cannot answer

Per-request ranking accuracy on the GORGO arm in isolation: its dispatches
are score-correlated, so non-chosen replicas' latencies are unobserved
(selection bias). Would require shadow probes. The random-arm rank analysis
(§1) and the cross-arm paired comparison (§5) together make this unnecessary
for the rebuttal.
