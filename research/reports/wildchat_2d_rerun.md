# WildChat rerun on the 2D cost model (Jul 27, 2026)

## Why this ran

The paper's WildChat appendix (Table `tab:wildchat`) was produced May 5 on a
pre-2D cost model: `score = rtt_seconds + t_prefill*uncached +
queued_tokens_weight*(queued+used)`. RTT entered raw in *seconds* against token
terms in the hundreds, so the RTT term was ~0.1% of every score — numerically
invisible. The ES could not rescue it: rescaling both token weights down ~1000x
to make RTT matter is a zero-gradient plateau under argmin scale-invariance
(routing is bit-identical along the whole path), ~13-17 sigma of coordinated
log-space travel, and the Rechenberg schedule froze sigma within the first ~3
minutes of the run. The published "gorgo (online)" weights (t_prefill=0.195,
queued_tokens_weight=0.043) are a ~90-evaluation random walk from the seed, not
a converged optimum. Trace-level analysis of the archived run also showed the
real TTFT deficit was dispatch *bursts* (mean same-target run length 3.5 vs 1.5
for random; TTFT climbs monotonically with position inside a burst) — not
aggregate imbalance (shares were ~uniform) and not proxy decision overhead
(~16 ms p50).

So the appendix numbers reflect a cost model the paper no longer describes.
This rerun puts the v3 2D model (`w_rtt*rtt_ms + uncached + w_queue*queued`)
on the same workload.

## Setup

- Windows: WildChat window1 (tune) -> window2 (held out, never used in the
  paper). Both are Mooncake-format traces with bodies on
  `GORGO-glm5-completions`; arrivals are synthetic Poisson at 11.4 rps over 30
  min (WildChat has no usable arrival timestamps). ~840 avg input tokens,
  block reuse 4.95% — the cache term has essentially nothing to discriminate.
- Fleet: per-arm 3x H100:1 in us-west4 / CANADA-2 / sines-2 (the paper's L40S
  regions were capacity-blocked; no Asia H100 capacity existed at launch).
  Arms: gorgo-2d, random, simple-session-affinity, each on its own fleet,
  same window in parallel.
- Client: c=64 open loop (the May run's c=32 backlogged ~21 min at the median
  and degenerated into closed-loop c=32).
- ES: window 256 / hop 64, sigma 0.5 -> 0.02, seeded RNG, paper's box
  unchanged (`w_rtt [0.05,2.0]`, `w_queue [0.05,0.5]`). Seeds at the physical
  point: on WildChat the uncached term cancels across replicas, so routing is
  set by w_rtt/w_queue = queued tokens per ms of RTT; H100 prefills ~0.04
  ms/token so 1 ms RTT ~ 25 tokens. Seeded rtt=1.2, queue=0.06 (ratio 20).
- Runs: `costmodel_wildchat_2d_v2` (tune ts=1.0 + eval ts=1.0) and
  `costmodel_wildchat_2d_ts2_v1` (static eval at ts=2.0, same frozen weights,
  matching the paper's tune-at-1x / eval-at-ts2 protocol). The sass arm of the
  ts2 run was lost to a proxy container death at teardown.

Learned weights: `w_rtt = 2.0` (box ceiling), `w_queue = 0.104` — ratio 19.2,
i.e. the ES kept the seeded physical ratio and wanted *more* RTT emphasis than
the box allows on this workload.

## Results (window2, held out, n≈20,420 per arm, 0 failures)

ts=1.0 (saturated — 11.4 rps exceeds what the fleet drains; client slip ~15 min
in every arm, effectively closed-loop c=64):

| policy    | TTFT p50 | TTFT p95 | E2E p95  | slip p50 |
|-----------|---------:|---------:|---------:|---------:|
| gorgo-2d  | 0.459    | 1.194    | **13.64**| **892 s**|
| random    | **0.263**| **0.572**| 17.90    | 1069 s   |
| sass      | 0.275    | 0.662    | 17.47    | 1004 s   |

ts=2.0 (unsaturated — open-loop schedule honored exactly, queues ~empty):

| policy    | TTFT p50 | TTFT p95 | TTFT p99 | E2E p95 |
|-----------|---------:|---------:|---------:|--------:|
| gorgo-2d  | **0.215**| **0.506**| **0.728**| **2.42**|
| random    | 0.244    | 0.611    | 1.166    | 2.51    |

Gorgo wins every TTFT percentile at ts=2.0 (p99 −38%) with no cache signal
involved — purely the RTT and queue terms.

## The main finding: saturation decouples TTFT from replica health

**Under saturation gorgo still wins E2E decisively but loses TTFT, and the
mechanism is continuous batching + chunked prefill: an overloaded replica keeps
handing out cheap first tokens (new prefill slots into the running batch)
while its decode drowns. The backlog is visible in E2E, invisible in TTFT.**

Measured directly at ts=1.0: every arm's us-west4 (spot) replica decoded slow.
Random/sass kept feeding it 33% blindly; its queue pinned at 34-39k tokens,
E2E p50 on that third of traffic hit 16-17 s, and it captured ~50 of the 64
client concurrency slots — yet its TTFT p50 stayed at 0.270 s,
indistinguishable from healthy. Their two idle-ish replicas (in-flight 4-5)
served ultra-fast first tokens, so the baselines' TTFT is subsidized by their
own E2E failure. Gorgo's queue term saw the backlog (39k vs 3k), cut the
straggler to 29%, held every queue under 14k, kept in-flight balanced (9-14),
won E2E p95 by 24%, and paid for it in visible TTFT: every diverted request
buys real RTT and joins a genuinely busier prefill queue.

This is the same tradeoff as the paper's load sweep (`tab:loadsweep`: at
ts=1.0 gorgo E2E 15.76 vs sass 17.94 while losing TTFT p95) — now reproduced
on a completely different workload (short prompts, no reuse), which makes it a
property of continuous batching under saturation, not of the trace.
One-sentence form: **gorgo's weights don't collapse under saturation; the
metric does — chunked prefill makes overloaded replicas look fast on TTFT, so
a policy that routes around overload looks slow exactly when it is doing its
job.**

Residual gorgo-specific cost, both loads: deterministic-argmin herding (mean
same-target run 3.8 vs 1.5; within-burst TTFT penalty +86 ms p50 from burst
start to position 6+, down from +178 ms under the old model). Randomized
tie-breaking / power-of-two-choices when scores are within noise would remove
it.

## Rebuttal-usable claims

1. The published WildChat numbers came from a superseded cost model whose RTT
   term was numerically dead (units bug, later documented in
   `policy/COST_MODEL.md` and fixed by the 2D gauge choice).
2. Rerun under the v3 model on a held-out WildChat window: gorgo beats random
   on all TTFT percentiles when the fleet has headroom, and converts to a 24%
   E2E p95 advantage under saturation by preventing queue collapse on a slow
   replica.
3. Neither result uses the cache term — consistent with (and sharpening) the
   paper's claim that *cache-aware* gains require long shared prefixes.
4. Caveat to state before a reviewer does: learned w_rtt railed at the box
   ceiling (2.0); the kept ratio (19.2) is the seeded physical prior, not a
   discovered interior optimum. On short-prompt workloads the paper's box
   binds.

## Artifacts

- Volume `GORGO-bench-results` (alessio-dev): `workload_runs/` and
  `proxy_traces/` under `costmodel_wildchat_2d_v2`,
  `costmodel_wildchat_2d_v2_eval`, `costmodel_wildchat_2d_ts2_v1`.
- Specs: `specs/c64/costmodel/tune_wildchat_w1_2d.json`,
  `eval_wildchat_w2_2d.json`, `eval_wildchat_w2_2d_ts2.json`;
  manifests `specs/c64/manifests/manifest_wildchat_window{1,2}.json`.
- May-run archaeology: results deleted at `cc35b43` ("Prepare for release");
  recover with `git show cc35b43^:results/analysis/wildchat_w1.md`,
  `git show cc35b43^:paper.md`, `git show cc35b43^:policy/COST_MODEL.md`.
