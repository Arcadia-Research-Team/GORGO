Figures extracted from 33143_GORGO_Online_Tuning_for_.pdf

The PNG files are the embedded figure images from the PDF, preserved at their original resolution.

Figure_1.png — source page 3
Figure 1: Left: Token composition per dataset. GLM-5.1’s reuse is overwhelmingly intra-user (54%), reflecting multi-turn dialogue; WildChat’s is predominantly cross-user (29%, shared templates); LMSYS has minimal reuse (9%, all cross-user). Right: Radar chart with six axes normalized to the dataset maximum. GLM-5.1 dominates on every axis relevant to routing: long prompts, high multi-turn density, concentrated users, and strong intra-user prefix reuse.

Figure_2.png — source page 5
Figure 2: Proxy-to-replica RTT over the W1 tuning window. Bold lines show the EWMA-smoothed signal (α=0.3) that GORGO uses for routing; faint lines show raw probe samples. Dashed lines show per-replica means. Seoul exhibits periodic spikes from routing-table churn; Frankfurt is moderately variable; Ashburn is near-constant. The 18× spread motivates network-aware routing.

Figure_3.png — source page 7
Figure 3: W1 TTFT broken out by percentile (sorted by p95, worst at top). gorgo-hillclimb (gold outline) wins all three percentiles during the tuning window.

Figure_4.png — source page 7
Figure 4: Left: KV-cache hit rate vs. p95 TTFT on W2. Bottom-right is ideal (high cache, low latency). gorgo-static and gorgo-hillclimb achieve both; simple-session-affinity achieves the highest cache hit rate but the worst p95 because it cannot rebalance load. Right: routing concentration (share of requests on the most-used replica). The dashed line marks uniform (33%). GORGO variants concentrate moderately on cache-warm replicas without the pathological imbalance of simple-session-affinity.

Figure_5.png — source page 8
Figure 5: W2 TTFT broken out by percentile (sorted by W1 p95, worst at top). Gold outline marks the winner in each panel. gorgo-static (frozen W1 weights) wins p50 and p95; gorgo-hillclimb wins p99. simple-session-affinity degrades sharply at the tail (4.92 s p99).

Figure_6.png — source page 14
Figure 6: WildChat-4.8M replay, per-percentile TTFT (p50/p95/p99) across the eight policies. GORGO variants (dark blues) are not competitive: gorgo-static and gorgo-autotune are the slowest two policies at the tail, and gorgo-hillclimb sits mid-pack. The ranking inverts the GLM-5.1 result, consistent with the regime argument in §2.

Figure_7.png — source page 15
Figure 7: Achieved prefix hit rate over the W1 tuning window: the fraction of each request’s input tokens that were already cached on the replica the policy routed to (not the total cache available across all replicas). A higher rate means the routing decision exploited more of the available cache. Bold lines are EWMA-smoothed (α=0.06); faint lines are 64-request rolling averages. gorgo-hillclimb converges from ∼45% to ∼95% within the first 5 minutes as the ES learns to route requests to their best-cached replica.
