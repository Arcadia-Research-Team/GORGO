

<div align="center">
  <img src="assets/gorgo-logo.png" width="220" alt="GORGO" />
  <p>
    <a href="https://arxiv.org/abs/2602.11688"><strong>Paper</strong></a>
    | <a href="https://huggingface.co/datasets/alessiotoniolo/ART-Chat-2.5M"><strong>Dataset</strong></a>
    | <a href="https://romethorstenson.com/writing/network-aware-llm-routing"><strong>Blog</strong></a>
  </p>

  [![arXiv](https://img.shields.io/badge/arXiv-2602.11688-b31b1b.svg)](https://arxiv.org/abs/2602.11688)
  [![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-ART--Chat--2.5M-yellow.svg)](https://huggingface.co/datasets/alessiotoniolo/ART-Chat-2.5M)
</div>
<br/>

GORGO is an online-tuned, network-aware routing policy for cross-region LLM serving. It jointly accounts for network latency, prefix-cache reuse, and replica queueing to minimize TTFT, and learns its weights via an evolutionary strategy on live traffic.

We release [**ART-Chat-2.5M**](https://huggingface.co/datasets/alessiotoniolo/ART-Chat-2.5M), a synthetic long-context, high prefix-reuse chat dataset (2.5M requests, 19× the intra-user prefix reuse of WildChat-4.8M), used to tune and evaluate GORGO's routing policy in the paper.

## Goal
Decrease TTFT from standard methods for LLM load balancing (least-load, session-affinity, vtc-basic) by >2× using GORGO + tuning.

## Getting Started

Setup your Python environment:
```
uv venv
source .venv/bin/activate
uv sync
```

Note that for the following `run` commands, you can optionally specify a `--env=<your-modal-env>` following the `run` subcommand.

To launch a model replica:
```bash
REGION=us-east GPU_TYPE=H100 MODEL_ORG=Qwen MODEL_NAME=Qwen3.5-35B-A3B-FP8 modal deploy engine/modal_sglang.py
``` 

> Full list of regions [here](https://modal.com/docs/guide/region-selection).

To launch the proxy:
```bash
REGION=us-east modal run proxy/modal_proxy.py::proxy
```

> See the list of API routes [here](https://github.com/Arcadia-Research-Team/GORGO/blob/main/proxy/openapi.yaml)

To run an example workload on lmsys-chat-1m:
```bash
modal run proxy/workload.py --proxy-url https://... \
  --source hf --preset lmsys --num-requests 1000 --stream true
```

Or on Wildchat-4.8M with a specified data path:
```bash
modal run proxy/workload.py --proxy-url https://... \
  --source hf --data-path /datasets/allenai__WildChat-4.8M --stream true --num-requests 1000
```

> Note that the Modal volume names are specified in [app.py](https://github.com/Arcadia-Research-Team/GORGO/blob/main/app.py)


## Project Structure

- *proxy*: Request handling, workload streaming, and parameter tuning code that all runs on CPU instances in the same region.
- *engine*: LLM inference engine backend. Currently sglang is supported with DeepGEMM kernels built into the image and volume weight loading.
- *data_processing*: Scripts for reading from HF/local volumes and saving data + statistics + serialized radix trees to volumes.
- *policy*: Various load-balancing policies constructed from both Arcadia Research's GORGO paper and vLLM's AI-Brix model gateway.
- *utils*: Helpful util classes including RadixTrie, which is used for storing KV-cache state across sglang servers in the proxy.

## Running Policy Experiments

The experiment runner spawns isolated per-policy engine fleets + proxies across regions, replays a Mooncake trace, and saves results. See `experiment_runner/BENCHMARK_PLAN.md` for full methodology.

```bash
# 1. Build a trace (production example; also supports --source lmsys / --source wildchat)
modal run  data_processing/build_mooncake_trace.py::main \
  --source prod --start-time 2026-04-01T00:30:00 --end-time 2026-04-01T01:00:00 \
  --num-requests 200000 --include-bodies --max-input-tokens 24000 --time-scale 1.0 \
  --output-path mooncake_traces/my_trace/with_bodies/prod.jsonl

# 2. Launch (spec defines policies/regions/concurrency; manifest points at the trace)
modal run --detach  experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/c64/tuning/policy_matrix_c64_tuning_p95ttft_2d.json \
  --sweep-manifest-path specs/c64/manifests/manifest_prod_decoded_apr5_1615_1645.json \
  --experiment-id my_experiment_v1 --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/my_experiment_v1
# 3. Monitor / early-stop
python scripts/experiment_status.py --experiment-id my_experiment_v1 --env alessio-dev
python scripts/stop_experiment.py --experiment-id my_experiment_v1  # saves partial results

# 4. Pull + analyze
modal volume get  --force GORGO-bench-results /workload_runs results/
python scripts/analyze_results.py --prefix <run_prefix> --label "My Run"
python scripts/plot_policy_summary.py --results-dir results --run-prefix <run_prefix> --out results/analysis/summary.png
```

> To reproduce the paper's tuning + eval runs, use the specs/manifests in `specs/c64/`: `tuning/policy_matrix_c64_tuning_p95ttft_2d.json` (tuning) and `eval_ts2/{apr6,apr7_ts3}.json` (held-out eval), each paired with the matching file in `specs/c64/manifests/`. `specs/c64/loadsweep_apr7/` holds the `time_scale` sweep used for the load-sweep results.

### Using the public ART-Chat-2.5M dataset

The paper's traces above live on a private Modal volume, but [ART-Chat-2.5M](https://huggingface.co/datasets/alessiotoniolo/ART-Chat-2.5M) ships the same data publicly as one Mooncake FAST'25 JSONL per day (`jsonl/artchat_week_<YYYYMMDD>.jsonl`, April 1st–7th 2026) — already in the `request`/`timestamp`/`hash_ids` shape the proxy replays directly, so no conversion script is needed.

```bash
# 1. Download a day's trace and land it on the GORGO-hf-datasets volume
huggingface-cli download alessiotoniolo/ART-Chat-2.5M \
  jsonl/artchat_week_20260405.jsonl --repo-type dataset --local-dir /tmp/art-chat
modal volume put  GORGO-hf-datasets \
  /tmp/art-chat/jsonl/artchat_week_20260405.jsonl \
  mooncake_traces/art_chat/artchat_week_20260405.jsonl

# 2. Point a manifest at it (mirrors the shape of specs/c64/manifests/*.json)
cat > specs/c64/manifests/manifest_hf_apr5.json <<'EOF'
{"top": [{"result": {"output_path": "/datasets/mooncake_traces/art_chat/artchat_week_20260405.jsonl"}}]}
EOF

# 3. Launch with any spec from specs/c64/, same as above
modal run --detach  experiment_runner/policy_matrix_app.py::main \
  --base-spec-path specs/c64/tuning/policy_matrix_c64_tuning_p95ttft_2d.json \
  --sweep-manifest-path specs/c64/manifests/manifest_hf_apr5.json \
  --experiment-id my_hf_run --start-index 0 --top-k 1 \
  --output-dir /results/policy_matrix_sweep/my_hf_run
```

> Row `timestamp` in the released trace is milliseconds since the start of the whole week, not the start of that day, so an `"arrival_mode": "open-loop"` spec (used by the paper's specs, tuned for a pre-sliced 30-minute window) will sit idle for days before the first request. For a full day file, either slice/rebase it yourself to the window you want, or set `"arrival_mode": "bounded"` in your spec copy to replay requests back-to-back at the configured concurrency and ignore the raw schedule entirely.

## Tuning Parameters

The tuning script is now a lightweight client for the running proxy. It
submits a batch tuning request to the proxy's embedded `/tuning/*` API; the
proxy runs the workload locally against `http://127.0.0.1:8000` so the tuning
metric does not include client-to-proxy tunnel latency. Start the proxy with
the `GORGO-completions` and `GORGO-bench-results` volumes available (the
default `proxy/modal_proxy.py::proxy` deployment does this) and set the active
policy to `gorgo` before launching a run.

The tuning script will present a TUI allowing you to adjust default parameters
before starting the proxy-managed workload steps.
```bash
modal run -q proxy/tuning.py::tune_interactive --proxy-url https://your-proxy.modal.run
```

Alternatively, you can specify the specific parameters/settings via CLI args:
```bash
modal run proxy/tuning.py::tune_cli --proxy-url https://your-proxy.modal.run \
  --start-time 2026-04-01T12:00:00 \
  --num-requests 200 \
  --concurrency 32 \
  --metric output_throughput \
  --algorithm gaussian-es \
  --max-steps 16 \
  --seed 0 \
  --t-prefill-min 1e-4 --t-prefill-max 0.1 \
  --queued-tokens-weight-min 1e-3 --queued-tokens-weight-max 0.05
```

## Citation

```bibtex
@misc{gorgo2026,
  title         = {GORGO: Online Tuning for Cross-Region Network-Aware LLM Serving},
  author        = {Toniolo, Alessio Ricci and Thorstenson, Rome and Dinesh, Abinaya},
  year          = {2026},
  eprint        = {2602.11688},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC}
}
```
