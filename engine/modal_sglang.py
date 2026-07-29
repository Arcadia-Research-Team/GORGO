import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request

import modal

from app import app, bench_results_volume, ENVIRONMENT_NAME

replicas = modal.Dict.from_name(
    "GORGO-replicas", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu13-20260411-0011d2ae")
    .run_commands("rm -rf /root/.cache/huggingface")
    .entrypoint(
        []  # silence chatty logs on container start
    )
)
# NOTE: the local source is added as the LAST image layer (see below, after
# compile_deep_gemm). compile_deep_gemm doesn't need our code, and baking the
# source in before it would make every code edit invalidate that ~20-min
# compile layer. Keeping the copy last means edits only rebuild the cheap
# final layer.

REGION = os.getenv("REGION", "us-east")
GPU_TYPE = os.getenv("GPU_TYPE", "H100")
MODEL_ORG = os.getenv("MODEL_ORG", "Qwen")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3.5-35B-A3B-FP8")
HF_REPO_ID = f"{MODEL_ORG}/{MODEL_NAME}"
MODEL_REVISION = (  # pin revision id to avoid nasty surprises!
    "0b2752837483aa34b3db6e83e151b150c0e00e49"  # latest commit as of 2026-04-03, from release
)
N_GPUS = os.getenv("N_GPUS", 1)
GPU = f"{GPU_TYPE}:{N_GPUS}"
PORT = 8000
# Pinning ``--context-length`` lets ``proxy/workload.py`` auto-detect a
# safe input-token cap from ``/get_server_info`` (which would otherwise
# return ``context_length: null`` and force operators to remember
# ``--max-input-tokens N`` on every workload/tuning run). Default is
# Qwen3's native 32k window; bump via env var if the GPU has the KV
# headroom to support longer prompts.
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", 32768))
HF_CACHE_VOL = modal.Volume.from_name(
    f"{MODEL_NAME}-huggingface-cache", create_if_missing=True, environment_name=ENVIRONMENT_NAME
)
HF_CACHE_PATH = "/root/.cache/huggingface"
FULL_MODEL_NAME = f"{MODEL_ORG}/{MODEL_NAME}"
MODEL_PATH = f"{HF_CACHE_PATH}/{FULL_MODEL_NAME}"
# Default to scale-to-zero: ``modal deploy`` parks the function with no
# warm replicas, the first inbound /v1/chat/completions cold-starts a
# container and ``wait_ready`` does an in-process warmup before
# registering the tunnel URL in ``replicas[REGION]`` (the workload
# client also has its own client-side warmup phase). Once warm, the
# container survives ``SCALEDOWN_WINDOW_SECONDS`` of idle so back-to-back
# experiments avoid paying cold-start again. Set ``MIN_CONTAINERS`` to a
# positive integer if you want a permanently-warm pool instead.
MIN_CONTAINERS = os.getenv("MIN_CONTAINERS", 0)
SCALEDOWN_WINDOW_SECONDS = int(os.getenv("SCALEDOWN_WINDOW_SECONDS", 15 * 60))
WAIT_READY_TIMEOUT = os.getenv("WAIT_READY_TIMEOUT", 1200)

# ---------------------------------------------------------------------------
# Per-request engine timing log (cost-model validation only; off by default)
# ---------------------------------------------------------------------------
# Set ``REQUEST_TIMING_LOG=1`` to make SGLang emit one JSON line per request
# carrying the scheduler's own per-request timestamps. This is what lets the
# TTFT decomposition be validated against *measured* components rather than
# inferred ones. Nothing here changes how requests are served or scheduled --
# it is pure observability, and the routing policy still consumes only the
# HTTP-exposed signals (``/metrics`` + the proxy's RTT probe).
#
# Mechanism, all stock SGLang flags (verified against this image with
# ``scripts/introspect_sglang_engine.py``):
#
#   --enable-metrics          already passed below; gates the scheduler ->
#                             tokenizer propagation of per-request timing
#                             (``SchedulerReqTimeStats.__getstate__`` returns
#                             ``{}`` without it) and the merge into meta_info
#                             at ``tokenizer_manager.py:1665``.
#   --log-requests            enables RequestLogger.
#   --log-requests-level 0    skips ``text`` / ``input_ids`` / ``output_ids``,
#                             so prompts never reach the log -- privacy-safe
#                             and small, while ``meta_info`` is retained.
#   --log-requests-format json  one JSON object per line.
#   --log-requests-target DIR   a *directory*; SGLang writes
#                               ``<hostname>_<rank>.log`` inside it
#                               (``srt/utils/log_utils.py``).
#
# Each ``event: "request.finished"`` line carries ``rid`` plus ``out.meta_info``
# with the fields the decomposition needs:
#
#   queue_time            = forward_entry_time - wait_queue_entry_time   (Q)
#   forward_entry_time    absolute engine wall-clock at scheduler admission
#   prefill_finished_time absolute engine wall-clock at end of prefill
#                         => P = prefill_finished_time - forward_entry_time
#   request_received_ts / response_sent_to_client_ts  (API-server side)
#
# Q and P are differences between two timestamps taken on the *same* host, so
# neither needs cross-region clock alignment. The proxy sends its own request
# id as SGLang's ``rid`` (a first-class field on the chat request), which is
# the join key back to the proxy trace.
REQUEST_TIMING_LOG = os.getenv("REQUEST_TIMING_LOG", "") not in ("", "0", "false", "False")
# Root of the per-replica log directories on the bench-results volume.
REQUEST_TIMING_LOG_ROOT = os.getenv("REQUEST_TIMING_LOG_ROOT", "/results/engine_req_logs")
# The logging handler writes continuously but a Modal volume only publishes on
# commit, so a background thread commits on this interval (and once more on
# shutdown). Engines are torn down by the controller, so waiting for exit would
# risk losing the whole run.
REQUEST_TIMING_COMMIT_SECONDS = float(os.getenv("REQUEST_TIMING_COMMIT_SECONDS", 15.0))

sglang_image = sglang_image.env(
    {
        "HF_HUB_CACHE": HF_CACHE_PATH,
        "HF_XET_HIGH_PERFORMANCE": "1",
        "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
    }
)
sglang_image = sglang_image.run_commands(
    f"python3 -m sglang.compile_deep_gemm --model-path {FULL_MODEL_NAME} --revision {MODEL_REVISION} --tp {N_GPUS}",
    # Do not mount the DeepGEMM cache here; compiled kernels should be written
    # into the image layer. The HF cache remains a volume so model files are not
    # baked into the image.
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    gpu=GPU,
)
# Install the per-request timing serialization fix (engine/sglang_timing_patch.py)
# so SGLang's own scheduler timings survive the scheduler -> detokenizer ->
# tokenizer hops and reach meta_info. Auto-imported through a ``.pth`` file --
# NOT a ``sitecustomize.py``, which would shadow Modal's own ``/pkg/
# sitecustomize.py`` (PYTHONPATH is ``/pkg/:/root/``) and break the container.
# A ``.pth`` line that isn't an ``import`` statement is added to ``sys.path``,
# so the file adds ``/opt/gorgo`` and then imports the module from it. The patch
# is inert unless REQUEST_TIMING_LOG is set, so normal deploys are unaffected.
# Placed after compile_deep_gemm to keep that layer's hash stable.
sglang_image = sglang_image.add_local_file(
    "engine/sglang_timing_patch.py", "/opt/gorgo/sglang_timing_patch.py", copy=True
).run_commands(
    "python3 -c \"import os, site; "
    "d = site.getsitepackages()[0]; "
    "open(os.path.join(d, 'zz_gorgo_timing_patch.pth'), 'w')"
    ".write('/opt/gorgo\\nimport sglang_timing_patch\\n')\""
)
# Propagate the timing-log toggle into the container: the module is re-imported
# there, so ``REQUEST_TIMING_LOG`` would otherwise always read empty and the
# flags would silently never be passed. Deliberately a *separate* ``.env()``
# call placed AFTER compile_deep_gemm -- folding it into the env block above
# would change that layer's hash and force a ~20-minute kernel recompile on
# every toggle. Deploy-time env controls it, matching REGION / GPU_TYPE.
sglang_image = sglang_image.env(
    {
        "REQUEST_TIMING_LOG": os.getenv("REQUEST_TIMING_LOG", ""),
        "REQUEST_TIMING_LOG_ROOT": REQUEST_TIMING_LOG_ROOT,
        "REQUEST_TIMING_COMMIT_SECONDS": str(REQUEST_TIMING_COMMIT_SECONDS),
    }
)
# Local source goes LAST so editing app/engine only rebuilds this cheap copy
# layer, not the expensive compile_deep_gemm step above.
sglang_image = sglang_image.add_local_python_source("app", "engine", copy=True)


@app.function(
    image=sglang_image,
    timeout=24 * 60 * 60,
    region=REGION,
    gpu=GPU,
    min_containers=int(MIN_CONTAINERS),
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    # ``/results`` carries the per-request engine timing log when
    # REQUEST_TIMING_LOG is on. Mounted unconditionally so toggling the env var
    # doesn't change the image/function signature (and cost nothing when idle).
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, "/results": bench_results_volume},
)
def model_endpoint(registry_key: str = REGION):
    import os

    os.environ["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "1"
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        HF_REPO_ID,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        HF_REPO_ID,
        "--host",
        "0.0.0.0",
        "--port",
        f"{PORT}",
        "--tp",  # use all GPUs to split up tensor-parallel operations
        f"{N_GPUS}",
        "--cuda-graph-max-bs",  # only capture CUDA graphs for batch sizes we're likely to observe
        f"{10 * 2}",
        "--enable-metrics",  # expose metrics endpoints for telemetry
        # Populate ``usage.prompt_tokens_details.cached_tokens`` on every
        # OpenAI-compatible response (off by default in SGLang; verified
        # null without it on this build). The proxy's cache-eviction
        # feedback compares this served-cache truth against its radix-trie
        # prediction, so without the flag the whole feedback path is inert.
        "--enable-cache-report",
        "--decode-log-interval",  # how often to log during decoding, in tokens
        "100",
        "--mem-fraction",  # leave space for speculative model
        "0.8",
        "--context-length",  # surfaces in /get_server_info; drives workload pre-filter
        f"{CONTEXT_LENGTH}",
    ]

    # Per-request timing log: stock flags only, see REQUEST_TIMING_LOG above.
    timing_args, timing_log_dir = request_timing_log_args(registry_key)
    cmd += timing_args

    # SGLang exposes OpenAI-compatible routes plus control endpoints; RadixAttention
    # KV state can be cleared with POST /flush_cache on this server (same port).
    with modal.forward(PORT) as tunnel:
        print(f"tunnel.url        = {tunnel.url}")
        print(f"tunnel.tls_socket = {tunnel.tls_socket}")
        process = subprocess.Popen(cmd)
        stop_committing, committer = start_timing_log_committer(timing_log_dir)
        try:
            wait_ready(process)
            replicas[registry_key] = tunnel.url
            print(replicas[registry_key])
            process.wait()
        finally:
            stop_timing_log_committer(stop_committing, committer)
            if replicas.get(registry_key) == tunnel.url:
                replicas[registry_key] = ""
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def request_timing_log_args(registry_key: str) -> tuple[list[str], str | None]:
    """Extra ``sglang.launch_server`` args for the per-request timing log.

    Returns ``([], None)`` when ``REQUEST_TIMING_LOG`` is off so callers can
    splice unconditionally. Shared by ``model_endpoint`` here and
    ``experiment_runner/policy_matrix_app.py::_serve_model`` so the single-engine
    and fleet launch paths cannot drift.
    """
    if not REQUEST_TIMING_LOG:
        return [], None
    safe_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in registry_key)
    log_dir = os.path.join(REQUEST_TIMING_LOG_ROOT, safe_key)
    os.makedirs(log_dir, exist_ok=True)
    args = [
        "--log-requests",
        "--log-requests-level",
        "0",  # no prompt text / token ids in the log
        "--log-requests-format",
        "json",
        "--log-requests-target",
        log_dir,
    ]
    print(f"request timing log -> {log_dir}", flush=True)
    return args, log_dir


def start_timing_log_committer(log_dir: str | None) -> tuple[threading.Event, object]:
    """Start the periodic volume-commit thread for the timing log.

    Returns ``(stop_event, thread_or_None)``; pass both to
    :func:`stop_timing_log_committer` in a ``finally``.
    """
    stop = threading.Event()
    if log_dir is None:
        return stop, None
    thread = threading.Thread(
        target=_commit_timing_log_periodically, args=(stop,), daemon=True
    )
    thread.start()
    return stop, thread


def stop_timing_log_committer(stop: threading.Event, thread: object) -> None:
    """Stop the committer and flush the tail of the log.

    The controller tears fleets down, so a commit-on-exit-only policy would
    routinely lose the last interval of every run.
    """
    stop.set()
    if thread is not None:
        thread.join(timeout=30)
        _commit_timing_log()


def _commit_timing_log() -> None:
    """Publish whatever the request-timing handler has written so far.

    A Modal volume only makes writes visible to other containers on commit, and
    the analysis reads these files after the fleet is gone.
    """
    try:
        bench_results_volume.commit()
    except Exception as e:  # never let observability kill the engine
        print(f"[timing-log] commit failed: {e!r}", flush=True)


def _commit_timing_log_periodically(stop: threading.Event) -> None:
    while not stop.wait(REQUEST_TIMING_COMMIT_SECONDS):
        _commit_timing_log()


def _check_process(process: subprocess.Popen):
    if (rc := process.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=process.args)


def wait_ready(process: subprocess.Popen, timeout: int = WAIT_READY_TIMEOUT):
    deadline = time.time() + timeout

    while time.time() < deadline:
        _check_process(process)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/model_info"):
                break
        except urllib.error.URLError:
            time.sleep(2)
    else:
        raise TimeoutError(f"SGLang server not ready within {timeout} seconds")

    warmup_body = json.dumps(
        {
            "model": HF_REPO_ID,
            "messages": [{"role": "user", "content": "warmup"}],
            "max_tokens": 1,
        }
    ).encode()
    warmup_req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=warmup_body,
        headers={"Content-Type": "application/json"},
    )
    while time.time() < deadline:
        _check_process(process)
        try:
            with urllib.request.urlopen(warmup_req):
                return
        except (urllib.error.URLError, urllib.error.HTTPError):
            time.sleep(2)
    raise TimeoutError(f"SGLang server not ready within {timeout} seconds")


if __name__ == "__main__":
    model_endpoint.remote(REGION)
