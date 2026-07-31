"""Read-only introspection of the SGLang engine image's per-request timing internals.

Companion to ``scripts/probe_sglang_metrics.py``. That script probes a *live*
replica over HTTP; this one inspects the *installed package* inside the engine
image, which is the only way to answer the question that blocks per-request
cost-model validation:

    ``/metrics`` exposes ``sglang:queue_time_seconds``,
    ``sglang:per_stage_req_latency_seconds{stage="prefill_forward"}`` and
    ``sglang:time_to_first_token_seconds`` as *aggregate histograms*, so the
    engine demonstrably computes each of those per request before folding them
    into a histogram. Where is that choke point, what are the per-request
    timestamp fields called, and can we thread a proxy-side request id through
    ``/v1/chat/completions`` so the per-request record can be joined back to a
    proxy trace row?

The answers determine how ``engine/modal_sglang.py`` is instrumented to emit
``t_engine_admission`` / ``t_prefill_complete`` per request (measurement only --
the routing policy keeps using HTTP-exposed signals).

Runs on a CPU container on the bare SGLang registry image: no GPU, no
``compile_deep_gemm`` layer, no model weights, nothing imported from sglang
(pure file inspection, so a heavyweight ``import torch`` is never triggered).

Usage:
    modal run scripts/introspect_sglang_engine.py::main
    modal run scripts/introspect_sglang_engine.py::main --out /tmp/sglang_introspection.txt

Keep ``SGLANG_IMAGE_REF`` in sync with ``engine/modal_sglang.py``; re-run this
after any image bump, since the tap point is version-sensitive.
"""

from __future__ import annotations

import modal

from app import app

# Must match engine/modal_sglang.py's ``from_registry`` ref. Declared without
# the entrypoint/env/compile layers so this stays a cheap CPU pull.
SGLANG_IMAGE_REF = "lmsysorg/sglang:nightly-dev-cu13-20260411-0011d2ae"

introspect_image = (
    modal.Image.from_registry(SGLANG_IMAGE_REF)
    .entrypoint([])
    .add_local_python_source("app", "scripts")
)

# Per-pattern match caps keep the report readable; a pattern that blows past its
# cap is reported as truncated rather than silently cut.
MAX_MATCHES_PER_PATTERN = 40
MAX_BLOCK_LINES = 90

# Substrings we grep for across the installed package. Ordered by what each one
# is meant to establish, because the report is read top-to-bottom.
GREP_PATTERNS: tuple[tuple[str, str], ...] = (
    # -- 1. the per-request timing record itself -------------------------------
    ("class TimeStats", "the per-request timing record's definition"),
    ("time_stats", "where the per-request timing record is attached / read"),
    ("wait_queue_entry_time", "queue-entry timestamp field name"),
    ("forward_entry_time", "scheduler-admission timestamp field name"),
    ("first_token_time", "first-token timestamp field name"),
    ("completion_time", "completion timestamp field name"),
    # -- 2. the metrics choke point (definitely runs on this build) ------------
    ("observe_queue_time", "queue_time histogram observation"),
    ("per_stage_req_latency", "per-stage latency histogram observation"),
    ("observe_time_to_first_token", "TTFT histogram observation"),
    ("def observe_", "all histogram observation entry points"),
    # -- 3. request-id threading (join key to the proxy trace) ----------------
    ("x-request-id", "header-based request id propagation"),
    ("x_request_id", "header-based request id propagation (snake_case)"),
    ('rid"', "explicit rid field in a request/response schema"),
    ("rid:", "rid as a typed field / annotation"),
    # -- 4. existing per-request logging we may be able to reuse --------------
    ("request_time_stats", "the --enable-request-time-stats-logging flag"),
    ("enable_trace", "OpenTelemetry-style tracing flag"),
    ("enable_metrics", "metrics flag plumbing"),
)

# Blocks dumped in full when their defining line is found. Keyed by the search
# string; the value is a label for the report.
FULL_BLOCK_TARGETS: tuple[tuple[str, str], ...] = (
    ("class TimeStats", "TimeStats (per-request timing record)"),
    ("def observe_queue_time", "queue_time observation"),
    ("def observe_per_stage_req_latency", "per-stage latency observation"),
    ("def observe_time_to_first_token", "TTFT observation"),
)

# Files whose full inventory of ``rid`` / id plumbing is worth listing, since
# the join key has to survive the OpenAI-compatible entrypoint.
ENTRYPOINT_FILE_HINTS: tuple[str, ...] = (
    "entrypoints/openai/protocol.py",
    "entrypoints/openai/serving_chat.py",
    "entrypoints/http_server.py",
    "managers/io_struct.py",
    "managers/tokenizer_manager.py",
)


@app.function(image=introspect_image, timeout=900)
def introspect() -> str:
    """Walk the installed sglang package and report everything needed to
    design the per-request timing tap. Returns the report as text."""
    import os
    import sys

    out: list[str] = []

    def emit(line: str = "") -> None:
        out.append(line)

    def header(title: str) -> None:
        emit()
        emit("=" * 78)
        emit(title)
        emit("=" * 78)

    # ---- locate the package without importing it ---------------------------
    # ``import sglang`` would drag in torch and the whole srt stack; we only
    # need to read source, so resolve the directory on disk instead. Modal runs
    # its own interpreter inside the image, so the engine's sglang install is
    # typically NOT on our sys.path -- hence the glob + bounded-walk fallbacks.
    import glob

    def looks_like_sglang(path: str) -> bool:
        # Distinguish the real package from a stray namespace dir / egg-link.
        return os.path.isdir(os.path.join(path, "srt"))

    searched: list[str] = []
    root: str | None = None

    for entry in sys.path:
        if not entry:
            continue
        candidate = os.path.join(entry, "sglang")
        searched.append(candidate)
        if looks_like_sglang(candidate):
            root = candidate
            break

    if root is None:
        for pattern in (
            "/usr/local/lib/python3*/site-packages/sglang",
            "/usr/local/lib/python3*/dist-packages/sglang",
            "/usr/lib/python3*/site-packages/sglang",
            "/usr/lib/python3*/dist-packages/sglang",
            "/opt/conda/lib/python3*/site-packages/sglang",
            "/opt/venv/lib/python3*/site-packages/sglang",
            "/root/.venv/lib/python3*/site-packages/sglang",
            "/sgl-workspace/*/python/sglang",
            "/sgl-workspace/sglang/python/sglang",
        ):
            searched.append(pattern)
            for candidate in sorted(glob.glob(pattern)):
                if looks_like_sglang(candidate):
                    root = candidate
                    break
            if root is not None:
                break

    if root is None:
        # Last resort: bounded walk of the roots an image would plausibly use.
        for base in ("/usr", "/opt", "/root", "/sgl-workspace", "/workspace"):
            if root is not None:
                break
            if not os.path.isdir(base):
                continue
            base_depth = base.rstrip("/").count("/")
            for dirpath, dirnames, _ in os.walk(base):
                if dirpath.count("/") - base_depth > 8:
                    dirnames[:] = []
                    continue
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                if "sglang" in dirnames and looks_like_sglang(os.path.join(dirpath, "sglang")):
                    root = os.path.join(dirpath, "sglang")
                    break

    if root is None:
        return (
            "FATAL: could not locate an 'sglang' package directory.\n"
            "Searched:\n  " + "\n  ".join(searched)
        )

    header("ENVIRONMENT")
    emit(f"image:          {SGLANG_IMAGE_REF}")
    emit(f"python:         {sys.version.split()[0]}")
    emit(f"sglang package: {root}")
    for version_file in ("version.py", "srt/version.py"):
        vpath = os.path.join(root, version_file)
        if os.path.exists(vpath):
            with open(vpath, "r", errors="replace") as f:
                emit(f"{version_file}: {f.read().strip()[:200]}")

    # ---- read every .py once ------------------------------------------------
    files: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", "test", "tests"}]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", errors="replace") as f:
                    files[path] = f.read().splitlines()
            except OSError:
                continue
    emit(f"python files scanned: {len(files)}")

    def rel(path: str) -> str:
        return os.path.relpath(path, root)

    def grep(needle: str) -> list[tuple[str, int, str]]:
        hits: list[tuple[str, int, str]] = []
        for path, lines in files.items():
            for i, line in enumerate(lines, start=1):
                if needle in line:
                    hits.append((path, i, line.strip()))
        return hits

    def extract_block(path: str, start_idx: int) -> list[str]:
        """Lines of the def/class block starting at 0-based ``start_idx``,
        ending at the first non-blank line dedented to or past its indent."""
        lines = files[path]
        first = lines[start_idx]
        base_indent = len(first) - len(first.lstrip())
        block = [first]
        for line in lines[start_idx + 1 :]:
            if len(block) >= MAX_BLOCK_LINES:
                block.append("        ... [truncated]")
                break
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent <= base_indent:
                    break
            block.append(line)
        return block

    # ---- 1. pattern inventory ----------------------------------------------
    header("PATTERN INVENTORY")
    emit("Each block: what the pattern establishes, then <file>:<line>: <source>")
    for needle, why in GREP_PATTERNS:
        hits = grep(needle)
        emit()
        emit(f"--- {needle!r}  ({why}) --- {len(hits)} match(es)")
        if not hits:
            emit("    (none -- not available on this build)")
            continue
        for path, lineno, text in hits[:MAX_MATCHES_PER_PATTERN]:
            emit(f"    {rel(path)}:{lineno}: {text[:180]}")
        if len(hits) > MAX_MATCHES_PER_PATTERN:
            emit(f"    ... {len(hits) - MAX_MATCHES_PER_PATTERN} more match(es) suppressed")

    # ---- 2. full source of the tap candidates ------------------------------
    header("FULL SOURCE OF TAP CANDIDATES")
    for needle, label in FULL_BLOCK_TARGETS:
        emit()
        emit(f"--- {label}  (searched for {needle!r}) ---")
        found = False
        for path, lines in files.items():
            for i, line in enumerate(lines):
                if needle in line and line.strip().startswith(("class ", "def ", "async def ")):
                    found = True
                    emit(f"  # {rel(path)}:{i + 1}")
                    for bl in extract_block(path, i):
                        emit(f"  {bl}")
                    emit()
        if not found:
            emit("  (definition not found on this build)")

    # ---- 3. request-id plumbing in the entrypoint files --------------------
    header("REQUEST-ID PLUMBING IN ENTRYPOINT FILES")
    emit("Determines whether a proxy-side request id can be threaded through")
    emit("/v1/chat/completions to serve as the join key.")
    for hint in ENTRYPOINT_FILE_HINTS:
        matching = [p for p in files if p.endswith(hint)]
        emit()
        emit(f"--- {hint} ---")
        if not matching:
            emit("    (file not present on this build)")
            continue
        for path in matching:
            for i, line in enumerate(files[path], start=1):
                low = line.lower()
                if ("rid" in low or "request_id" in low or "request-id" in low) and (
                    "=" in line or ":" in line
                ):
                    emit(f"    {rel(path)}:{i}: {line.strip()[:180]}")

    # ---- 4. launch flags ---------------------------------------------------
    header("LAUNCH FLAGS MENTIONING TIME / STATS / TRACE")
    emit("Sourced from the server-args definitions; a flag here may already emit")
    emit("per-request timings without any patching.")
    for path in sorted(p for p in files if p.endswith("server_args.py")):
        emit()
        emit(f"--- {rel(path)} ---")
        for i, line in enumerate(files[path], start=1):
            stripped = line.strip()
            if not stripped.startswith(('"--', "'--", "add_argument")):
                continue
            low = stripped.lower()
            if any(k in low for k in ("time", "stat", "trace", "metric", "log-request", "profil")):
                emit(f"    {rel(path)}:{i}: {stripped[:180]}")

    return "\n".join(out)


# Files (optionally ``:start-end`` line ranges, 1-based inclusive) dumped by
# ``dump``. These are the ones that decide how the timing tap is built: the
# per-request timing record, the completion-time logging site behind
# --enable-request-time-stats-logging, and the rid plumbing on the OpenAI path.
DEFAULT_DUMP_TARGETS = ",".join(
    (
        "srt/observability/req_time_stats.py",
        "srt/managers/scheduler_output_processor_mixin.py:1100-1200",
        "srt/entrypoints/openai/serving_base.py:100-200",
        "srt/managers/tokenizer_manager.py:1760-1830",
        "srt/managers/tokenizer_manager.py:2090-2130",
        "srt/server_args.py:4640-4680",
    )
)


@app.function(image=introspect_image, timeout=900)
def dump(targets: str) -> str:
    """Return the verbatim source of ``targets`` (comma-separated
    ``relpath[:start-end]``, relative to the sglang package root)."""
    import glob
    import os
    import sys

    def looks_like_sglang(path: str) -> bool:
        return os.path.isdir(os.path.join(path, "srt"))

    root: str | None = None
    candidates: list[str] = [os.path.join(e, "sglang") for e in sys.path if e]
    candidates += sorted(glob.glob("/sgl-workspace/*/python/sglang"))
    for candidate in candidates:
        if looks_like_sglang(candidate):
            root = candidate
            break
    if root is None:
        return "FATAL: could not locate an 'sglang' package directory"

    out: list[str] = [f"sglang package: {root}", ""]
    for target in targets.split(","):
        target = target.strip()
        if not target:
            continue
        rel, _, span = target.partition(":")
        path = os.path.join(root, rel)
        out.append("=" * 78)
        out.append(f"{target}")
        out.append("=" * 78)
        if not os.path.exists(path):
            out.append("  (file not present on this build)")
            continue
        with open(path, "r", errors="replace") as f:
            lines = f.read().splitlines()
        start, end = 1, len(lines)
        if span:
            lo, _, hi = span.partition("-")
            start = max(1, int(lo))
            end = min(len(lines), int(hi) if hi else len(lines))
        for i in range(start, end + 1):
            out.append(f"{i:5d}  {lines[i - 1]}")
        out.append("")
    return "\n".join(out)


@app.function(image=introspect_image, timeout=900)
def run_shell(command: str) -> str:
    """Run a read-only shell command inside the engine image and return its
    output. For image-level questions the package walk can't answer (does a
    ``sitecustomize`` already exist? what is PYTHONPATH? which interpreter?)."""
    import subprocess

    proc = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=600)
    return f"$ {command}\n--- exit={proc.returncode} ---\n{proc.stdout}\n" + (
        f"--- stderr ---\n{proc.stderr}\n" if proc.stderr.strip() else ""
    )


@app.local_entrypoint()
def shell(command: str) -> None:
    """Ad-hoc read-only shell in the engine image."""
    print(run_shell.remote(command))


@app.function(image=introspect_image, timeout=900)
def grep_engine(patterns: str, context: int = 0) -> str:
    """Grep the installed package for ``patterns`` (comma-separated), with
    ``context`` lines of surrounding source per hit. Ad-hoc follow-up tool for
    tracing a call chain the canned report didn't resolve."""
    import glob
    import os
    import sys

    def looks_like_sglang(path: str) -> bool:
        return os.path.isdir(os.path.join(path, "srt"))

    root: str | None = None
    candidates: list[str] = [os.path.join(e, "sglang") for e in sys.path if e]
    candidates += sorted(glob.glob("/sgl-workspace/*/python/sglang"))
    for candidate in candidates:
        if looks_like_sglang(candidate):
            root = candidate
            break
    if root is None:
        return "FATAL: could not locate an 'sglang' package directory"

    files: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "r", errors="replace") as f:
                        files[path] = f.read().splitlines()
                except OSError:
                    continue

    out: list[str] = []
    for pattern in patterns.split(","):
        pattern = pattern.strip()
        if not pattern:
            continue
        out.append("=" * 78)
        out.append(f"pattern: {pattern!r}")
        out.append("=" * 78)
        n = 0
        for path, lines in sorted(files.items()):
            for i, line in enumerate(lines):
                if pattern not in line:
                    continue
                n += 1
                rel = os.path.relpath(path, root)
                if context <= 0:
                    out.append(f"{rel}:{i + 1}: {line.strip()}")
                    continue
                out.append(f"--- {rel}:{i + 1}")
                lo = max(0, i - context)
                hi = min(len(lines), i + context + 1)
                for j in range(lo, hi):
                    marker = ">" if j == i else " "
                    out.append(f"  {marker} {j + 1:5d}  {lines[j]}")
                out.append("")
        out.append(f"({n} match(es))")
        out.append("")
    return "\n".join(out)


@app.local_entrypoint()
def grep_source(patterns: str, context: int = 0, out: str = "") -> None:
    """Ad-hoc grep of the engine source (comma-separated patterns)."""
    text = grep_engine.remote(patterns, context)
    if out:
        with open(out, "w") as f:
            f.write(text)
        print(f"[introspect] wrote {len(text)} chars to {out}")
    else:
        print(text)


@app.local_entrypoint()
def main(out: str = "") -> None:
    """Run the introspection remotely and print (optionally save) the report."""
    report = introspect.remote()
    print(report)
    if out:
        with open(out, "w") as f:
            f.write(report)
        print(f"\n[introspect] wrote {len(report)} chars to {out}")


@app.local_entrypoint()
def dump_source(targets: str = DEFAULT_DUMP_TARGETS, out: str = "") -> None:
    """Dump verbatim engine source for the timing-tap design."""
    text = dump.remote(targets)
    if out:
        with open(out, "w") as f:
            f.write(text)
        print(f"[introspect] wrote {len(text)} chars to {out}")
    else:
        print(text)
