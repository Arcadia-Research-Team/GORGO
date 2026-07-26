"""Restore SGLang's per-request scheduler timings across process hops.

Auto-imported in every interpreter in the engine container via a ``.pth`` file
(see ``engine/modal_sglang.py``), and **inert unless ``REQUEST_TIMING_LOG`` is
set**, so ordinary deployments behave exactly as before.

Why this exists
---------------
SGLang computes per-request scheduler timings and intends to ship them to the
client in ``meta_info`` (``tokenizer_manager.py`` merges
``SchedulerReqTimeStats.convert_to_output_meta_info()`` whenever
``--enable-metrics`` is on). Those timings are what a TTFT decomposition needs:

    queue_time            = forward_entry_time - wait_queue_entry_time
    prefill_finished_time - forward_entry_time = prefill duration

They never arrive. ``ReqTimeStatsBase.__getstate__`` hardcodes
``"enable_metrics": False`` when serializing (reasonably: a metrics-collector
handle must not cross a process boundary), while
``SchedulerReqTimeStats.__getstate__`` opens with

    if not self.enable_metrics:
        return {}

Text generation crosses **two** hops -- scheduler -> detokenizer -> tokenizer
manager. Hop 1 carries the timestamps but clears the flag; on hop 2 the guard
sees ``enable_metrics == False`` and serializes ``{}``. Because these are
dataclass fields with plain defaults, the receiver's attribute lookups fall back
to the class attributes, so every timestamp silently reads ``0.0``: the client
sees ``queue_time: 0.0`` and no ``forward_entry_time`` /
``prefill_finished_time`` keys at all (both are emitted only when ``> 0.0``).
Measured on this image: 8/8 concurrent requests reported ``queue_time == 0.0``
with those two keys absent, which reads as "unsupported" but is really "zeroed
in transit".

The patch drops *only* that early-return, so the subclass always serializes its
own timestamps. It does not change ``enable_metrics`` (so every
``observe_*`` / metrics path behaves identically), and it does not touch
scheduling, batching, admission, or routing -- it changes what the engine
*reports*, not what it does. Routing still consumes only ``/metrics`` plus the
proxy's own RTT probe.

``__setstate__`` already remaps any key ending in ``time`` between the sender's
and receiver's ``perf_counter`` domains, so the restored timestamps are
translated per hop. Note it does this unconditionally, so a field that was never
set (0.0) arrives as a small nonzero clock-offset artifact rather than 0.0 --
callers must treat only plausibly-large values as real (the analysis requires
``forward_entry_time > 0`` and a positive, bounded duration).
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys

_TARGET = "sglang.srt.observability.req_time_stats"
_ENV_FLAG = "REQUEST_TIMING_LOG"
# Fields SchedulerReqTimeStats.__getstate__ ships when it isn't short-circuited.
# Mirrors the upstream list so the patch stays a pure guard-removal.
_TIMESTAMP_FIELDS = (
    "wait_queue_entry_time",
    "forward_entry_time",
    "prefill_run_batch_start_time",
    "prefill_run_batch_end_time",
    "prefill_finished_time",
)


def _enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "") not in ("", "0", "false", "False")


def _patch_module(module) -> None:
    """Replace ``SchedulerReqTimeStats.__getstate__`` with the guard removed."""
    stats_cls = getattr(module, "SchedulerReqTimeStats", None)
    base_cls = getattr(module, "ReqTimeStatsBase", None)
    if stats_cls is None or base_cls is None:
        print(
            "[gorgo-timing-patch] unexpected module shape; leaving SGLang untouched",
            file=sys.stderr,
        )
        return
    if getattr(stats_cls, "_gorgo_timing_patched", False):
        return

    def __getstate__(self):  # noqa: N807 - dunder on purpose
        state = {name: getattr(self, name, 0.0) for name in _TIMESTAMP_FIELDS}
        state["diff_realtime_monotonic"] = module.global_diff_realtime_monotonic
        # Keep the base's contract intact -- notably ``enable_metrics: False``,
        # so a metrics-collector handle still never crosses a process boundary.
        state.update(base_cls.__getstate__(self))
        return state

    stats_cls.__getstate__ = __getstate__
    stats_cls._gorgo_timing_patched = True
    print(
        "[gorgo-timing-patch] SchedulerReqTimeStats.__getstate__ patched "
        "(per-request timings will survive scheduler -> detokenizer -> tokenizer)",
        file=sys.stderr,
    )


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Resolve the target module normally, then patch it once it has executed.

    Deliberately lazy: importing sglang (and torch) from interpreter startup
    would be both slow and circular. Resolution uses the ``path`` the import
    system hands us -- i.e. the parent package's ``__path__`` -- so the parent
    is never re-imported and there is no recursion back into this finder.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        except Exception:
            return None
        if spec is None or spec.loader is None:
            return None
        original_exec_module = spec.loader.exec_module

        def exec_module(module):
            original_exec_module(module)
            try:
                _patch_module(module)
            except Exception as e:  # never break the engine over observability
                print(f"[gorgo-timing-patch] patch failed: {e!r}", file=sys.stderr)

        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    if not _enabled():
        return
    if any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
        return
    # Already imported (we lost the race)? Patch in place instead.
    existing = sys.modules.get(_TARGET)
    if existing is not None:
        _patch_module(existing)
        return
    sys.meta_path.insert(0, _PatchOnImport())


try:
    install()
except Exception as e:  # a broken hook must not stop the container booting
    print(f"[gorgo-timing-patch] install failed: {e!r}", file=sys.stderr)
