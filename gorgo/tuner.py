"""Online tuning for the GORGO cost-model weights.

Extracted from ``proxy/modal_proxy.py`` so any router embedding the
GORGO policy (the GORGO proxy itself, or a downstream application) can
tune weights on live traffic without importing the proxy. Everything
here is stdlib-only and HTTP-free; the caller owns the sample buffer,
the hyperparameter store, and when to invoke the hooks.

Three layers, lowest first:

* :class:`GaussianESTuner` -- a generic (1+1)-Evolution Strategy with
  Rechenberg's 1/5 success rule, mutating in log-space within
  per-key ranges.
* Calibration -- online OLS for the physical-rate model
  ``ttft_ms ~ intercept_r + P*uncached + Q*queued`` via normal-equation
  sufficient statistics (:func:`new_calibration_state`,
  :func:`accumulate_calibration`, :func:`calibrated_rates_payload`;
  :class:`Calibration` is a thin stateful wrapper).
* :class:`OnlineTuner` -- the windowed auto-tune state machine: every
  ``hop_size`` samples, score the trailing ``window_size`` with an
  :data:`ONLINE_SCORE_FUNCTIONS` metric, report to the ES, and propose
  the next candidate weights.

The per-request sample shape consumed by :data:`ONLINE_SCORE_FUNCTIONS`
and :class:`OnlineTuner` is the one produced by the proxy's
``_record_request_sample`` (and by :func:`gorgo.measure.
measure_chat_completion`): a dict with at least ``ttft_seconds`` and
``total_seconds``.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable, Sequence

from gorgo.policy.gorgo import DEFAULT_GORGO_HYPERPARAMETERS, merge_update

DEFAULT_TUNE_WINDOW_SIZE = 100
DEFAULT_TUNE_HOP_SIZE = 50  # recompute every N new samples once warm
DEFAULT_TUNE_APPLY = True  # default to actually mutating hyperparameters

HYPERPARAM_RANGES: dict[str, tuple[float, float]] = {
    "prefill_weight": (1e-5, 5.0),
    "rtt_weight": (1e-5, 5.0),
}


def validated_ranges(
    overrides: dict[str, tuple[float, float]],
    *,
    merge_defaults: bool = True,
) -> dict[str, tuple[float, float]]:
    """Merge ``overrides`` with defaults and check each pair is ``0 < lo < hi``."""
    merged = {k: tuple(v) for k, v in HYPERPARAM_RANGES.items()} if merge_defaults else {}
    merged.update({k: tuple(v) for k, v in overrides.items()})
    for k, (lo, hi) in merged.items():
        if lo <= 0:
            raise ValueError(f"{k} lower bound must be > 0 (log-space sampling), got {lo}")
        if lo >= hi:
            raise ValueError(f"{k} range invalid: lo ({lo}) must be < hi ({hi})")
    return merged


def _percentile_of(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))]


# Online tuning metric functions: operate on a list of per-request
# samples and return a "higher is better" score.
ONLINE_SCORE_FUNCTIONS: dict[str, Callable[[list[dict]], float]] = {
    "neg_p50_ttft": lambda w: -_percentile_of([s["ttft_seconds"] for s in w], 0.50),
    "neg_p95_ttft": lambda w: -_percentile_of([s["ttft_seconds"] for s in w], 0.95),
    "neg_p99_ttft": lambda w: -_percentile_of([s["ttft_seconds"] for s in w], 0.99),
    "neg_avg_ttft": lambda w: -(sum(s["ttft_seconds"] for s in w) / max(1, len(w))),
    "neg_p95_e2e": lambda w: -_percentile_of([s["total_seconds"] for s in w], 0.95),
    "neg_p99_e2e": lambda w: -_percentile_of([s["total_seconds"] for s in w], 0.99),
}


SUPPORTED_AUTO_TUNE_MODES: frozenset[str] = frozenset({"online-es", "calibrate"})

# Policies whose hyperparameters the tuner writes; under any other
# active policy the writes would be inert, so enabling is rejected.
TUNABLE_POLICIES: frozenset[str] = frozenset({"gorgo", "gorgo-2d"})


class GaussianESTuner:
    """(1+1)-Evolution Strategy with Rechenberg's 1/5 success rule."""

    name = "gaussian-es-1plus1-1over5"

    def __init__(
        self,
        initial_params: dict[str, float],
        ranges: dict[str, tuple[float, float]],
        *,
        sigma: float = 0.5,
        sigma_min: float = 0.02,
        sigma_decay: float = 0.817,
        success_window: int = 8,
        target_rate: float = 0.2,
        tol: float = 0.005,
        max_steps: int = 16,
        seed: int | None = None,
    ) -> None:
        self.ranges = ranges
        self.keys = list(ranges.keys())
        self.best_params: dict[str, float] = {
            k: self._clamp(k, float(initial_params.get(k, sum(ranges[k]) / 2))) for k in ranges
        }
        self.best_score: float | None = None
        self.sigma = float(sigma)
        self.sigma_min = float(sigma_min)
        self.sigma_decay = float(sigma_decay)
        self.success_window = int(success_window)
        self.target_rate = float(target_rate)
        self.tol = float(tol)
        self.max_steps = int(max_steps)
        self.evaluated_after_baseline = 0
        self._recent: list[bool] = []
        self._rng = random.Random(seed)

    def _clamp(self, key: str, v: float) -> float:
        lo, hi = self.ranges[key]
        return max(lo, min(hi, v))

    def propose(self) -> dict[str, float] | None:
        if self.best_score is None:
            return dict(self.best_params)
        if self.evaluated_after_baseline >= self.max_steps:
            return None
        if self.sigma < self.sigma_min:
            return None
        cand: dict[str, float] = {}
        for key in self.keys:
            v = self.best_params[key]
            log_new = math.log(max(v, 1e-300)) + self.sigma * self._rng.gauss(0.0, 1.0)
            cand[key] = self._clamp(key, math.exp(log_new))
        return cand

    def report(self, candidate: dict[str, float], score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_params = dict(candidate)
            return True
        self.evaluated_after_baseline += 1
        accepted = score > self.best_score * (1.0 + self.tol)
        if accepted:
            self.best_score = score
            self.best_params = dict(candidate)
        self._recent.append(accepted)
        if len(self._recent) > self.success_window:
            self._recent.pop(0)
        if len(self._recent) >= self.success_window:
            rate = sum(self._recent) / len(self._recent)
            if rate > self.target_rate:
                self.sigma /= self.sigma_decay
            elif rate < self.target_rate:
                self.sigma *= self.sigma_decay
        return accepted

    @property
    def state(self) -> dict:
        recent_rate = sum(self._recent) / len(self._recent) if self._recent else None
        return {
            "sigma": self.sigma,
            "recent_success_rate": recent_rate,
            "recent_window_filled": len(self._recent),
        }


# ---------------------------------------------------------------------------
# Calibration: online OLS for ``ttft_ms ~ intercept_r + P*uncached + Q*queued``
# ---------------------------------------------------------------------------


def new_calibration_state() -> dict:
    """Fresh sufficient-statistics accumulator for the rate regression."""
    return {
        "n": 0,
        "sum_unc2": 0.0,
        "sum_q2": 0.0,
        "sum_uncq": 0.0,
        "sum_unc_ttft": 0.0,
        "sum_q_ttft": 0.0,
        # target -> {n, sum_ttft, sum_unc, sum_q}
        "per_target": {},
        "skipped": 0,
    }


def accumulate_calibration(
    cal: dict,
    *,
    target: str,
    uncached_at_dispatch: int,
    queued_at_dispatch: int,
    ttft_ms: float | None,
) -> None:
    """Fold one successful request into the online regression accumulator.

    Updates the normal-equation sufficient statistics for the model::

        ttft_ms ≈ intercept_r + P * uncached + Q * queued

    using only the caller-measured TTFT, the request's uncached prompt
    tokens, and the caller's ``queued_tokens_at_dispatch`` load counter.
    Regressing on ``uncached`` AND ``queued`` jointly, with a per-replica
    intercept absorbing RTT + fixed overhead, is what keeps queue/RTT from
    contaminating P (a raw ``ttft/uncached`` ratio explodes under load).
    Solved on demand in :func:`calibrated_rates_payload`.
    """
    if ttft_ms is None or ttft_ms <= 0.0:
        cal["skipped"] += 1
        return
    u = float(max(1, uncached_at_dispatch))
    q = float(max(0, queued_at_dispatch))
    y = float(ttft_ms)

    cal["n"] += 1
    cal["sum_unc2"] += u * u
    cal["sum_q2"] += q * q
    cal["sum_uncq"] += u * q
    cal["sum_unc_ttft"] += u * y
    cal["sum_q_ttft"] += q * y
    per = cal["per_target"].setdefault(
        target, {"n": 0, "sum_ttft": 0.0, "sum_unc": 0.0, "sum_q": 0.0}
    )
    per["n"] += 1
    per["sum_ttft"] += y
    per["sum_unc"] += u
    per["sum_q"] += q


def solve_spd(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Solve ``A x = b`` for a small symmetric system via Gauss-Jordan with
    partial pivoting. Dependency-free (avoids numpy in the hot path); the
    system is tiny (n_targets + 2). Returns None if singular."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        pivval = a[col][col]
        a[col] = [v / pivval for v in a[col]]
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            if factor != 0.0:
                a[r] = [v - factor * a[col][i] for i, v in enumerate(a[r])]
    return [a[i][n] for i in range(n)]


def calibrated_rates_payload(cal: dict) -> dict:
    """Solve the accumulated regression for the shared physical rates.

    Builds the normal equations for ``ttft ~ intercept_r + P*uncached +
    Q*queued`` from the online sufficient statistics and solves for
    ``[intercept_1..intercept_T, P, Q]``. ``prefill_rate``/``queue_rate``
    are the fleet-shared P/Q callers patch into the ``gorgo-2d`` weights.
    """
    targets = sorted(cal["per_target"].keys())
    n_t = len(targets)
    dim = n_t + 2
    result: dict = {
        "prefill_rate": None,
        "queue_rate": None,
        "diagnostics": {
            "model": "ols: ttft_ms ~ intercept_r + P*uncached + Q*queued",
            "samples": cal["n"],
            "skipped": cal["skipped"],
            "n_targets": n_t,
            "per_target_n": {t: cal["per_target"][t]["n"] for t in targets},
            "per_replica_intercept_ms": None,
            "warnings": [],
        },
    }
    # Need at least a few samples and >1 distinct (uncached, queued) pattern.
    if cal["n"] < dim + 2 or n_t == 0:
        result["diagnostics"]["warnings"].append("insufficient samples")
        return result

    # Assemble the symmetric normal matrix A and rhs b for coef ordering
    # [intercept_t0..t{T-1}, P, Q].
    p_i, q_i = n_t, n_t + 1
    a = [[0.0] * dim for _ in range(dim)]
    b = [0.0] * dim
    for idx, t in enumerate(targets):
        pt = cal["per_target"][t]
        a[idx][idx] = float(pt["n"])  # intercept diagonal
        a[idx][p_i] = a[p_i][idx] = pt["sum_unc"]
        a[idx][q_i] = a[q_i][idx] = pt["sum_q"]
        b[idx] = pt["sum_ttft"]
    a[p_i][p_i] = cal["sum_unc2"]
    a[q_i][q_i] = cal["sum_q2"]
    a[p_i][q_i] = a[q_i][p_i] = cal["sum_uncq"]
    b[p_i] = cal["sum_unc_ttft"]
    b[q_i] = cal["sum_q_ttft"]

    coef = solve_spd(a, b)
    if coef is None:
        result["diagnostics"]["warnings"].append("singular normal matrix")
        return result

    prefill_rate = coef[p_i]
    queue_rate = coef[q_i]
    result["prefill_rate"] = prefill_rate
    result["queue_rate"] = queue_rate
    result["diagnostics"]["per_replica_intercept_ms"] = {
        t: coef[i] for i, t in enumerate(targets)
    }
    # Negative coefficients are unphysical (collinearity / too little
    # independent variation in this window); surface rather than ship them.
    for name, val in (("prefill_rate", prefill_rate), ("queue_rate", queue_rate)):
        if val < 0.0:
            result["diagnostics"]["warnings"].append(f"{name} negative ({val:.4g})")
    return result


class Calibration:
    """Stateful wrapper over the calibration accumulator functions."""

    def __init__(self) -> None:
        self.state = new_calibration_state()

    def add(
        self,
        *,
        target: str,
        uncached_at_dispatch: int,
        queued_at_dispatch: int,
        ttft_ms: float | None,
    ) -> None:
        accumulate_calibration(
            self.state,
            target=target,
            uncached_at_dispatch=uncached_at_dispatch,
            queued_at_dispatch=queued_at_dispatch,
            ttft_ms=ttft_ms,
        )

    def rates(self) -> dict:
        return calibrated_rates_payload(self.state)

    def reset(self) -> None:
        self.state = new_calibration_state()


# ---------------------------------------------------------------------------
# OnlineTuner: the windowed auto-tune state machine
# ---------------------------------------------------------------------------


class OnlineTuner:
    """Windowed online auto-tune driver for the GORGO weights.

    The caller owns the sample buffer (a bounded deque of per-request
    sample dicts) and the hyperparameter store; this class owns the
    hop/window gating, the ES lifecycle, and the calibrate-mode routing.
    Call :meth:`on_sample` after every sample append; it returns the new
    hyperparameter store when the tuner decided to write weights, else
    ``None`` (the caller then assigns ``store = new_store or store``).

    ``on_event`` (optional) receives one dict per recompute -- the same
    shape the GORGO proxy writes to its tune trace -- for logging.
    """

    def __init__(
        self,
        *,
        objective_metric: str = "neg_p95_ttft",
        window_size: int = DEFAULT_TUNE_WINDOW_SIZE,
        hop_size: int = DEFAULT_TUNE_HOP_SIZE,
        apply: bool = DEFAULT_TUNE_APPLY,
        mode: str = "online-es",
        enabled: bool = False,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.window_size = int(window_size)
        self.hop_size = int(hop_size)
        self.apply = bool(apply)
        self.mode = mode
        self.objective_metric = objective_metric
        self.on_event = on_event
        self.samples_since_last_apply = 0
        self.applied_count = 0
        self.last_applied_at_monotonic: float | None = None
        self.last_recommendation: dict | None = None
        self.enabled_at_monotonic: float | None = None
        self.hyperparam_ranges: dict[str, tuple[float, float]] = HYPERPARAM_RANGES
        self.online_tuner: GaussianESTuner | None = None
        self.pending_candidate: dict[str, float] | None = None
        self.last_score: float | None = None
        self.total_samples_seen = 0
        self.calibration = Calibration()
        if self.mode not in SUPPORTED_AUTO_TUNE_MODES:
            raise ValueError(f"mode must be one of {sorted(SUPPORTED_AUTO_TUNE_MODES)}")
        if self.objective_metric not in ONLINE_SCORE_FUNCTIONS:
            raise ValueError(
                f"objective_metric must be one of {sorted(ONLINE_SCORE_FUNCTIONS)}"
            )

    # -- configuration ------------------------------------------------

    def configure(
        self,
        data: dict,
        *,
        active_policy: str,
        current_defaults: dict[str, float] | None = None,
    ) -> str | None:
        """Merge a ``POST /tune``-shaped body into the live config.

        Validates every requested key before mutating anything (a bad
        ``hop_size`` after a good ``window_size`` must not leave the
        tuner half-configured). Returns an error string, or ``None`` on
        success. ``active_policy`` gates enabling: the tuner only writes
        gorgo hyperparameters, so enabling under any other policy is an
        error (disabling always works).
        """
        if not isinstance(data, dict):
            return "body must be a JSON object"

        new_window = self.window_size
        new_hop = self.hop_size
        new_apply = self.apply
        new_mode = self.mode
        new_metric = self.objective_metric

        if "window_size" in data:
            try:
                new_window = int(data["window_size"])
            except (TypeError, ValueError):
                return "window_size must be an integer"
            if new_window <= 0:
                return "window_size must be positive"
        if "hop_size" in data:
            try:
                new_hop = int(data["hop_size"])
            except (TypeError, ValueError):
                return "hop_size must be an integer"
            if new_hop <= 0:
                return "hop_size must be > 0"
        if "apply" in data:
            new_apply = bool(data["apply"])
        if "mode" in data:
            mv = data["mode"]
            if not isinstance(mv, str) or mv not in SUPPORTED_AUTO_TUNE_MODES:
                return f"mode must be one of {sorted(SUPPORTED_AUTO_TUNE_MODES)}"
            new_mode = mv
        if "objective_metric" in data:
            mv = data["objective_metric"]
            if not isinstance(mv, str) or mv not in ONLINE_SCORE_FUNCTIONS:
                return f"objective_metric must be one of {sorted(ONLINE_SCORE_FUNCTIONS)}"
            new_metric = mv

        # Default to enabling so a bare ``{}`` body turns the tuner on.
        new_enabled = bool(data.get("enabled", True))
        if new_enabled and active_policy not in TUNABLE_POLICIES:
            return (
                "auto-tuning can only be enabled when the active policy is "
                f"one of {sorted(TUNABLE_POLICIES)}"
            )

        custom_ranges = data.get("hyperparam_ranges")
        active_ranges = self.hyperparam_ranges
        if new_enabled and new_mode == "online-es" and custom_ranges:
            try:
                active_ranges = validated_ranges(
                    {k: tuple(v) for k, v in custom_ranges.items()},
                    merge_defaults=False,
                )
            except (TypeError, ValueError) as exc:
                return f"invalid hyperparam_ranges: {exc}"

        was_enabled = self.enabled
        was_mode = self.mode
        self.window_size = new_window
        self.hop_size = new_hop
        self.apply = new_apply
        self.enabled = new_enabled
        self.mode = new_mode
        self.objective_metric = new_metric

        # Lifecycle for the online-ES tuner instance:
        #   - fresh enable into online-es      -> create tuner from current
        #     defaults, reset pending state
        #   - reconfiguration within online-es -> keep tuner and pending state
        if new_enabled and new_mode == "online-es":
            self.hyperparam_ranges = active_ranges
            seed_defaults = current_defaults or DEFAULT_GORGO_HYPERPARAMETERS
            seed = {k: float(seed_defaults.get(k, sum(v) / 2)) for k, v in active_ranges.items()}
            need_new_tuner = (
                self.online_tuner is None or was_mode != "online-es" or not was_enabled
            )
            if need_new_tuner:
                self.online_tuner = GaussianESTuner(
                    initial_params=seed,
                    ranges=active_ranges,
                    sigma=0.5,
                    sigma_min=0.05,
                    max_steps=10_000,
                )
                self.pending_candidate = None
                self.last_score = None

        if new_enabled and not was_enabled:
            # Fresh enable: zero the per-window counter so the first
            # recompute is measured from this moment.
            self.samples_since_last_apply = 0
            self.enabled_at_monotonic = time.monotonic()
        return None

    # -- per-sample hook ----------------------------------------------

    def on_sample(
        self,
        samples: Sequence[dict],
        hyperparameters: dict[str, Any],
        *,
        policy: str,
    ) -> dict[str, Any] | None:
        """Hop/window-gated recompute; call after each sample append.

        Returns the *new* hyperparameter store when weights were written
        (``apply`` on and the ES proposed or converged), else ``None``.
        """
        self.total_samples_seen += 1
        if not self.enabled:
            return None
        self.samples_since_last_apply += 1
        if self.samples_since_last_apply < self.hop_size:
            return None
        if len(samples) < self.window_size:
            # Don't fire until the buffer holds at least one full window;
            # earlier recomputes would over-weight whichever short prefix
            # of the run happens to have landed first.
            return None
        if policy not in TUNABLE_POLICIES:
            # Keep the counter pinned so a switch back to gorgo
            # immediately resumes recomputing on the next sample.
            self.samples_since_last_apply = self.hop_size
            return None

        window = list(samples)[-self.window_size :]

        if self.mode == "calibrate":
            # Calibration accumulates per-request (caller feeds
            # ``self.calibration``); the windowed recompute is a
            # deliberate no-op and never writes weights.
            return None

        if self.mode != "online-es":
            return None

        score_fn = ONLINE_SCORE_FUNCTIONS.get(self.objective_metric)
        tuner = self.online_tuner
        if score_fn is None or tuner is None:
            # Defensive: shouldn't happen because configure() validates,
            # but if state is corrupted don't crash the request loop.
            self.samples_since_last_apply = 0
            return None

        score = float(score_fn(window))
        pending = self.pending_candidate
        if pending is not None:
            accepted = tuner.report(pending, score)
        else:
            accepted = tuner.report(dict(tuner.best_params), score)
        self.last_score = score

        proposal = tuner.propose()
        new_store: dict[str, Any] | None = None
        if proposal is None:
            # Converged (or step/sigma budget exhausted): pin the best
            # params and go dormant.
            if self.apply:
                new_store = merge_update(
                    hyperparameters,
                    {"defaults": dict(tuner.best_params)},
                    replace=False,
                )
            self.enabled = False
            self.pending_candidate = None
        else:
            if self.apply:
                new_store = merge_update(
                    hyperparameters,
                    {"defaults": dict(proposal)},
                    replace=False,
                )
            self.pending_candidate = dict(proposal)
            self.applied_count += 1
            self.last_applied_at_monotonic = time.monotonic()
            self.last_recommendation = {"defaults": dict(proposal), "per_target": {}}
            self.samples_since_last_apply = 0

        if self.on_event is not None:
            try:
                self.on_event(
                    {
                        "kind": "tune",
                        "mode": "online-es",
                        "monotonic_s": time.monotonic(),
                        "step": self.applied_count,
                        "total_samples": self.total_samples_seen,
                        "window_size": len(window),
                        "converged": proposal is None,
                        "accepted": accepted,
                        "candidate": pending,
                        "score": score,
                        "best_score": tuner.best_score,
                        "best_params": dict(tuner.best_params),
                        "proposal": dict(proposal) if proposal is not None else None,
                        "sigma": tuner.sigma,
                        "objective_metric": self.objective_metric,
                    }
                )
            except Exception:
                pass
        return new_store

    # -- diagnostics ---------------------------------------------------

    def status(self, *, buffered_samples: int = 0) -> dict:
        """Snapshot of the live config + diagnostics (``GET /tune`` shape)."""
        tuner = self.online_tuner
        tuner_state = None
        if tuner is not None:
            tuner_state = {
                "name": tuner.name,
                "best_score": tuner.best_score,
                "best_params": tuner.best_params,
                "evaluated_after_baseline": tuner.evaluated_after_baseline,
                **tuner.state,
            }
        return {
            "enabled": self.enabled,
            "window_size": self.window_size,
            "hop_size": self.hop_size,
            "apply": self.apply,
            "mode": self.mode,
            "objective_metric": self.objective_metric,
            "online_tuner_state": tuner_state,
            "pending_candidate": self.pending_candidate,
            "last_score": self.last_score,
            "buffered_samples": buffered_samples,
            "samples_since_last_apply": self.samples_since_last_apply,
            "samples_until_next_apply": (
                max(0, self.hop_size - self.samples_since_last_apply)
                if self.enabled
                else None
            ),
            "applied_count": self.applied_count,
            "last_applied_at_monotonic": self.last_applied_at_monotonic,
            "last_recommendation": self.last_recommendation,
            "enabled_at_monotonic": self.enabled_at_monotonic,
        }
