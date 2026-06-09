"""
analysis.py
===========
Phase-4 analysis utility (programmatic, no CLI). Loads the raw artifacts produced
by the pipeline and emits metric tables (CSV) + figures (PNG):

  inputs  : outputs/rollouts/<scenario>/<ts>/  (manifest.json + per-method .npz)
            outputs/models/<scenario>/<run_label>[/seed<k>]/  (run.json + training_log.npz)
  outputs : outputs/analysis/<scenario>/<ts>/  (CSV tables + PNG figures)

Design:
  - Loading, metric computation, and plotting are separate. Metric functions are
    pure (numpy in, numbers out); figure functions own their matplotlib lifecycle.
  - Identity/grouping comes from typed metadata (schema.MethodRecord / RunSpec via
    the manifest and run.json) — never from parsing a filename/slug.
  - TWO seed axes: each rollout method is one (condition, TRAINING-seed) model with
    N deterministic EVAL episodes. Within-model we reduce over eval episodes; the
    across-seed (robust) statistics are over the training seeds.

Entry points: `analyse_scenario(...)` (one scenario) and `aggregate_scenarios(...)`
(cross-env normalised aggregate via rliable).
"""

import enum
import json
import os
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from constraints import constraint_metrics
from schema import MethodRecord, MethodRole, RunSpec, Scenario


class Claim(enum.StrEnum):
    """Dissertation claims, used to tag output files."""
    C1_SAFETY = "c1_safety"
    C2_PERFORMANCE = "c2_performance"
    C3_TAKEOVER = "c3_takeover"


# ===========================================================================
# Typed containers + loaders
# ===========================================================================

_ROLLOUT_ARRAYS = ("states", "obs", "actions", "actions_agent", "actions_baseline",
                   "rewards", "takeover", "q_gap", "violations")


@dataclass(frozen=True)
class RolloutArrays:
    """Per-method rollout arrays, shape [N_episodes, T, ...]."""
    states: np.ndarray
    obs: np.ndarray
    actions: np.ndarray
    actions_agent: np.ndarray
    actions_baseline: np.ndarray
    rewards: np.ndarray
    takeover: np.ndarray
    q_gap: np.ndarray
    violations: np.ndarray

    @classmethod
    def from_npz(cls, npz) -> "RolloutArrays":
        return cls(**{name: npz[name] for name in _ROLLOUT_ARRAYS})


@dataclass(frozen=True)
class Method:
    """One rolled-out method: its structured record + arrays."""
    record: MethodRecord
    arrays: RolloutArrays


@dataclass(frozen=True)
class ScenarioRollouts:
    """A loaded rollout directory for one scenario."""
    scenario: Scenario
    n_steps: int
    dt: float
    plot_config: list[dict]
    setpoints: dict[str, list[float]]
    constraint_spec: list[dict]
    methods: list[Method]

    @classmethod
    def load(cls, rollout_dir: str | os.PathLike) -> "ScenarioRollouts":
        rd = Path(rollout_dir)
        manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
        methods: list[Method] = []
        for m in manifest["methods"]:
            record = MethodRecord.from_json(m)
            with np.load(rd / record.npz_file) as npz:
                methods.append(Method(record, RolloutArrays.from_npz(npz)))
        return cls(
            scenario=Scenario(manifest["scenario"]),
            n_steps=int(manifest["n_steps"]),
            dt=float(manifest["dt"]),
            plot_config=manifest["plot_config"],
            setpoints=manifest["setpoints"],
            constraint_spec=manifest.get("constraints", []),
            methods=methods,
        )

    def reference(self, role: MethodRole) -> Method | None:
        return next((m for m in self.methods if m.record.role is role), None)

    def models_by_condition(self) -> dict[str, list[Method]]:
        """Group MODEL methods by their condition label (one entry per training seed)."""
        out: dict[str, list[Method]] = {}
        for m in self.methods:
            if m.record.role is MethodRole.MODEL and m.record.run is not None:
                out.setdefault(m.record.run.condition_label, []).append(m)
        return out


_TRAIN_ARRAYS = ("beh_steps", "beh_return", "beh_viol_count", "beh_viol_rate",
                 "beh_viol_max", "beh_takeover", "eval_steps", "eval_return", "eval_takeover")


@dataclass(frozen=True)
class TrainingArrays:
    """One run's behaviour-time (per training episode) + eval (per boundary) logs."""
    beh_steps: np.ndarray
    beh_return: np.ndarray
    beh_viol_count: np.ndarray
    beh_viol_rate: np.ndarray
    beh_viol_max: np.ndarray
    beh_takeover: np.ndarray
    eval_steps: np.ndarray
    eval_return: np.ndarray
    eval_takeover: np.ndarray

    @classmethod
    def from_npz(cls, npz) -> "TrainingArrays":
        return cls(**{name: npz[name] for name in _TRAIN_ARRAYS})


@dataclass(frozen=True)
class TrainingRun:
    """One training run's behaviour-time + eval log, paired with its RunSpec."""
    run: RunSpec
    warmup_steps: int
    arrays: TrainingArrays

    @classmethod
    def load(cls, run_dir: str | os.PathLike) -> "TrainingRun | None":
        rd = Path(run_dir)
        run_json, log_npz = rd / "run.json", rd / "training_log.npz"
        if not (run_json.exists() and log_npz.exists()):
            return None
        run = RunSpec.from_json(json.loads(run_json.read_text(encoding="utf-8")))
        with np.load(log_npz, allow_pickle=True) as npz:
            arrays = TrainingArrays.from_npz(npz)
            warmup = int(json.loads(str(npz["meta"]))["warmup_steps"])
        return cls(run=run, warmup_steps=warmup, arrays=arrays)


def load_training_runs(models_dir: str | os.PathLike, scenario: Scenario) -> list[TrainingRun]:
    """All training runs for a scenario (single-run and per-seed layouts)."""
    root = Path(models_dir) / str(scenario)
    if not root.is_dir():
        return []
    runs = [TrainingRun.load(p.parent) for p in root.rglob("training_log.npz")]
    return [r for r in runs if r is not None]


def training_runs_by_condition(runs: list[TrainingRun]) -> dict[str, list[TrainingRun]]:
    out: dict[str, list[TrainingRun]] = {}
    for r in runs:
        out.setdefault(r.run.condition_label, []).append(r)
    return out


# ===========================================================================
# Pure metric helpers
# ===========================================================================

def median_mad(x: np.ndarray) -> tuple[float, float]:
    """Robust centre + dispersion (the dissertation's preferred stats)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(x))
    return med, float(np.median(np.abs(x - med)))


def _stack_to_min(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack 1-D arrays of possibly-different length, truncated to the shortest."""
    if not arrays:
        return np.empty((0, 0))
    m = min(len(a) for a in arrays)
    return np.stack([np.asarray(a)[:m] for a in arrays], axis=0)


def episodic_returns(arrays: RolloutArrays) -> np.ndarray:
    """Undiscounted return per eval episode -> [N]."""
    return arrays.rewards.sum(axis=1)


def condition_return_per_seed(methods: list[Method]) -> np.ndarray:
    """One robust return summary (median over eval episodes) per training seed."""
    return np.array([float(np.median(episodic_returns(m.arrays))) for m in methods])


def integral_errors(arrays: RolloutArrays, state_idx: int, sp_idx: int, dt: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Per-episode IAE and ISE for one controlled output -> ([N], [N])."""
    err = arrays.states[:, :, state_idx] - arrays.states[:, :, sp_idx]
    iae = np.abs(err).sum(axis=1) * dt
    ise = (err ** 2).sum(axis=1) * dt
    return iae, ise


def setpoint_segments(sp: list[float]) -> list[tuple[int, int, float]]:
    """Split a setpoint schedule into (start, end_exclusive, value) constant segments."""
    sp = np.asarray(sp, dtype=float)
    if sp.size == 0:
        return []
    change = np.flatnonzero(np.diff(sp) != 0) + 1
    bounds = [0, *change.tolist(), len(sp)]
    return [(bounds[i], bounds[i + 1], float(sp[bounds[i]])) for i in range(len(bounds) - 1)]


def steady_state_offset(arrays: RolloutArrays, state_idx: int, sp: list[float],
                        tail_frac: float = 0.25) -> np.ndarray:
    """Mean |x - sp| over the last `tail_frac` of each SP segment, averaged -> [N]."""
    x = arrays.states[:, :, state_idx]
    offsets = []
    for start, end, value in setpoint_segments(sp):
        tail = max(start + 1, int(end - tail_frac * (end - start)))
        offsets.append(np.abs(x[:, tail:end] - value).mean(axis=1))
    return np.mean(np.stack(offsets, axis=1), axis=1) if offsets else np.full(x.shape[0], np.nan)


def _settle_steps(outside_band: np.ndarray) -> np.ndarray:
    """Per row of a [N, L] 'outside-band' mask: # samples until it settles (= index
    after the LAST out-of-band sample); 0 if never out of band, L if never settles."""
    out = np.zeros(outside_band.shape[0], dtype=float)
    for i, row in enumerate(outside_band):
        idx = np.flatnonzero(row)
        out[i] = float(idx[-1] + 1) if idx.size else 0.0
    return out


def overshoot_settling(arrays: RolloutArrays, state_idx: int, sp: list[float], dt: float,
                       band_frac: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-episode overshoot (% of the setpoint step) and settling time (time units),
    averaged over the SP segments that have a non-zero step (so the FIRST segment,
    with no prior setpoint, is skipped). Returns ([N], [N]); NaN if no such segment.
    """
    x = arrays.states[:, :, state_idx]
    n = x.shape[0]
    over_segs, settle_segs = [], []
    prev: float | None = None
    for start, end, value in setpoint_segments(sp):
        if prev is not None and end - start >= 2:
            step = value - prev
            if abs(step) > 1e-9:
                seg = x[:, start:end]
                peak = (seg.max(axis=1) - value) if step > 0 else (value - seg.min(axis=1))
                over_segs.append(np.maximum(0.0, peak) / abs(step) * 100.0)
                outside = np.abs(seg - value) > band_frac * abs(step)
                settle_segs.append(_settle_steps(outside) * dt)
        prev = value
    if not over_segs:
        return np.full(n, np.nan), np.full(n, np.nan)
    return (np.mean(np.stack(over_segs, axis=1), axis=1),
            np.mean(np.stack(settle_segs, axis=1), axis=1))


def recovery_times(violations: np.ndarray) -> np.ndarray:
    """Durations (in steps) of contiguous constraint-violation episodes, pooled over
    episodes and constraints. Empty array if there are no violations."""
    if violations.size == 0 or violations.shape[-1] == 0:
        return np.array([])
    viol = (violations > 0).any(axis=-1)            # [N, T]
    runs: list[int] = []
    for row in viol:
        edges = np.diff(np.concatenate(([0], row.astype(int), [0])))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        runs.extend((ends - starts).tolist())
    return np.array(runs, dtype=float)


def steps_to_threshold(steps: np.ndarray, returns: np.ndarray, threshold: float) -> float:
    """First training step at which the eval return reaches `threshold`; NaN if never."""
    if not np.isfinite(threshold):
        return float("nan")
    idx = np.flatnonzero(np.asarray(returns, dtype=float) >= threshold)
    return float(np.asarray(steps)[idx[0]]) if idx.size else float("nan")


def learning_auc(steps: np.ndarray, returns: np.ndarray) -> float:
    """Mean eval return over training = area under the eval curve / step span."""
    steps = np.asarray(steps, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if steps.size < 2:
        return float("nan")
    return float(np.trapezoid(returns, steps) / (steps[-1] - steps[0]))


def setpoint_change_steps(setpoints: dict[str, list[float]]) -> list[int]:
    """Steps at which any controlled setpoint changes (union across SP variables)."""
    changes: set[int] = set()
    for sp in setpoints.values():
        arr = np.asarray(sp, dtype=float)
        changes.update((np.flatnonzero(np.diff(arr) != 0) + 1).tolist())
    return sorted(changes)


def takeover_by_phase(takeover: np.ndarray, change_steps: list[int], window: int
                      ) -> tuple[float, float]:
    """
    Mean agent-takeover fraction in TRANSIENT (within `window` steps of the episode
    start or a setpoint change) vs STEADY-STATE regions. NaN entries (non-RL) are
    ignored. Returns (transient_frac, steady_frac).
    """
    T = takeover.shape[1]
    transient = np.zeros(T, dtype=bool)
    for c in (0, *change_steps):                    # 0 = the start-up transient
        transient[c:min(c + window, T)] = True
    tr, st = takeover[:, transient], takeover[:, ~transient]
    return (float(np.nanmean(tr)) if tr.size and np.isfinite(tr).any() else float("nan"),
            float(np.nanmean(st)) if st.size and np.isfinite(st).any() else float("nan"))


def deployment_safety(method: Method, constraint_spec: list[dict]) -> dict:
    """Reuse constraints.constraint_metrics for one method's deployment rollouts."""
    if not constraint_spec:
        return {"rate": 0.0, "count": 0, "max_magnitude": 0.0}
    return constraint_metrics(method.arrays.violations, constraint_spec)["overall"]


def normalised_scores(cond_returns: np.ndarray, j_pid: float, j_oracle: float) -> np.ndarray:
    """Map returns onto [PID=0, oracle=1]; NaN if the span is degenerate."""
    span = j_oracle - j_pid
    if not np.isfinite(span) or abs(span) < 1e-9:
        return np.full_like(cond_returns, np.nan, dtype=float)
    return (cond_returns - j_pid) / span


# ===========================================================================
# rliable adapter (isolated; used only by the cross-scenario aggregate)
# ===========================================================================

def iqm_with_ci(score_dict: dict[str, np.ndarray], reps: int = 2000
                ) -> dict[str, tuple[float, float, float]]:
    """
    Interquartile mean + 95% stratified-bootstrap CI per algorithm.
    score_dict: {algorithm: scores[n_runs, n_tasks]}. Returns {algo: (iqm, lo, hi)}.
    """
    from rliable import library as rly
    from rliable import metrics as rly_metrics
    point, interval = rly.get_interval_estimates(
        score_dict, lambda s: np.array([rly_metrics.aggregate_iqm(s)]), reps=reps)
    return {a: (float(point[a][0]), float(interval[a][0, 0]), float(interval[a][1, 0]))
            for a in score_dict}


def prob_improvement(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """P(a > b) averaged over tasks (rliable). 1-D arrays = single task."""
    from rliable import metrics as rly_metrics
    a = np.asarray(scores_a, dtype=float).reshape(-1, 1)
    b = np.asarray(scores_b, dtype=float).reshape(-1, 1)
    return float(rly_metrics.probability_of_improvement(a, b).mean())


# ===========================================================================
# Metric report (per condition) + CSV
# ===========================================================================

@dataclass(frozen=True)
class ConditionReport:
    condition: str
    n_seeds: int
    return_median: float
    return_mad: float
    iae_median: float
    ise_median: float
    offset_median: float
    overshoot_median: float        # % of setpoint step
    settling_median: float         # time units
    deploy_viol_rate: float
    deploy_viol_count: int
    recovery_median: float         # steps; NaN if no violations
    norm_score_median: float       # NaN if no oracle
    opt_gap_median: float          # J(oracle) - J(method); NaN if no oracle
    steps_to_pid: float            # training steps for eval return to reach PID level
    auc_median: float              # mean eval return over training
    takeover_transient: float      # agent-takeover fraction in transients
    takeover_steady: float         # agent-takeover fraction in steady state

    @classmethod
    def csv_header(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def csv_row(self) -> list:
        def fmt(v):
            return v if isinstance(v, (str, int)) and not isinstance(v, bool) else f"{v:.4g}"
        return [fmt(getattr(self, f.name)) for f in fields(self)]


def _primary_output(plot_config: list[dict]) -> dict:
    """The first controlled output (with a setpoint index) — the headline tracking var."""
    return next(p for p in plot_config if "sp_idx" in p)


def _nanmedian(x: np.ndarray) -> float:
    """Median ignoring NaN; NaN if all-NaN (without the numpy all-NaN warning)."""
    x = np.asarray(x, dtype=float)
    return float(np.median(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")


def condition_reports(roll: ScenarioRollouts,
                      runs_by_cond: dict[str, list[TrainingRun]] | None = None
                      ) -> list[ConditionReport]:
    runs_by_cond = runs_by_cond or {}
    pid = roll.reference(MethodRole.PID)
    oracle = roll.reference(MethodRole.NMPC_ORACLE)
    j_pid = float(np.median(condition_return_per_seed([pid]))) if pid else float("nan")
    j_oracle = float(np.median(condition_return_per_seed([oracle]))) if oracle else float("nan")
    out = _primary_output(roll.plot_config)
    sp_values = roll.setpoints[out["label"]]
    change_steps = setpoint_change_steps(roll.setpoints)
    window = max(1, roll.n_steps // 10)

    reports: list[ConditionReport] = []
    for label, methods in roll.models_by_condition().items():
        rets = condition_return_per_seed(methods)
        ret_med, ret_mad = median_mad(rets)
        errs = [integral_errors(m.arrays, out["state_idx"], out["sp_idx"], roll.dt) for m in methods]
        iae = np.concatenate([e[0] for e in errs])
        ise = np.concatenate([e[1] for e in errs])
        offs = np.concatenate([steady_state_offset(m.arrays, out["state_idx"], sp_values)
                               for m in methods])
        ov_set = [overshoot_settling(m.arrays, out["state_idx"], sp_values, roll.dt) for m in methods]
        overshoot = np.concatenate([o[0] for o in ov_set])
        settling = np.concatenate([o[1] for o in ov_set])
        safety = [deployment_safety(m, roll.constraint_spec) for m in methods]
        recov = np.concatenate([recovery_times(m.arrays.violations) for m in methods]) \
            if roll.constraint_spec else np.array([])
        phase = [takeover_by_phase(m.arrays.takeover, change_steps, window) for m in methods]
        norm = normalised_scores(rets, j_pid, j_oracle)

        cond_runs = runs_by_cond.get(label, [])
        s2p = [steps_to_threshold(r.arrays.eval_steps, r.arrays.eval_return, j_pid) for r in cond_runs]
        auc = [learning_auc(r.arrays.eval_steps, r.arrays.eval_return) for r in cond_runs]

        reports.append(ConditionReport(
            condition=label, n_seeds=len(methods),
            return_median=ret_med, return_mad=ret_mad,
            iae_median=float(np.median(iae)), ise_median=float(np.median(ise)),
            offset_median=float(np.median(offs)),
            overshoot_median=_nanmedian(overshoot), settling_median=_nanmedian(settling),
            deploy_viol_rate=float(np.mean([s["rate"] for s in safety])),
            deploy_viol_count=int(np.sum([s["count"] for s in safety])),
            recovery_median=_nanmedian(recov) if recov.size else float("nan"),
            norm_score_median=_nanmedian(norm),
            opt_gap_median=(j_oracle - ret_med) if oracle is not None else float("nan"),
            steps_to_pid=_nanmedian(np.array(s2p)) if s2p else float("nan"),
            auc_median=_nanmedian(np.array(auc)) if auc else float("nan"),
            takeover_transient=_nanmedian(np.array([p[0] for p in phase])),
            takeover_steady=_nanmedian(np.array([p[1] for p in phase])),
        ))
    return reports


def write_reports_csv(reports: list[ConditionReport], path: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(ConditionReport.csv_header())
        for r in reports:
            w.writerow(r.csv_row())


# ===========================================================================
# Figures (each owns its fig lifecycle; honours the project's plot preferences)
# ===========================================================================

_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def fig_learning_curves(runs_by_cond: dict[str, list[TrainingRun]], path: str,
                        scenario: str) -> None:
    """Deterministic eval return vs steps: median + IQR band per condition (C2)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, runs) in enumerate(sorted(runs_by_cond.items())):
        steps = runs[0].arrays.eval_steps
        ev = _stack_to_min([r.arrays.eval_return for r in runs])
        if ev.size == 0:
            continue
        steps = steps[:ev.shape[1]]
        med = np.median(ev, axis=0)
        lo, hi = np.percentile(ev, [25, 75], axis=0)
        c = _COLORS[i % len(_COLORS)]
        ax.plot(steps, med, color=c, lw=2, label=f"{label} (median)")
        ax.fill_between(steps, lo, hi, color=c, alpha=0.2, edgecolor="none")
    ax.set_xlabel("training steps"); ax.set_ylabel("deterministic eval return")
    ax.set_title(f"Learning curves — {scenario}"); ax.grid(alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_safety_during_training(runs_by_cond: dict[str, list[TrainingRun]], path: str,
                               scenario: str) -> None:
    """Cumulative behaviour-time constraint violations vs steps (C1)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, runs) in enumerate(sorted(runs_by_cond.items())):
        steps = runs[0].arrays.beh_steps
        cum = _stack_to_min([np.cumsum(r.arrays.beh_viol_count) for r in runs])
        if cum.size == 0:
            continue
        steps = steps[:cum.shape[1]]
        med = np.median(cum, axis=0)
        lo, hi = np.percentile(cum, [25, 75], axis=0)
        c = _COLORS[i % len(_COLORS)]
        ax.plot(steps, med, color=c, lw=2, label=f"{label} (median)")
        ax.fill_between(steps, lo, hi, color=c, alpha=0.2, edgecolor="none")
    ax.set_xlabel("training steps"); ax.set_ylabel("cumulative behaviour violations")
    ax.set_title(f"Safety during training — {scenario}"); ax.grid(alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_optimality_gap(reports: list[ConditionReport], path: str, scenario: str) -> None:
    """Normalised score [PID=0, oracle=1] per condition (C2). Skipped if no oracle."""
    scored = [r for r in reports if np.isfinite(r.norm_score_median)]
    if not scored:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [r.condition for r in scored]
    ax.bar(labels, [r.norm_score_median for r in scored], color=_COLORS[:len(scored)])
    ax.axhline(0, color="black", lw=1, ls="--", label="PID")
    ax.axhline(1, color="tab:blue", lw=1, ls="--", label="NMPC oracle")
    ax.set_ylabel("normalised score  [PID=0, oracle=1]")
    ax.set_title(f"Optimality gap — {scenario}"); ax.grid(alpha=0.3, axis="y"); ax.legend(loc="best")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_takeover_over_training(runs_by_cond: dict[str, list[TrainingRun]], path: str,
                               scenario: str) -> None:
    """Deployment (greedy) takeover fraction vs steps, shadow conditions only (C3)."""
    shadow = {l: rs for l, rs in runs_by_cond.items() if rs and rs[0].run.shadow}
    if not shadow:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, runs) in enumerate(sorted(shadow.items())):
        steps = runs[0].arrays.eval_steps
        tk = _stack_to_min([r.arrays.eval_takeover for r in runs])
        if tk.size == 0:
            continue
        steps = steps[:tk.shape[1]]
        c = _COLORS[i % len(_COLORS)]
        ax.plot(steps, np.median(tk, axis=0), color=c, lw=2, label=label)
    ax.set_xlabel("training steps"); ax.set_ylabel("agent takeover (%)"); ax.set_ylim(-2, 102)
    ax.set_title(f"Earned takeover over training — {scenario}")
    ax.grid(alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_trajectory(roll: ScenarioRollouts, state_idx: int, label: str, unit: str,
                   sp_values: list[float] | None, path: str, scenario: str) -> None:
    """
    One state variable vs time: median + IQR band per method (models pooled across
    seeds; PID + oracle), with the setpoint step (if given) and any constraint lines
    on that state. Used for both the tracking output and the constrained variable.
    """
    t = np.arange(roll.n_steps) * roll.dt
    series: list[tuple[str, np.ndarray]] = [
        (cond, np.concatenate([m.arrays.states[:, :, state_idx] for m in methods], axis=0))
        for cond, methods in roll.models_by_condition().items()
    ]
    for role in (MethodRole.PID, MethodRole.NMPC_ORACLE):
        ref = roll.reference(role)
        if ref is not None:
            series.append((ref.record.label, ref.arrays.states[:, :, state_idx]))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, x) in enumerate(series):
        tt = t[:x.shape[1]]
        med = np.median(x, axis=0)
        lo, hi = np.percentile(x, [25, 75], axis=0)
        c = _COLORS[i % len(_COLORS)]
        ax.plot(tt, med, color=c, lw=2, label=name)
        ax.fill_between(tt, lo, hi, color=c, alpha=0.15, edgecolor="none")
    if sp_values is not None:
        sp = np.asarray(sp_values, dtype=float)
        ax.step(t[:len(sp)], sp, where="post", color="black", ls="--", lw=1.5, label="setpoint")
    for con in roll.constraint_spec:
        if con.get("state_idx") == state_idx:
            ax.axhline(con["bound"], color="red", ls=":", lw=1.3,
                       label=f"constraint ({con['type']} {con['bound']})")
    ax.set_xlabel("time"); ax.set_ylabel(f"{label} ({unit})" if unit else label)
    ax.set_title(f"{label} trajectories — {scenario}")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ===========================================================================
# Orchestration
# ===========================================================================

def analyse_scenario(rollout_dir: str | os.PathLike, models_dir: str = "outputs/models",
                     output_dir: str = "outputs/analysis") -> str:
    """Produce the CSV table + C1/C2/C3 figures for one scenario's rollout dir."""
    roll = ScenarioRollouts.load(rollout_dir)
    runs_by_cond = training_runs_by_condition(load_training_runs(models_dir, roll.scenario))

    out = Path(output_dir) / str(roll.scenario) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    sc = str(roll.scenario)
    reports = condition_reports(roll, runs_by_cond)
    write_reports_csv(reports, str(out / "metrics.csv"))
    fig_optimality_gap(reports, str(out / f"{Claim.C2_PERFORMANCE}_optimality_gap.png"), sc)

    # Trajectory overlays: the tracking output (with setpoint), and any constrained
    # variable on a different state (with its bound lines — the safety view).
    primary = _primary_output(roll.plot_config)
    fig_trajectory(roll, primary["state_idx"], primary["label"], primary.get("unit", ""),
                   roll.setpoints[primary["label"]],
                   str(out / f"{Claim.C2_PERFORMANCE}_trajectory_{primary['label']}.png"), sc)
    for con_state in dict.fromkeys(c["state_idx"] for c in roll.constraint_spec
                                   if c.get("state_idx") != primary["state_idx"]):
        name = next(c.get("label", c["name"]) for c in roll.constraint_spec if c["state_idx"] == con_state)
        unit = next(c.get("unit", "") for c in roll.constraint_spec if c["state_idx"] == con_state)
        fig_trajectory(roll, con_state, name, unit, None,
                       str(out / f"{Claim.C1_SAFETY}_trajectory_state{con_state}.png"), sc)

    if runs_by_cond:
        fig_learning_curves(runs_by_cond, str(out / f"{Claim.C2_PERFORMANCE}_learning_curves.png"), sc)
        fig_safety_during_training(runs_by_cond, str(out / f"{Claim.C1_SAFETY}_safety_during_training.png"), sc)
        fig_takeover_over_training(runs_by_cond, str(out / f"{Claim.C3_TAKEOVER}_takeover.png"), sc)

    print(f"  Analysis written to {out}  ({len(reports)} conditions)")
    return str(out)


def aggregate_scenarios(rollout_dirs: list[str | os.PathLike], output_dir: str = "outputs/analysis"
                        ) -> dict[str, tuple[float, float, float]]:
    """
    Cross-env aggregate: pool each condition's per-seed normalised scores across
    scenarios (tasks) and report rliable IQM + 95% CI. Returns {condition: (iqm, lo, hi)}.
    """
    per_cond: dict[str, list[np.ndarray]] = {}
    for rd in rollout_dirs:
        roll = ScenarioRollouts.load(rd)
        pid, oracle = roll.reference(MethodRole.PID), roll.reference(MethodRole.NMPC_ORACLE)
        if not (pid and oracle):
            continue
        j_pid = float(np.median(condition_return_per_seed([pid])))
        j_oracle = float(np.median(condition_return_per_seed([oracle])))
        for label, methods in roll.models_by_condition().items():
            per_cond.setdefault(label, []).append(
                normalised_scores(condition_return_per_seed(methods), j_pid, j_oracle))
    if not per_cond:
        return {}
    n_tasks = min(len(v) for v in per_cond.values())
    n_runs = min(min(len(a) for a in v) for v in per_cond.values())
    score_dict = {label: np.stack([a[:n_runs] for a in arrs[:n_tasks]], axis=1)
                  for label, arrs in per_cond.items()}
    return iqm_with_ci(score_dict)
