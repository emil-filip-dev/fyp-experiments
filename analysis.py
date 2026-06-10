"""
analysis.py
===========
Phase-4 metrics + plotting utility. Loads a rollout directory written by
`deploy.run_rollouts` (one `.npz` per method×stage + `manifest.json`) and emits:

  - `metrics_summary.csv`  — one row per method, every metric family below.
  - figures (PNG): tracking trajectories, return bar (median+MAD), normalized
    optimality score, safety (violation rate), takeover, and return distribution.
  - optional learning curves from a run's `training_log.npz`.

It NEVER parses identity from a filename — method identity comes from the typed
`MethodRecord` in each `.npz`'s `meta` and the manifest (see schema.py). Model
runs are aggregated across seeds by (condition, deployment stage).

Robust statistics throughout (median + MAD / IQR, not mean ± std), per the
project's stated preference; `rliable` is used for IQM + bootstrap CIs of the
normalized score when installed.

Programmatic API (no CLI):
  analyse_rollout_dir(rollout_dir, out_dir=None) -> (metrics rows, out_dir)
  latest_rollout_dir(scenario)                   -> newest rollout dir
  plot_training_curve(run_dir, out_path=None)    -> learning-curve PNG
"""

import copy
import csv
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from constraints import constraint_metrics  # noqa: E402
from schema import DeploymentStage, ExpertKind, MethodRecord, MethodRole  # noqa: E402


# ---------------------------------------------------------------------------
# Robust stats helpers
# ---------------------------------------------------------------------------

def _median(x): return float(np.median(x)) if len(x) else float("nan")


def _mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def _iqr(x):
    x = np.asarray(x, float)
    if not x.size:
        return (float("nan"), float("nan"))
    return (float(np.percentile(x, 25)), float(np.percentile(x, 75)))


# ---------------------------------------------------------------------------
# Loading + grouping
# ---------------------------------------------------------------------------

_ARRAY_KEYS = ("states", "obs", "actions", "actions_agent", "actions_expert",
               "rewards", "takeover", "q_gap", "divergence", "violations")


@dataclass
class MethodData:
    """One method (references) or one (condition × stage) group (models),
    with its per-seed rollouts concatenated along the episode axis."""
    key: str
    role: MethodRole
    stage: DeploymentStage | None
    arrays: dict = field(default_factory=dict)         # [E, T, ...] pooled episodes
    per_seed_returns: list = field(default_factory=list)  # mean return per source .npz

    @property
    def returns(self) -> np.ndarray:
        return self.arrays["rewards"].sum(axis=1)        # [E]

    def state_traj(self, idx: int) -> np.ndarray:
        return self.arrays["states"][:, :, idx]          # [E, T]


def _group_key(rec: MethodRecord) -> str:
    if rec.role is MethodRole.MODEL and rec.run is not None:
        return f"{rec.run.condition_label} [{rec.stage.value}]"
    return rec.label


def load_rollout(rollout_dir: str) -> tuple[dict, dict[str, MethodData]]:
    """Load manifest + every method .npz, grouping model runs across seeds."""
    with open(os.path.join(rollout_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    buckets: dict[str, list] = defaultdict(list)
    for m in manifest["methods"]:
        rec = MethodRecord.from_json(m)
        z = np.load(os.path.join(rollout_dir, rec.npz_file))
        buckets[_group_key(rec)].append((rec, {k: z[k] for k in _ARRAY_KEYS if k in z}))

    methods: dict[str, MethodData] = {}
    for key, items in buckets.items():
        rec0 = items[0][0]
        arrays = {k: np.concatenate([a[k] for _, a in items], axis=0)
                  for k in items[0][1]}
        per_seed = [float(np.mean(a["rewards"].sum(axis=1))) for _, a in items]
        methods[key] = MethodData(key=key, role=rec0.role, stage=rec0.stage,
                                  arrays=arrays, per_seed_returns=per_seed)
    return manifest, methods


def latest_rollout_dir(scenario: str, root: str = "outputs/rollouts") -> str:
    dirs = sorted(glob.glob(os.path.join(root, scenario, "*")))
    if not dirs:
        raise FileNotFoundError(f"no rollouts under {os.path.join(root, scenario)}")
    return dirs[-1]


# ---------------------------------------------------------------------------
# Tracking metrics (IAE/ISE + per-segment overshoot/settling/offset)
# ---------------------------------------------------------------------------

def _segments(sp: np.ndarray):
    """Contiguous constant stretches of a setpoint schedule -> (start, end, value)."""
    sp = np.asarray(sp, float)
    segs, start = [], 0
    for t in range(1, len(sp)):
        if abs(sp[t] - sp[t - 1]) > 1e-9 * max(1.0, abs(sp[t - 1])):
            segs.append((start, t, sp[start]))
            start = t
    segs.append((start, len(sp), sp[start]))
    return segs


def tracking_metrics(mean_traj: np.ndarray, sp: np.ndarray, dt: float) -> dict:
    """IAE/ISE over the episode + averaged per-segment overshoot/settling/offset,
    computed on the across-episode MEDIAN trajectory (physical units) — consistent
    with the median trace drawn in plot_trajectories and robust to RL outliers."""
    L = min(len(mean_traj), len(sp))
    y, s = mean_traj[:L], np.asarray(sp[:L], float)
    e = y - s
    out = {"iae": float(np.sum(np.abs(e)) * dt), "ise": float(np.sum(e ** 2) * dt)}

    offsets, overs, settles = [], [], []
    rng = max(np.ptp(s), 1e-9)
    for (a, b, val) in _segments(s):
        if b - a < 3:
            continue
        seg = y[a:b]
        last = max(1, (b - a) // 5)                      # last 20%
        offsets.append(float(np.mean(np.abs(seg[-last:] - val))))
        prev = s[a - 1] if a > 0 else y[0]
        step = val - prev
        if abs(step) > 0.02 * rng:
            peak = (seg.max() - val) if step > 0 else (val - seg.min())
            overs.append(max(0.0, peak) / abs(step) * 100.0)
            band = 0.05 * abs(step)
            settled = np.nan
            for t in range(b - a):
                if np.all(np.abs(seg[t:] - val) <= band):
                    settled = t * dt
                    break
            settles.append(settled)
    out["offset"] = float(np.mean(offsets)) if offsets else float("nan")
    out["overshoot_pct"] = float(np.nanmean(overs)) if overs else float("nan")
    valid = [v for v in settles if not np.isnan(v)]
    out["settling_time"] = float(np.mean(valid)) if valid else float("nan")
    return out


def _control_effort(actions: np.ndarray) -> float:
    """Mean per-step |Δu| (L1), averaged over episodes — total control movement."""
    du = np.abs(np.diff(actions, axis=1)).sum(axis=-1)   # [E, T-1]
    return float(du.mean()) if du.size else 0.0


# ---------------------------------------------------------------------------
# Metric table
# ---------------------------------------------------------------------------

def compute_metrics(manifest: dict, methods: dict[str, MethodData]) -> list[dict]:
    dt = manifest["dt"]
    cspec = manifest.get("constraints", [])
    plot_cfg = manifest["plot_config"]
    setpoints = manifest.get("setpoints", {})

    # Reference anchors for the normalized optimality score [PID = 0, NMPC = 1].
    pid_med = next((m.returns for m in methods.values() if m.role is MethodRole.PID), None)
    nmpc_med = next((m.returns for m in methods.values() if m.role is MethodRole.NMPC), None)
    pid_med = _median(pid_med) if pid_med is not None else None
    nmpc_med = _median(nmpc_med) if nmpc_med is not None else None
    # Normalized score [PID=0, NMPC=1] is only meaningful when NMPC is the upper
    # anchor. If the oracle is missing or did not beat PID, skip it (rather than
    # emitting silently sign-inverted scores).
    if (pid_med is not None and nmpc_med is not None
            and (nmpc_med - pid_med) > 1e-6 * max(1.0, abs(pid_med))):
        span = nmpc_med - pid_med
    else:
        span = None
        if pid_med is not None and nmpc_med is not None:
            print(f"  [analysis] NMPC median ({nmpc_med:.1f}) not meaningfully above PID "
                  f"({pid_med:.1f}); normalized score omitted (degenerate/inverted anchor)")

    rows = []
    for key, m in methods.items():
        R = m.returns
        row = {
            "method": key, "role": m.role.value,
            "stage": m.stage.value if m.stage else "",
            "n_episodes": int(R.size), "n_seeds": len(m.per_seed_returns),
            "return_median": _median(R), "return_mad": _mad(R),
            "return_mean": float(np.mean(R)), "return_std": float(np.std(R)),
        }
        if span is not None:
            row["norm_score"] = (_median(R) - pid_med) / span
            row["norm_score_seeds"] = (np.mean(m.per_seed_returns) - pid_med) / span

        # tracking per controlled output
        for pc in plot_cfg:
            sp = np.asarray(setpoints.get(pc["label"], []), float)
            if sp.size == 0:
                continue
            tm = tracking_metrics(np.median(m.state_traj(pc["state_idx"]), axis=0), sp, dt)
            lab = pc["label"]
            row[f"IAE_{lab}"] = tm["iae"]
            row[f"ISE_{lab}"] = tm["ise"]
            row[f"offset_{lab}"] = tm["offset"]
            row[f"overshoot%_{lab}"] = tm["overshoot_pct"]
            row[f"settle_{lab}"] = tm["settling_time"]

        # control effort
        row["ctrl_effort_dU"] = _control_effort(m.arrays["actions"])

        # safety
        if cspec:
            cm = constraint_metrics(m.arrays["violations"], cspec)["overall"]
            row["viol_rate"] = cm["rate"]
            row["viol_count_median"] = cm["median_count"]
            row["viol_max"] = cm["max_magnitude"]
            row["viol_first_step"] = cm["first_step_median"]

        # takeover / divergence (models only)
        tk = m.arrays.get("takeover")
        if tk is not None and np.isfinite(tk).any():
            row["takeover_frac"] = float(np.nanmean(tk))
        dv = m.arrays.get("divergence")
        if dv is not None and m.role is MethodRole.MODEL:
            row["divergence_mean"] = float(np.mean(dv))
        rows.append(row)

    _maybe_rliable(rows, methods, pid_med, span)
    # stable order: references first, then models by stage
    order = {MethodRole.NMPC.value: 0, MethodRole.PID.value: 1, MethodRole.MODEL.value: 2}
    rows.sort(key=lambda r: (order.get(r["role"], 3), r.get("stage", ""), r["method"]))
    return rows


def _maybe_rliable(rows, methods, pid_med, span):
    """Add IQM + 95% bootstrap CI of the normalized score across seeds (models),
    if rliable is installed. Best-effort; silently skipped otherwise."""
    if span is None:
        return
    try:
        from rliable import library as rly
        from rliable import metrics
    except Exception:
        return
    by_key = {r["method"]: r for r in rows}
    for key, m in methods.items():
        if m.role is not MethodRole.MODEL or len(m.per_seed_returns) < 2:
            continue
        scores = (np.asarray(m.per_seed_returns) - pid_med) / span
        scores = scores.reshape(-1, 1)
        try:
            iqm, cis = rly.get_interval_estimates(
                {"m": scores}, lambda s: np.array([metrics.aggregate_iqm(s)]), reps=2000)
            by_key[key]["norm_iqm"] = float(iqm["m"][0])
            by_key[key]["norm_ci_lo"] = float(cis["m"][0][0])
            by_key[key]["norm_ci_hi"] = float(cis["m"][1][0])
        except Exception:
            pass


def write_csv(rows: list[dict], path: str) -> str:
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    return path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _palette(n):
    return plt.cm.tab10(np.linspace(0, 1, max(n, 1)))


def plot_trajectories(manifest, methods, out_path) -> str:
    """Controlled variable(s) vs setpoint over time: each method's median trajectory
    with IQR band, the setpoint schedule, and constraint bounds shaded."""
    dt = manifest["dt"]
    plot_cfg = manifest["plot_config"]
    setpoints = manifest.get("setpoints", {})
    cspec = manifest.get("constraints", [])
    keys = list(methods)
    colors = dict(zip(keys, _palette(len(keys))))

    fig, axes = plt.subplots(len(plot_cfg), 1, figsize=(10, 3.4 * len(plot_cfg)),
                             squeeze=False)
    for ax, pc in zip(axes[:, 0], plot_cfg):
        idx, lab = pc["state_idx"], pc["label"]
        T = next(iter(methods.values())).state_traj(idx).shape[1]
        t = np.arange(T) * dt
        for key in keys:
            tr = methods[key].state_traj(idx)
            med = np.median(tr, axis=0)
            lo, hi = np.percentile(tr, 25, axis=0), np.percentile(tr, 75, axis=0)
            ax.plot(t, med, color=colors[key], lw=1.8, label=key)
            ax.fill_between(t, lo, hi, color=colors[key], alpha=0.12)
        sp = np.asarray(setpoints.get(lab, []), float)
        if sp.size:
            ax.plot(np.arange(len(sp)) * dt, sp, "k--", lw=1.3, label="setpoint")
        for c in cspec:
            if c["state_idx"] == idx:
                ax.axhline(c["bound"], color="red", ls=":", lw=1.2)
                ax.text(t[-1], c["bound"], f" {c['name']}", color="red",
                        va="bottom", ha="right", fontsize=7)
        ax.set_ylabel(f"{lab} [{pc.get('unit','')}]")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time")
    axes[0, 0].legend(loc="best", fontsize=7, ncol=2)
    fig.suptitle(f"Deployment trajectories — {manifest['scenario']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_return_bar(rows, out_path) -> str:
    names = [r["method"] for r in rows]
    med = [r["return_median"] for r in rows]
    mad = [r.get("return_mad", 0) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(names) + 1.5))
    y = np.arange(len(names))
    ax.barh(y, med, xerr=mad, color=_palette(len(names)), alpha=0.85,
            error_kw={"elinewidth": 1, "capsize": 3})
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("median episodic return (± MAD)")
    ax.set_title("Deployment performance — return"); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def plot_normalized(rows, out_path) -> str:
    rows = [r for r in rows if "norm_score" in r]
    if not rows:
        return ""
    names = [r["method"] for r in rows]
    # Plot the IQM where rliable provided one (else the median score), and anchor
    # the CI whiskers on THAT value (clamped non-negative) — per row, so a single
    # CI-less row doesn't disable whiskers for all.
    vals, lo, hi = [], [], []
    for r in rows:
        v = r.get("norm_iqm", r["norm_score"])
        vals.append(v)
        if "norm_ci_lo" in r and "norm_ci_hi" in r:
            lo.append(max(0.0, v - r["norm_ci_lo"]))
            hi.append(max(0.0, r["norm_ci_hi"] - v))
        else:
            lo.append(0.0)
            hi.append(0.0)
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(names) + 1.5))
    y = np.arange(len(names))
    xerr = [lo, hi] if any(l or h for l, h in zip(lo, hi)) else None
    ax.barh(y, vals, xerr=xerr, color=_palette(len(names)), alpha=0.85,
            error_kw={"elinewidth": 1, "capsize": 3})
    ax.axvline(0, color="gray", lw=1); ax.axvline(1, color="green", lw=1, ls="--")
    ax.text(0, -0.7, "PID", color="gray", ha="center", fontsize=8)
    ax.text(1, -0.7, "NMPC", color="green", ha="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("normalized optimality score  [PID = 0, NMPC = 1]")
    ax.set_title("Optimality gap"); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def plot_safety(rows, out_path) -> str:
    rows = [r for r in rows if "viol_rate" in r]
    if not rows:
        return ""
    names = [r["method"] for r in rows]
    rate = [r["viol_rate"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(names) + 1.5))
    y = np.arange(len(names))
    ax.barh(y, rate, color=_palette(len(names)), alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("constraint violation rate (% of steps)")
    ax.set_title("Safety"); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def plot_takeover(manifest, methods, out_path) -> str:
    """Agent takeover fraction per model group (bar) + over-time timeline."""
    model_keys = [k for k, m in methods.items() if m.role is MethodRole.MODEL
                  and np.isfinite(m.arrays.get("takeover", np.array([np.nan]))).any()]
    if not model_keys:
        return ""
    dt = manifest["dt"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = dict(zip(model_keys, _palette(len(model_keys))))
    fracs = [float(np.nanmean(methods[k].arrays["takeover"])) * 100 for k in model_keys]
    y = np.arange(len(model_keys))
    ax1.barh(y, fracs, color=[colors[k] for k in model_keys], alpha=0.85)
    ax1.set_yticks(y); ax1.set_yticklabels(model_keys, fontsize=8); ax1.invert_yaxis()
    ax1.set_xlabel("agent takeover (% of steps)"); ax1.set_xlim(0, 100)
    ax1.set_title("Takeover by method"); ax1.grid(alpha=0.3, axis="x")
    for k in model_keys:
        tk = methods[k].arrays["takeover"]
        t = np.arange(tk.shape[1]) * dt
        ax2.plot(t, np.nanmean(tk, axis=0) * 100, color=colors[k], lw=1.6, label=k)
    ax2.set_xlabel("time"); ax2.set_ylabel("takeover (% of episodes)")
    ax2.set_ylim(-2, 102); ax2.set_title("Takeover over time"); ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    fig.suptitle(f"Earned takeover — {manifest['scenario']}")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def plot_return_box(rows, methods, out_path) -> str:
    names = [r["method"] for r in rows]
    data = [methods[n].returns for n in names]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(names) + 1.5))
    try:                                   # matplotlib >= 3.9 renamed labels -> tick_labels
        ax.boxplot(data, vert=False, tick_labels=names, showmeans=True)
    except TypeError:
        ax.boxplot(data, vert=False, labels=names, showmeans=True)
    ax.set_xlabel("episodic return"); ax.set_title("Return distribution")
    ax.grid(alpha=0.3, axis="x"); ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def analyse_rollout_dir(rollout_dir: str, out_dir: str | None = None) -> tuple[list[dict], str]:
    """Load a rollout dir, compute all metrics (CSV) and figures (PNG)."""
    manifest, methods = load_rollout(rollout_dir)
    out_dir = out_dir or os.path.join(rollout_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    rows = compute_metrics(manifest, methods)
    csv_path = write_csv(rows, os.path.join(out_dir, "metrics_summary.csv"))

    figs = {
        "trajectories.png": lambda p: plot_trajectories(manifest, methods, p),
        "return_bar.png": lambda p: plot_return_bar(rows, p),
        "normalized_score.png": lambda p: plot_normalized(rows, p),
        "safety.png": lambda p: plot_safety(rows, p),
        "takeover.png": lambda p: plot_takeover(manifest, methods, p),
        "return_box.png": lambda p: plot_return_box(rows, methods, p),
    }
    written = []
    for name, fn in figs.items():
        try:
            p = fn(os.path.join(out_dir, name))
            if p:
                written.append(os.path.basename(p))
        except Exception as e:
            print(f"  [plot skip] {name}: {e}")

    _print_table(manifest, rows)
    print(f"\n  metrics -> {csv_path}")
    print(f"  figures -> {out_dir}  ({', '.join(written)})")
    return rows, out_dir


def _print_table(manifest, rows):
    has_norm = any("norm_score" in r for r in rows)
    print(f"\n{'='*94}")
    print(f"  DEPLOYMENT METRICS — {manifest['scenario']}  (expert: {manifest.get('expert_kind','?')})")
    print(f"{'='*94}")
    hdr = f"  {'method':<28}{'med ret':>10}{'MAD':>8}"
    if has_norm:
        hdr += f"{'norm':>7}"
    hdr += f"{'viol%':>8}{'takeover%':>11}{'dU':>8}"
    print(hdr)
    print("-" * 94)
    for r in rows:
        line = f"  {r['method']:<28}{r['return_median']:>10.1f}{r.get('return_mad',0):>8.1f}"
        if has_norm:
            line += f"{r.get('norm_score', float('nan')):>7.2f}"
        tk = r.get("takeover_frac")
        tk_s = f"{tk*100:8.1f}" if tk is not None else "     n/a"
        line += f"{r.get('viol_rate',0)*100:>8.1f}{tk_s:>11}{r.get('ctrl_effort_dU',0):>8.2f}"
        print(line)
    print("=" * 94)


# ---------------------------------------------------------------------------
# Takeover map — Q-advantage heatmap over a 2D state-space slice, evolving over
# training. Each cell s is coloured by ΔQ(s) = Q(s, π_RL(s)) − Q(s, a_MPC(s)):
# orange where the RL agent would TAKE OVER (ΔQ > 0), blue where the MPC wins.
# This is exactly the switching/takeover decision boundary the ShadowController
# uses (q_gap > margin), visualised as a diverging field.
# ---------------------------------------------------------------------------

@dataclass
class StateSlice:
    """A 2D slice of state space for the takeover map: two state indices vary over
    physical ranges, the rest are held fixed (physical values)."""
    x_idx: int
    y_idx: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    fixed: dict[int, float]
    x_label: str
    y_label: str


def default_slice(scenario: str, sp_value: float) -> StateSlice:
    """The hand-tuned takeover-map state slice for a scenario. CSTR: Ca (x) ×
    reactor T (y), with the Ca setpoint held at sp_value.

    Only scenarios with a deliberately chosen slice are supported — a generic
    "first two obs dims" fallback produced semantically meaningless maps (the two
    axes might be uncontrolled states and the held setpoint wrong), so we raise
    instead. Add a branch here (axes + physical ranges + fixed setpoint) to support
    another scenario."""
    if scenario == "cstr":
        return StateSlice(x_idx=0, y_idx=1, x_range=(0.80, 0.95), y_range=(316.0, 332.0),
                          fixed={2: sp_value}, x_label="Ca [mol/L]", y_label="T [K]")
    raise NotImplementedError(
        f"No takeover-map state slice defined for scenario {scenario!r}; add one to "
        f"analysis.default_slice (the generic fallback was removed as it was meaningless).")


def _takeover_cmap():
    """Diverging colormap centred at ΔQ=0: dark blue (MPC ≫) → light blue → ~white
    (tie) → light orange → dark orange (RL ≫)."""
    cmap = LinearSegmentedColormap.from_list("takeover_blue_orange", [
        (0.00, "#08306b"), (0.25, "#6baed6"), (0.50, "#f7f7f7"),
        (0.75, "#fdae6b"), (1.00, "#7f2704"),
    ])
    cmap.set_bad("#d9d9d9")   # masked cells (MPC infeasible) -> grey
    return cmap


def _build_obs_grid(cfg: dict, sl: StateSlice, grid_res: int):
    """Return normalised observation grid [G*G, obs_dim] (row-major: row=y, col=x)
    plus the physical X, Y axis vectors."""
    o = cfg["env_params"]["o_space"]
    low, high = np.asarray(o["low"], float), np.asarray(o["high"], float)
    X = np.linspace(*sl.x_range, grid_res)
    Y = np.linspace(*sl.y_range, grid_res)
    xs, ys = np.meshgrid(X, Y)            # [G, G]
    phys = np.zeros((grid_res * grid_res, len(low)), float)
    for i in range(len(low)):
        if i == sl.x_idx:
            phys[:, i] = xs.ravel()
        elif i == sl.y_idx:
            phys[:, i] = ys.ravel()
        else:
            phys[:, i] = sl.fixed.get(i, (low[i] + high[i]) / 2.0)
    span = np.where(high > low, high - low, 1.0)   # avoid 0/0 on degenerate dims
    obs = 2.0 * (phys - low) / span - 1.0
    return np.clip(obs, -1.0, 1.0).astype(np.float32), X, Y


def _mpc_action_grid(scenario: str, cfg: dict, sl: StateSlice, obs_grid: np.ndarray,
                     sp_value: float, mpc_horizon: int) -> np.ndarray:
    """Expert action per cell (computed ONCE; the expert doesn't learn). For NMPC
    scenarios a viz controller with a CONSTANT setpoint = sp_value is built so the
    map reflects a single setpoint. Infeasible/failed solves -> NaN (masked)."""
    from experts import expert_kind_for
    from models import NMPCController
    kind = expert_kind_for(scenario)
    a_dim = cfg["action_dim"]
    if kind is ExpertKind.NMPC:
        cfg2 = copy.deepcopy(cfg)
        for k in cfg2["env_params"]["SP"]:
            n = len(cfg2["env_params"]["SP"][k])
            cfg2["env_params"]["SP"][k] = [sp_value] * n
        expert = NMPCController(cfg2, horizon=mpc_horizon)
    else:
        expert = cfg["baseline_cls"]()

    n = len(obs_grid)
    out = np.full((n, a_dim), np.nan, np.float32)
    for i, obs in enumerate(obs_grid):
        try:
            if hasattr(expert, "reset"):
                expert.reset()
            act, _ = expert.predict(obs)
            out[i] = np.asarray(act, np.float32)
        except Exception:
            pass
        if i % max(1, n // 10) == 0:
            print(f"    MPC action grid {i}/{n}")
    return out


def _cached_mpc_action_grid(scenario: str, cfg: dict, sl: StateSlice, obs_grid: np.ndarray,
                            sp_value: float, mpc_horizon: int, grid_res: int,
                            cache_dir: str = "outputs/cache/mpc_grids") -> np.ndarray:
    """The MPC action grid depends only on (scenario, slice, sp_value, horizon,
    grid_res) — NOT on the agent — so it's identical across snapshots and across
    conditions of the same scenario. Cache it so the 2,500+ IPOPT solves run once."""
    import hashlib
    sig = json.dumps({"x": sl.x_idx, "y": sl.y_idx, "xr": list(sl.x_range),
                      "yr": list(sl.y_range), "fx": {str(k): v for k, v in sl.fixed.items()}},
                     sort_keys=True)
    h = hashlib.md5(sig.encode()).hexdigest()[:8]
    path = os.path.join(cache_dir, f"{scenario}_g{grid_res}_sp{sp_value:g}_h{mpc_horizon}_{h}.npz")
    if os.path.exists(path):
        print(f"  [mpc-grid cache] hit -> {path}")
        return np.load(path)["a_mpc"]
    a_mpc = _mpc_action_grid(scenario, cfg, sl, obs_grid, sp_value, mpc_horizon)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, a_mpc=a_mpc)
    print(f"  [mpc-grid cache] saved -> {path}")
    return a_mpc


def _dq_grid(agent, obs_grid: np.ndarray, a_mpc: np.ndarray, grid_res: int) -> np.ndarray:
    """ΔQ(s) = Q(s, π(s)) − Q(s, a_MPC) over the grid (batched), reshaped [G, G].
    Cells where the MPC action is NaN (infeasible) are returned as NaN."""
    import torch
    dev = agent.device
    obs_t = torch.as_tensor(obs_grid, dtype=torch.float32, device=dev)
    mpc_t = torch.as_tensor(np.nan_to_num(a_mpc), dtype=torch.float32, device=dev)
    with torch.no_grad():
        a_ag = agent.actor(obs_t)
        q_ag = agent.q(obs_t, a_ag).squeeze(-1)
        q_mp = agent.q(obs_t, mpc_t).squeeze(-1)
    dq = (q_ag - q_mp).detach().cpu().numpy().astype(float)
    dq[np.isnan(a_mpc).any(axis=-1)] = np.nan
    return dq.reshape(grid_res, grid_res)


def _load_snapshots(run_dir: str) -> list[dict]:
    """Ordered snapshot list from snapshots/snapshots.json (offline before online,
    by step). Falls back to best.pt as a single 'final' snapshot."""
    idx_path = os.path.join(run_dir, "snapshots", "snapshots.json")
    if os.path.exists(idx_path):
        snaps = json.load(open(idx_path, encoding="utf-8"))
        for s in snaps:
            s["path"] = os.path.join(run_dir, "snapshots", s["file"])
        order = {"offline": 0, "online": 1}
        return sorted(snaps, key=lambda s: (order.get(s["phase"], 2), s["step"]))
    best = os.path.join(run_dir, "best.pt")
    if os.path.exists(best):
        return [{"file": "best.pt", "phase": "final", "step": 0, "path": best}]
    raise FileNotFoundError(f"no snapshots or best.pt under {run_dir}")


def _draw_constraints(ax, cfg, sl):
    for c in cfg.get("constraint_spec", []):
        if c["state_idx"] == sl.y_idx:
            ax.axhline(c["bound"], color="k", ls="--", lw=1.1)
        if c["state_idx"] == sl.x_idx:
            ax.axvline(c["bound"], color="k", ls="--", lw=1.1)


def plot_takeover_map(run_dir: str, *, grid_res: int = 60, cell_px: int = 12,
                      sp_value: float = 0.90, mpc_horizon: int = 20,
                      out_dir: str | None = None, ncols: int = 4) -> str:
    """
    Render the RL–MPC takeover map (ΔQ heatmap over a state-space slice) for every
    training snapshot of a run, showing how the takeover region evolves. Writes one
    PNG per snapshot AND a combined small-multiples grid, to <run_dir>/takeover_maps/.

    grid_res  — cells per axis (detail). cell_px — rendered pixels per cell (square
    size). Works for any run whose scenario has a tuned slice (see default_slice;
    CSTR is provided) across all training modes: offline (x = gradient step), o2o /
    online (x = env step) — including online-only runs.
    """
    from scenarios import SCENARIOS
    from models import get_agent
    import torch

    run = json.load(open(os.path.join(run_dir, "run.json"), encoding="utf-8"))
    scenario = run["scenario"]
    cfg = SCENARIOS[scenario]
    sl = default_slice(scenario, sp_value)
    snaps = _load_snapshots(run_dir)
    out_dir = out_dir or os.path.join(run_dir, "takeover_maps")
    os.makedirs(out_dir, exist_ok=True)

    obs_grid, X, Y = _build_obs_grid(cfg, sl, grid_res)
    a_mpc = _cached_mpc_action_grid(scenario, cfg, sl, obs_grid, sp_value, mpc_horizon, grid_res)
    feasible = float(np.mean(~np.isnan(a_mpc).any(axis=-1))) * 100
    print(f"  MPC feasible on {feasible:.0f}% of cells")

    results = []
    for s in snaps:
        ckpt = torch.load(s["path"], weights_only=False, map_location="cpu")
        agent = get_agent(ckpt["type"]).load(ckpt, device=torch.device("cpu"))
        results.append((s, _dq_grid(agent, obs_grid, a_mpc, grid_res)))

    allv = np.concatenate([dq.ravel() for _, dq in results])
    allv = allv[np.isfinite(allv)]
    vmax = float(np.percentile(np.abs(allv), 98)) if allv.size else 1.0
    vmax = vmax or 1.0
    norm, cmap = Normalize(-vmax, vmax), _takeover_cmap()
    extent = [sl.x_range[0], sl.x_range[1], sl.y_range[0], sl.y_range[1]]
    cbar_label = "ΔQ = Q(s, π_RL) − Q(s, a_MPC)\n(orange: RL takes over · blue: MPC drives)"

    # individual PNGs
    side = grid_res * cell_px / 120.0
    items = []
    for s, dq in results:
        label = f"{s['phase']} · step {s['step']:,}"
        fig, ax = plt.subplots(figsize=(side + 2.6, side + 1.4))
        im = ax.imshow(dq, origin="lower", extent=extent, aspect="auto",
                       cmap=cmap, norm=norm, interpolation="nearest")
        _draw_constraints(ax, cfg, sl)
        ax.set_xlabel(sl.x_label); ax.set_ylabel(sl.y_label)
        ax.set_title(f"Takeover map — {scenario}\n{label}", fontsize=10)
        fig.colorbar(im, ax=ax, label=cbar_label)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"takeover_{s['phase']}_{s['step']:08d}.png"), dpi=120)
        plt.close(fig)
        items.append((label, dq))

    # combined small-multiples
    n = len(items)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), squeeze=False)
    im = None
    for k, (label, dq) in enumerate(items):
        ax = axes[k // ncols][k % ncols]
        im = ax.imshow(dq, origin="lower", extent=extent, aspect="auto",
                       cmap=cmap, norm=norm, interpolation="nearest")
        _draw_constraints(ax, cfg, sl)
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label=cbar_label)
    fig.suptitle(f"RL–MPC takeover region over training — {scenario}", fontsize=12)
    fig.savefig(os.path.join(out_dir, "takeover_grid.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  takeover maps -> {out_dir}  ({n} snapshots + takeover_grid.png)")
    return out_dir


def plot_training_curve(run_dir: str, out_path: str | None = None) -> str:
    """Learning curve(s) from a run's training_log.npz — mode-specific (offline:
    eval return + violation vs grad step; o2o/online: behaviour return + violation
    vs env step). Training-return and safety on separate axes, zoomed to data."""
    z = np.load(os.path.join(run_dir, "training_log.npz"), allow_pickle=False)
    mode = str(z["mode"])
    out_path = out_path or os.path.join(run_dir, "training_curve.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if mode == "offline":
        ax1.plot(z["eval_step"], z["eval_return"], "b-o", ms=3)
        ax1.set_xlabel("gradient step"); ax1.set_ylabel("autonomous eval return")
        ax2.plot(z["eval_step"], np.asarray(z["eval_viol_rate"]) * 100, "r-o", ms=3)
        ax2.set_ylabel("eval violation rate (%)")
    else:
        xs = z["beh_env_step"]
        ax1.plot(xs, z["beh_return"], color="tab:blue", alpha=0.5, lw=1)
        if "eval_return" in z and len(z["eval_return"]):
            ax1.plot(z["eval_env_step"], z["eval_return"], "b-o", ms=3, label="eval")
            ax1.legend(fontsize=8)
        ax1.set_xlabel("env step"); ax1.set_ylabel("return")
        ax2.plot(xs, np.asarray(z["beh_viol_rate"]) * 100, color="tab:red", lw=1.2)
        ax2.set_ylabel("behaviour violation rate (%)")
    ax1.grid(alpha=0.3); ax2.grid(alpha=0.3); ax2.set_xlabel(ax1.get_xlabel())
    fig.suptitle(f"Training curve ({mode}) — {os.path.basename(run_dir)}")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path
