"""
constraints.py
==============
Constraint-violation detection and metrics for PC-Gym rollouts.

Two jobs:
  1. DETECTION (used by evaluate.py at capture time): given recorded physical
     states and a scenario's `constraint_spec` (see scenarios.py), compute the
     per-step, per-constraint violation magnitude. We compute this ourselves from
     `env.state` rather than reading PC-Gym's `info["cons_info"]`, which (a) is not
     re-zeroed on reset (so magnitudes leak across episodes) and (b) records only
     the FIRST violated constraint per step (states short-circuit inputs). Direct
     computation is complete (every constraint, every step) and contamination-free.
  2. METRICS (post-hoc): given the stacked [n_seeds, T, n_con] violation array,
     report HOW MANY violations occur (count, rate, magnitude, robust per-seed
     dispersion) and WHEN they occur (per-step timeline, median first-violation
     step, transient-vs-steady split).

CLI — summarise the violations recorded in a rollout directory across methods:
  .venv/Scripts/python constraints.py outputs/rollouts/cstr/<timestamp>
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def violation_magnitudes(states, spec: list[dict]) -> np.ndarray:
    """
    Per-constraint violation magnitude (>= 0; 0 if satisfied).

    states : array [..., >= state_dim] of PHYSICAL states (env.state).
    spec   : list of constraint dicts with keys state_idx, bound, type.
    returns: array [..., n_con]; column k is max(0, distance past the bound), i.e.
             how far state[state_idx] exceeds a "<=" bound or falls below a ">=" bound.
    """
    states = np.asarray(states, dtype=float)
    out = np.zeros(states.shape[:-1] + (len(spec),), dtype=float)
    for k, c in enumerate(spec):
        v = states[..., c["state_idx"]]
        if c["type"] == "<=":
            out[..., k] = np.maximum(0.0, v - c["bound"])
        elif c["type"] == ">=":
            out[..., k] = np.maximum(0.0, c["bound"] - v)
        else:
            raise ValueError(
                f"constraint {c.get('name')!r}: bad type {c['type']!r} (use '<=' or '>=')"
            )
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _mad(x) -> float:
    """Median absolute deviation — robust dispersion (matches the dissertation's stats)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.median(np.abs(x - np.median(x))))


def _stats(viol_bool: np.ndarray, mag: np.ndarray) -> dict:
    """HOW-MANY + WHEN stats for a single [n_seeds, T] violation mask + magnitude."""
    n_seeds, T = viol_bool.shape
    per_seed = viol_bool.sum(axis=1).astype(int)              # [n_seeds] count per seed

    first = np.full(n_seeds, -1, dtype=int)                   # first violated step per seed
    for s in range(n_seeds):
        idx = np.flatnonzero(viol_bool[s])
        first[s] = int(idx[0]) if idx.size else -1
    first_valid = first[first >= 0]
    half = T // 2

    return {
        # HOW MANY
        "count":          int(viol_bool.sum()),
        "rate":           float(viol_bool.mean()),                # fraction of all steps
        "per_seed_count": per_seed.tolist(),
        "median_count":   float(np.median(per_seed)),
        "mad_count":      _mad(per_seed),
        "n_seeds_with_violation": int((per_seed > 0).sum()),
        "max_magnitude":  float(mag.max()) if mag.size else 0.0,
        "mean_magnitude": float(mag[viol_bool].mean()) if viol_bool.any() else 0.0,
        # WHEN
        "step_rate":         viol_bool.mean(axis=0).tolist(),     # [T] fraction of seeds per step
        "first_step_median": float(np.median(first_valid)) if first_valid.size else None,
        "frac_first_half":   float(viol_bool[:, :half].mean()) if half else 0.0,
        "frac_second_half":  float(viol_bool[:, half:].mean()) if T - half else 0.0,
    }


def constraint_metrics(violations, spec: list[dict]) -> dict:
    """
    Compute violation metrics from a [n_seeds, T, n_con] magnitude array.

    Returns a nested dict:
      n_seeds, n_steps, n_con
      per_constraint : {name: {label, bound, type, unit, <stats>}}
      overall        : {<stats>}   (a step counts as violated if ANY constraint is)
    where <stats> covers HOW MANY (count/rate/magnitude/median+MAD per seed) and
    WHEN (per-step rate timeline, median first-violation step, half split).
    """
    V = np.asarray(violations, dtype=float)
    if V.ndim != 3:
        raise ValueError(f"violations must be [n_seeds, T, n_con]; got shape {V.shape}")
    n_seeds, T, n_con = V.shape

    per_constraint = {}
    for k, c in enumerate(spec):
        mag = V[..., k]
        per_constraint[c["name"]] = {
            "label": c.get("label", c["name"]),
            "bound": c["bound"], "type": c["type"], "unit": c.get("unit", ""),
            **_stats(mag > 0, mag),
        }

    any_bool = (V > 0).any(axis=-1) if n_con else np.zeros((n_seeds, T), dtype=bool)
    any_mag  = V.max(axis=-1)       if n_con else np.zeros((n_seeds, T))

    return {
        "n_seeds": n_seeds, "n_steps": T, "n_con": n_con,
        "per_constraint": per_constraint,
        "overall": _stats(any_bool, any_mag),
    }


def format_summary(label: str, metrics: dict) -> str:
    """One-method human-readable violation summary (counts + timing)."""
    o = metrics["overall"]
    lines = [f"{label}"]
    lines.append(
        f"  overall : {o['rate']*100:5.1f}% of steps | {o['count']} total | "
        f"{o['n_seeds_with_violation']}/{metrics['n_seeds']} seeds | "
        f"median/seed {o['median_count']:.0f} (MAD {o['mad_count']:.0f})"
    )
    if o["first_step_median"] is not None:
        lines.append(
            f"            when : first ~step {o['first_step_median']:.0f}/{metrics['n_steps']} | "
            f"first-half {o['frac_first_half']*100:.1f}% vs second-half {o['frac_second_half']*100:.1f}%"
        )
    for name, m in metrics["per_constraint"].items():
        lines.append(
            f"  - {m['label']:<26} ({name} {m['type']} {m['bound']}{m['unit']}): "
            f"{m['rate']*100:5.1f}% of steps | {m['count']} | max over {m['max_magnitude']:.3g}{m['unit']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rollout-directory loader + CLI
# ---------------------------------------------------------------------------

def load_rollout_metrics(rollout_dir) -> tuple[list[dict], dict]:
    """
    Load a rollout directory (written by evaluate.run_rollouts) and compute
    per-method constraint metrics. Returns (constraint_spec, {label: metrics}).
    """
    rollout_dir = Path(rollout_dir)
    manifest = json.loads((rollout_dir / "manifest.json").read_text(encoding="utf-8"))
    spec = manifest.get("constraints", [])
    results: dict[str, dict] = {}
    for m in manifest["methods"]:
        data = np.load(rollout_dir / m["file"])
        if "violations" not in data.files:
            continue
        results[m["label"]] = constraint_metrics(data["violations"], spec)
    return spec, results


def main():
    ap = argparse.ArgumentParser(
        description="Summarise constraint violations recorded in a rollout directory."
    )
    ap.add_argument("rollout_dir", help="outputs/rollouts/<scenario>/<timestamp>")
    args = ap.parse_args()

    spec, results = load_rollout_metrics(args.rollout_dir)
    print(f"\n{'='*64}")
    print(f"  Constraint violations - {args.rollout_dir}")
    if spec:
        print("  Constraints: " + "; ".join(
            f"{c['name']} ({c.get('label', c['name'])} {c['type']} {c['bound']}{c.get('unit','')})"
            for c in spec))
    else:
        print("  (no constraints defined for this scenario)")
    print(f"{'='*64}")
    for label, metrics in results.items():
        print(format_summary(label, metrics))
        print()


if __name__ == "__main__":
    main()
