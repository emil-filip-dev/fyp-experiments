"""
experiments.py
==============
Declarative experiment grids for the dissertation's empirical study (see
`dissertation_plan.md`). This module is *configuration*, not orchestration: it
describes WHAT to run (environments × conditions × seeds + the step budget and
rollout settings) as version-controllable data. The to-be-built
`run_experiments.py` (Phase 3) consumes a grid and actually launches `train()` +
`run_rollouts()`; the analysis utility (Phase 4) reads back the resulting
checkpoints/rollouts using the same path helpers defined here.

Design notes
------------
- A `Condition` is a single *learned* training configuration (one training job
  per seed). Its `train_kwargs` are forwarded verbatim to `train.train()`, so the
  training interface stays the single source of truth — we never re-declare
  hyperparameters here.
- The reference controllers (PID, NMPC oracle) are NOT conditions: they require
  no training and `evaluate.run_rollouts()` already injects them into every
  rollout run. Listing them here would only invite special-casing.
- Output-directory naming is delegated to `train.run_label_for()` so the producer
  (train) and the consumers (orchestrator / analysis) cannot drift apart. The
  per-seed leaf (`.../<slug>/seed<k>/`) matches `train_model(per_seed_dir=True)`.

Reproducibility comes from `write_provenance()`, which snapshots the resolved
grid + git commit + library versions next to a run's outputs, so a config edited
later cannot silently re-interpret old results.

CLI
---
  .venv/Scripts/python experiments.py                       # print the default grid
  .venv/Scripts/python experiments.py --grid phase1_cstr_fourtank
  .venv/Scripts/python experiments.py --provenance outputs/experiments/phase1   # write provenance.json
"""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterator, Mapping

from train import run_label_for
from util import configure_utf8_output


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# eq=False keeps Condition hashable (identity) despite the unhashable dict field,
# and we never need value-equality of conditions — only their (string) slugs.
@dataclass(frozen=True, eq=False)
class Condition:
    """
    One *learned* condition = a training configuration run once per seed.

    label        human-readable name for tables/figures (e.g. "Shadow DDPG (Q-value)").
    train_kwargs forwarded verbatim to train.train(); must include 'model_type'.
                 'shadow', 'mode', 'lambda_reg', 'switch_critic', 'eta_agent' optional.
    """
    label: str
    train_kwargs: Mapping[str, Any]

    @property
    def slug(self) -> str:
        """Output-directory name — the single source of truth is train.run_label_for()."""
        tk = self.train_kwargs
        return run_label_for(
            tk["model_type"],
            shadow=bool(tk.get("shadow", False)),
            mode=tk.get("mode", "qvalue"),
            lambda_reg=tk.get("lambda_reg", 0.0),
            switch_critic=tk.get("switch_critic", "q1"),
        )


@dataclass(frozen=True)
class EnvSpec:
    """A scenario plus its per-environment training budget (steps)."""
    name: str
    total_steps: int


@dataclass(frozen=True)
class ExperimentGrid:
    """
    A full experiment specification: every (env, condition, seed) combination is
    one training job; rollouts are then recorded over `n_rollout_seeds` episodes.

    seeds            independently *trained* models per condition — the axis the
                     across-seed (rliable) statistics are computed over.
    n_rollout_seeds  deterministic eval episodes recorded per trained model.
    """
    name: str
    envs: tuple[EnvSpec, ...]
    conditions: tuple[Condition, ...]
    seeds: tuple[int, ...]
    eval_freq: int = 1_000
    n_rollout_seeds: int = 20
    include_oracle: bool = True
    mpc_horizon: int = 20

    @property
    def n_training_jobs(self) -> int:
        return len(self.envs) * len(self.conditions) * len(self.seeds)


# ---------------------------------------------------------------------------
# Orchestration helpers (consumed by the Phase 3 runner / Phase 4 analysis)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingJob:
    """A single resolved training job: one condition, on one env, at one seed."""
    scenario: str
    total_steps: int
    condition: Condition
    seed: int
    eval_freq: int

    def train_call_kwargs(self) -> dict:
        """
        Exact kwargs for train.train(). per_seed_dir=True so multi-seed runs write
        to distinct .../<slug>/seed<k>/ dirs instead of overwriting each other.
        """
        return {
            "scenario": self.scenario,
            "total_steps": self.total_steps,
            "seed": self.seed,
            "eval_freq": self.eval_freq,
            "per_seed_dir": True,
            **self.condition.train_kwargs,
        }


def iter_training_jobs(grid: ExperimentGrid) -> Iterator[TrainingJob]:
    """Yield every (env × condition × seed) training job in a stable order."""
    for env in grid.envs:
        for condition in grid.conditions:
            for seed in grid.seeds:
                yield TrainingJob(
                    scenario=env.name, total_steps=env.total_steps,
                    condition=condition, seed=seed, eval_freq=grid.eval_freq,
                )


def checkpoint_path(
    scenario: str,
    condition: Condition,
    seed: int,
    output_dir: str = "outputs/models",
    filename: str = "best.pt",
) -> str:
    """
    Canonical checkpoint location for a (scenario, condition, seed) — mirrors the
    path train_model() writes with per_seed_dir=True. Used by the orchestrator to
    find each best.pt and by the analysis utility to load trained models.
    """
    return os.path.join(output_dir, scenario, condition.slug, f"seed{seed}", filename)


# ---------------------------------------------------------------------------
# Provenance (reproducibility)
# ---------------------------------------------------------------------------

_PROVENANCE_PACKAGES = ("torch", "numpy", "pcgym", "do-mpc", "rliable", "arch", "casadi")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _package_versions() -> dict:
    versions = {}
    for pkg in _PROVENANCE_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def grid_to_dict(grid: ExperimentGrid) -> dict:
    """JSON-serialisable view of a grid, with each condition's resolved slug."""
    return {
        "name": grid.name,
        "envs": [{"name": e.name, "total_steps": e.total_steps} for e in grid.envs],
        "conditions": [
            {"label": c.label, "slug": c.slug, "train_kwargs": dict(c.train_kwargs)}
            for c in grid.conditions
        ],
        "seeds": list(grid.seeds),
        "eval_freq": grid.eval_freq,
        "n_rollout_seeds": grid.n_rollout_seeds,
        "include_oracle": grid.include_oracle,
        "mpc_horizon": grid.mpc_horizon,
        "n_training_jobs": grid.n_training_jobs,
    }


def write_provenance(grid: ExperimentGrid, out_dir: str) -> str:
    """
    Snapshot the resolved grid + git commit + library versions to
    out_dir/provenance.json so a run's outputs carry their exact origin even if
    this config changes later. Returns the written path.
    """
    os.makedirs(out_dir, exist_ok=True)
    provenance = {
        "grid": grid_to_dict(grid),
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "package_versions": _package_versions(),
    }
    path = os.path.join(out_dir, "provenance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Grid registry
# ---------------------------------------------------------------------------

def _phase1_cstr_fourtank() -> ExperimentGrid:
    """
    First-pass headline grid (dissertation_plan.md Phase 0/5): the core ablation
    Pure DDPG vs Shadow DDPG (q-value) on two dynamically different processes.
    PID + NMPC oracle references are added automatically at rollout time.
    Seeds start at 5 for a complete fast pipeline run; scale to 10–20 later.
    four_tank gets a larger budget (longer, harder-coupled episodes than CSTR).
    """
    return ExperimentGrid(
        name="phase1_cstr_fourtank",
        envs=(
            EnvSpec("cstr",      total_steps=50_000),
            EnvSpec("four_tank", total_steps=100_000),
        ),
        conditions=(
            Condition("DDPG", {"model_type": "ddpg", "shadow": False}),
            Condition("Shadow DDPG (Q-value)",
                      {"model_type": "ddpg", "shadow": True, "mode": "qvalue"}),
        ),
        seeds=(0, 1, 2, 3, 4),
        eval_freq=1_000,
        n_rollout_seeds=20,
        include_oracle=True,
        mpc_horizon=20,
    )


# Registry of named grids.
GRIDS: dict[str, ExperimentGrid] = {
    "phase1_cstr_fourtank": _phase1_cstr_fourtank(),
}


# ---------------------------------------------------------------------------
# CLI — describe a grid (and optionally write provenance)
# ---------------------------------------------------------------------------

def _describe(grid: ExperimentGrid) -> None:
    print(f"\n{'='*64}")
    print(f"  Experiment grid: {grid.name}")
    print(f"{'='*64}")
    print(f"  Envs ({len(grid.envs)}):")
    for e in grid.envs:
        print(f"    - {e.name:<24} budget = {e.total_steps:,} steps")
    print(f"  Learned conditions ({len(grid.conditions)}):")
    for c in grid.conditions:
        print(f"    - {c.label:<26} -> {c.slug}/   {dict(c.train_kwargs)}")
    print(f"  References (auto-added at rollout): PID"
          f"{' + NMPC oracle' if grid.include_oracle else ''}")
    print(f"  Seeds ({len(grid.seeds)}): {list(grid.seeds)}")
    print(f"  Eval freq: every {grid.eval_freq:,} steps  |  rollout seeds: {grid.n_rollout_seeds}")
    print(f"  Training jobs (envs × conditions × seeds): {grid.n_training_jobs}")
    print(f"{'='*64}\n")


def main():
    configure_utf8_output()
    ap = argparse.ArgumentParser(
        description="Describe an experiment grid; optionally write provenance.json."
    )
    ap.add_argument("--grid", default="phase1_cstr_fourtank", choices=list(GRIDS),
                    help="Named grid to inspect (default: phase1_cstr_fourtank).")
    ap.add_argument("--provenance", metavar="DIR", default=None,
                    help="Write provenance.json (resolved grid + git + versions) to DIR and exit.")
    args = ap.parse_args()

    grid = GRIDS[args.grid]
    _describe(grid)

    if args.provenance:
        path = write_provenance(grid, args.provenance)
        print(f"  Provenance written: {path}\n")


if __name__ == "__main__":
    main()
