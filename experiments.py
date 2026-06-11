"""
experiments.py
==============
Declarative experiment grids for the offline process-control study (see
`dissertation_plan.md`). Configuration, not orchestration: it describes WHAT to
run (environments × conditions × seeds + budgets and deployment settings) as
version-controllable data. `pipeline.run_pipeline(grid)` consumes a grid and runs
dataset generation -> offline pretraining -> (fine-tuning) -> staged deployment.

A `Condition` (schema.py) is the single factory for both a run's directory slug
(`Condition.slug`) and its metadata (`Condition.to_run_spec()`), so producer
(pretrain) and consumers (orchestrator / analysis) cannot drift apart.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Iterator

from models import DeploymentStage
from schema import (Algorithm, Condition, ExpertKind, RunSpec, Scenario,
                    TrainingMode, offline, offline_to_online, online_contrast)


# ---------------------------------------------------------------------------
# Grid schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvSpec:
    """A scenario plus its per-environment budgets."""
    name: str
    offline_steps: int        # offline gradient steps
    online_steps: int = 0     # env steps for o2o fine-tuning / online contrast


@dataclass(frozen=True)
class ExperimentGrid:
    """
    A full experiment specification: every (env, condition, seed) is one training
    job; rollouts are then recorded over `n_rollout_seeds` episodes across the
    requested deployment stages.
    """
    name: str
    envs: tuple[EnvSpec, ...]
    conditions: tuple[Condition, ...]
    seeds: tuple[int, ...]
    eval_freq: int = 2_000
    dataset_episodes: int = 200
    n_rollout_seeds: int = 20
    include_oracle: bool = True
    mpc_horizon: int = 20
    stages: tuple[DeploymentStage, ...] = (
        DeploymentStage.SHADOW, DeploymentStage.AUTONOMOUS)
    shadow_margins: tuple[float, ...] = (0.0,)

    @property
    def n_training_jobs(self) -> int:
        return len(self.envs) * len(self.conditions) * len(self.seeds)


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingJob:
    """A single resolved training job: one condition, on one env, at one seed."""
    scenario: str
    offline_steps: int
    online_steps: int
    condition: Condition
    seed: int


def iter_training_jobs(grid: ExperimentGrid) -> Iterator[TrainingJob]:
    for env in grid.envs:
        for condition in grid.conditions:
            for seed in grid.seeds:
                yield TrainingJob(scenario=env.name, offline_steps=env.offline_steps,
                                  online_steps=env.online_steps, condition=condition, seed=seed)


@dataclass(frozen=True)
class ModelRef:
    """Locates one trained model in the grid and carries its typed metadata."""
    scenario: str
    condition: Condition
    seed: int
    offline_steps: int
    online_steps: int
    checkpoint: str
    expert_kind: ExpertKind

    def run_spec(self) -> RunSpec:
        ofs = 0 if self.condition.training_mode in (
            TrainingMode.ONLINE_CONTRAST, TrainingMode.ONLINE_SHADOW) else self.offline_steps
        return self.condition.to_run_spec(
            Scenario(self.scenario), self.seed,
            offline_steps=ofs, online_steps=self.online_steps, expert_kind=self.expert_kind)


def iter_model_refs(grid: ExperimentGrid, output_dir: str = "outputs/models",
                    filename: str = "best.pt") -> Iterator[ModelRef]:
    from experts import expert_kind_for
    for env in grid.envs:
        ek = expert_kind_for(env.name)
        for condition in grid.conditions:
            for seed in grid.seeds:
                yield ModelRef(
                    scenario=env.name, condition=condition, seed=seed,
                    offline_steps=env.offline_steps, online_steps=env.online_steps,
                    checkpoint=checkpoint_path(env.name, condition, seed, output_dir, filename),
                    expert_kind=ek)


def checkpoint_path(scenario: str, condition: Condition, seed: int,
                    output_dir: str = "outputs/models", filename: str = "best.pt") -> str:
    """Canonical checkpoint location (mirrors pretrain.run_condition per_seed_dir=True)."""
    return os.path.join(output_dir, scenario, condition.slug, f"seed{seed}", filename)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

_PROVENANCE_PACKAGES = ("torch", "numpy", "pcgym", "do-mpc", "rliable", "casadi")


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
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
    return {
        "name": grid.name,
        "envs": [{"name": e.name, "offline_steps": e.offline_steps,
                  "online_steps": e.online_steps} for e in grid.envs],
        "conditions": [{"label": c.label, "slug": c.slug, "mode": c.training_mode.value,
                        "algorithm": c.algorithm.value, "bc_alpha": c.bc_alpha}
                       for c in grid.conditions],
        "seeds": list(grid.seeds),
        "eval_freq": grid.eval_freq, "dataset_episodes": grid.dataset_episodes,
        "n_rollout_seeds": grid.n_rollout_seeds, "include_oracle": grid.include_oracle,
        "mpc_horizon": grid.mpc_horizon,
        "stages": [s.value for s in grid.stages],
        "shadow_margins": list(grid.shadow_margins),
        "n_training_jobs": grid.n_training_jobs,
    }


def write_provenance(grid: ExperimentGrid, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    provenance = {
        "grid": grid_to_dict(grid), "git_commit": _git_commit(),
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

def _phase1_offline() -> ExperimentGrid:
    """
    Headline grid (dissertation_plan.md). DDPG is the PRIMARY model (aligning with
    Gassert & Althoff's shadow-mode paper, which uses DDPG): offline DDPG+BC,
    deployed at both the shadow (MPC-guarded, earned takeover) and autonomous
    stages, plus its conservative offline-to-online variant and the naive online
    DDPG contrast. TD3+BC is kept as the offline robustness comparison (stronger
    value-overestimation control). On two dynamically different setpoint-tracking
    processes (NMPC expert). PID + NMPC references are added automatically at
    rollout time.
    """
    return ExperimentGrid(
        name="phase1_offline",
        envs=(
            EnvSpec("cstr", offline_steps=50_000, online_steps=20_000),
            EnvSpec("four_tank", offline_steps=80_000, online_steps=30_000),
        ),
        conditions=(
            # PRIMARY: DDPG, deployed in shadow (MPC-guarded) and autonomous (normal).
            offline(Algorithm.DDPG, bc_alpha=2.5, label="Offline DDPG+BC"),
            offline_to_online(Algorithm.DDPG, label="O2O DDPG"),
            online_contrast(Algorithm.DDPG, label="Online DDPG (contrast)"),
            # KEPT: TD3 offline robustness comparison.
            offline(Algorithm.TD3, bc_alpha=2.5, label="Offline TD3+BC"),
        ),
        seeds=(0, 1, 2, 3, 4),
        eval_freq=2_000,
        dataset_episodes=200,
        n_rollout_seeds=20,
        include_oracle=True,
        shadow_margins=(0.0,),
    )


GRIDS: dict[str, ExperimentGrid] = {
    "phase1_offline": _phase1_offline(),
}


def describe_grid(grid: ExperimentGrid) -> None:
    print(f"\n{'='*64}")
    print(f"  Experiment grid: {grid.name}")
    print(f"{'='*64}")
    print(f"  Envs ({len(grid.envs)}):")
    for e in grid.envs:
        print(f"    - {e.name:<24} offline={e.offline_steps:,}  online={e.online_steps:,}")
    print(f"  Conditions ({len(grid.conditions)}):")
    for c in grid.conditions:
        print(f"    - {c.label:<26} -> {c.slug}/   mode={c.training_mode.value} bc={c.bc_alpha}")
    print(f"  References (auto): PID{' + NMPC' if grid.include_oracle else ''}")
    print(f"  Stages: {[s.value for s in grid.stages]}  shadow_margins={list(grid.shadow_margins)}")
    print(f"  Seeds ({len(grid.seeds)}): {list(grid.seeds)}")
    print(f"  Training jobs: {grid.n_training_jobs}")
    print(f"{'='*64}\n")
