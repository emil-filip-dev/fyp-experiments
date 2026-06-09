"""
run_experiments.py
==================
Phase-3 orchestrator: execute an experiment grid (envs x conditions x seeds) by
driving the existing single-run pieces — `train.train_condition()` then
`evaluate.run_rollouts()`. It adds NO training/eval logic of its own.

Programmatic entry (no CLI): call `run_grid(GRIDS["phase1_cstr_fourtank"])`, or the
individual stages `train_grid(...)` / `rollout_grid(...)`. `override_grid(...)`
returns a subset grid (fewer envs/seeds, smaller budget) for smoke runs.

Design (see dissertation_plan.md Phase 3):
  - The grid is declarative config from `experiments.py` (the source of truth for
    WHAT to run). This module only EXECUTES it.
  - Metadata flows as typed objects: rollout ModelSpecs are built straight from the
    grid's `Condition.to_run_spec(...)` (in-memory source of truth) — never read
    back from a directory slug. Slugs only *locate* files by their known path.
  - Resumable: a job whose best.pt already exists is skipped (unless force=True), so
    a killed multi-hour run resumes cleanly. A failing job is isolated (logged) and
    does not abort the rest of the grid.
  - Reproducible: write_provenance() snapshots the resolved grid + git commit +
    library versions next to the outputs.
"""

import dataclasses
import enum
import os

import torch

from evaluate import run_rollouts
from experiments import (
    EnvSpec,
    ExperimentGrid,
    ModelRef,
    checkpoint_path,
    describe_grid,
    iter_model_refs,
    iter_training_jobs,
    write_provenance,
)
from schema import Device, ModelSpec, Scenario
from train import train_condition
from util import device_label, resolve_device


class Stage(enum.StrEnum):
    """Which stage(s) of the pipeline to run."""
    ALL = "all"
    TRAIN = "train"
    ROLLOUTS = "rollouts"


# ---------------------------------------------------------------------------
# Grid overrides (subsetting) — returns a new frozen grid
# ---------------------------------------------------------------------------

def override_grid(
    grid: ExperimentGrid,
    *,
    env_names: list[str] | None = None,
    seeds: list[int] | None = None,
    steps: int | None = None,
) -> ExperimentGrid:
    """Apply optional subsetting (env subset, seed list, uniform step budget)."""
    envs: tuple[EnvSpec, ...] = grid.envs
    if env_names:
        envs = tuple(e for e in envs if e.name in env_names)
        if not envs:
            raise ValueError(f"env_names {env_names} matched none of {[e.name for e in grid.envs]}")
    if steps is not None:
        envs = tuple(dataclasses.replace(e, total_steps=steps) for e in envs)

    changes: dict[str, object] = {}
    if envs != grid.envs:
        changes["envs"] = envs
    if seeds is not None:
        changes["seeds"] = tuple(seeds)
    return dataclasses.replace(grid, **changes) if changes else grid


# ---------------------------------------------------------------------------
# Stage 1 — training
# ---------------------------------------------------------------------------

def train_grid(
    grid: ExperimentGrid,
    output_dir: str,
    device: Device,
    *,
    force: bool = False,
    checkpoint_freq: int = 0,
) -> list[str]:
    """
    Train every (env x condition x seed) job; skip those already on disk. A failed
    job is logged and isolated (the grid continues). Returns the failed-job tags.
    train_condition() takes the grid's Condition object directly (no kwargs
    round-trip), and resolves the device per job.
    """
    jobs = list(iter_training_jobs(grid))
    failures: list[str] = []
    for i, job in enumerate(jobs, start=1):
        ckpt = checkpoint_path(job.scenario, job.condition, job.seed, output_dir)
        tag = f"[{i}/{len(jobs)}] {job.scenario} | {job.condition.label} | seed {job.seed}"
        if not force and os.path.exists(ckpt):
            print(f"{tag}  ->  skip (exists)")
            continue
        print(f"{tag}  ->  train {job.total_steps:,} steps")
        try:
            train_condition(
                job.condition, Scenario(job.scenario), job.total_steps, job.seed,
                eval_freq=job.eval_freq, checkpoint_freq=checkpoint_freq,
                device=device, output_dir=output_dir, per_seed_dir=True,
            )
        except Exception as exc:  # isolate one job's failure from the rest of the grid
            print(f"{tag}  ->  FAILED: {exc!r}")
            failures.append(tag)
    if failures:
        print(f"\n  [train] {len(failures)}/{len(jobs)} job(s) FAILED:")
        for f in failures:
            print(f"    - {f}")
    return failures


# ---------------------------------------------------------------------------
# Stage 2 — rollouts (one timestamped dir per env, all conditions x seeds)
# ---------------------------------------------------------------------------

def rollout_grid(
    grid: ExperimentGrid,
    model_output_dir: str,
    rollout_output_dir: str,
    device: torch.device,
    *,
    use_oracle: bool = True,
) -> None:
    """Record deployment rollouts for every trained model in the grid, per env."""
    refs_by_env: dict[str, list[ModelRef]] = {}
    for ref in iter_model_refs(grid, model_output_dir):
        refs_by_env.setdefault(ref.scenario, []).append(ref)

    for env in grid.envs:
        specs: list[ModelSpec] = []
        for ref in refs_by_env.get(env.name, []):
            if not os.path.exists(ref.checkpoint):
                print(f"  [warn] missing checkpoint, skipped: {ref.checkpoint}")
                continue
            # ModelSpec built from the grid's Condition (in-memory truth), not run.json.
            specs.append(ModelSpec(checkpoint=ref.checkpoint, run=ref.run_spec()))
        run_rollouts(
            scenario=Scenario(env.name),
            model_specs=specs,
            n_seeds=grid.n_rollout_seeds,
            use_oracle=use_oracle and grid.include_oracle,
            mpc_horizon=grid.mpc_horizon,
            output_dir=rollout_output_dir,
            device=device,
        )


# ---------------------------------------------------------------------------
# Full grid execution + dry-run description
# ---------------------------------------------------------------------------

def describe_plan(grid: ExperimentGrid, model_output_dir: str = "outputs/models") -> None:
    """Print the grid plus how many jobs already exist on disk (would be skipped)."""
    describe_grid(grid)
    jobs = list(iter_training_jobs(grid))
    n_exist = sum(os.path.exists(checkpoint_path(j.scenario, j.condition, j.seed, model_output_dir))
                  for j in jobs)
    print(f"  already-trained (would skip): {n_exist}/{len(jobs)}\n")


def run_grid(
    grid: ExperimentGrid,
    *,
    model_output_dir: str = "outputs/models",
    rollout_output_dir: str = "outputs/rollouts",
    provenance_dir: str = "outputs/experiments",
    device: Device = Device.CPU,
    stage: Stage = Stage.ALL,
    force: bool = False,
    use_oracle: bool = True,
) -> None:
    """Write provenance, then run the requested stage(s) of the grid end-to-end."""
    prov = write_provenance(grid, os.path.join(provenance_dir, grid.name))
    torch_device = resolve_device(device)   # resolved once for the print + rollouts
    print(f"  Provenance: {prov}  |  device: {device_label(torch_device)}")

    if stage in (Stage.ALL, Stage.TRAIN):
        train_grid(grid, model_output_dir, device, force=force)
    if stage in (Stage.ALL, Stage.ROLLOUTS):
        rollout_grid(grid, model_output_dir, rollout_output_dir, torch_device, use_oracle=use_oracle)
