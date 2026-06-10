"""
pipeline.py
===========
Phase-3 orchestrator for the offline pipeline (programmatic, no CLI). Drives the
full study end to end:

  generate dataset -> offline pretrain -> (conservative fine-tune) -> staged deploy

`run_pipeline(grid)` writes provenance, then:
  - train_grid : runs pretrain.run_condition for every (env × condition × seed).
                 Resumable — skips a job whose best.pt already exists. Per-job
                 failures are isolated and reported, not fatal.
  - rollout_grid : builds ModelSpecs straight from the grid and records staged
                   rollouts (deploy.run_rollouts) per scenario, including the PID
                   and NMPC references.

`override_grid(...)` subsets a grid for smoke runs; Stage {all, train, rollouts}
selects which phases run.
"""

import enum
import json
import os
import traceback
from dataclasses import replace

from experiments import (ExperimentGrid, EnvSpec, checkpoint_path, describe_grid,
                         iter_model_refs, iter_training_jobs, write_provenance)
from schema import Device, ModelSpec, RunSpec, Scenario


class Stage(enum.StrEnum):
    ALL = "all"
    TRAIN = "train"
    ROLLOUTS = "rollouts"
    ANALYSE = "analyse"


def override_grid(grid: ExperimentGrid, *, env_names=None, seeds=None,
                  offline_steps=None, online_steps=None, conditions=None,
                  dataset_episodes=None, n_rollout_seeds=None) -> ExperimentGrid:
    """Return a subset/smaller copy of a grid for smoke runs."""
    envs = grid.envs
    if env_names is not None:
        envs = tuple(e for e in envs if e.name in set(env_names))
    if offline_steps is not None or online_steps is not None:
        envs = tuple(EnvSpec(e.name,
                             offline_steps if offline_steps is not None else e.offline_steps,
                             online_steps if online_steps is not None else e.online_steps)
                     for e in envs)
    kw = {}
    if seeds is not None:
        kw["seeds"] = tuple(seeds)
    if conditions is not None:
        kw["conditions"] = tuple(conditions)
    if dataset_episodes is not None:
        kw["dataset_episodes"] = dataset_episodes
    if n_rollout_seeds is not None:
        kw["n_rollout_seeds"] = n_rollout_seeds
    return replace(grid, envs=envs, **kw)


def train_grid(grid: ExperimentGrid, *, device: Device = Device.AUTO,
               output_dir: str = "outputs/models", skip_existing: bool = True) -> None:
    """Run every training job in the grid (resumable, failure-isolated)."""
    from pretrain import run_condition

    jobs = list(iter_training_jobs(grid))
    print(f"\n[train_grid] {len(jobs)} job(s)\n")
    for i, job in enumerate(jobs, 1):
        ckpt = checkpoint_path(job.scenario, job.condition, job.seed, output_dir)
        if skip_existing and os.path.exists(ckpt):
            print(f"  [{i}/{len(jobs)}] skip (exists): {ckpt}")
            continue
        print(f"  [{i}/{len(jobs)}] {job.scenario} | {job.condition.slug} | seed {job.seed}")
        try:
            run_condition(
                job.condition, job.scenario, job.seed,
                offline_steps=job.offline_steps, online_steps=job.online_steps,
                eval_freq=grid.eval_freq, device=device, output_dir=output_dir,
                per_seed_dir=True, dataset_episodes=grid.dataset_episodes,
                mpc_horizon=grid.mpc_horizon,
            )
        except Exception:
            print(f"  [!] job failed ({job.scenario}/{job.condition.slug}/seed{job.seed}):")
            traceback.print_exc()


def rollout_grid(grid: ExperimentGrid, *, device=None, output_models: str = "outputs/models",
                 output_rollouts: str = "outputs/rollouts") -> list[str]:
    """Record staged rollouts for every trained model, grouped by scenario.
    Returns the list of rollout output directories produced."""
    import torch

    from deploy import run_rollouts

    dev = device or torch.device("cpu")
    by_scenario: dict[str, list[ModelSpec]] = {}
    for ref in iter_model_refs(grid, output_dir=output_models):
        if not os.path.exists(ref.checkpoint):
            print(f"  [rollouts] missing checkpoint, skipping: {ref.checkpoint}")
            continue
        # Prefer the RunSpec actually written next to the checkpoint (the trained
        # identity) over re-deriving it from the current grid, which may have been
        # edited between the train and rollout phases.
        run_json = os.path.join(os.path.dirname(ref.checkpoint), "run.json")
        if os.path.exists(run_json):
            with open(run_json, encoding="utf-8") as f:
                spec = RunSpec.from_json(json.load(f))
        else:
            spec = ref.run_spec()
        by_scenario.setdefault(ref.scenario, []).append(
            ModelSpec(checkpoint=ref.checkpoint, run=spec))

    dirs = []
    for scenario, specs in by_scenario.items():
        dirs.append(run_rollouts(
            scenario, specs, stages=grid.stages, shadow_margins=grid.shadow_margins,
            n_seeds=grid.n_rollout_seeds, use_oracle=grid.include_oracle,
            mpc_horizon=grid.mpc_horizon, output_dir=output_rollouts, device=dev,
        ))
    return dirs


# Scenarios with a hand-tuned takeover-map state slice (others use a generic
# slice that may not be meaningful, so the pipeline skips them by default).
_TAKEOVER_SCENARIOS = {"cstr"}


def analyse_grid(grid: ExperimentGrid, *, output_rollouts: str = "outputs/rollouts",
                 output_models: str = "outputs/models", rollout_dirs: list[str] | None = None,
                 takeover_maps: bool = True, takeover_grid_res: int = 50) -> None:
    """Compute metrics (CSV) + figures (PNG) for each scenario's rollouts, then (if
    `takeover_maps`) render the RL–MPC takeover-map evolution for one representative
    seed per (scenario, condition). If `rollout_dirs` is None, analyse the newest
    rollout dir per scenario in the grid."""
    from analysis import analyse_rollout_dir, latest_rollout_dir, plot_takeover_map

    if rollout_dirs is None:
        scenarios = {e.name for e in grid.envs}
        rollout_dirs = []
        for s in scenarios:
            try:
                rollout_dirs.append(latest_rollout_dir(s, root=output_rollouts))
            except FileNotFoundError:
                print(f"  [analyse] no rollouts for {s}, skipping")
    for d in rollout_dirs:
        try:
            analyse_rollout_dir(d)
        except Exception:
            print(f"  [analyse skip] {d}")
            traceback.print_exc()

    if not takeover_maps:
        return
    seen: set[tuple[str, str]] = set()
    for ref in iter_model_refs(grid, output_dir=output_models):
        key = (ref.scenario, ref.condition.slug)
        if key in seen or ref.scenario not in _TAKEOVER_SCENARIOS:
            continue
        run_dir = os.path.dirname(ref.checkpoint)
        if not os.path.exists(os.path.join(run_dir, "run.json")):
            continue   # this seed didn't train; try the next seed for the condition
        seen.add(key)
        print(f"  [takeover-map] {ref.scenario} / {ref.condition.slug} (seed {ref.seed})")
        try:
            plot_takeover_map(run_dir, grid_res=takeover_grid_res, mpc_horizon=grid.mpc_horizon)
        except Exception:
            print(f"  [takeover-map skip] {run_dir}")
            traceback.print_exc()


def run_pipeline(grid: ExperimentGrid, *, stage: Stage = Stage.ALL,
                 device: Device = Device.AUTO, output_root: str = "outputs",
                 skip_existing: bool = True) -> None:
    """Run the requested pipeline stages for a grid."""
    from util import resolve_device

    describe_grid(grid)
    models_dir = os.path.join(output_root, "models")
    rollouts_dir = os.path.join(output_root, "rollouts")

    # Snapshot provenance only when (re)training — an analyse/rollouts-only run must
    # not clobber the train-time provenance with a possibly-edited grid.
    if stage in (Stage.ALL, Stage.TRAIN):
        prov_dir = os.path.join(output_root, "experiments", grid.name)
        print(f"[provenance] {write_provenance(grid, prov_dir)}")

    rollout_dirs = None
    if stage in (Stage.ALL, Stage.TRAIN):
        train_grid(grid, device=device, output_dir=models_dir, skip_existing=skip_existing)
    if stage in (Stage.ALL, Stage.ROLLOUTS):
        rollout_dirs = rollout_grid(grid, device=resolve_device(device), output_models=models_dir,
                                    output_rollouts=rollouts_dir)
    if stage in (Stage.ALL, Stage.ANALYSE):
        analyse_grid(grid, output_rollouts=rollouts_dir, output_models=models_dir,
                     rollout_dirs=rollout_dirs)
