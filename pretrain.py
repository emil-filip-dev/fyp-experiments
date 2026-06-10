"""
pretrain.py
===========
Training entry points for the offline pipeline. Three training modes, dispatched
by a Condition's TrainingMode (see schema.py):

  offline           — pretrain the agent purely from the STATIC expert(+perturbed)
                      dataset (no env interaction). TD3+BC / DDPG+BC. The headline
                      method: the agent learns to control the process from logged
                      data, never touching the plant during training.
  offline_to_online — offline pretrain, then conservative fine-tuning from
                      expert-GUARDED online transitions, sweeping the deployment
                      stage shadow -> autonomous as the agent earns
                      more authority. Exploration is low and the expert catches
                      un-earned actions, so training-time safety is preserved.
  online_contrast   — naive online RL on the live plant from scratch (full
                      exploration, no expert guard). The unsafe foil whose
                      training-time violations motivate the offline approach.

Outputs: outputs/models/<scenario>/<run_label>[/seed<k>]/
  best.pt + run.json + training_log.npz  (+ dataset.npz for offline modes).

Programmatic API (no CLI):
  run_condition(condition, scenario, seed, ...) -> (agent, save_path)
"""

import json
import os

import numpy as np
import torch
from tqdm import tqdm

from constraints import violation_magnitudes
from data import (describe_dataset, dataset_to_buffer, get_or_make_dataset,
                  save_dataset)
from deploy import evaluate_deploy
from experts import make_expert
from models import DeploymentStage, ShadowController, get_agent
from scenarios import SCENARIOS, make_env_for
from schema import (Condition, Device, ExpertKind, RunSpec, Scenario,
                    TrainingMode)
from util import device_label, resolve_device


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _save_log(save_path: str, mode: str, meta: dict, **arrays) -> str:
    path = os.path.join(save_path, "training_log.npz")
    np.savez(path, mode=np.array(mode), meta=np.array(json.dumps(meta)),
             **{k: np.asarray(v) for k, v in arrays.items()})
    return path


# ---------------------------------------------------------------------------
# Periodic model snapshots (weights only; for the takeover-map evolution viz)
# ---------------------------------------------------------------------------

def _clear_snapshots(save_path: str) -> None:
    import shutil
    shutil.rmtree(os.path.join(save_path, "snapshots"), ignore_errors=True)


def _save_snapshot(agent, save_path: str, phase: str, step: int) -> None:
    """Save a lightweight weights-only checkpoint + append to snapshots.json. `phase`
    is 'offline' (x-axis = gradient step) or 'online' (x-axis = env step)."""
    snap_dir = os.path.join(save_path, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    fn = f"snap_{phase}_{int(step):08d}.pt"
    agent.save(os.path.join(snap_dir, fn))
    idx_path = os.path.join(snap_dir, "snapshots.json")
    idx = []
    if os.path.exists(idx_path):
        try:
            with open(idx_path, encoding="utf-8") as f:
                idx = json.load(f)
        except Exception:
            idx = []   # corrupt index from a killed run — start fresh rather than abort
    idx.append({"file": fn, "phase": phase, "step": int(step)})
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)


# ---------------------------------------------------------------------------
# Offline pretraining (TD3+BC / DDPG+BC)
# ---------------------------------------------------------------------------

def pretrain_offline(agent, dataset_buffer, run_spec: RunSpec, expert, *,
                     save_path: str, eval_freq: int = 2_000, n_eval: int = 5,
                     snapshot_freq: int = 0) -> str:
    """
    Gradient-descent the agent on the static dataset for run_spec.offline_steps
    gradient steps. Periodically evaluates STANDALONE (autonomous) deployment and
    saves best.pt on eval-return improvement. Writes training_log.npz. If
    snapshot_freq > 0, saves a weights-only snapshot every that-many gradient steps
    (phase='offline') for the takeover-map evolution visualisation.
    """
    agent.load_dataset(dataset_buffer)
    total = run_spec.offline_steps
    if snapshot_freq:
        _save_snapshot(agent, save_path, "offline", 0)   # random-init snapshot

    g_step, c_loss, a_loss = [], [], []
    e_step, e_ret, e_vrate, e_vmax, e_tk, e_div = [], [], [], [], [], []
    best = -np.inf
    last_a = float("nan")

    bar = tqdm(total=total, unit="grad", dynamic_ncols=True, desc=run_spec.run_label)
    while agent.total_updates < total:
        out = agent.update()
        bar.update(1)
        if out and "actor_loss" in out:
            last_a = out["actor_loss"]
        if snapshot_freq and agent.total_updates % snapshot_freq == 0:
            _save_snapshot(agent, save_path, "offline", agent.total_updates)

        if agent.total_updates % eval_freq == 0 or agent.total_updates >= total:
            ev = evaluate_deploy(agent, run_spec.scenario, expert,
                                 stage=DeploymentStage.AUTONOMOUS, n_episodes=n_eval)
            g_step.append(int(agent.total_updates))
            c_loss.append(float(out["critic_loss"]) if out else float("nan"))
            a_loss.append(float(last_a))
            e_step.append(int(agent.total_updates))
            e_ret.append(ev["return_mean"]); e_vrate.append(ev["viol_rate"])
            e_vmax.append(ev["viol_max"]); e_tk.append(ev["takeover_frac"])
            e_div.append(ev["divergence_mean"])
            if ev["return_mean"] > best:
                best = ev["return_mean"]
                agent.save(os.path.join(save_path, "best.pt"))
            bar.set_postfix({"ret": f"{ev['return_mean']:.0f}", "best": f"{best:.0f}",
                             "viol%": f"{ev['viol_rate']*100:.1f}"})
    bar.close()
    if snapshot_freq and agent.total_updates % snapshot_freq != 0:
        _save_snapshot(agent, save_path, "offline", agent.total_updates)

    if not os.path.exists(os.path.join(save_path, "best.pt")):
        agent.save(os.path.join(save_path, "best.pt"))

    meta = _log_meta(run_spec, 0)
    meta["schema"] = {
        "grad_step": "gradient step at eval boundary",
        "critic_loss": "critic loss", "actor_loss": "actor (TD3+BC) loss",
        "eval_step": "gradient step", "eval_return": "standalone (autonomous) mean return",
        "eval_viol_rate": "autonomous deployment violation rate",
        "eval_viol_max": "autonomous deployment max violation magnitude",
        "eval_takeover": "autonomous takeover frac (1.0 unless safety fallback)",
        "eval_divergence": "mean ||a_agent - a_expert|| in autonomous deployment",
    }
    path = _save_log(
        save_path, "offline", meta,
        grad_step=g_step, critic_loss=c_loss, actor_loss=a_loss,
        eval_step=e_step, eval_return=e_ret, eval_viol_rate=e_vrate,
        eval_viol_max=e_vmax, eval_takeover=e_tk, eval_divergence=e_div,
    )
    tqdm.write(f"  Offline pretraining done. Best autonomous return: {best:.1f}")
    tqdm.write(f"  Training log: {path}")
    return best


# ---------------------------------------------------------------------------
# Conservative offline-to-online fine-tuning (staged, expert-guarded)
# ---------------------------------------------------------------------------

def _stage_for(progress: float, frac_shadow: float):
    """Map training progress in [0,1] to a deployment stage (rising autonomy)."""
    return DeploymentStage.SHADOW if progress < frac_shadow else DeploymentStage.AUTONOMOUS


def finetune_o2o(agent, run_spec: RunSpec, expert, *, save_path: str,
                 eval_freq: int = 2_000, n_eval: int = 5,
                 frac_shadow: float = 0.7, margin: float = 0.0,
                 snapshot_freq: int = 0, init_best: float = -np.inf) -> str:
    """
    Continue training the (already offline-pretrained) agent from expert-guarded
    online transitions, sweeping shadow -> autonomous over run_spec.online_steps
    env steps. In the shadow phase the expert executes for un-earned actions, so
    behaviour-time safety is preserved while the agent fine-tunes on-distribution.
    Records the behaviour-time curve (the C1/C3 evidence) and a deployment curve.

    `init_best` carries the offline phase's best eval return so best.pt is only
    overwritten if fine-tuning genuinely IMPROVES on the offline policy — otherwise
    the (better) offline checkpoint is kept. best.pt is thus best-of-both, never
    silently replaced by a worse fine-tuned policy.
    """
    scenario = run_spec.scenario
    cfg = SCENARIOS[str(scenario)]
    n_steps = cfg["n_steps"]
    constraint_spec = cfg.get("constraint_spec", [])
    env = make_env_for(str(scenario))
    controller = ShadowController(agent, expert, safety_fallback=True)
    total = run_spec.online_steps

    beh = {"env_step": [], "return": [], "viol_rate": [], "viol_max": [],
           "takeover": [], "stage": []}
    evl = {"env_step": [], "return": [], "viol_rate": [], "takeover": []}
    e_step_best = init_best       # keep the offline best; only beat it to overwrite best.pt
    episode = 0
    next_snap = snapshot_freq

    bar = tqdm(total=total, unit="step", dynamic_ncols=True, desc=f"{run_spec.run_label} (o2o)")
    while agent.total_env_steps < total:
        progress = agent.total_env_steps / max(total, 1)
        stage = _stage_for(progress, frac_shadow)
        explore = True

        obs, _ = env.reset(seed=episode + run_spec.seed * 10_000)
        controller.reset()
        done, step = False, 0
        ep_ret = 0.0
        flags, states = [], []
        start = agent.total_env_steps
        while not done and step < n_steps:
            a_exec, used, _, _ = controller.decide(obs, stage, margin=margin, explore=explore)
            next_obs, r, term, trunc, _ = env.step(a_exec)
            done = bool(term or trunc)
            agent.store(obs, a_exec, r, next_obs, done)
            states.append(env.state.copy().astype(np.float32))
            flags.append(1.0 if used else 0.0)
            ep_ret += r
            obs = next_obs
            step += 1
            agent.total_env_steps += 1
        for _ in range(step):
            agent.update()
        bar.update(agent.total_env_steps - start)
        if snapshot_freq and agent.total_env_steps >= next_snap:
            _save_snapshot(agent, save_path, "online", agent.total_env_steps)
            next_snap += snapshot_freq

        bv = violation_magnitudes(np.asarray(states, dtype=np.float32), constraint_spec)
        any_v = (bv > 0).any(axis=-1) if bv.shape[-1] else np.zeros(bv.shape[0], dtype=bool)
        beh["env_step"].append(int(agent.total_env_steps))
        beh["return"].append(float(ep_ret))
        beh["viol_rate"].append(float(any_v.mean()) if any_v.size else 0.0)
        beh["viol_max"].append(float(bv.max()) if bv.size else 0.0)
        beh["takeover"].append(float(np.mean(flags)) * 100.0)
        beh["stage"].append(stage.value)

        if agent.total_env_steps >= eval_freq * (len(evl["env_step"]) + 1):
            ev = evaluate_deploy(agent, scenario, expert,
                                 stage=DeploymentStage.AUTONOMOUS, n_episodes=n_eval)
            evl["env_step"].append(int(agent.total_env_steps))
            evl["return"].append(ev["return_mean"]); evl["viol_rate"].append(ev["viol_rate"])
            evl["takeover"].append(ev["takeover_frac"])
            if ev["return_mean"] > e_step_best:
                e_step_best = ev["return_mean"]
                agent.save(os.path.join(save_path, "best.pt"))
        episode += 1
        bar.set_postfix({"stage": stage.value[:4], "ret": f"{ep_ret:.0f}",
                         "agent%": f"{np.mean(flags)*100:.0f}"})
    bar.close()

    if not os.path.exists(os.path.join(save_path, "best.pt")):
        agent.save(os.path.join(save_path, "best.pt"))

    meta = _log_meta(run_spec, 0)
    meta["schema"] = {
        "beh_env_step": "cumulative env step at episode end",
        "beh_return": "behaviour return (expert-guarded, incl. exploration)",
        "beh_viol_rate": "behaviour violation rate (training-time safety, C1)",
        "beh_viol_max": "behaviour max violation magnitude",
        "beh_takeover": "agent takeover % over the episode (rising autonomy, C3)",
        "beh_stage": "deployment stage active during the episode",
        "eval_*": "periodic standalone (autonomous) deployment metrics",
    }
    path = _save_log(
        save_path, "offline_to_online", meta,
        beh_env_step=beh["env_step"], beh_return=beh["return"],
        beh_viol_rate=beh["viol_rate"], beh_viol_max=beh["viol_max"],
        beh_takeover=beh["takeover"], beh_stage=np.array(beh["stage"]),
        eval_env_step=evl["env_step"], eval_return=evl["return"],
        eval_viol_rate=evl["viol_rate"], eval_takeover=evl["takeover"],
    )
    tqdm.write(f"  Offline-to-online fine-tuning done. Log: {path}")
    return path


# ---------------------------------------------------------------------------
# Naive online contrast (the unsafe foil)
# ---------------------------------------------------------------------------

def train_online_contrast(agent, run_spec: RunSpec, expert, *, save_path: str,
                          eval_freq: int = 2_000, n_eval: int = 5,
                          warmup_steps: int = 1_000, snapshot_freq: int = 0) -> str:
    """
    Train the agent online on the live plant from scratch — full exploration, the
    agent's action ALWAYS executed (no expert guard). Records the behaviour-time
    violation curve: the unsafe-exploration evidence that motivates offline RL.
    """
    scenario = run_spec.scenario
    cfg = SCENARIOS[str(scenario)]
    n_steps = cfg["n_steps"]
    constraint_spec = cfg.get("constraint_spec", [])
    env = make_env_for(str(scenario))
    action_dim = cfg["action_dim"]
    total = run_spec.online_steps

    beh = {"env_step": [], "return": [], "viol_rate": [], "viol_max": [], "viol_count": []}
    evl = {"env_step": [], "return": [], "viol_rate": []}
    best = -np.inf
    episode = 0
    next_snap = snapshot_freq
    if snapshot_freq:
        _save_snapshot(agent, save_path, "online", 0)   # random-init snapshot

    bar = tqdm(total=total, unit="step", dynamic_ncols=True, desc=f"{run_spec.run_label} (online)")
    while agent.total_env_steps < total:
        obs, _ = env.reset(seed=episode + run_spec.seed * 10_000)
        done, step = False, 0
        ep_ret = 0.0
        states = []
        start = agent.total_env_steps
        while not done and step < n_steps:
            if agent.total_env_steps < warmup_steps:
                a = np.random.uniform(-1.0, 1.0, action_dim).astype(np.float32)
            else:
                a = agent.act(obs, explore=True)
            next_obs, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            agent.store(obs, a, r, next_obs, done)
            states.append(env.state.copy().astype(np.float32))
            ep_ret += r
            obs = next_obs
            step += 1
            agent.total_env_steps += 1
        if agent.total_env_steps >= warmup_steps:
            for _ in range(step):
                agent.update()
        bar.update(agent.total_env_steps - start)
        if snapshot_freq and agent.total_env_steps >= next_snap:
            _save_snapshot(agent, save_path, "online", agent.total_env_steps)
            next_snap += snapshot_freq

        bv = violation_magnitudes(np.asarray(states, dtype=np.float32), constraint_spec)
        any_v = (bv > 0).any(axis=-1) if bv.shape[-1] else np.zeros(bv.shape[0], dtype=bool)
        beh["env_step"].append(int(agent.total_env_steps))
        beh["return"].append(float(ep_ret))
        beh["viol_rate"].append(float(any_v.mean()) if any_v.size else 0.0)
        beh["viol_max"].append(float(bv.max()) if bv.size else 0.0)
        beh["viol_count"].append(int(any_v.sum()))

        if agent.total_env_steps >= eval_freq * (len(evl["env_step"]) + 1):
            ev = evaluate_deploy(agent, scenario, expert,
                                 stage=DeploymentStage.AUTONOMOUS, n_episodes=n_eval)
            evl["env_step"].append(int(agent.total_env_steps))
            evl["return"].append(ev["return_mean"]); evl["viol_rate"].append(ev["viol_rate"])
            if ev["return_mean"] > best:
                best = ev["return_mean"]
                agent.save(os.path.join(save_path, "best.pt"))
        episode += 1
        bar.set_postfix({"ret": f"{ep_ret:.0f}", "viol%": f"{beh['viol_rate'][-1]*100:.0f}"})
    bar.close()

    if not os.path.exists(os.path.join(save_path, "best.pt")):
        agent.save(os.path.join(save_path, "best.pt"))

    meta = _log_meta(run_spec, warmup_steps)
    meta["schema"] = {
        "beh_env_step": "cumulative env step at episode end",
        "beh_return": "behaviour return (full exploration, no guard)",
        "beh_viol_rate": "behaviour violation rate (UNSAFE online exploration, C1 foil)",
        "beh_viol_max": "behaviour max violation magnitude",
        "beh_viol_count": "# violated steps in the episode",
        "eval_*": "periodic standalone deployment metrics",
    }
    path = _save_log(
        save_path, "online_contrast", meta,
        beh_env_step=beh["env_step"], beh_return=beh["return"],
        beh_viol_rate=beh["viol_rate"], beh_viol_max=beh["viol_max"],
        beh_viol_count=beh["viol_count"],
        eval_env_step=evl["env_step"], eval_return=evl["return"], eval_viol_rate=evl["viol_rate"],
    )
    tqdm.write(f"  Online contrast done. Log: {path}")
    return path


# ---------------------------------------------------------------------------
# Metadata + dispatch
# ---------------------------------------------------------------------------

def _log_meta(run_spec: RunSpec, warmup: int) -> dict:
    cfg = SCENARIOS[str(run_spec.scenario)]
    constraint_spec = cfg.get("constraint_spec", [])
    return {
        "scenario": str(run_spec.scenario), "run_label": run_spec.run_label,
        "seed": run_spec.seed, "training_mode": run_spec.training_mode.value,
        "expert_kind": run_spec.expert_kind.value,
        "offline_steps": run_spec.offline_steps, "online_steps": run_spec.online_steps,
        "warmup_steps": int(warmup), "n_con": len(constraint_spec),
        "constraints": [{k: c[k] for k in ("name", "label", "bound", "type", "unit") if k in c}
                        for c in constraint_spec],
    }


def run_condition(
    condition: Condition,
    scenario: Scenario | str,
    seed: int,
    *,
    offline_steps: int = 50_000,
    online_steps: int = 20_000,
    eval_freq: int = 2_000,
    device: Device = Device.AUTO,
    output_dir: str = "outputs/models",
    per_seed_dir: bool = False,
    dataset: dict | None = None,
    dataset_episodes: int = 200,
    mpc_horizon: int = 20,
    n_snapshots: int = 8,
) -> tuple[object, str]:
    """
    Train one Condition on one (scenario, seed), dispatching on its TrainingMode.
    The Condition is the single source of the agent's hyperparameters, the run
    slug, and the RunSpec. Returns (agent, save_path).
    """
    scenario = Scenario(scenario)
    cfg = SCENARIOS[str(scenario)]
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = resolve_device(device)
    state_dim, action_dim = cfg["state_dim"], cfg["action_dim"]

    expert, expert_kind = make_expert(scenario, mpc_horizon=mpc_horizon)
    run_spec = condition.to_run_spec(
        scenario, seed,
        offline_steps=offline_steps if condition.training_mode is not TrainingMode.ONLINE_CONTRAST else 0,
        online_steps=online_steps,
        expert_kind=expert_kind,
    )

    save_path = os.path.join(output_dir, str(scenario), condition.slug)
    if per_seed_dir:
        save_path = os.path.join(save_path, f"seed{seed}")
    os.makedirs(save_path, exist_ok=True)
    _clear_snapshots(save_path)   # fresh evolution series for the takeover-map viz
    with open(os.path.join(save_path, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_spec.to_json(), f, indent=2)

    snap_offline = max(1, offline_steps // n_snapshots) if n_snapshots else 0
    snap_online = max(1, online_steps // n_snapshots) if n_snapshots else 0

    agent = get_agent(condition.algorithm.value)(
        state_dim=state_dim, action_dim=action_dim, device=dev, **condition.agent_kwargs())

    print(f"\n{'='*60}")
    print(f"  Scenario : {scenario}   |  expert: {expert_kind.value}")
    print(f"  Condition: {condition.label}  ({condition.slug})")
    print(f"  Mode     : {condition.training_mode.value}   |  device: {device_label(dev)}  seed={seed}")
    print(f"{'='*60}\n")

    mode = condition.training_mode
    if mode in (TrainingMode.OFFLINE, TrainingMode.OFFLINE_TO_ONLINE):
        if dataset is None:
            dataset = get_or_make_dataset(scenario, expert, seed=seed,
                                          n_episodes=dataset_episodes, expert_kind=expert_kind)
        print(f"  {describe_dataset(dataset)}")
        save_dataset(dataset, os.path.join(save_path, "dataset.npz"))
        buffer = dataset_to_buffer(dataset)
        best_offline = pretrain_offline(agent, buffer, run_spec, expert, save_path=save_path,
                                        eval_freq=eval_freq, snapshot_freq=snap_offline)
        if mode is TrainingMode.OFFLINE_TO_ONLINE:
            finetune_o2o(agent, run_spec, expert, save_path=save_path,
                         eval_freq=eval_freq, snapshot_freq=snap_online, init_best=best_offline)
    elif mode is TrainingMode.ONLINE_CONTRAST:
        train_online_contrast(agent, run_spec, expert, save_path=save_path,
                              eval_freq=eval_freq, snapshot_freq=snap_online)
    else:
        raise ValueError(f"Unhandled training mode {mode!r}")

    print(f"\n  Done. Artifacts in {save_path}")
    return agent, save_path
