"""
deploy.py
=========
Staged deployment of an offline-pretrained agent alongside its expert, and the
raw per-step rollout recorder. This is the evaluation half of the pipeline: it
serialises rollouts to disk (no plotting / summary metrics — that is the analysis
utility's job).

The proposal's staged introduction is realised as DeploymentStages:
  shadow      — agent takes over wherever it has EARNED it (q_gap > margin); the
                expert handles the rest. The headline deployment mode.
  autonomous  — agent controls alone (expert only as optional safety fallback).

Reference controllers (PID, and NMPC where available) are always recorded too, so
the analysis utility has the baseline floor and the optimality ceiling.

Programmatic API (no CLI):
  evaluate_deploy(agent, scenario, expert, stage, margin, ...) -> summary dict
      cheap, in-memory; used by pretrain.py for periodic evaluation.
  run_rollouts(scenario, model_specs, ...) -> output dir
      full disk serialisation (one .npz per method×stage + manifest.json).
"""

import json
import os
from datetime import datetime

import numpy as np
import torch

# Evaluation/rollout episodes use seeds in a range DISJOINT from training, so the
# reported deployment performance is genuinely held-out. Training seeds episodes as
# `seed*10_000 + episode` (run seeds are small), so this offset is safely clear of them.
EVAL_SEED_OFFSET = 1_000_000

from constraints import constraint_metrics, violation_magnitudes
from experts import make_expert
from models import DeploymentStage, ShadowController, NMPCController, get_agent
from scenarios import SCENARIOS, make_env_for
from schema import (ExpertKind, MethodRecord, MethodRole, ModelSpec, Scenario)


# ---------------------------------------------------------------------------
# Per-episode recorder
# ---------------------------------------------------------------------------

def _record_episode(env, step_fn, reset_fn, seed: int, n_steps: int,
                    constraint_spec: list) -> dict:
    """
    Run one deterministic episode. `step_fn(obs) -> (a_exec, used_agent, a_agent,
    a_expert, q_gap)` produces the per-step decision; `reset_fn()` resets any
    stateful controller. Returns per-step arrays [T, ...].
    """
    obs, _ = env.reset(seed=seed)
    reset_fn()

    states, observations = [], []
    a_exec_l, a_agent_l, a_exp_l = [], [], []
    rewards, takeover, q_gap, divergence = [], [], [], []
    done, step = False, 0

    while not done and step < n_steps:
        a_exec, used_agent, a_agent, a_expert, qg = step_fn(obs)
        a_exec = np.asarray(a_exec, dtype=np.float32)
        a_agent = np.asarray(a_agent, dtype=np.float32)
        a_expert = np.asarray(a_expert, dtype=np.float32)

        observations.append(np.asarray(obs, dtype=np.float32))
        obs, reward, terminated, truncated, _ = env.step(a_exec)
        done = bool(terminated or truncated)

        states.append(env.state.copy().astype(np.float32))
        a_exec_l.append(a_exec)
        a_agent_l.append(a_agent)
        a_exp_l.append(a_expert)
        rewards.append(float(reward))
        takeover.append(1.0 if used_agent else (0.0 if used_agent is False else float("nan")))
        q_gap.append(np.float32(qg))
        divergence.append(np.float32(np.linalg.norm(a_agent - a_expert)))
        step += 1

    states_arr = np.asarray(states, dtype=np.float32)
    violations = violation_magnitudes(states_arr, constraint_spec or []).astype(np.float32)
    return {
        "states": states_arr,
        "obs": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(a_exec_l, dtype=np.float32),
        "actions_agent": np.asarray(a_agent_l, dtype=np.float32),
        "actions_expert": np.asarray(a_exp_l, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "takeover": np.asarray(takeover, dtype=np.float32),
        "q_gap": np.asarray(q_gap, dtype=np.float32),
        "divergence": np.asarray(divergence, dtype=np.float32),
        "violations": violations,
    }


def _agent_step_fn(controller: ShadowController, stage: DeploymentStage, margin: float):
    """Build a step_fn for an agent deployed at a given stage/margin."""
    def step_fn(obs):
        a_exec, used, a_agent, a_expert = controller.decide(obs, stage, margin=margin)
        qg = controller.agent.q_gap(obs, a_expert)
        return a_exec, used, a_agent, a_expert, qg
    return step_fn


def _reference_step_fn(controller):
    """Build a step_fn for a plain reference controller (PID / NMPC)."""
    def step_fn(obs):
        a, _ = controller.predict(obs)
        a = np.asarray(a, dtype=np.float32)
        return a, None, a, a, float("nan")   # used_agent=None -> takeover NaN
    return step_fn


def _stack(episodes: list[dict]) -> dict:
    lengths = {e["rewards"].shape[0] for e in episodes}
    t_min = min(lengths)
    if len(lengths) > 1:
        print(f"  [warn] episodes have differing lengths {sorted(lengths)}; "
              f"truncating all to {t_min} (a short episode biases this method's metrics)")
    return {k: np.stack([e[k][:t_min] for e in episodes], axis=0) for k in episodes[0]}


# ---------------------------------------------------------------------------
# Cheap in-memory evaluation (used by pretrain.py)
# ---------------------------------------------------------------------------

def evaluate_deploy(agent, scenario: Scenario | str, expert, *,
                    stage: DeploymentStage = DeploymentStage.AUTONOMOUS,
                    margin: float = 0.0, n_episodes: int = 5,
                    safety_fallback: bool = False) -> dict:
    """
    Roll the agent at a deployment stage for `n_episodes` and return summary
    stats (no disk). Used for periodic evaluation during pretraining.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    scenario = Scenario(scenario)
    cfg = SCENARIOS[str(scenario)]
    n_steps = cfg["n_steps"]
    constraint_spec = cfg.get("constraint_spec", [])
    env = make_env_for(str(scenario))
    controller = ShadowController(agent, expert, safety_fallback=safety_fallback)

    eps = [_record_episode(env, _agent_step_fn(controller, stage, margin),
                           controller.reset, seed=EVAL_SEED_OFFSET + s, n_steps=n_steps,
                           constraint_spec=constraint_spec)
           for s in range(n_episodes)]
    data = _stack(eps)
    returns = data["rewards"].sum(axis=1)
    any_v = (data["violations"] > 0).any(axis=-1) if constraint_spec else np.zeros_like(data["rewards"], dtype=bool)
    tk = data["takeover"]
    return {
        "return_mean": float(np.mean(returns)),
        "return_median": float(np.median(returns)),
        "viol_rate": float(any_v.mean()),
        "viol_max": float(data["violations"].max()) if constraint_spec else 0.0,
        "takeover_frac": float(np.nanmean(tk)) if np.isfinite(tk).any() else float("nan"),
        "divergence_mean": float(np.mean(data["divergence"])),
    }


# ---------------------------------------------------------------------------
# Full rollout serialisation
# ---------------------------------------------------------------------------

_ARRAY_SCHEMA = {
    "states": "[N, T, n_states]   env.state each step (physical)",
    "obs": "[N, T, obs_dim]    normalised observation",
    "actions": "[N, T, a_dim]      executed action (normalised)",
    "actions_agent": "[N, T, a_dim]      agent's proposed action",
    "actions_expert": "[N, T, a_dim]      expert's action",
    "rewards": "[N, T]             per-step reward",
    "takeover": "[N, T]             1=agent, 0=expert, NaN=reference",
    "q_gap": "[N, T]             Q(s,a_agent)-Q(s,a_expert) (NaN for references)",
    "divergence": "[N, T]             ||a_agent - a_expert||",
    "violations": "[N, T, n_con]      per-constraint violation magnitude (>=0)",
}


def run_rollouts(
    scenario: Scenario | str,
    model_specs: list[ModelSpec],
    *,
    stages: tuple[DeploymentStage, ...] = (
        DeploymentStage.SHADOW, DeploymentStage.AUTONOMOUS),
    shadow_margins: tuple[float, ...] = (0.0,),
    n_seeds: int = 20,
    use_oracle: bool = True,
    mpc_horizon: int = 20,
    output_dir: str = "outputs/rollouts",
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Roll every reference controller and every trained model (across the requested
    deployment stages) on the scenario for `n_seeds` seeds, serialising the raw
    rollouts to output_dir/<scenario>/<timestamp>/ as one .npz per method×stage +
    manifest.json. Method identity flows through typed MethodRecord objects.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    scenario = Scenario(scenario)
    cfg = SCENARIOS[str(scenario)]
    n_steps = cfg["n_steps"]
    constraint_spec = cfg.get("constraint_spec", [])

    env = make_env_for(str(scenario))
    expert, expert_kind = make_expert(scenario, mpc_horizon=mpc_horizon)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, str(scenario), timestamp)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Scenario : {scenario}  |  expert: {expert_kind.value}  "
          f"|  models: {len(model_specs)}  |  seeds: {n_seeds}")
    print(f"  Output   : {save_dir}")
    print(f"{'='*64}\n")

    method_records: list[MethodRecord] = []

    def _write(record: MethodRecord, step_fn_factory, reset_fn):
        eps = [_record_episode(env, step_fn_factory(), reset_fn, seed=EVAL_SEED_OFFSET + s,
                               n_steps=n_steps, constraint_spec=constraint_spec)
               for s in range(n_seeds)]
        data = _stack(eps)
        np.savez(os.path.join(save_dir, record.npz_file),
                 meta=np.array(json.dumps(record.to_json())), **data)
        mean_total = float(np.mean(data["rewards"].sum(axis=1)))
        note = ""
        if constraint_spec:
            vo = constraint_metrics(data["violations"], constraint_spec)["overall"]
            note = f"  |  viol {vo['rate']*100:4.1f}%"
        print(f"  {record.label:<34} mean return = {mean_total:9.1f}{note}  -> {record.npz_file}")
        method_records.append(record)

    # --- reference controllers ------------------------------------------
    pid = cfg["baseline_cls"]()
    _write(
        MethodRecord(role=MethodRole.PID, label="PID", npz_file="pid.npz", scenario=scenario),
        lambda: _reference_step_fn(pid), getattr(pid, "reset", lambda: None),
    )
    nmpc = None
    if use_oracle:
        try:
            nmpc = NMPCController(cfg, horizon=mpc_horizon)
            _write(
                MethodRecord(role=MethodRole.NMPC, label="NMPC", npz_file="nmpc.npz",
                             scenario=scenario),
                lambda: _reference_step_fn(nmpc), nmpc.reset,
            )
        except NotImplementedError as e:
            print(f"  [skip NMPC] {e}")

    # --- trained models across stages -----------------------------------
    for spec in model_specs:
        ckpt = torch.load(spec.checkpoint, weights_only=False, map_location=device)
        agent = get_agent(ckpt["type"]).load(ckpt, device=device)
        controller = ShadowController(agent, expert, safety_fallback=False)
        for stage in stages:
            margins = shadow_margins if stage is DeploymentStage.SHADOW else (0.0,)
            for margin in margins:
                mtag = f"_m{margin:g}" if stage is DeploymentStage.SHADOW else ""
                label = f"{agent.label} [{stage.value}{f' m={margin:g}' if mtag else ''}]"
                npz = f"{spec.run.artifact_stem}__{stage.value}{mtag}.npz"
                _write(
                    MethodRecord(role=MethodRole.MODEL, label=label, npz_file=npz,
                                 scenario=scenario, run=spec.run, stage=stage),
                    (lambda st=stage, mg=margin: _agent_step_fn(controller, st, mg)),
                    controller.reset,
                )

    manifest = {
        "scenario": str(scenario),
        "timestamp": timestamp,
        "expert_kind": expert_kind.value,
        "n_seeds": n_seeds,
        "n_steps": n_steps,
        "tsim": cfg["env_params"]["tsim"],
        "dt": cfg["env_params"]["tsim"] / cfg["env_params"]["N"],
        "plot_config": cfg["plot_config"],
        "setpoints": {k: list(map(float, v)) for k, v in cfg["env_params"]["SP"].items()},
        "constraints": constraint_spec,
        "stages": [s.value for s in stages],
        "shadow_margins": list(shadow_margins),
        "methods": [r.to_json() for r in method_records],
        "array_schema": _ARRAY_SCHEMA,
    }
    with open(os.path.join(save_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Rollouts + manifest.json written to {save_dir}")
    return save_dir
