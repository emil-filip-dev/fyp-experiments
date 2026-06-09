"""
evaluate.py
===========
Run trained models (and the reference controllers) on a PC-Gym scenario and
SERIALISE the raw per-step rollout outputs to disk. It does NOT plot or compute
summary metrics — a separate (to-be-built) plotting/metrics utility loads these
rollout files and produces figures + metric tables.

Each run records, per step: physical state (env.state), observation, executed
action, the agent's proposed action, the PID baseline's action, reward, and the
shadow takeover flag. N seeds per method are stacked and written as one .npz per
method under outputs/rollouts/<scenario>/<timestamp>/, plus a manifest.json
describing the scenario (timing, plot_config, setpoint schedule, method list).

Reference controllers always included:
  - PID         — the scenario's baseline controller acting on its own
  - NMPC Oracle — do-mpc + IPOPT nonlinear MPC on the env's exact dynamics
                  (best-achievable reference; disable with --no-oracle)

Also exports run_episode and evaluate (reused by trainer.py).

Usage
-----
  # Auto-discover all models under outputs/models/<scenario>/ and write rollouts
  .venv/Scripts/python evaluate.py --scenario cstr --n-seeds 20

  # Specific checkpoints, no oracle
  .venv/Scripts/python evaluate.py --scenario cstr --no-oracle \\
      --models outputs/models/cstr/ddpg/best.pt

Output: outputs/rollouts/<scenario>/<timestamp>/  (<method>.npz + manifest.json)
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from constraints import constraint_metrics, violation_magnitudes
from models import NMPCController, ShadowModels, get_shadow_model, get_standard_model
from scenarios import SCENARIOS, make_env_for
from util import resolve_device


# ---------------------------------------------------------------------------
# Episode runner + evaluator  (reused by trainer.py)
# ---------------------------------------------------------------------------

def run_episode(env, agent, baseline, training: bool,
                seed: int, n_steps: int, collect_states: bool = False):
    """
    Run one full episode. Compatible with any scenario and any ShadowDDPG-like
    instance (Pure* / Shadow* / SB3 adapters).

    Returns
    -------
    total_reward : float
    agent_flags  : list[bool] — True at each step the agent was in control
    states       : np.ndarray [T, state_dim] of physical env.state per step —
                   ONLY when collect_states=True (appended as a third element).
                   Used by the training loop to measure behaviour-time constraint
                   violations (the trajectory that actually ran on the "plant").
    """
    obs, _ = env.reset(seed=seed)
    baseline.reset()

    t_warmup = (
        int(np.random.uniform(0, agent.max_t_train_frac) * n_steps)
        if training else 0
    )

    total_reward = 0.0
    agent_flags: list[bool] = []
    states: list = []
    transitions: list = []
    done = False
    step = 0

    while not done:
        a_baseline, _ = baseline.predict(obs)

        force_baseline = step < t_warmup or (training and agent.total_steps < agent.warmup_steps)
        a_exec, used_agent, a_agent = agent.decide_action(
            obs, a_baseline, training=training, force_baseline=force_baseline
        )

        next_obs, reward, terminated, truncated, _ = env.step(a_exec)
        done = terminated or truncated

        agent_flags.append(used_agent)
        if collect_states:
            states.append(env.state.copy().astype(np.float32))
        transitions.append((obs, a_exec, reward, next_obs, done, a_agent, a_baseline))
        total_reward += reward
        obs   = next_obs
        step += 1
        if training:
            agent.total_steps += 1

    if training:
        for t in transitions:
            agent.store(*t)
        if agent.total_steps >= agent.warmup_steps:
            for _ in transitions:
                agent.update()

    if collect_states:
        return total_reward, agent_flags, np.asarray(states, dtype=np.float32)
    return total_reward, agent_flags


def evaluate(env, agent, baseline, n_seeds: int, n_steps: int) -> float:
    """Mean reward over n_seeds deterministic evaluation episodes."""
    rewards = [
        run_episode(env, agent, baseline, training=False, seed=s, n_steps=n_steps)[0]
        for s in range(n_seeds)
    ]
    return float(np.mean(rewards))


# ---------------------------------------------------------------------------
# Rollout recorder + writer  (the sole job of this module's CLI)
# ---------------------------------------------------------------------------

def _record_rollout(env, controller, scenario_baseline, seed: int,
                    n_steps: int, constraint_spec: list | None = None) -> dict:
    """
    Run one deterministic episode, recording the full per-step model outputs.

    `controller` may be an RL agent (has .decide_action) or a plain controller
    (PID / NMPC, .predict only). `scenario_baseline` is the scenario's PID, used
    both as the shadow switching reference and to record a consistent baseline
    action column for every method.

    Returns arrays of shape [T, ...] (T = episode length):
      states, obs, actions, actions_agent, actions_baseline, rewards, takeover,
      violations
    `takeover` is 1.0 (agent) / 0.0 (baseline) for RL agents, NaN otherwise.
    `violations` is [T, n_con] per-constraint violation magnitude (>=0; computed
    from the physical states against `constraint_spec`, [T, 0] if none defined).
    """
    is_agent = hasattr(controller, "decide_action")

    obs, _ = env.reset(seed=seed)
    if hasattr(controller, "reset"):
        controller.reset()
    scenario_baseline.reset()

    states, observations = [], []
    a_exec_l, a_agent_l, a_base_l = [], [], []
    rewards, takeover, q_gap = [], [], []
    done, step = False, 0

    while not done and step < n_steps:
        a_baseline, _ = scenario_baseline.predict(obs)
        a_baseline = np.asarray(a_baseline, dtype=np.float32)

        if is_agent:
            a_exec, used_agent, a_agent = controller.decide_action(
                obs, a_baseline, training=False
            )
            a_exec  = np.asarray(a_exec,  dtype=np.float32)
            # Agent-decision mode returns the augmented action (a^a, a^decision);
            # record only the physical action so the schema stays [.., action_dim].
            a_agent = np.asarray(a_agent, dtype=np.float32)[:a_exec.shape[-1]]
            tk = 1.0 if used_agent else 0.0
        else:
            a_exec, _ = controller.predict(obs)
            a_exec  = np.asarray(a_exec, dtype=np.float32)
            a_agent = a_exec.copy()
            tk = float("nan")          # takeover not applicable

        # Takeover advantage signal: Q(s,a_agent) - Q(s,a_baseline) under the
        # switching critic (q-value shadow only; NaN for agent-mode / non-RL).
        qg = controller.q_gap(obs, a_baseline) if hasattr(controller, "q_gap") else float("nan")

        observations.append(np.asarray(obs, dtype=np.float32))
        obs, reward, terminated, truncated, _ = env.step(a_exec)
        done = terminated or truncated

        states.append(env.state.copy().astype(np.float32))
        a_exec_l.append(a_exec)
        a_agent_l.append(a_agent)
        a_base_l.append(a_baseline)
        rewards.append(float(reward))
        takeover.append(tk)
        q_gap.append(np.float32(qg))
        step += 1

    states_arr = np.asarray(states, dtype=np.float32)
    # Per-step, per-constraint violation magnitude, computed directly from the
    # physical states (see constraints.py for why we avoid PC-Gym's cons_info).
    violations = violation_magnitudes(states_arr, constraint_spec or []).astype(np.float32)

    return {
        "states":           states_arr,
        "obs":              np.asarray(observations,  dtype=np.float32),
        "actions":          np.asarray(a_exec_l,      dtype=np.float32),
        "actions_agent":    np.asarray(a_agent_l,     dtype=np.float32),
        "actions_baseline": np.asarray(a_base_l,      dtype=np.float32),
        "rewards":          np.asarray(rewards,       dtype=np.float32),
        "takeover":         np.asarray(takeover,      dtype=np.float32),
        "q_gap":            np.asarray(q_gap,         dtype=np.float32),
        "violations":       violations,
    }


def _stack_episodes(episodes: list[dict]) -> dict:
    """Stack a list of per-episode [T,...] dicts into [N, T, ...] (truncate to min T)."""
    t_min = min(e["rewards"].shape[0] for e in episodes)
    return {k: np.stack([e[k][:t_min] for e in episodes], axis=0)
            for k in episodes[0]}


def run_rollouts(
    scenario:    str,
    model_paths: list[str],
    n_seeds:     int  = 10,
    use_oracle:  bool = True,
    mpc_horizon: int  = 20,
    output_dir:  str  = "outputs/rollouts",
    device:      torch.device = torch.device("cpu")
):
    """
    Run every method (PID, NMPC oracle, and the given/discovered models) on the
    scenario for `n_seeds` seeds, serialising the raw rollouts to
    output_dir/<scenario>/<timestamp>/ as one .npz per method + manifest.json.
    No plotting or metric computation — that is the plotting utility's job.
    """
    cfg     = SCENARIOS[scenario]
    n_steps = cfg["n_steps"]

    # (slug, label, controller). References first.
    entries: list[tuple[str, str, object]] = [("pid", "PID", cfg["baseline_cls"]())]
    if use_oracle:
        print(f"  Building NMPC oracle (do-mpc + IPOPT, horizon={mpc_horizon})...")
        try:
            entries.append(("nmpc_oracle", "NMPC Oracle", NMPCController(cfg, horizon=mpc_horizon)))
        except NotImplementedError as e:
            # e.g. delta-u scenarios (crystallization) — the oracle can't model them.
            print(f"  [skip oracle] {e}")

    for path in model_paths:
        slug = Path(path).parent.name
        ckpt = torch.load(path, weights_only=False, map_location=device)
        if isinstance(ckpt["type"], ShadowModels):
            model = get_shadow_model(ckpt["type"]).load(ckpt, device=device)
        else:
            model = get_standard_model(ckpt["type"]).load(ckpt, device=device)
        entries.append((slug, model.label, model))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = os.path.join(output_dir, scenario, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Scenario : {scenario}   |   methods: {len(entries)}   |   seeds: {n_seeds}")
    print(f"  Output   : {save_dir}")
    print(f"{'='*62}\n")

    env = make_env_for(scenario)
    scenario_baseline = cfg["baseline_cls"]()   # shadow switching + baseline column
    constraint_spec = cfg.get("constraint_spec", [])
    methods_meta: list[dict] = []

    for slug, label, controller in entries:
        episodes = [
            _record_rollout(env, controller, scenario_baseline, seed=s, n_steps=n_steps,
                            constraint_spec=constraint_spec)
            for s in range(n_seeds)
        ]
        data = _stack_episodes(episodes)
        meta = {"slug": slug, "label": label, "scenario": scenario,
                "n_seeds": n_seeds, "seeds": list(range(n_seeds))}
        npz_path = os.path.join(save_dir, f"{slug}.npz")
        np.savez(npz_path, meta=np.array(json.dumps(meta)), **data)

        mean_total = float(np.mean(data["rewards"].sum(axis=1)))
        viol_note = ""
        if constraint_spec:
            vo = constraint_metrics(data["violations"], constraint_spec)["overall"]
            viol_note = f"  |  viol {vo['rate']*100:4.1f}% ({vo['count']})"
        print(f"  {label:<28}  mean total reward = {mean_total:9.1f}{viol_note}   -> {slug}.npz")
        methods_meta.append({"slug": slug, "label": label, "file": f"{slug}.npz"})

    # Manifest: everything the plotting utility needs to interpret the .npz files.
    manifest = {
        "scenario":    scenario,
        "timestamp":   timestamp,
        "n_seeds":     n_seeds,
        "n_steps":     n_steps,
        "tsim":        cfg["env_params"]["tsim"],
        "dt":          cfg["env_params"]["tsim"] / cfg["env_params"]["N"],
        "plot_config": cfg["plot_config"],
        "setpoints":   {k: list(map(float, v)) for k, v in cfg["env_params"]["SP"].items()},
        "constraints": constraint_spec,
        "methods":     methods_meta,
        "array_schema": {
            "states":           "[N, T, n_physical_states]  env.state each step (physical)",
            "obs":              "[N, T, obs_dim]            normalised observation",
            "actions":          "[N, T, action_dim]         executed action (normalised)",
            "actions_agent":    "[N, T, action_dim]         agent's proposed action",
            "actions_baseline": "[N, T, action_dim]         PID baseline action",
            "rewards":          "[N, T]                     per-step reward",
            "takeover":         "[N, T]                     1=agent, 0=baseline, NaN=N/A",
            "q_gap":            "[N, T]                     Q(s,a_agent)-Q(s,a_baseline) under switching critic (q-value shadow only; NaN otherwise)",
            "violations":       "[N, T, n_con]              per-constraint violation magnitude (>=0; 0 if ok)",
        },
    }
    with open(os.path.join(save_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Rollouts + manifest.json written to {save_dir}")
    return save_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run models on a PC-Gym scenario and serialise per-step rollouts."
    )
    parser.add_argument(
        "--scenario", type=str, default="cstr", choices=list(SCENARIOS.keys()),
        help="PC-Gym scenario to run on",
    )
    parser.add_argument(
        "--models", type=str, nargs="*", default=[], metavar="PATH",
        help="Checkpoint paths to run (default: auto-discover under "
             "outputs/models/<scenario>/)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Number of seeds (episodes) recorded per method (default: 10)",
    )
    parser.add_argument(
        "--no-oracle", action="store_true",
        help="Skip the NMPC oracle (faster; included by default)",
    )
    parser.add_argument(
        "--mpc-horizon", type=int, default=20,
        help="NMPC oracle prediction horizon in steps (default: 20)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/rollouts",
        help="Root directory for rollout outputs (default: outputs/rollouts/)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force to use the CPU."
    )
    args = parser.parse_args()

    run_rollouts(
        scenario=args.scenario,
        model_paths=args.models,
        n_seeds=args.n_seeds,
        use_oracle=not args.no_oracle,
        mpc_horizon=args.mpc_horizon,
        output_dir=args.output_dir,
        device=resolve_device("cpu" if args.cpu else "gpu"),
    )


if __name__ == "__main__":
    main()
