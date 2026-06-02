"""
evaluate.py
===========
Load, run, and compare trained RL models on a PC-Gym scenario.

Also exports run_episode, evaluate, and plot_training_curves for use by
train.py and train_shadow.py.

Standalone usage
----------------
  # Auto-discover all models saved under outputs/models/<scenario>/
  .venv/Scripts/python evaluate.py --scenario cstr

  # Evaluate specific checkpoints
  .venv/Scripts/python evaluate.py --scenario cstr \\
      --models outputs/models/cstr/ddpg/best.pt \\
               outputs/models/cstr/shadow_qvalue/best.pt

  # More evaluation seeds and a specific trajectory seed
  .venv/Scripts/python evaluate.py --scenario cstr --n-seeds 20 --eval-seed 7

Output is written to outputs/runs/<scenario>/<timestamp>/
"""

import argparse
import glob
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DDPG as SB3DDPG
from stable_baselines3 import PPO, TD3

from models import ShadowDDPG, ShadowTD3
from scenarios import SCENARIOS, make_env_for


# ---------------------------------------------------------------------------
# PureDDPG — agent always acts, defined here so evaluate.py and train.py
# are both self-contained without a shared dependency between them
# ---------------------------------------------------------------------------

class PureDDPG(ShadowDDPG):
    """Standard DDPG: always executes the agent's action, no switching."""

    def decide_action(self, obs, baseline_action, training=True):
        noise = (
            np.random.normal(0, self.noise_std, self.action_dim).astype(np.float32)
            if training else np.zeros(self.action_dim, dtype=np.float32)
        )
        action_agent, _ = self._get_agent_action(obs)
        action_noisy = np.clip(action_agent + noise, -1.0, 1.0)
        self.agent_takeover_count += 1
        return action_noisy, True, action_noisy


# ---------------------------------------------------------------------------
# SB3 adapter — wraps a Stable Baselines 3 model so it works in run_episode
# ---------------------------------------------------------------------------

class _SB3Adapter:
    """
    Thin wrapper making a loaded SB3 model compatible with run_episode().
    The agent always applies its own action (no shadow switching).
    """

    def __init__(self, sb3_model):
        self._model          = sb3_model
        self.total_steps     = 0
        self.warmup_steps    = 0
        self.max_t_train_frac = 0.0
        self.action_dim      = sb3_model.action_space.shape[0]

    def decide_action(self, obs, baseline_action, training=False):
        action, _ = self._model.predict(obs, deterministic=True)
        return action, True, action

    def store(self, *args):  pass
    def update(self):        pass
    def reset(self):         pass


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _label_from_path(checkpoint_path: str) -> str:
    """Human-readable label inferred from the checkpoint's parent directory."""
    name = Path(checkpoint_path).parent.name

    # Exact matches for standard SB3 models
    exact = {"ddpg": "DDPG", "td3": "TD3", "ppo": "PPO"}
    if name in exact:
        return exact[name]

    # Shadow model names: shadow_<model>_<mode>[_reg<λ>]
    if name.startswith("shadow_"):
        parts = name[len("shadow_"):]          # e.g. "td3_qvalue" or "ddpg_agent_reg2.0"
        model = parts.split("_")[0].upper()    # DDPG or TD3
        if "agent_reg" in parts:
            lam = parts.split("agent_reg")[-1]
            return f"Shadow {model} (Agent, λ={lam})"
        if "agent" in parts:
            return f"Shadow {model} (Agent)"
        return f"Shadow {model} (Q-value)"

    return name


def load_model(checkpoint_path: str, state_dim: int, action_dim: int):
    """
    Load a trained model from a checkpoint file.

    Accepts both:
      *.pt        — custom ShadowDDPG / ShadowTD3 / PureDDPG checkpoints
      *.zip       — SB3 (DDPG / TD3 / PPO) checkpoints saved by train.py

    The model type is inferred from the checkpoint's parent directory name.
    """
    name = Path(checkpoint_path).parent.name.lower()
    ext  = Path(checkpoint_path).suffix.lower()

    if ext == ".zip":
        if "ppo" in name:
            return _SB3Adapter(PPO.load(checkpoint_path))
        if "td3" in name:
            return _SB3Adapter(TD3.load(checkpoint_path))
        return _SB3Adapter(SB3DDPG.load(checkpoint_path))

    # .pt — custom shadow-mode or pure-DDPG checkpoint
    mode = "agent" if "agent" in name else "qvalue"

    if "td3" in name:
        agent = ShadowTD3(state_dim=state_dim, action_dim=action_dim, mode=mode)
    elif "shadow" not in name:
        agent = PureDDPG(state_dim=state_dim, action_dim=action_dim, mode="qvalue")
    else:
        agent = ShadowDDPG(state_dim=state_dim, action_dim=action_dim, mode=mode)

    agent.load(checkpoint_path)
    return agent


def discover_models(scenario: str, models_dir: str = "outputs/models") -> list[str]:
    """
    Return all checkpoint paths found under models_dir/<scenario>/*/,
    accepting both best.pt (custom) and best_model.zip (SB3 EvalCallback).
    """
    paths = []
    for pattern in ("best.pt", "best_model.zip"):
        paths.extend(glob.glob(os.path.join(models_dir, scenario, "*", pattern)))
    return sorted(paths)


# ---------------------------------------------------------------------------
# Episode runner  (also used by train.py and train_shadow.py)
# ---------------------------------------------------------------------------

def run_episode(env, agent, baseline, training: bool,
                seed: int, n_steps: int) -> tuple[float, list[bool]]:
    """
    Run one full episode.  Compatible with any scenario and any ShadowDDPG
    instance (including PureDDPG subclass).

    Returns
    -------
    total_reward : float
    agent_flags  : list[bool] — True at each step the agent was in control
    """
    obs, _ = env.reset(seed=seed)
    baseline.reset()

    t_warmup = (
        int(np.random.uniform(0, agent.max_t_train_frac) * n_steps)
        if training else 0
    )

    total_reward = 0.0
    agent_flags: list[bool] = []
    transitions: list = []
    done = False
    step = 0

    while not done:
        a_baseline, _ = baseline.predict(obs)

        if step < t_warmup or (training and agent.total_steps < agent.warmup_steps):
            a_exec, a_agent, used_agent = a_baseline, a_baseline.copy(), False
        else:
            a_exec, used_agent, a_agent = agent.decide_action(
                obs, a_baseline, training=training
            )

        next_obs, reward, terminated, truncated, _ = env.step(a_exec)
        done = terminated or truncated

        agent_flags.append(used_agent)
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

    return total_reward, agent_flags


# ---------------------------------------------------------------------------
# Evaluator  (also used by train.py and train_shadow.py)
# ---------------------------------------------------------------------------

def evaluate(env, agent, baseline, n_seeds: int, n_steps: int) -> float:
    """Mean reward over n_seeds deterministic evaluation episodes."""
    rewards = [
        run_episode(env, agent, baseline, training=False, seed=s, n_steps=n_steps)[0]
        for s in range(n_seeds)
    ]
    return float(np.mean(rewards))


# ---------------------------------------------------------------------------
# Training-curve plotter  (used by train.py and train_shadow.py)
# ---------------------------------------------------------------------------

def plot_training_curves(
    rewards_log:   list[tuple[int, float]],
    eval_log:      list[tuple[int, float]],
    save_path:     str,
    scenario:      str,
    run_label:     str,
    agent_pct_log: list[float] | None = None,
):
    """
    Save a training-curve figure to save_path/training_curves.png.
    Renders 2 panels for standard DDPG or 3 panels (adds agent takeover %)
    when agent_pct_log is provided (shadow mode).
    """
    n_panels = 3 if agent_pct_log is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), sharex=False)

    steps, rews = zip(*rewards_log) if rewards_log else ([0], [0])
    window = 50

    ax = axes[0]
    ax.plot(steps, rews, alpha=0.25, color="steelblue", linewidth=0.7)
    if len(rews) >= window:
        ma = np.convolve(rews, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], ma, color="steelblue", linewidth=2,
                label=f"MA({window})")
    ax.set_ylabel("Episode Reward")
    ax.set_title(f"{scenario} / {run_label} — Training Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if eval_log:
        es, er = zip(*eval_log)
        ax.plot(es, er, "o-", color="green", linewidth=2, markersize=4)
    ax.set_ylabel("Eval Reward (mean 5 seeds)")
    ax.set_title("Evaluation Reward")
    ax.grid(True, alpha=0.3)

    if agent_pct_log is not None:
        ax = axes[2]
        ax.plot(agent_pct_log, color="red", alpha=0.4, linewidth=0.7)
        if len(agent_pct_log) >= window:
            ma = np.convolve(agent_pct_log, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(agent_pct_log)), ma, color="red", linewidth=2)
        ax.set_ylabel("Agent Control (%)")
        ax.set_xlabel("Episode")
        ax.set_title("Agent Takeover Fraction")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_path, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [Plot] {out}")


# ---------------------------------------------------------------------------
# Trajectory recorder  (evaluation-only, captures physical env.state)
# ---------------------------------------------------------------------------

def _record_trajectory(env, agent_or_baseline, baseline, seed: int,
                        n_steps: int) -> dict:
    """
    Run one deterministic episode, recording the physical state at every step.

    If agent_or_baseline is a baseline controller (has .predict but no
    .decide_action), it is run directly.  Otherwise it is run as an RL agent
    in evaluation mode with the baseline alongside for switching.

    Returns a dict with keys:
      states   : np.ndarray [T, n_physical_states]  — env.state each step
      rewards  : np.ndarray [T]
      total    : float
    """
    is_agent = hasattr(agent_or_baseline, "decide_action")

    obs, _ = env.reset(seed=seed)
    if hasattr(agent_or_baseline, "reset"):
        agent_or_baseline.reset()
    baseline.reset()

    states_list:  list = []
    rewards_list: list = []
    done = False
    step = 0

    while not done and step < n_steps:
        if is_agent:
            a_baseline, _ = baseline.predict(obs)
            a_exec, _, _  = agent_or_baseline.decide_action(
                obs, a_baseline, training=False
            )
        else:
            a_exec, _ = agent_or_baseline.predict(obs)

        obs, reward, terminated, truncated, _ = env.step(a_exec)
        done = terminated or truncated
        states_list.append(env.state.copy())
        rewards_list.append(reward)
        step += 1

    states  = np.array(states_list)
    rewards = np.array(rewards_list)
    return {"states": states, "rewards": rewards, "total": float(rewards.sum())}


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    scenario:    str,
    model_paths: list[str],
    n_seeds:     int = 10,
    eval_seed:   int = 42,
    models_dir:  str = "outputs/models",
    output_dir:  str = "outputs/runs",
):
    """
    Evaluate a list of models (plus the scenario baseline) on the given scenario,
    then write a comparison plot and results summary to output_dir/<scenario>/<timestamp>/.

    Parameters
    ----------
    scenario    : key in SCENARIOS
    model_paths : list of checkpoint paths; pass [] to auto-discover
    n_seeds     : number of seeds for reward statistics
    eval_seed   : seed used for the trajectory plot
    models_dir  : root directory where model checkpoints are stored
    output_dir  : root directory for run outputs
    """
    cfg     = SCENARIOS[scenario]
    n_steps = cfg["n_steps"]
    env_fn  = lambda: make_env_for(scenario)

    if not model_paths:
        model_paths = discover_models(scenario, models_dir)
        if not model_paths:
            print(f"  No models found under {models_dir}/{scenario}/. "
                  "Train some models first.")
            return

    # Build list of (label, agent-or-baseline) entries; baseline is always first
    entries: list[tuple[str, object]] = [("Baseline", cfg["baseline_cls"]())]
    for path in model_paths:
        label = _label_from_path(path)
        agent = load_model(path, cfg["state_dim"], cfg["action_dim"])
        entries.append((label, agent))

    print(f"\n{'='*62}")
    print(f"  Scenario  : {scenario}")
    print(f"  Models    : {len(entries) - 1} loaded + baseline")
    print(f"  Eval seeds: {n_seeds}  |  trajectory seed: {eval_seed}")
    print(f"{'='*62}\n")

    # Multi-seed reward statistics
    eval_env  = env_fn()
    baseline  = cfg["baseline_cls"]()
    stats: dict[str, dict] = {}

    for label, agent in entries:
        rewards = []
        for s in range(n_seeds):
            if label == "Baseline":
                obs, _ = eval_env.reset(seed=s)
                agent.reset()
                total = 0.0
                done  = False
                while not done:
                    a, _ = agent.predict(obs)
                    obs, r, terminated, truncated, _ = eval_env.step(a)
                    done   = terminated or truncated
                    total += r
                rewards.append(total)
            else:
                r, _ = run_episode(eval_env, agent, baseline,
                                   training=False, seed=s, n_steps=n_steps)
                rewards.append(r)
        arr = np.array(rewards)
        stats[label] = {"mean": float(arr.mean()), "std": float(arr.std()),
                        "rewards": arr}
        print(f"  {label:<30}  mean={arr.mean():9.1f}  std={arr.std():7.1f}")

    # Trajectory for the plot seed
    traj_env = env_fn()
    trajectories: dict[str, dict] = {}
    for label, agent in entries:
        trajectories[label] = _record_trajectory(
            traj_env, agent, baseline, seed=eval_seed, n_steps=n_steps
        )

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = os.path.join(output_dir, scenario, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    _write_results(stats, scenario, n_seeds, eval_seed, save_dir)
    _plot_comparison(trajectories, stats, cfg, scenario, eval_seed, save_dir)

    print(f"\n  Results written to {save_dir}")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_results(stats: dict, scenario: str, n_seeds: int,
                   eval_seed: int, save_dir: str):
    baseline_mean = stats.get("Baseline", {}).get("mean", 0.0)
    lines = [
        f"Scenario    : {scenario}",
        f"Eval seeds  : {n_seeds}",
        f"Traj. seed  : {eval_seed}",
        f"Timestamp   : {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"{'Model':<32} {'Mean Reward':>12} {'Std':>8} {'vs Baseline':>12}",
        "-" * 68,
    ]
    for label, s in stats.items():
        delta = s["mean"] - baseline_mean if label != "Baseline" else 0.0
        delta_str = f"{delta:+.1f}" if label != "Baseline" else "—"
        lines.append(
            f"{label:<32} {s['mean']:>12.1f} {s['std']:>8.1f} {delta_str:>12}"
        )
    text = "\n".join(lines) + "\n"

    out = os.path.join(save_dir, "results.txt")
    with open(out, "w") as f:
        f.write(text)
    print(f"  [Results] {out}")
    print()
    print(text)


def _plot_comparison(trajectories: dict, stats: dict, cfg: dict,
                     scenario: str, eval_seed: int, save_dir: str):
    plot_config = cfg["plot_config"]
    n_outputs   = len(plot_config)
    tsim        = cfg["env_params"]["tsim"]
    n_steps     = cfg["n_steps"]
    time_axis   = np.linspace(0, tsim, n_steps)

    # Generate one colour per entry; tab10 cycles cleanly for up to 10,
    # beyond that we tile it so no entry is ever silently dropped.
    n_entries = len(trajectories)
    cmap      = plt.get_cmap("tab10")
    colours   = [cmap(i % 10) for i in range(n_entries)]

    fig, axes = plt.subplots(n_outputs + 1, 1,
                             figsize=(13, 4 * (n_outputs + 1)), sharex=False)
    if n_outputs + 1 == 1:
        axes = [axes]

    # --- Trajectory panels ---
    for panel_idx, pc in enumerate(plot_config):
        ax      = axes[panel_idx]
        si      = pc["state_idx"]
        sp_i    = pc["sp_idx"]
        label_y = f"{pc['label']} ({pc['unit']})"

        # Setpoint from baseline trajectory (all models see the same setpoint)
        sp_traj = trajectories["Baseline"]["states"][:, sp_i]
        t_sp    = time_axis[:len(sp_traj)]
        ax.plot(t_sp, sp_traj, "k--", linewidth=1.8, label="Setpoint", zorder=5)

        for colour, (name, traj) in zip(colours, trajectories.items()):
            y = traj["states"][:, si]
            t = time_axis[:len(y)]
            ax.plot(t, y, linewidth=1.8, color=colour,
                    label=f"{name}  (reward {traj['total']:.0f})")

        ax.set_ylabel(label_y, fontsize=11)
        ax.set_title(f"{scenario} — {pc['label']} tracking  (seed={eval_seed})",
                     fontsize=11)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    # --- Reward bar chart ---
    ax      = axes[n_outputs]
    labels  = list(stats.keys())
    means   = [stats[l]["mean"] for l in labels]
    stds    = [stats[l]["std"]  for l in labels]

    bar_colours = [cmap(i % 10) for i in range(len(labels))]
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=bar_colours, alpha=0.85, edgecolor="black")
    for bar, mean in zip(bars, means):
        # For negative bars, place label below the bar bottom; for positive, above top
        y = bar.get_height()
        offset = abs(y) * 0.02 if y >= 0 else -abs(y) * 0.02
        va = "bottom" if y >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y + offset,
                f"{mean:.0f}", ha="center", va=va, fontsize=9)

    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title(f"Reward Comparison ({len(stats[labels[0]]['rewards'])} seeds)",
                 fontsize=11)
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(save_dir, "comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [Plot] {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load and compare trained RL models on a PC-Gym scenario."
    )
    parser.add_argument(
        "--scenario", type=str, default="cstr",
        choices=list(SCENARIOS.keys()),
        help="PC-Gym scenario to evaluate on",
    )
    parser.add_argument(
        "--models", type=str, nargs="*", default=[],
        metavar="PATH",
        help="Checkpoint paths to evaluate (default: auto-discover under "
             "outputs/models/<scenario>/)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Number of seeds for reward statistics (default: 10)",
    )
    parser.add_argument(
        "--eval-seed", type=int, default=42,
        help="Seed used for the trajectory plot (default: 42)",
    )
    parser.add_argument(
        "--models-dir", type=str, default="outputs/models",
        help="Root directory where model checkpoints are stored",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/runs",
        help="Root directory for run outputs (default: outputs/runs/)",
    )
    args = parser.parse_args()

    run_comparison(
        scenario=args.scenario,
        model_paths=args.models,
        n_seeds=args.n_seeds,
        eval_seed=args.eval_seed,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
