"""
train_shadow.py
===============
Train a Shadow-Mode RL agent on a PC-Gym process-control environment.

The agent operates in shadow: at each step it proposes an action compared
against a baseline controller, and a switching criterion decides which to apply.

Supported scenarios: cstr, four_tank, multistage_extraction, crystallization
Supported models:    ddpg, td3
  (PPO is not supported — shadow mode requires a deterministic off-policy actor-critic)

Switching modes (--mode):
  qvalue  — execute agent action if Q(s, a_agent) > Q(s, a_baseline)
            TD3 uses min(Q1, Q2) for a more conservative comparison
  agent   — agent outputs an explicit control-authority probability;
            execute agent action if that probability > --eta-agent

Outputs
-------
  outputs/models/<scenario>/shadow_<model>_<mode>/           — Q-value or agent mode
  outputs/models/<scenario>/shadow_<model>_agent_reg<λ>/     — agent mode with regularisation

Usage
-----
  .venv/Scripts/python train_shadow.py --scenario cstr --model ddpg
  .venv/Scripts/python train_shadow.py --scenario cstr --model td3
  .venv/Scripts/python train_shadow.py --scenario cstr --model td3 --mode agent --lambda-reg 2.0
  .venv/Scripts/python train_shadow.py --scenario four_tank --model td3 --steps 300000
  .venv/Scripts/python train_shadow.py --scenario cstr --model ddpg --cpu
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from models import SHADOW_MODELS, create_shadow_agent, device_label, resolve_device
from scenarios import SCENARIOS, make_env_for
from evaluate import run_episode, evaluate, plot_training_curves


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    scenario:    str   = "cstr",
    model_type:  str   = "ddpg",
    mode:        str   = "qvalue",
    total_steps: int   = 200_000,
    seed:        int   = 42,
    lambda_reg:  float = 0.0,
    eta_agent:   float = 0.5,
    force_cpu:   bool  = False,
    output_dir:  str   = "outputs/models",
):
    cfg     = SCENARIOS[scenario]
    n_steps = cfg["n_steps"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    if mode == "agent" and lambda_reg > 0.0:
        run_label = f"shadow_{model_type}_agent_reg{lambda_reg}"
    else:
        run_label = f"shadow_{model_type}_{mode}"

    save_path = os.path.join(output_dir, scenario, run_label)
    os.makedirs(save_path, exist_ok=True)

    device   = resolve_device(force_cpu)
    env      = make_env_for(scenario)
    eval_env = make_env_for(scenario)
    baseline = cfg["baseline_cls"]()

    agent = create_shadow_agent(
        model_type=model_type,
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        mode=mode,
        lambda_reg=lambda_reg,
        eta_agent=eta_agent,
        device=device,
    )

    print(f"\n{'='*60}")
    print(f"  Scenario : {scenario}")
    print(f"  Model    : Shadow {model_type.upper()}  "
          f"(mode={mode}, λ={lambda_reg}, η={eta_agent})")
    print(f"  Device   : {device_label(device)}")
    print(f"  Steps    : {total_steps:,}  |  seed={seed}")
    print(f"  Output   : {save_path}")
    print(f"{'='*60}\n")

    episode        = 0
    best_reward    = -np.inf
    rewards_log:   list[tuple[int, float]] = []
    eval_log:      list[tuple[int, float]] = []
    agent_pct_log: list[float] = []
    next_eval      = 10_000

    bar = tqdm(total=total_steps, unit="step", dynamic_ncols=True,
               desc=f"Shadow {model_type.upper()}/{mode}")

    while agent.total_steps < total_steps:
        ep_seed     = episode + seed * 10_000
        steps_before = agent.total_steps

        reward, flags = run_episode(env, agent, baseline,
                                    training=True, seed=ep_seed, n_steps=n_steps)

        steps_taken = agent.total_steps - steps_before
        rewards_log.append((agent.total_steps, reward))
        agent_pct_log.append(float(np.mean(flags)) * 100)

        bar.update(steps_taken)

        # Periodic evaluation — loop so no boundary is skipped if episode spans multiple intervals
        while agent.total_steps >= next_eval:
            next_eval += 10_000
            eval_r = evaluate(eval_env, agent, baseline, n_seeds=5, n_steps=n_steps)
            eval_log.append((agent.total_steps, eval_r))
            if eval_r > best_reward:
                best_reward = eval_r
                agent.save(os.path.join(save_path, "best.pt"))
                tqdm.write(f"  [Save] step {agent.total_steps:,} — new best eval: {best_reward:.1f}")

        # Update progress bar postfix
        recent = float(np.mean([r for _, r in rewards_log[-50:]])) if rewards_log else 0.0
        postfix: dict[str, str] = {
            "reward": f"{reward:.0f}",
            "avg50":  f"{recent:.0f}",
            "agent%": f"{agent_pct_log[-1]:.1f}",
        }
        if eval_log:
            postfix["eval"] = f"{eval_log[-1][1]:.0f}"
        if best_reward > -np.inf:
            postfix["best"] = f"{best_reward:.0f}"
        bar.set_postfix(postfix)

        episode += 1

    bar.close()
    tqdm.write(f"\n  Training complete.  Best eval reward: {best_reward:.1f}")
    plot_training_curves(rewards_log, eval_log, save_path, scenario, run_label,
                         agent_pct_log=agent_pct_log)
    return agent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a Shadow-Mode RL agent on a PC-Gym environment."
    )
    parser.add_argument(
        "--scenario", type=str, default="cstr",
        choices=list(SCENARIOS.keys()),
        help="PC-Gym environment to train on",
    )
    parser.add_argument(
        "--model", type=str, default="ddpg",
        choices=SHADOW_MODELS,
        help=f"Shadow RL algorithm: {', '.join(SHADOW_MODELS)}  (default: ddpg)",
    )
    parser.add_argument(
        "--mode", type=str, default="qvalue", choices=["qvalue", "agent"],
        help="Switching mechanism: qvalue (recommended) or agent",
    )
    parser.add_argument(
        "--steps", type=int, default=200_000,
        help="Total training environment steps",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed",
    )
    parser.add_argument(
        "--lambda-reg", type=float, default=0.0,
        help="Regularisation strength λ — penalises distance from baseline (agent mode only)",
    )
    parser.add_argument(
        "--eta-agent", type=float, default=0.5,
        help="Control authority threshold η — agent acts when decision prob > η (agent mode only)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/models",
        help="Root directory for checkpoints and plots (default: outputs/models/)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU training even when a CUDA GPU is available",
    )
    args = parser.parse_args()

    train(
        scenario=args.scenario,
        model_type=args.model,
        mode=args.mode,
        total_steps=args.steps,
        seed=args.seed,
        lambda_reg=args.lambda_reg,
        eta_agent=args.eta_agent,
        force_cpu=args.cpu,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
