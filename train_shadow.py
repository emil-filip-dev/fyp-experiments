"""
train_shadow.py
===============
Train a Shadow-Mode RL agent on a PC-Gym process-control environment.

The agent operates in shadow: at each step it proposes an action compared
against a baseline controller, and a switching criterion decides which to apply.
It shares its entire core (network, hyperparameters, PID-assisted exploration)
with the standard agents in train.py — the ONLY difference is the switching, so
the two scripts form a clean ablation pair.

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

import numpy as np
import torch

from models import SHADOW_MODELS, create_shadow_agent, device_label, resolve_device
from scenarios import SCENARIOS, make_env_for
from trainer import configure_utf8_output, train_custom


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    scenario:    str   = "cstr",
    model_type:  str   = "ddpg",
    mode:        str   = "qvalue",
    backend:     str   = "custom",
    total_steps: int   = 200_000,
    seed:        int   = 42,
    lambda_reg:  float = 0.0,
    eta_agent:   float = 0.5,
    eval_freq:   int   = 1_000,
    force_cpu:   bool  = False,
    output_dir:  str   = "outputs/models",
):
    cfg = SCENARIOS[scenario]

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(force_cpu)

    if backend == "sb3":
        if mode != "qvalue":
            raise ValueError("SB3 shadow backend supports only --mode qvalue "
                             "(agent-decision mode needs a custom policy head).")
        from stable_baselines3.common.monitor import Monitor

        from models import create_shadow_sb3_agent
        from trainer import train_sb3

        env       = Monitor(make_env_for(scenario))
        baseline  = cfg["baseline_cls"]()
        model     = create_shadow_sb3_agent(model_type, env, baseline,
                                            seed=seed, device=device)
        run_label = f"shadow_sb3_{model_type}"

        print(f"\n{'='*60}")
        print(f"  Scenario : {scenario}")
        print(f"  Model    : Shadow SB3 {model_type.upper()}  (mode=qvalue)")
        print(f"  Device   : {device_label(device)}")
        print(f"  Steps    : {total_steps:,}  |  seed={seed}")
        print(f"{'='*60}\n")

        return train_sb3(
            model, scenario=scenario, model_type=model_type, run_label=run_label,
            total_steps=total_steps, seed=seed, output_dir=output_dir,
            is_shadow=True, eval_freq=eval_freq,
            meta_extra={"mode": "qvalue", "backend": "sb3"},
        )

    if mode == "agent" and lambda_reg > 0.0:
        run_label = f"shadow_{model_type}_agent_reg{lambda_reg}"
    else:
        run_label = f"shadow_{model_type}_{mode}"

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
          f"(mode={mode}, lambda={lambda_reg}, eta={eta_agent})")
    print(f"  Device   : {device_label(device)}")
    print(f"  Steps    : {total_steps:,}  |  seed={seed}")
    print(f"{'='*60}\n")

    return train_custom(
        agent,
        scenario=scenario,
        model_type=model_type,
        run_label=run_label,
        total_steps=total_steps,
        seed=seed,
        output_dir=output_dir,
        is_shadow=True,
        eval_freq=eval_freq,
        meta_extra={"mode": mode, "lambda_reg": lambda_reg, "eta_agent": eta_agent},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    configure_utf8_output()
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
        "--backend", type=str, default="custom", choices=["custom", "sb3"],
        help="custom = project's core (labelled Shadow DDPG/TD3); sb3 = "
             "Stable-Baselines3 (labelled Shadow SB3 DDPG/TD3, qvalue only). "
             "Default: custom",
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
        help="Regularisation strength (lambda) — penalises distance from baseline (agent mode only)",
    )
    parser.add_argument(
        "--eta-agent", type=float, default=0.5,
        help="Control authority threshold (eta) — agent acts when decision prob > eta (agent mode only)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=1_000,
        help="Evaluate (and snapshot metrics) every N env steps (default: 1000)",
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
        backend=args.backend,
        total_steps=args.steps,
        seed=args.seed,
        lambda_reg=args.lambda_reg,
        eta_agent=args.eta_agent,
        eval_freq=args.eval_freq,
        force_cpu=args.cpu,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
