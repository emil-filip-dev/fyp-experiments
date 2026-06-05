"""
train.py
========
Train an RL agent on a PC-Gym process-control environment — standard or
shadow-mode, custom core or Stable-Baselines3, all from one entry point.

Mode (--shadow):
  standard (default) — the agent always executes its own action.
  shadow             — at each step the agent's action is compared against a PID
                       baseline and a switching criterion decides which to apply.
  Standard and shadow share the SAME core/hyperparameters, so they form a clean
  ablation isolating the effect of shadow switching.

Backend (--backend):
  custom (default) — the project's DDPG/TD3 core. Labelled "DDPG"/"TD3"
                     (standard) or "Shadow DDPG"/"Shadow TD3" (shadow).
  sb3              — Stable-Baselines3 DDPG/TD3. Labelled "SB3 DDPG" /
                     "Shadow SB3 DDPG". SB3 shadow supports --mode qvalue only.

Shadow switching modes (--mode, custom backend supports both):
  qvalue  — execute agent action if Q(s, a_agent) > Q(s, a_baseline)
  agent   — agent outputs a control-authority probability; act if > --eta-agent
            (optional L1 regularisation toward the baseline via --lambda-reg)

Supported scenarios: cstr, four_tank, multistage_extraction, crystallization
Supported models:    ddpg, td3   (PPO unsupported — shadow needs a deterministic
                                   off-policy actor-critic)

Outputs  (outputs/models/<scenario>/<run_label>/ — best.pt custom, best_model.zip sb3)
  standard custom : <model>/                    e.g. ddpg/
  shadow   custom : shadow_<model>_<mode>/       (+ _reg<λ> for agent mode w/ reg)
  standard sb3    : sb3_<model>/
  shadow   sb3    : shadow_sb3_<model>/

Usage
-----
  .venv/Scripts/python train.py --scenario cstr --model ddpg                       # DDPG
  .venv/Scripts/python train.py --scenario cstr --model ddpg --shadow              # Shadow DDPG
  .venv/Scripts/python train.py --scenario cstr --model ddpg --backend sb3         # SB3 DDPG
  .venv/Scripts/python train.py --scenario cstr --model ddpg --shadow --backend sb3  # Shadow SB3 DDPG
  .venv/Scripts/python train.py --scenario cstr --model td3 --shadow --mode agent --lambda-reg 2.0
"""

import argparse

import numpy as np
import torch

from models import (
    PURE_MODELS,
    create_pure_agent,
    create_shadow_agent,
    device_label,
    resolve_device,
)
from scenarios import SCENARIOS, make_env_for
from trainer import configure_utf8_output, train_custom


def _print_header(scenario, model_desc, device, total_steps, seed):
    print(f"\n{'='*60}")
    print(f"  Scenario : {scenario}")
    print(f"  Model    : {model_desc}")
    print(f"  Device   : {device_label(device)}")
    print(f"  Steps    : {total_steps:,}  |  seed={seed}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    scenario:    str   = "cstr",
    model_type:  str   = "ddpg",
    shadow:      bool  = False,
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

    # -- SB3 backend ---------------------------------------------------------
    if backend == "sb3":
        from stable_baselines3.common.monitor import Monitor
        from trainer import train_sb3

        env = Monitor(make_env_for(scenario))

        if shadow:
            if mode != "qvalue":
                raise ValueError("SB3 shadow backend supports only --mode qvalue "
                                 "(agent-decision mode needs a custom policy head).")
            from models import create_shadow_sb3_agent
            model      = create_shadow_sb3_agent(model_type, env, cfg["baseline_cls"](),
                                                 seed=seed, device=device)
            run_label  = f"shadow_sb3_{model_type}"
            model_desc = f"Shadow SB3 {model_type.upper()}  (mode=qvalue)"
            meta_extra = {"mode": "qvalue", "backend": "sb3"}
        else:
            from models import create_sb3_agent
            model      = create_sb3_agent(model_type, env, seed=seed, device=device)
            run_label  = f"sb3_{model_type}"
            model_desc = f"SB3 {model_type.upper()} (standard, no shadow)"
            meta_extra = {"backend": "sb3"}

        _print_header(scenario, model_desc, device, total_steps, seed)
        return train_sb3(
            model, scenario=scenario, model_type=model_type, run_label=run_label,
            total_steps=total_steps, seed=seed, output_dir=output_dir,
            is_shadow=shadow, eval_freq=eval_freq, meta_extra=meta_extra,
        )

    # -- Custom core ---------------------------------------------------------
    if shadow:
        agent = create_shadow_agent(
            model_type=model_type, state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"], mode=mode, lambda_reg=lambda_reg,
            eta_agent=eta_agent, device=device,
        )
        if mode == "agent" and lambda_reg > 0.0:
            run_label = f"shadow_{model_type}_agent_reg{lambda_reg}"
        else:
            run_label = f"shadow_{model_type}_{mode}"
        model_desc = f"Shadow {model_type.upper()}  (mode={mode}, lambda={lambda_reg}, eta={eta_agent})"
        meta_extra = {"mode": mode, "lambda_reg": lambda_reg, "eta_agent": eta_agent}
    else:
        agent = create_pure_agent(
            model_type=model_type, state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"], device=device,
        )
        run_label  = model_type            # "ddpg" / "td3" -> labelled DDPG / TD3
        model_desc = f"{model_type.upper()} (standard, no shadow)"
        meta_extra = None

    _print_header(scenario, model_desc, device, total_steps, seed)
    return train_custom(
        agent, scenario=scenario, model_type=model_type, run_label=run_label,
        total_steps=total_steps, seed=seed, output_dir=output_dir,
        is_shadow=shadow, eval_freq=eval_freq, meta_extra=meta_extra,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Train a standard or shadow-mode RL agent on a PC-Gym environment."
    )
    parser.add_argument(
        "--scenario", type=str, default="cstr",
        choices=list(SCENARIOS.keys()),
        help="PC-Gym environment to train on",
    )
    parser.add_argument(
        "--model", type=str, default="ddpg",
        choices=PURE_MODELS,
        help=f"RL algorithm: {', '.join(PURE_MODELS)}  (default: ddpg)",
    )
    parser.add_argument(
        "--shadow", action="store_true",
        help="Train in shadow mode (agent action gated against the PID baseline). "
             "Default: standard (agent always acts).",
    )
    parser.add_argument(
        "--mode", type=str, default="qvalue", choices=["qvalue", "agent"],
        help="Shadow switching mechanism: qvalue (recommended) or agent (--shadow only)",
    )
    parser.add_argument(
        "--backend", type=str, default="custom", choices=["custom", "sb3"],
        help="custom = project's DDPG/TD3 core; sb3 = Stable-Baselines3. Default: custom",
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
        help="Regularisation strength (lambda) — penalises distance from baseline "
             "(shadow agent mode only)",
    )
    parser.add_argument(
        "--eta-agent", type=float, default=0.5,
        help="Control authority threshold (eta) — agent acts when decision prob > eta "
             "(shadow agent mode only)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=1_000,
        help="Evaluate every N env steps (default: 1000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/models",
        help="Root directory for checkpoints (default: outputs/models/)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU training even when a CUDA GPU is available",
    )
    args = parser.parse_args()

    train(
        scenario=args.scenario,
        model_type=args.model,
        shadow=args.shadow,
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
