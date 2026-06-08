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

Shadow switching modes (--mode, custom backend supports both):
  qvalue  — execute agent action if Q(s, a_agent) > Q(s, a_baseline)
  agent   — agent outputs a control-authority probability; act if > --eta-agent
            (optional L1 regularisation toward the baseline via --lambda-reg)

Supported scenarios: cstr, four_tank, multistage_extraction, crystallization
Supported models:    ddpg, td3

Outputs  (outputs/models/<scenario>/<run_label>/ — best.pt custom, best_model.zip sb3)
  standard custom : <model>/                    e.g. ddpg/
  shadow   custom : shadow_<model>_<mode>/       (+ _reg<λ> for agent mode w/ reg)

Usage
-----
  .venv/Scripts/python train.py --scenario cstr --model ddpg
  .venv/Scripts/python train.py --scenario cstr --model ddpg --shadow
  .venv/Scripts/python train.py --scenario cstr --model td3 --shadow --mode agent --lambda-reg 2.0
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from evaluate import run_episode, evaluate
from models import StandardModels, get_shadow_model, get_standard_model, SwitchingMode, TD3SwitchCritic
from scenarios import SCENARIOS, make_env_for
from util import configure_utf8_output, device_label, resolve_device


def _print_train_header(scenario, model_desc, device, total_steps, seed):
    print(f"\n{'='*60}")
    print(f"  Scenario : {scenario}")
    print(f"  Model    : {model_desc}")
    print(f"  Device   : {device_label(device)}")
    print(f"  Steps    : {total_steps:,}  |  seed={seed}")
    print(f"{'='*60}\n")

def train(
    scenario:    str   = "cstr",
    model_type:  str   = "ddpg",
    shadow:      bool  = False,
    mode:        str   = "qvalue",
    total_steps: int   = 200_000,
    seed:        int   = 42,
    lambda_reg:  float = 0.0,
    eta_agent:   float = 0.5,
    switch_critic: str = "q1",
    eval_freq:   int   = 1_000,
    device:      str   = "cpu",
    output_dir:  str   = "outputs/models",
):
    cfg = SCENARIOS[scenario]

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(device)

    state_dim = cfg["state_dim"]
    action_dim = cfg["action_dim"]

    if shadow:
        model_class = get_shadow_model(model_type)
        mode = SwitchingMode(mode)
        # switch_critic only applies to ShadowTD3's Q-value switching.
        is_td3_qvalue = model_type == "td3" and mode is SwitchingMode.Q_VALUE
        extra_kwargs = {"switch_critic": switch_critic} if is_td3_qvalue else {}
        agent = model_class(
            state_dim=state_dim, action_dim=action_dim, switching_mode=mode, eta_agent=eta_agent, lambda_reg=lambda_reg,
            device=device, **extra_kwargs
        )
        if mode is SwitchingMode.AGENT and lambda_reg > 0.0:
            run_label = f"shadow_{model_type}_agent_reg{lambda_reg}"
        elif is_td3_qvalue:
            run_label = f"shadow_{model_type}_{mode.value}_{switch_critic}"
        else:
            run_label = f"shadow_{model_type}_{mode.value}"

        model_desc = f"Shadow {model_type.upper()}  (mode={mode.value}, lambda={lambda_reg}, eta={eta_agent}"
        model_desc += f", switch_critic={switch_critic})" if is_td3_qvalue else ")"
        meta_extra = {"switching_mode": mode.value, "lambda_reg": lambda_reg, "eta_agent": eta_agent}
    else:
        model_class = get_standard_model(model_type)
        agent = model_class(
            state_dim=state_dim, action_dim=action_dim, lambda_reg=lambda_reg, device=device
        )
        run_label  = model_type
        model_desc = f"{model_type.upper()} (standard, no shadow)"
        meta_extra = None

    _print_train_header(scenario, model_desc, device, total_steps, seed)
    return train_model(
        agent, scenario=scenario, run_label=run_label,
        total_steps=total_steps, seed=seed, output_dir=output_dir,
        is_shadow=shadow, eval_freq=eval_freq, meta_extra=meta_extra,
    )


def train_model(
    agent,
    *,
    scenario:    str,
    run_label:   str,
    total_steps: int,
    seed:        int,
    output_dir:  str  = "outputs/models",
    is_shadow:   bool = False,
    eval_freq:   int  = 1_000,
    n_eval:      int  = 5,
    meta_extra:  dict | None = None
):
    """
    Run the shared episode-based training loop for any ShadowDDPG-like agent
    (PureDDPG / PureTD3 / ShadowDDPG / ShadowTD3). Saves best.pt whenever the
    periodic evaluation reward improves.
    """
    cfg       = SCENARIOS[scenario]
    n_steps   = cfg["n_steps"]
    save_path = os.path.join(output_dir, scenario, run_label)
    os.makedirs(save_path, exist_ok=True)

    env      = make_env_for(scenario)
    eval_env = make_env_for(scenario)
    baseline = cfg["baseline_cls"]()

    episode        = 0
    best_reward    = -np.inf
    recent_rewards: list[float] = []
    next_eval      = eval_freq

    bar = tqdm(total=total_steps, unit="step", dynamic_ncols=True, desc=run_label)

    while agent.total_steps < total_steps:
        ep_seed      = episode + seed * 10_000
        steps_before = agent.total_steps

        reward, flags = run_episode(env, agent, baseline,
                                    training=True, seed=ep_seed, n_steps=n_steps)
        recent_rewards.append(reward)
        bar.update(agent.total_steps - steps_before)

        # Periodic evaluation - loop so no boundary is skipped
        while agent.total_steps >= next_eval:
            next_eval += eval_freq
            eval_r = evaluate(eval_env, agent, baseline, n_seeds=n_eval, n_steps=n_steps)
            if eval_r > best_reward:
                best_reward = eval_r
                agent.save(os.path.join(save_path, "best.pt"))
                tqdm.write(f"  [Save] step {agent.total_steps:,} - new best eval: {best_reward:.1f}")

        recent = float(np.mean(recent_rewards[-50:])) if recent_rewards else 0.0
        postfix: dict[str, str] = {"reward": f"{reward:.0f}", "avg50": f"{recent:.0f}"}
        if is_shadow:
            postfix["agent%"] = f"{float(np.mean(flags)) * 100:.1f}"
        if best_reward > -np.inf:
            postfix["best"] = f"{best_reward:.0f}"
        bar.set_postfix(postfix)

        episode += 1

    bar.close()
    tqdm.write(f"\n  Training complete.  Best eval reward: {best_reward:.1f}")
    tqdm.write(f"  Saved: {os.path.join(save_path, 'best.pt')}")
    return agent


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
        choices=[m.value for m in StandardModels],
        help=f"RL algorithm: {', '.join(StandardModels)}  (default: ddpg)",
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
        "--switch-critic", type=str, default=TD3SwitchCritic.Q1.value,
        choices=[c.value for c in TD3SwitchCritic],
        help="Twin-critic estimate for Q-value switching (Shadow TD3 + qvalue only): "
             "q1 (actor-consistent, default) or qmin (conservative)",
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
        "--device", default="cpu", choices=["cpu", "gpu"],
        help="Choose device: either CPU or Cuda GPU (defaults to CPU if GPU not available)",
    )
    args = parser.parse_args()

    train(
        scenario=args.scenario,
        model_type=args.model,
        shadow=args.shadow,
        mode=args.mode,
        total_steps=args.steps,
        seed=args.seed,
        lambda_reg=args.lambda_reg,
        eta_agent=args.eta_agent,
        switch_critic=args.switch_critic,
        eval_freq=args.eval_freq,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
