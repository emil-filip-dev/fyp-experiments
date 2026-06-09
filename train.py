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


def run_label_for(
    model_type:    str,
    *,
    shadow:        bool  = False,
    mode:          str   = "qvalue",
    lambda_reg:    float = 0.0,
    switch_critic: str   = "q1",
) -> str:
    """
    Canonical output-directory name for a training condition.

    This is the SINGLE source of truth for run-label naming: train() uses it to
    decide where to save, and the experiment orchestrator / analysis utilities
    import it to locate the resulting checkpoints. Keeping all naming logic here
    stops the producer (train) and the consumers (experiments / evaluate) from
    drifting apart. Arguments mirror the train() CLI (plain strings, not enums),
    so callers need not import the models enums.

      standard            -> "<model>"                         e.g. "ddpg"
      shadow agent + reg  -> "shadow_<model>_agent_reg<λ>"
      shadow td3 qvalue   -> "shadow_<model>_qvalue_<switch_critic>"
      shadow (other)      -> "shadow_<model>_<mode>"
    """
    if not shadow:
        return model_type
    if mode == "agent" and lambda_reg > 0.0:
        return f"shadow_{model_type}_agent_reg{lambda_reg}"
    if model_type == "td3" and mode == "qvalue":
        return f"shadow_{model_type}_{mode}_{switch_critic}"
    return f"shadow_{model_type}_{mode}"


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
    checkpoint_freq: int = 1_000,
    device:      str   = "cpu",
    output_dir:  str   = "outputs/models",
    per_seed_dir: bool = False,
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
        run_label = run_label_for(
            model_type, shadow=True, mode=mode.value,
            lambda_reg=lambda_reg, switch_critic=switch_critic,
        )

        model_desc = f"Shadow {model_type.upper()}  (mode={mode.value}, lambda={lambda_reg}, eta={eta_agent}"
        model_desc += f", switch_critic={switch_critic})" if is_td3_qvalue else ")"
        meta_extra = {"switching_mode": mode.value, "lambda_reg": lambda_reg, "eta_agent": eta_agent}
    else:
        model_class = get_standard_model(model_type)
        agent = model_class(
            state_dim=state_dim, action_dim=action_dim, lambda_reg=lambda_reg, device=device
        )
        run_label  = run_label_for(model_type)
        model_desc = f"{model_type.upper()} (standard, no shadow)"
        meta_extra = None

    _print_train_header(scenario, model_desc, device, total_steps, seed)
    return train_model(
        agent, scenario=scenario, run_label=run_label,
        total_steps=total_steps, seed=seed, output_dir=output_dir,
        is_shadow=shadow, eval_freq=eval_freq, checkpoint_freq=checkpoint_freq,
        meta_extra=meta_extra, per_seed_dir=per_seed_dir,
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
    checkpoint_freq: int = 1_000,
    n_eval:      int  = 5,
    meta_extra:  dict | None = None,
    per_seed_dir: bool = False,
):
    """
    Run the shared episode-based training loop for any ShadowDDPG-like agent
    (PureDDPG / PureTD3 / ShadowDDPG / ShadowTD3). Saves best.pt whenever the
    periodic evaluation reward improves, plus a periodic snapshot epN.pt every
    `checkpoint_freq` episodes (set 0 to disable the snapshots).

    `per_seed_dir` opts into a per-seed leaf directory (.../<run_label>/seed<k>/)
    so multi-seed orchestration does not overwrite checkpoints across seeds. It
    defaults to False to preserve the single-run layout (.../<run_label>/).
    """
    cfg       = SCENARIOS[scenario]
    n_steps   = cfg["n_steps"]
    save_path = os.path.join(output_dir, scenario, run_label)
    if per_seed_dir:
        save_path = os.path.join(save_path, f"seed{seed}")
    os.makedirs(save_path, exist_ok=True)

    env      = make_env_for(scenario)
    eval_env = make_env_for(scenario)
    baseline = cfg["baseline_cls"]()

    episode        = 0
    best_reward    = -np.inf
    recent_rewards: list[float] = []
    next_eval      = eval_freq
    train_tk_log: list[tuple[int, float]] = []   # (step, per-episode behaviour takeover %)
    eval_tk_log:  list[tuple[int, float]] = []   # (step, deployment greedy takeover %)
    last_eval_tk: float | None = None

    bar = tqdm(total=total_steps, unit="step", dynamic_ncols=True, desc=run_label)

    while agent.total_steps < total_steps:
        ep_seed      = episode + seed * 10_000
        steps_before = agent.total_steps

        reward, flags = run_episode(env, agent, baseline,
                                    training=True, seed=ep_seed, n_steps=n_steps)
        recent_rewards.append(reward)
        train_tk_log.append((agent.total_steps, float(np.mean(flags)) * 100.0))
        bar.update(agent.total_steps - steps_before)

        # Periodic evaluation - loop so no boundary is skipped
        while agent.total_steps >= next_eval:
            next_eval += eval_freq
            # Deterministic (deployment) rollouts: reward + takeover fraction.
            e_rewards, e_takeovers = [], []
            for es in range(n_eval):
                er, eflags = run_episode(eval_env, agent, baseline,
                                         training=False, seed=es, n_steps=n_steps)
                e_rewards.append(er)
                e_takeovers.append(float(np.mean(eflags)))
            eval_r  = float(np.mean(e_rewards))
            eval_tk = float(np.mean(e_takeovers)) * 100.0
            eval_tk_log.append((agent.total_steps, eval_tk))
            last_eval_tk = eval_tk
            if eval_r > best_reward:
                best_reward = eval_r
                agent.save(os.path.join(save_path, "best.pt"))
                tqdm.write(f"  [Save] step {agent.total_steps:,} - new best eval: {best_reward:.1f}")
            if is_shadow:
                tqdm.write(f"  [Eval] step {agent.total_steps:,} | reward {eval_r:.1f} | "
                           f"deploy takeover {eval_tk:.1f}%")
                _plot_takeover(train_tk_log, eval_tk_log, agent.warmup_steps, save_path, run_label)

        recent = float(np.mean(recent_rewards[-50:])) if recent_rewards else 0.0
        postfix: dict[str, str] = {"reward": f"{reward:.0f}", "avg50": f"{recent:.0f}"}
        if is_shadow:
            postfix["agent%"] = f"{float(np.mean(flags)) * 100:.1f}"
            if last_eval_tk is not None:
                postfix["deploy%"] = f"{last_eval_tk:.0f}"
        if best_reward > -np.inf:
            postfix["best"] = f"{best_reward:.0f}"
        bar.set_postfix(postfix)

        episode += 1

        # Periodic snapshot every `checkpoint_freq` episodes (best.pt is handled
        # separately above and always holds the best-evaluating model).
        if checkpoint_freq > 0 and episode % checkpoint_freq == 0:
            ckpt_path = os.path.join(save_path, f"ep{episode}.pt")
            agent.save(ckpt_path)
            tqdm.write(f"  [Checkpoint] episode {episode:,} (step {agent.total_steps:,}) -> {os.path.basename(ckpt_path)}")

    bar.close()

    # Shadow runs: emit the takeover graph (training behaviour + deployment) + raw log.
    if is_shadow:
        tk_png = _plot_takeover(train_tk_log, eval_tk_log, agent.warmup_steps, save_path, run_label)
        ts, tv = (zip(*train_tk_log) if train_tk_log else ((), ()))
        es, ev = (zip(*eval_tk_log)  if eval_tk_log  else ((), ()))
        np.savez(
            os.path.join(save_path, "takeover.npz"),
            train_steps=np.array(ts), train_takeover=np.array(tv),
            eval_steps=np.array(es),  eval_takeover=np.array(ev),
        )
        if tk_png:
            tqdm.write(f"  Takeover graph: {tk_png}")

    tqdm.write(f"\n  Training complete.  Best eval reward: {best_reward:.1f}")
    tqdm.write(f"  Saved: {os.path.join(save_path, 'best.pt')}")
    return agent


def _plot_takeover(train_log, eval_log, warmup, save_path, run_label):
    """
    Save takeover.png showing the agent-takeover fraction over training:
      - training behaviour takeover (per training episode, incl. exploration + warmup)
      - deployment takeover (deterministic greedy eval, no exploration)
    Returns the PNG path (or None if there is nothing to plot).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not train_log and not eval_log:
        return None

    max_step = 0
    if train_log:
        max_step = max(max_step, max(s for s, _ in train_log))
    if eval_log:
        max_step = max(max_step, max(s for s, _ in eval_log))

    plt.figure(figsize=(9, 5))
    if train_log:
        ts = np.array([s for s, _ in train_log])
        tv = np.array([v for _, v in train_log])
        plt.plot(ts, tv, color="tab:blue", alpha=0.25, lw=1,
                 label="training (per-episode behaviour)")
        if len(tv) >= 5:
            w = max(1, len(tv) // 50)
            ma = np.convolve(tv, np.ones(w) / w, mode="valid")
            plt.plot(ts[w - 1:], ma, color="tab:blue", lw=2, label="training (moving avg)")
    if eval_log:
        es = np.array([s for s, _ in eval_log])
        ev = np.array([v for _, v in eval_log])
        plt.plot(es, ev, color="tab:red", marker="o", ms=3, lw=1.5,
                 label="deployment (greedy eval)")
    if warmup and warmup < max_step:
        plt.axvline(warmup, color="gray", ls="--", lw=1, label=f"warmup end ({warmup:,})")

    plt.xlabel("training steps")
    plt.ylabel("agent takeover (%)")
    plt.ylim(-2, 102)
    plt.grid(alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.title(f"Agent takeover over training — {run_label}")
    plt.tight_layout()
    out = os.path.join(save_path, "takeover.png")
    plt.savefig(out, dpi=120)
    plt.close()
    return out


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
        "--checkpoint-freq", type=int, default=1_000,
        help="Save a snapshot epN.pt every N episodes (default: 1000; 0 disables). "
             "best.pt is always kept separately as the best-evaluating model.",
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
        checkpoint_freq=args.checkpoint_freq,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
