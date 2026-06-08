"""
demo_takeover.py
================
Demonstration script: train Shadow DDPG (q-value switching) on a PC-Gym scenario
using the *real* training loop (evaluate.run_episode), while logging the
agent-takeover fraction of the DETERMINISTIC (greedy) policy over training and
saving a plot.

This exists because the current train.py deliberately does not serialise or plot
training metrics. It reuses the exact same agent + run_episode code path, so it is
a faithful demonstration that the training loop works — it only adds a periodic
deterministic "takeover probe" and a live-updating plot.

Expected behaviour (per Gassert & Althoff, 2024, Fig. 4): with q-value switching
the agent takes over a lot early (randomly-initialised critic), then the takeover
fraction trends DOWNWARD as the critic learns the baseline is good — i.e. control
is "earned".

Usage
-----
  .venv/Scripts/python demo_takeover.py --scenario cstr --steps 40000
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")            # headless: just write a PNG
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from evaluate import run_episode
from models import ShadowDDPG, SwitchingMode
from scenarios import SCENARIOS, make_env_for
from util import configure_utf8_output, device_label, resolve_device


def deterministic_takeover(env, agent, baseline, n_steps, n_probe, seed0=900_000):
    """Mean takeover fraction + mean reward of the greedy policy (no exploration)."""
    fracs, rewards = [], []
    for k in range(n_probe):
        r, flags = run_episode(env, agent, baseline, training=False,
                               seed=seed0 + k, n_steps=n_steps)
        fracs.append(float(np.mean(flags)))
        rewards.append(r)
    return float(np.mean(fracs)), float(np.mean(rewards))


def _plot(steps, tk, warmup, scenario, out_path):
    if not steps:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(steps, tk, marker="o", ms=3, lw=1.5, color="tab:blue",
             label="agent takeover (deterministic policy)")
    if warmup and warmup < max(steps):
        plt.axvline(warmup, color="gray", ls="--", lw=1,
                    label=f"warmup end ({warmup:,})")
    plt.xlabel("training steps")
    plt.ylabel("agent takeover fraction (%)")
    plt.title(f"Shadow DDPG (q-value) — agent takeover over training [{scenario}]")
    plt.ylim(-2, 102)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    configure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="cstr", choices=list(SCENARIOS.keys()))
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--eval-freq", type=int, default=1_000)
    ap.add_argument("--n-probe", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--out", default="outputs/takeover_cstr.png")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    cfg = SCENARIOS[args.scenario]
    n_steps = cfg["n_steps"]
    env = make_env_for(args.scenario)
    probe_env = make_env_for(args.scenario)
    baseline = cfg["baseline_cls"]()

    agent = ShadowDDPG(
        state_dim=cfg["state_dim"], action_dim=cfg["action_dim"],
        switching_mode=SwitchingMode.Q_VALUE, device=device,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"\n{'=' * 62}")
    print(f"  DEMO: Shadow DDPG (q-value) takeover trend on '{args.scenario}'")
    print(f"  steps={args.steps:,} | warmup={agent.warmup_steps:,} | "
          f"device={device_label(device)}")
    print(f"  plot -> {args.out}")
    print(f"{'=' * 62}\n")

    steps_log, tk_log, rew_log = [], [], []
    next_eval = args.eval_freq
    episode = 0

    bar = tqdm(total=args.steps, unit="step", dynamic_ncols=True, desc="shadow_ddpg_qvalue")
    while agent.total_steps < args.steps:
        ep_seed = episode + args.seed * 10_000
        before = agent.total_steps
        run_episode(env, agent, baseline, training=True, seed=ep_seed, n_steps=n_steps)
        bar.update(agent.total_steps - before)
        episode += 1

        while agent.total_steps >= next_eval:
            next_eval += args.eval_freq
            tk, r = deterministic_takeover(probe_env, agent, baseline, n_steps, args.n_probe)
            steps_log.append(int(agent.total_steps))
            tk_log.append(tk * 100.0)
            rew_log.append(r)
            tqdm.write(f"  step {agent.total_steps:6,} | takeover {tk * 100:5.1f}% "
                       f"| eval reward {r:8.1f}")
            _plot(steps_log, tk_log, agent.warmup_steps, args.scenario, args.out)

        bar.set_postfix({"ep": episode,
                         "takeover%": f"{tk_log[-1]:.0f}" if tk_log else "-"})
    bar.close()

    np.savez(args.out.replace(".png", ".npz"),
             steps=np.array(steps_log), takeover_pct=np.array(tk_log),
             eval_reward=np.array(rew_log))
    _plot(steps_log, tk_log, agent.warmup_steps, args.scenario, args.out)
    print(f"\n  Done. Takeover plot saved to {args.out}")
    print(f"  Log saved to {args.out.replace('.png', '.npz')}")


if __name__ == "__main__":
    main()
