"""
train.py
========
Train a standard (non-shadow) RL agent on a PC-Gym process-control environment.

Uses Stable Baselines 3.  The agent always acts — no baseline switching.
For shadow-mode training see train_shadow.py.

Supported scenarios: cstr, four_tank, multistage_extraction, crystallization
Supported models:    ddpg, td3, ppo

Outputs
-------
  outputs/models/<scenario>/<model>/   — best_model.zip + training_curves.png

Usage
-----
  .venv/Scripts/python train.py --scenario cstr --model td3
  .venv/Scripts/python train.py --scenario four_tank --model ppo --steps 300000
  .venv/Scripts/python train.py --scenario cstr --model td3 --cpu
"""

import argparse
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from models import STANDARD_MODELS, create_standard_agent, device_label, resolve_device
from scenarios import SCENARIOS, make_env_for
from evaluate import plot_training_curves


# ---------------------------------------------------------------------------
# Combined training callback
# ---------------------------------------------------------------------------

class _TrainingCallback(BaseCallback):
    """
    Single callback that handles everything during model.learn():
      - tqdm progress bar (one bar, total timesteps)
      - per-episode reward logging (via Monitor's info dict)
      - periodic evaluation every eval_freq steps
      - best-model saving on improved eval reward
    """

    def __init__(self, total_steps: int, eval_env, save_path: str,
                 eval_freq: int = 10_000, n_eval: int = 5):
        super().__init__(verbose=0)
        self._eval_env  = eval_env
        self._save_path = save_path
        self._eval_freq = eval_freq
        self._n_eval    = n_eval
        self._next_eval = eval_freq

        self.rewards_log: list[tuple[int, float]] = []
        self.eval_log:    list[tuple[int, float]] = []

        self._bar       = tqdm(total=total_steps, unit="step",
                               dynamic_ncols=True, desc="Training")
        self._last_ep_r = float("nan")
        self._best_eval = -float("inf")

    def _on_step(self) -> bool:
        self._bar.update(1)

        # Collect episode reward when Monitor signals episode end
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._last_ep_r = float(info["episode"]["r"])
                self.rewards_log.append((self.num_timesteps, self._last_ep_r))

        # Periodic evaluation — while loop so no window is skipped
        while self.num_timesteps >= self._next_eval:
            self._next_eval += self._eval_freq
            rewards = []
            for s in range(self._n_eval):
                obs, _ = self._eval_env.reset(seed=s)
                total, done = 0.0, False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, terminated, truncated, _ = self._eval_env.step(action)
                    done   = terminated or truncated
                    total += r
                rewards.append(total)
            mean_r = float(np.mean(rewards))
            self.eval_log.append((self.num_timesteps, mean_r))
            if mean_r > self._best_eval:
                self._best_eval = mean_r
                self.model.save(os.path.join(self._save_path, "best_model"))
                tqdm.write(f"  [Save] step {self.num_timesteps:,} — new best eval: {mean_r:.1f}")

        # Update postfix with latest metrics
        postfix: dict[str, str] = {}
        if not np.isnan(self._last_ep_r):
            postfix["ep_r"] = f"{self._last_ep_r:.0f}"
        if self.eval_log:
            postfix["eval"] = f"{self.eval_log[-1][1]:.0f}"
        if self._best_eval > -float("inf"):
            postfix["best"] = f"{self._best_eval:.0f}"
        self._bar.set_postfix(postfix)
        return True

    def on_training_end(self) -> None:
        self._bar.close()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    scenario:    str  = "cstr",
    model_type:  str  = "td3",
    total_steps: int  = 200_000,
    seed:        int  = 42,
    force_cpu:   bool = False,
    output_dir:  str  = "outputs/models",
):
    cfg       = SCENARIOS[scenario]
    save_path = os.path.join(output_dir, scenario, model_type)
    os.makedirs(save_path, exist_ok=True)

    device    = resolve_device(force_cpu)
    train_env = Monitor(make_env_for(scenario))
    eval_env  = make_env_for(scenario)
    model     = create_standard_agent(model_type, train_env, seed=seed, device=device)

    print(f"\n{'='*60}")
    print(f"  Scenario : {scenario}")
    print(f"  Model    : {model_type.upper()}")
    print(f"  Device   : {device_label(device)}")
    print(f"  Steps    : {total_steps:,}  |  seed={seed}")
    print(f"  Output   : {save_path}")
    print(f"{'='*60}\n")

    cb = _TrainingCallback(total_steps, eval_env, save_path,
                           eval_freq=10_000, n_eval=5)
    model.learn(total_timesteps=total_steps, callback=cb)

    tqdm.write(f"\n  Training complete.  Best eval reward: {cb._best_eval:.1f}")
    plot_training_curves(cb.rewards_log, cb.eval_log, save_path, scenario, model_type)
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a standard RL agent on a PC-Gym environment."
    )
    parser.add_argument(
        "--scenario", type=str, default="cstr",
        choices=list(SCENARIOS.keys()),
        help="PC-Gym environment to train on",
    )
    parser.add_argument(
        "--model", type=str, default="td3",
        choices=STANDARD_MODELS,
        help=f"RL algorithm: {', '.join(STANDARD_MODELS)}  (default: td3)",
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
        total_steps=args.steps,
        seed=args.seed,
        force_cpu=args.cpu,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
