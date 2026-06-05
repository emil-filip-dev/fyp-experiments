"""
trainer.py
==========
Shared training loops used by both entry points:

  train.py        — standard (no-shadow) agents: PureDDPG / PureTD3  (+ SB3 backend)
  train_shadow.py — shadow-mode agents:          ShadowDDPG / ShadowTD3 (+ SB3 backend)

`train_custom` drives the custom PyTorch core (episode-based loop); `train_sb3`
drives a Stable-Baselines3 model via its own model.learn() with a callback. Both
periodically evaluate and save the best checkpoint.

NOTE: training-metric serialisation (training_metrics.npz) and learning-curve
plotting were intentionally removed (see CLAUDE.md → "Removed: training-metrics
& plotting subsystem"). These loops now only train and save the best checkpoint;
a redesigned metrics/plotting pipeline will be reintroduced separately.
"""

import os
import sys

import numpy as np
from tqdm import tqdm

from evaluate import evaluate, run_episode
from scenarios import SCENARIOS, make_env_for

# stable_baselines3 is imported lazily inside train_sb3 so the custom-core path
# never pays the import cost.


def configure_utf8_output() -> None:
    """
    Force stdout/stderr to UTF-8 so non-ASCII output doesn't crash when stdout is
    redirected to a file/pipe on Windows (default cp1252 can't encode some chars).
    Call once at the start of a script's main().
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def train_custom(
    agent,
    *,
    scenario:    str,
    model_type:  str,
    run_label:   str,
    total_steps: int,
    seed:        int,
    output_dir:  str  = "outputs/models",
    is_shadow:   bool = False,
    eval_freq:   int  = 1_000,
    n_eval:      int  = 5,
    meta_extra:  dict | None = None,   # reserved for the future metrics rewrite
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

        # Periodic evaluation — loop so no boundary is skipped
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
# Stable-Baselines3 training loop — SB3 agents have their own model.learn()
# rollout collection, so this drives it with a callback (instead of the custom
# run_episode loop). Shadow switching happens inside the agent's _sample_action;
# here we run periodic eval, save the best model, and reset the PID baseline at
# episode boundaries.
# ---------------------------------------------------------------------------

def train_sb3(
    model,
    *,
    scenario:    str,
    model_type:  str,
    run_label:   str,
    total_steps: int,
    seed:        int,
    output_dir:  str  = "outputs/models",
    is_shadow:   bool = False,
    eval_freq:   int  = 1_000,
    n_eval:      int  = 5,
    meta_extra:  dict | None = None,   # reserved for the future metrics rewrite
):
    save_path = os.path.join(output_dir, scenario, run_label)
    os.makedirs(save_path, exist_ok=True)

    cb = _SB3TrainingCallback(
        total_steps=total_steps, scenario=scenario, run_label=run_label,
        save_path=save_path, is_shadow=is_shadow, eval_freq=eval_freq, n_eval=n_eval,
    )
    model.learn(total_timesteps=total_steps, callback=cb)

    tqdm.write(f"\n  Training complete.  Best eval reward: {cb.best_eval:.1f}")
    tqdm.write(f"  Saved: {os.path.join(save_path, 'best_model.zip')}")
    return model


def _make_sb3_callback_cls():
    """Build the callback class lazily (needs stable_baselines3 imported)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _SB3TrainingCallback(BaseCallback):
        def __init__(self, *, total_steps, scenario, run_label, save_path,
                     is_shadow, eval_freq, n_eval):
            super().__init__()
            self.save_path  = save_path
            self.is_shadow  = is_shadow
            self.eval_freq  = eval_freq
            self.n_eval     = n_eval

            self.best_eval   = -float("inf")
            self._next_eval  = eval_freq
            self._last_ep_r  = float("nan")
            self._prev_dec   = 0
            self._prev_agent = 0
            self._last_takeover = float("nan")

            self._eval_env      = make_env_for(scenario)
            self._eval_baseline = (SCENARIOS[scenario]["baseline_cls"]()
                                   if is_shadow else None)
            self._bar = tqdm(total=total_steps, unit="step",
                             dynamic_ncols=True, desc=run_label)

        def _on_step(self) -> bool:
            self._bar.update(1)

            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [False] * len(infos))
            for info, done in zip(infos, dones):
                # Gate on done — PC-Gym's persistent info dict leaves a stale
                # "episode" key around between episodes.
                if done and "episode" in info:
                    self._last_ep_r = float(info["episode"]["r"])
                    if self.is_shadow and hasattr(self.model, "_n_decisions"):
                        d = self.model._n_decisions - self._prev_dec
                        a = self.model._n_agent     - self._prev_agent
                        self._last_takeover = 100.0 * a / d if d > 0 else float("nan")
                        self._prev_dec   = self.model._n_decisions
                        self._prev_agent = self.model._n_agent
                if done and hasattr(self.model, "reset_baseline"):
                    self.model.reset_baseline()

            while self.num_timesteps >= self._next_eval:
                self._next_eval += self.eval_freq
                r = self._evaluate()
                if r > self.best_eval:
                    self.best_eval = r
                    self.model.save(os.path.join(self.save_path, "best_model"))
                    tqdm.write(f"  [Save] step {self.num_timesteps:,} - new best eval: {r:.1f}")

            postfix: dict[str, str] = {}
            if not np.isnan(self._last_ep_r):
                postfix["ep_r"] = f"{self._last_ep_r:.0f}"
            if self.is_shadow and not np.isnan(self._last_takeover):
                postfix["agent%"] = f"{self._last_takeover:.1f}"
            if self.best_eval > -float("inf"):
                postfix["best"] = f"{self.best_eval:.0f}"
            self._bar.set_postfix(postfix)
            return True

        def _evaluate(self) -> float:
            """Deterministic eval. Shadow models are evaluated WITH switching."""
            rewards = []
            for s in range(self.n_eval):
                obs, _ = self._eval_env.reset(seed=s)
                if self.is_shadow:
                    self._eval_baseline.reset()
                total, done = 0.0, False
                while not done:
                    if self.is_shadow:
                        action = self.model.executed_action(obs, self._eval_baseline)
                    else:
                        action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, terminated, truncated, _ = self._eval_env.step(action)
                    done   = terminated or truncated
                    total += r
                rewards.append(total)
            return float(np.mean(rewards))

        def on_training_end(self) -> None:
            self._bar.close()

    return _SB3TrainingCallback


# Built on first use (keeps stable_baselines3 out of the import path otherwise).
def _SB3TrainingCallback(**kwargs):
    cls = _make_sb3_callback_cls()
    return cls(**kwargs)
