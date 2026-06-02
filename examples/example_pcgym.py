"""
PC-Gym example: CSTR environment with PPO training and classical controller baselines.

Demonstrates:
- Setting up a CSTR environment in PC-Gym
- Training a PPO agent via Stable Baselines 3
- Comparing PPO against a PID controller on the same episode
- Plotting true (unnormalised) state trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from pcgym import make_env
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

T = 26      # simulation time (minutes)
N = 100     # number of time steps

# Setpoint schedule: Ca target switches mid-episode
SP = {
    'Ca': [0.85] * (N // 2) + [0.9] * (N // 2),
}

action_space = {
    'low':  np.array([295.], dtype=np.float32),
    'high': np.array([302.], dtype=np.float32),
}

observation_space = {
    'low':  np.array([0.7, 300., 0.8], dtype=np.float32),
    'high': np.array([1.0, 350., 0.9], dtype=np.float32),
}

env_params = {
    'N':                  N,
    'tsim':               T,
    'SP':                 SP,
    'o_space':            observation_space,
    'a_space':            action_space,
    'x0':                 np.array([0.8, 330.0, 0.8]),
    'model':              'cstr',
    'r_scale':            {'Ca': 1e3},
    'normalise_a':        True,   # actions in [-1, 1]
    'normalise_o':        True,   # observations in [-1, 1]
    'noise':              True,
    'integration_method': 'casadi',
    'noise_percentage':   0.001,
}


def make_cstr():
    return make_env(env_params)


# ---------------------------------------------------------------------------
# PID controller — operates in normalised observation space
#
# With normalise_o=True the obs is scaled to [-1, 1]:
#   norm = 2*(x - x_low)/(x_high - x_low) - 1
#
# obs[0] = normalised Ca,  obs[2] = normalised Ca setpoint
# The action output must also be in [-1, 1] (normalise_a=True).
#
# Gains are tuned for normalised space. Anti-windup clamps the integral so
# the total output never leaves [-1, 1].
# ---------------------------------------------------------------------------

class PIDController:
    def __init__(self, kp=-2.0, ki=-0.3, kd=-0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def predict(self, obs, deterministic=True):
        error = obs[2] - obs[0]          # setpoint - Ca (normalised)
        derivative = error - self._prev_error
        self._prev_error = error

        u = self.kp * error + self.ki * self._integral + self.kd * derivative

        # Anti-windup: only accumulate integral when not saturated
        if -1.0 < u < 1.0:
            self._integral += error

        u = float(np.clip(u, -1.0, 1.0))
        return np.array([u]), None


# ---------------------------------------------------------------------------
# Episode runner — uses env.state for true unnormalised Ca values
# ---------------------------------------------------------------------------

def run_episode(env, policy, seed=0):
    """
    Run one episode. Returns true (unnormalised) Ca and setpoint trajectories,
    plus per-step rewards.

    env.state layout for CSTR: [Ca (mol/L), T (K), Ca_sp (mol/L)]
    env.state is always in physical units regardless of normalise_o.
    """
    obs, _ = env.reset(seed=seed)
    ca_values, sp_values, rewards = [], [], []

    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ca_values.append(env.state[0])   # true Ca in mol/L
        sp_values.append(env.state[2])   # true Ca setpoint
        rewards.append(reward)

    return np.array(ca_values), np.array(sp_values), np.array(rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    env = make_cstr()
    env.model.info()

    # --- Train PPO ---
    print("\nTraining PPO agent (100k steps)...")
    train_env = make_cstr()
    ppo = PPO('MlpPolicy', train_env, verbose=0, learning_rate=3e-4)
    ppo.learn(total_timesteps=int(1e5))
    print("Training complete.")

    # --- Evaluate both policies on the same episode (same seed) ---
    EVAL_SEED = 42
    eval_env = make_cstr()
    pid = PIDController()

    ca_ppo, sp_ppo, rew_ppo = run_episode(eval_env, ppo, seed=EVAL_SEED)
    pid.reset()
    ca_pid, sp_pid, rew_pid = run_episode(eval_env, pid, seed=EVAL_SEED)

    time = np.linspace(0, T, len(ca_ppo))

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(time, sp_ppo, 'k--', label='Setpoint', linewidth=1.5)
    ax.plot(time, ca_ppo, label=f'PPO  (total reward: {rew_ppo.sum():.0f})', linewidth=1.5)
    ax.plot(time, ca_pid, label=f'PID  (total reward: {rew_pid.sum():.0f})', linewidth=1.5, linestyle='--')
    ax.set_ylabel('Ca (mol/L)')
    ax.set_title('CSTR Setpoint Tracking — PPO vs PID')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(time, np.cumsum(rew_ppo), label='PPO cumulative reward')
    ax.plot(time, np.cumsum(rew_pid), label='PID cumulative reward', linestyle='--')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Cumulative Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pcgym_example_results.png', dpi=150)
    plt.show()

    print(f"\nTotal reward — PPO: {rew_ppo.sum():.1f}  |  PID: {rew_pid.sum():.1f}")


if __name__ == '__main__':
    main()
