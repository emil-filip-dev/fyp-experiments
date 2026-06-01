"""
Controller Comparison on PC-Gym CSTR
=====================================
Compares four controllers on the same CSTR setpoint-tracking task:
  1. PID  — classical proportional-integral-derivative
  2. NMPC — nonlinear MPC oracle (do-mpc + CasADi + IPOPT)
  3. DDPG — standard deep deterministic policy gradient (no shadow mode)
  4. Shadow DDPG — Q-value shadow-mode switching

Produces a single Ca(t) trajectory plot for all four vs the setpoint.

Usage:
    .venv/Scripts/python comparison.py
    .venv/Scripts/python comparison.py --steps 100000   # shorter training budget
    .venv/Scripts/python comparison.py --seed 7         # different eval seed
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import do_mpc
from casadi import exp as ca_exp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pcgym import make_env

# Re-use everything that shadow_mode.py already defines.
sys.path.insert(0, os.path.dirname(__file__))
from shadow_mode import (
    PIDController,
    ShadowDDPG,
    run_episode,         # shadow-mode episode runner (training=False → pure eval)
    N_STEPS, T_SIM, ENV_PARAMS, STATE_DIM, ACTION_DIM, make_cstr,
)

# Physical action bounds (needed for normalisation / denormalisation)
A_LOW  = ENV_PARAMS["a_space"]["low"][0]    # 295 K
A_HIGH = ENV_PARAMS["a_space"]["high"][0]   # 302 K

# Derived constants used by NMPC
_DT    = T_SIM / N_STEPS            # time step (min)
_SP_CA = ENV_PARAMS["SP"]["Ca"]     # setpoint schedule, list of N_STEPS floats


# ---------------------------------------------------------------------------
# 1.  NMPC controller — direct do-mpc setup (no oracle wrapper)
# ---------------------------------------------------------------------------

class NMPCController:
    """
    Online closed-loop NMPC for CSTR Ca tracking.

    Builds a do-mpc MPC directly from the CSTR model equations
    (matching pcgym/model_classes.py exactly).  At each timestep
    the MPC solves a finite-horizon OCP using the physical state
    from env.state and returns a physical Tc action.
    The caller normalises that action before env.step().
    """

    def __init__(self, n_horizon: int = 10):
        self._n_horizon = n_horizon
        self._mpc = self._build_mpc()

    # ------------------------------------------------------------------
    def _build_mpc(self):
        # ---- do-mpc continuous model ----------------------------------
        model = do_mpc.model.Model("continuous")

        Ca    = model.set_variable("_x",  "Ca")
        T_r   = model.set_variable("_x",  "T")
        Tc    = model.set_variable("_u",  "Tc")
        Ca_sp = model.set_variable("_p",  "Ca_sp")

        # CSTR parameters from pcgym model_classes.py
        q, V        = 100.0, 100.0
        rho, C_p    = 1000.0, 0.239
        deltaHr     = -5e4
        EA_over_R   = 8750.0
        k0          = 7.2e10
        UA          = 5e4
        Ti, Caf     = 350.0, 1.0

        rA  = k0 * ca_exp(-EA_over_R / T_r) * Ca
        dCa = q / V * (Caf - Ca) - rA
        dT  = (q / V * (Ti - T_r)
               + (-deltaHr) * rA / (rho * C_p)
               + UA * (Tc - T_r) / (rho * C_p * V))

        model.set_rhs("Ca", dCa)
        model.set_rhs("T",  dT)
        model.setup()

        # ---- MPC controller ------------------------------------------
        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self._n_horizon,
            t_step=_DT,
            n_robust=0,
            store_full_solution=True,
        )

        mpc.set_objective(lterm=(Ca - Ca_sp) ** 2,
                          mterm=(Ca - Ca_sp) ** 2)
        mpc.set_rterm(Tc=0.0)
        mpc.n_combinations = 1          # required before set_p_fun

        mpc.bounds["lower", "_u", "Tc"] = A_LOW
        mpc.bounds["upper", "_u", "Tc"] = A_HIGH

        sp_arr = _SP_CA  # captured for p_fun closure
        def p_fun(t_now):
            p   = mpc.get_p_template(1)
            # t_now may arrive as a numpy array; .flat[0] handles any shape
            t   = float(np.asarray(t_now).flat[0])
            idx = max(0, min(int(t / _DT), len(sp_arr) - 1))
            p["_p", 0, "Ca_sp"] = float(sp_arr[idx])
            return p

        mpc.set_p_fun(p_fun)
        mpc.set_param(nlpsol_opts={
            "ipopt.print_level": 0,
            "print_time":        0,
            "ipopt.sb":          "yes",
        })
        mpc.setup()
        return mpc

    # ------------------------------------------------------------------
    def reset(self, x0: np.ndarray | None = None):
        """Rebuild MPC (resets internal time to 0) and set initial state."""
        self._mpc = self._build_mpc()
        if x0 is None:
            x0 = ENV_PARAMS["x0"][:2]
        x0 = np.asarray(x0[:2], dtype=float)
        self._mpc.x0 = x0
        self._mpc.set_initial_guess()

    def step(self, x_physical: np.ndarray) -> float:
        """Given physical [Ca, T], return optimal physical Tc."""
        x = np.asarray(x_physical[:2], dtype=float)
        u0 = self._mpc.make_step(x)
        return float(np.asarray(u0).flat[0])


def run_episode_nmpc(env, nmpc: NMPCController, seed: int = 0):
    """Run one NMPC episode in PC-Gym (normalised action space)."""
    obs, _ = env.reset(seed=seed)
    nmpc.reset(x0=env.state[:2])

    ca_values, sp_values, rewards = [], [], []
    done = False

    while not done:
        Tc_phys = nmpc.step(env.state[:2])
        a_norm  = 2.0 * (Tc_phys - A_LOW) / (A_HIGH - A_LOW) - 1.0
        action  = np.array([np.clip(a_norm, -1.0, 1.0)], dtype=np.float32)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ca_values.append(env.state[0])
        sp_values.append(env.state[2])
        rewards.append(reward)

    return np.array(ca_values), np.array(sp_values), np.array(rewards)


# ---------------------------------------------------------------------------
# 2.  Pure DDPG — ShadowDDPG subclass that always uses the agent action
# ---------------------------------------------------------------------------

class PureDDPG(ShadowDDPG):
    """
    Standard DDPG with no shadow-mode switching.
    decide_action always returns the (noisy) agent action.
    The baseline action is still computed and stored in the replay buffer
    so the class remains compatible with run_episode() from shadow_mode.py.
    """

    def decide_action(self, obs: np.ndarray, baseline_action: np.ndarray,
                      training: bool = True):
        noise = (
            np.random.normal(0, self.noise_std, ACTION_DIM).astype(np.float32)
            if training
            else np.zeros(ACTION_DIM, dtype=np.float32)
        )
        action_agent, _ = self._get_agent_action(obs)
        action_noisy    = np.clip(action_agent + noise, -1.0, 1.0)
        self.agent_takeover_count += 1
        return action_noisy, True, action_noisy


# ---------------------------------------------------------------------------
# 3.  Training / checkpoint helpers
# ---------------------------------------------------------------------------

def train_agent(agent: ShadowDDPG, total_steps: int, save_path: str,
                seed: int = 42, label: str = "agent"):
    """Generic training loop for any ShadowDDPG-like agent."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(save_path, exist_ok=True)

    env = make_cstr()
    pid = PIDController()

    print(f"\n{'='*55}")
    print(f"Training {label}  ({total_steps:,} steps)")
    print(f"{'='*55}")

    episode      = 0
    rewards_log  = []
    best_reward  = -np.inf
    t_start      = time.time()

    while agent.total_steps < total_steps:
        ep_seed = episode + seed * 10_000
        reward, _, _, _ = run_episode(env, agent, pid, training=True,
                                      seed=ep_seed)
        rewards_log.append(reward)

        if episode % 50 == 0:
            recent  = float(np.mean(rewards_log[-50:])) if rewards_log else 0.0
            elapsed = time.time() - t_start
            print(f"  ep {episode:4d} | steps {agent.total_steps:7,} | "
                  f"reward {reward:7.1f} | recent {recent:7.1f} | {elapsed:.0f}s")

        if agent.total_steps % 10_000 < N_STEPS:
            eval_r = _quick_eval(agent, pid, n=5)
            if eval_r > best_reward:
                best_reward = eval_r
                agent.save(os.path.join(save_path, "best.pt"))

        episode += 1

    print(f"  Done. Best eval reward: {best_reward:.2f}")
    return agent


def _quick_eval(agent: ShadowDDPG, pid: PIDController, n: int = 5) -> float:
    env = make_cstr()
    rewards = [
        run_episode(env, agent, pid, training=False, seed=s)[0]
        for s in range(n)
    ]
    return float(np.mean(rewards))


def load_or_train(agent_cls, label: str, save_dir: str,
                  total_steps: int, seed: int, **agent_kwargs) -> ShadowDDPG:
    ckpt = os.path.join(save_dir, "best.pt")
    agent = agent_cls(**agent_kwargs)
    if os.path.exists(ckpt):
        print(f"  Loading {label} from {ckpt}")
        agent.load(ckpt)
    else:
        train_agent(agent, total_steps=total_steps, save_path=save_dir,
                    seed=seed, label=label)
        agent.load(ckpt)
    return agent


# ---------------------------------------------------------------------------
# 4.  Main comparison
# ---------------------------------------------------------------------------

def main(total_steps: int = 200_000, eval_seed: int = 42,
         train_seed: int = 42, n_horizon: int = 10):

    # --- Load / train agents ------------------------------------------------
    print("\n[1/4] Shadow DDPG (Q-value mode)")
    shadow_agent = load_or_train(
        ShadowDDPG, label="Shadow DDPG",
        save_dir="runs/shadow_qvalue",
        total_steps=total_steps, seed=train_seed,
        mode="qvalue",
    )

    print("\n[2/4] Pure DDPG (no shadow mode)")
    pure_agent = load_or_train(
        PureDDPG, label="Pure DDPG",
        save_dir="runs/pure_ddpg",
        total_steps=total_steps, seed=train_seed,
        mode="qvalue",          # mode field unused by PureDDPG but needed by __init__
    )

    # --- Set up NMPC --------------------------------------------------------
    print("\n[3/4] Setting up NMPC oracle (do-mpc + IPOPT) …")
    nmpc = NMPCController(n_horizon=n_horizon)
    print("  NMPC ready.")

    # --- Evaluation episodes (same seed for all) ----------------------------
    print(f"\n[4/4] Running evaluation episodes (seed={eval_seed}) …")
    pid = PIDController()
    env = make_cstr()

    # PID
    pid.reset()
    obs, _ = env.reset(seed=eval_seed)
    ca_pid, sp_pid, r_pid = [], [], []
    done = False
    while not done:
        action, _ = pid.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ca_pid.append(env.state[0])
        sp_pid.append(env.state[2])
        r_pid.append(reward)
    ca_pid = np.array(ca_pid);  sp_pid = np.array(sp_pid)
    print(f"  PID         total reward: {sum(r_pid):8.1f}")

    # NMPC
    ca_nmpc, sp_nmpc, r_nmpc = run_episode_nmpc(env, nmpc, seed=eval_seed)
    print(f"  NMPC        total reward: {r_nmpc.sum():8.1f}")

    # Pure DDPG
    r_pure, ca_pure, sp_pure, _ = run_episode(
        env, pure_agent, pid, training=False, seed=eval_seed
    )
    ca_pure = np.array(ca_pure)
    print(f"  Pure DDPG   total reward: {r_pure:8.1f}")

    # Shadow DDPG
    r_shadow, ca_shadow, sp_shadow, agent_flags = run_episode(
        env, shadow_agent, pid, training=False, seed=eval_seed
    )
    ca_shadow    = np.array(ca_shadow)
    agent_flags  = np.array(agent_flags)
    print(f"  Shadow DDPG total reward: {r_shadow:8.1f}")

    # Use sp from PID (all controllers see the same setpoint)
    setpoint  = sp_pid
    time_axis = np.linspace(0, T_SIM, N_STEPS)

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    # Shade regions where Shadow DDPG agent is in control
    in_agent = False
    start_t  = None
    first_shade = True
    for i, flag in enumerate(agent_flags):
        t = time_axis[i] if i < len(time_axis) else time_axis[-1]
        if flag and not in_agent:
            start_t  = t
            in_agent = True
        elif not flag and in_agent:
            ax.axvspan(start_t, t, alpha=0.10, color="#4878CF",
                       label="Shadow: agent in control" if first_shade else "")
            first_shade = False
            in_agent    = False
    if in_agent:
        ax.axvspan(start_t, time_axis[min(len(agent_flags) - 1, len(time_axis) - 1)],
                   alpha=0.10, color="#4878CF")

    ax.plot(time_axis[:len(setpoint)], setpoint,
            color="black", linestyle="--", linewidth=2.0, label="Setpoint",
            zorder=5)

    ax.plot(time_axis[:len(ca_pid)], ca_pid,
            color="#E8834E", linewidth=2.0,
            label=f"PID  (reward {sum(r_pid):.0f})")

    ax.plot(time_axis[:len(ca_nmpc)], ca_nmpc,
            color="#2CA02C", linewidth=2.0,
            label=f"NMPC (reward {r_nmpc.sum():.0f})")

    ax.plot(time_axis[:len(ca_pure)], ca_pure,
            color="#D62728", linewidth=2.0,
            label=f"DDPG (reward {r_pure:.0f})")

    ax.plot(time_axis[:len(ca_shadow)], ca_shadow,
            color="#4878CF", linewidth=2.0,
            label=f"Shadow DDPG (reward {r_shadow:.0f})")

    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel(r"$C_A$ (mol/L)", fontsize=12)
    ax.set_title("CSTR Setpoint Tracking — PID vs NMPC vs DDPG vs Shadow DDPG",
                 fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs("runs/comparison", exist_ok=True)
    out_path = "runs/comparison/controller_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\n[Plot] Saved to {out_path}")

    print("\n" + "="*55)
    print("SUMMARY  (seed={})".format(eval_seed))
    print("="*55)
    results = {
        "PID":         sum(r_pid),
        "NMPC":        r_nmpc.sum(),
        "Pure DDPG":   r_pure,
        "Shadow DDPG": r_shadow,
    }
    nmpc_r = r_nmpc.sum()
    for name, r in results.items():
        delta = r - nmpc_r
        print(f"  {name:<16} {r:8.1f}   delta vs NMPC: {delta:+.1f}")
    print("="*55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare PID / NMPC / DDPG / Shadow DDPG on PC-Gym CSTR"
    )
    parser.add_argument("--steps",       type=int,   default=200_000,
                        help="Training budget for DDPG agents (default 200k)")
    parser.add_argument("--eval-seed",   type=int,   default=42)
    parser.add_argument("--train-seed",  type=int,   default=42)
    parser.add_argument("--n-horizon",   type=int,   default=10,
                        help="MPC prediction horizon (default 10)")
    args = parser.parse_args()

    main(
        total_steps=args.steps,
        eval_seed=args.eval_seed,
        train_seed=args.train_seed,
        n_horizon=args.n_horizon,
    )
