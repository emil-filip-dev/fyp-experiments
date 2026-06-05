"""
evaluate.py
===========
Run trained models (and the reference controllers) on a PC-Gym scenario and
SERIALISE the raw per-step rollout outputs to disk. It does NOT plot or compute
summary metrics — a separate (to-be-built) plotting/metrics utility loads these
rollout files and produces figures + metric tables.

Each run records, per step: physical state (env.state), observation, executed
action, the agent's proposed action, the PID baseline's action, reward, and the
shadow takeover flag. N seeds per method are stacked and written as one .npz per
method under outputs/rollouts/<scenario>/<timestamp>/, plus a manifest.json
describing the scenario (timing, plot_config, setpoint schedule, method list).

Reference controllers always included:
  - PID         — the scenario's baseline controller acting on its own
  - NMPC Oracle — do-mpc + IPOPT nonlinear MPC on the env's exact dynamics
                  (best-achievable reference; disable with --no-oracle)

Also exports run_episode and evaluate (reused by trainer.py).

Usage
-----
  # Auto-discover all models under outputs/models/<scenario>/ and write rollouts
  .venv/Scripts/python evaluate.py --scenario cstr --n-seeds 20

  # Specific checkpoints, no oracle
  .venv/Scripts/python evaluate.py --scenario cstr --no-oracle \\
      --models outputs/models/cstr/ddpg/best.pt

Output: outputs/rollouts/<scenario>/<timestamp>/  (<method>.npz + manifest.json)
"""

import argparse
import copy
import glob
import io
import json
import os
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG as SB3DDPG
from stable_baselines3 import TD3 as SB3TD3

from models import (
    PureDDPG,
    PureTD3,
    ShadowDDPG,
    ShadowSB3DDPG,
    ShadowSB3TD3,
    ShadowTD3,
)
from scenarios import SCENARIOS, make_env_for


# ---------------------------------------------------------------------------
# SB3 adapters — make a loaded SB3 model usable by the rollout recorder
# ---------------------------------------------------------------------------

class _SB3Adapter:
    """Plain SB3 model wrapper — always applies its own action (no switching)."""

    def __init__(self, model):
        self._model           = model
        self.total_steps      = 0
        self.warmup_steps     = 0
        self.max_t_train_frac = 0.0
        self.action_dim       = model.action_space.shape[0]

    def decide_action(self, obs, baseline_action, training=False):
        action, _ = self._model.predict(obs, deterministic=True)
        return action, True, action

    def store(self, *args):  pass
    def update(self):        pass
    def reset(self):         pass


class _ShadowSB3Adapter(_SB3Adapter):
    """Shadow SB3 model wrapper — q-value switching vs the passed baseline action."""

    def decide_action(self, obs, baseline_action, training=False):
        a_agent, _ = self._model.predict(obs, deterministic=True)
        obs_2d     = np.asarray(obs, dtype=np.float32)[None]
        q_agent    = self._model._q1(obs_2d, np.asarray(a_agent)[None])[0]
        q_baseline = self._model._q1(obs_2d, np.asarray(baseline_action)[None])[0]
        if q_agent > q_baseline:
            return a_agent, True, a_agent
        return baseline_action, False, a_agent


# ---------------------------------------------------------------------------
# NMPC oracle — receding-horizon do-mpc controller (best-achievable reference)
# ---------------------------------------------------------------------------

class NMPCController:
    """
    Nonlinear MPC oracle for a PC-Gym scenario.

    Built on PC-Gym's own do-mpc `oracle` (CasADi + IPOPT) so the prediction
    model is *exactly* the environment's dynamics — the "best achievable"
    reference. Exposes .predict(obs) / .reset() like the PID baselines and runs
    in true receding-horizon fashion against the real (noisy) environment.
    """

    def __init__(self, cfg: dict, horizon: int = 20):
        # Imported lazily so --no-oracle runs never pay the do-mpc import cost.
        from pcgym import make_env
        from pcgym.oracle import oracle

        # oracle mutates env_params in place, so hand it a private copy.
        env_params = copy.deepcopy(cfg["env_params"])
        self._oracle = oracle(make_env, env_params, MPC_params={"N": horizon})

        self._nx          = self._oracle.env.Nx_oracle
        self._a_low       = np.asarray(env_params["a_space"]["low"],  dtype=np.float64)
        self._a_high      = np.asarray(env_params["a_space"]["high"], dtype=np.float64)
        self._o_low       = np.asarray(env_params["o_space"]["low"],  dtype=np.float64)
        self._o_high      = np.asarray(env_params["o_space"]["high"], dtype=np.float64)
        self._normalise_o = env_params.get("normalise_o", True)
        self._x0          = np.asarray(cfg["env_params"]["x0"][:self._nx], dtype=np.float64)

        # Build the NLP once; reset() re-initialises it cheaply each episode.
        with redirect_stdout(io.StringIO()):
            self._mpc, _ = self._oracle.setup_mpc()

        # PC-Gym's shipped p_fun does `int(t_now/dt - 1)`, which crashes on modern
        # numpy (do-mpc passes t_now as a 1-D array) and lags the setpoint by one
        # step. make_step looks up self.p_fun dynamically, so we override it with a
        # scalar-safe version that tracks the *current* setpoint. Only setpoint
        # scheduling is supported (no disturbances / delta-u).
        if self._oracle.has_disturbances or self._oracle.use_delta_u:
            raise NotImplementedError(
                "NMPCController supports setpoint-tracking scenarios only "
                "(no disturbances or delta-u)."
            )
        self._mpc.p_fun = self._make_p_fun()
        self.reset()

    def _make_p_fun(self):
        """Build a scalar-safe do-mpc parameter function for setpoint tracking."""
        sp_dict        = self._oracle.env_params["SP"]
        sp_arrays      = [np.asarray(v, dtype=float) for v in sp_dict.values()]
        dt             = self._oracle.env.dt
        get_p_template = self._mpc.get_p_template

        def p_fun(t_now):
            t = float(np.asarray(t_now).flatten()[0])
            k = int(round(t / dt))                       # current step index
            p_template = get_p_template(1)
            sp_vals = [arr[min(k, len(arr) - 1)] for arr in sp_arrays]
            p_template["_p", 0, "SP"] = np.array(sp_vals).reshape(-1, 1)
            return p_template

        return p_fun

    def reset(self):
        """Restart for a new episode: reset the MPC clock and initial guess."""
        self._mpc.reset_history()            # resets internal t0 -> 0 (SP indexing)
        self._mpc.x0 = self._x0
        with redirect_stdout(io.StringIO()):
            self._mpc.set_initial_guess()

    def predict(self, obs, deterministic: bool = True):
        obs = np.asarray(obs, dtype=np.float64)
        if self._normalise_o:
            phys = (obs + 1.0) / 2.0 * (self._o_high - self._o_low) + self._o_low
        else:
            phys = obs
        x0 = phys[:self._nx].reshape(-1, 1)        # physical model state (no setpoints)

        with redirect_stdout(io.StringIO()):
            u = np.asarray(self._mpc.make_step(x0)).flatten()   # physical input(s)

        # Re-normalise to the env's [-1, 1] action space.
        u_norm = 2.0 * (u - self._a_low) / (self._a_high - self._a_low) - 1.0
        return np.clip(u_norm, -1.0, 1.0).astype(np.float32), None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _label_from_path(checkpoint_path: str) -> str:
    """Human-readable label inferred from the checkpoint's parent directory."""
    name = Path(checkpoint_path).parent.name

    # Exact matches for the standard (no-shadow) custom agents
    exact = {"ddpg": "DDPG", "td3": "TD3"}
    if name in exact:
        return exact[name]

    # SB3 backend (check before the generic shadow_ branch — shadow_sb3_* also
    # starts with "shadow_").
    if name.startswith("shadow_sb3_"):
        return f"Shadow SB3 {'TD3' if 'td3' in name else 'DDPG'}"
    if name.startswith("sb3_"):
        return f"SB3 {'TD3' if 'td3' in name else 'DDPG'}"

    # Shadow model names: shadow_<model>_<mode>[_reg<lambda>]
    if name.startswith("shadow_"):
        parts = name[len("shadow_"):]          # e.g. "td3_qvalue" or "ddpg_agent_reg2.0"
        model = parts.split("_")[0].upper()    # DDPG or TD3
        if "agent_reg" in parts:
            lam = parts.split("agent_reg")[-1]
            return f"Shadow {model} (Agent, lambda={lam})"
        if "agent" in parts:
            return f"Shadow {model} (Agent)"
        return f"Shadow {model} (Q-value)"

    # Pure (no-shadow) custom ablation: pure_<model>
    if name.startswith("pure_"):
        model = "TD3" if "td3" in name else "DDPG"
        return f"{model} (no shadow)"

    return name


def load_model(checkpoint_path: str, state_dim: int, action_dim: int):
    """
    Load a trained model from a checkpoint, inferred from its parent dir name:
      *.pt   — custom Shadow{DDPG,TD3} / Pure{DDPG,TD3}
      *.zip  — SB3 backend: plain SB3 DDPG/TD3, or Shadow SB3 DDPG/TD3
    """
    name   = Path(checkpoint_path).parent.name.lower()
    ext    = Path(checkpoint_path).suffix.lower()
    is_td3 = "td3" in name

    if ext == ".zip":        # SB3 backend
        if "shadow" in name:
            cls = ShadowSB3TD3 if is_td3 else ShadowSB3DDPG
            return _ShadowSB3Adapter(cls.load(checkpoint_path))
        cls = SB3TD3 if is_td3 else SB3DDPG
        return _SB3Adapter(cls.load(checkpoint_path))

    # .pt — custom core. Check "shadow" before "td3" so a no-shadow td3 maps to
    # PureTD3, not ShadowTD3.
    mode = "agent" if "agent" in name else "qvalue"

    if "shadow" not in name:        # pure / no-shadow custom ablation
        cls   = PureTD3 if is_td3 else PureDDPG
        agent = cls(state_dim=state_dim, action_dim=action_dim, mode="qvalue")
    elif is_td3:
        agent = ShadowTD3(state_dim=state_dim, action_dim=action_dim, mode=mode)
    else:
        agent = ShadowDDPG(state_dim=state_dim, action_dim=action_dim, mode=mode)

    agent.load(checkpoint_path)
    return agent


def discover_models(scenario: str, models_dir: str = "outputs/models") -> list[str]:
    """
    Return all checkpoint paths under models_dir/<scenario>/*/ — best.pt (custom
    core) and best_model.zip (SB3 backend).
    """
    paths = []
    for pattern in ("best.pt", "best_model.zip"):
        paths.extend(glob.glob(os.path.join(models_dir, scenario, "*", pattern)))
    return sorted(paths)


# ---------------------------------------------------------------------------
# Episode runner + evaluator  (reused by trainer.py)
# ---------------------------------------------------------------------------

def run_episode(env, agent, baseline, training: bool,
                seed: int, n_steps: int) -> tuple[float, list[bool]]:
    """
    Run one full episode. Compatible with any scenario and any ShadowDDPG-like
    instance (Pure* / Shadow* / SB3 adapters).

    Returns
    -------
    total_reward : float
    agent_flags  : list[bool] — True at each step the agent was in control
    """
    obs, _ = env.reset(seed=seed)
    baseline.reset()

    t_warmup = (
        int(np.random.uniform(0, agent.max_t_train_frac) * n_steps)
        if training else 0
    )

    total_reward = 0.0
    agent_flags: list[bool] = []
    transitions: list = []
    done = False
    step = 0

    while not done:
        a_baseline, _ = baseline.predict(obs)

        if step < t_warmup or (training and agent.total_steps < agent.warmup_steps):
            a_exec, a_agent, used_agent = a_baseline, a_baseline.copy(), False
        else:
            a_exec, used_agent, a_agent = agent.decide_action(
                obs, a_baseline, training=training
            )

        next_obs, reward, terminated, truncated, _ = env.step(a_exec)
        done = terminated or truncated

        agent_flags.append(used_agent)
        transitions.append((obs, a_exec, reward, next_obs, done, a_agent, a_baseline))
        total_reward += reward
        obs   = next_obs
        step += 1
        if training:
            agent.total_steps += 1

    if training:
        for t in transitions:
            agent.store(*t)
        if agent.total_steps >= agent.warmup_steps:
            for _ in transitions:
                agent.update()

    return total_reward, agent_flags


def evaluate(env, agent, baseline, n_seeds: int, n_steps: int) -> float:
    """Mean reward over n_seeds deterministic evaluation episodes."""
    rewards = [
        run_episode(env, agent, baseline, training=False, seed=s, n_steps=n_steps)[0]
        for s in range(n_seeds)
    ]
    return float(np.mean(rewards))


# ---------------------------------------------------------------------------
# Rollout recorder + writer  (the sole job of this module's CLI)
# ---------------------------------------------------------------------------

def _record_rollout(env, controller, scenario_baseline, seed: int,
                    n_steps: int) -> dict:
    """
    Run one deterministic episode, recording the full per-step model outputs.

    `controller` may be an RL agent (has .decide_action) or a plain controller
    (PID / NMPC, .predict only). `scenario_baseline` is the scenario's PID, used
    both as the shadow switching reference and to record a consistent baseline
    action column for every method.

    Returns arrays of shape [T, ...] (T = episode length):
      states, obs, actions, actions_agent, actions_baseline, rewards, takeover
    `takeover` is 1.0 (agent) / 0.0 (baseline) for RL agents, NaN otherwise.
    """
    is_agent = hasattr(controller, "decide_action")

    obs, _ = env.reset(seed=seed)
    if hasattr(controller, "reset"):
        controller.reset()
    scenario_baseline.reset()

    states, observations = [], []
    a_exec_l, a_agent_l, a_base_l = [], [], []
    rewards, takeover = [], []
    done, step = False, 0

    while not done and step < n_steps:
        a_baseline, _ = scenario_baseline.predict(obs)
        a_baseline = np.asarray(a_baseline, dtype=np.float32)

        if is_agent:
            a_exec, used_agent, a_agent = controller.decide_action(
                obs, a_baseline, training=False
            )
            a_exec  = np.asarray(a_exec,  dtype=np.float32)
            a_agent = np.asarray(a_agent, dtype=np.float32)
            tk = 1.0 if used_agent else 0.0
        else:
            a_exec, _ = controller.predict(obs)
            a_exec  = np.asarray(a_exec, dtype=np.float32)
            a_agent = a_exec.copy()
            tk = float("nan")          # takeover not applicable

        observations.append(np.asarray(obs, dtype=np.float32))
        obs, reward, terminated, truncated, _ = env.step(a_exec)
        done = terminated or truncated

        states.append(env.state.copy().astype(np.float32))
        a_exec_l.append(a_exec)
        a_agent_l.append(a_agent)
        a_base_l.append(a_baseline)
        rewards.append(float(reward))
        takeover.append(tk)
        step += 1

    return {
        "states":           np.asarray(states,        dtype=np.float32),
        "obs":              np.asarray(observations,  dtype=np.float32),
        "actions":          np.asarray(a_exec_l,      dtype=np.float32),
        "actions_agent":    np.asarray(a_agent_l,     dtype=np.float32),
        "actions_baseline": np.asarray(a_base_l,      dtype=np.float32),
        "rewards":          np.asarray(rewards,       dtype=np.float32),
        "takeover":         np.asarray(takeover,      dtype=np.float32),
    }


def _stack_episodes(episodes: list[dict]) -> dict:
    """Stack a list of per-episode [T,...] dicts into [N, T, ...] (truncate to min T)."""
    t_min = min(e["rewards"].shape[0] for e in episodes)
    return {k: np.stack([e[k][:t_min] for e in episodes], axis=0)
            for k in episodes[0]}


def run_rollouts(
    scenario:    str,
    model_paths: list[str],
    n_seeds:     int  = 10,
    use_oracle:  bool = True,
    mpc_horizon: int  = 20,
    models_dir:  str  = "outputs/models",
    output_dir:  str  = "outputs/rollouts",
):
    """
    Run every method (PID, NMPC oracle, and the given/discovered models) on the
    scenario for `n_seeds` seeds, serialising the raw rollouts to
    output_dir/<scenario>/<timestamp>/ as one .npz per method + manifest.json.
    No plotting or metric computation — that is the plotting utility's job.
    """
    cfg     = SCENARIOS[scenario]
    n_steps = cfg["n_steps"]

    if not model_paths:
        model_paths = discover_models(scenario, models_dir)
        if not model_paths:
            print(f"  No models found under {models_dir}/{scenario}/. "
                  "Recording reference controllers only.")

    # (slug, label, controller). References first.
    entries: list[tuple[str, str, object]] = [("pid", "PID", cfg["baseline_cls"]())]
    if use_oracle:
        print(f"  Building NMPC oracle (do-mpc + IPOPT, horizon={mpc_horizon})...")
        entries.append(("nmpc_oracle", "NMPC Oracle", NMPCController(cfg, horizon=mpc_horizon)))
    for path in model_paths:
        slug  = Path(path).parent.name
        entries.append((slug, _label_from_path(path),
                        load_model(path, cfg["state_dim"], cfg["action_dim"])))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = os.path.join(output_dir, scenario, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Scenario : {scenario}   |   methods: {len(entries)}   |   seeds: {n_seeds}")
    print(f"  Output   : {save_dir}")
    print(f"{'='*62}\n")

    env               = make_env_for(scenario)
    scenario_baseline = cfg["baseline_cls"]()   # shadow switching + baseline column
    methods_meta: list[dict] = []

    for slug, label, controller in entries:
        episodes = [
            _record_rollout(env, controller, scenario_baseline, seed=s, n_steps=n_steps)
            for s in range(n_seeds)
        ]
        data = _stack_episodes(episodes)
        meta = {"slug": slug, "label": label, "scenario": scenario,
                "n_seeds": n_seeds, "seeds": list(range(n_seeds))}
        npz_path = os.path.join(save_dir, f"{slug}.npz")
        np.savez(npz_path, meta=np.array(json.dumps(meta)), **data)

        mean_total = float(np.mean(data["rewards"].sum(axis=1)))
        print(f"  {label:<28}  mean total reward = {mean_total:9.1f}   -> {slug}.npz")
        methods_meta.append({"slug": slug, "label": label, "file": f"{slug}.npz"})

    # Manifest: everything the plotting utility needs to interpret the .npz files.
    manifest = {
        "scenario":    scenario,
        "timestamp":   timestamp,
        "n_seeds":     n_seeds,
        "n_steps":     n_steps,
        "tsim":        cfg["env_params"]["tsim"],
        "dt":          cfg["env_params"]["tsim"] / cfg["env_params"]["N"],
        "plot_config": cfg["plot_config"],
        "setpoints":   {k: list(map(float, v)) for k, v in cfg["env_params"]["SP"].items()},
        "methods":     methods_meta,
        "array_schema": {
            "states":           "[N, T, n_physical_states]  env.state each step (physical)",
            "obs":              "[N, T, obs_dim]            normalised observation",
            "actions":          "[N, T, action_dim]         executed action (normalised)",
            "actions_agent":    "[N, T, action_dim]         agent's proposed action",
            "actions_baseline": "[N, T, action_dim]         PID baseline action",
            "rewards":          "[N, T]                     per-step reward",
            "takeover":         "[N, T]                     1=agent, 0=baseline, NaN=N/A",
        },
    }
    with open(os.path.join(save_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Rollouts + manifest.json written to {save_dir}")
    return save_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run models on a PC-Gym scenario and serialise per-step rollouts."
    )
    parser.add_argument(
        "--scenario", type=str, default="cstr", choices=list(SCENARIOS.keys()),
        help="PC-Gym scenario to run on",
    )
    parser.add_argument(
        "--models", type=str, nargs="*", default=[], metavar="PATH",
        help="Checkpoint paths to run (default: auto-discover under "
             "outputs/models/<scenario>/)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Number of seeds (episodes) recorded per method (default: 10)",
    )
    parser.add_argument(
        "--no-oracle", action="store_true",
        help="Skip the NMPC oracle (faster; included by default)",
    )
    parser.add_argument(
        "--mpc-horizon", type=int, default=20,
        help="NMPC oracle prediction horizon in steps (default: 20)",
    )
    parser.add_argument(
        "--models-dir", type=str, default="outputs/models",
        help="Root directory where model checkpoints are stored",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/rollouts",
        help="Root directory for rollout outputs (default: outputs/rollouts/)",
    )
    args = parser.parse_args()

    run_rollouts(
        scenario=args.scenario,
        model_paths=args.models,
        n_seeds=args.n_seeds,
        use_oracle=not args.no_oracle,
        mpc_horizon=args.mpc_horizon,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
