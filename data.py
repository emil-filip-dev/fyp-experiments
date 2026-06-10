"""
data.py
=======
Offline dataset generation — the "historical and simulated process data" the RL
agent is pretrained on (project_proposal.md). Because the agent must learn
WITHOUT interacting with the (real) plant, we manufacture a STATIC dataset by
logging the expert controller operating the simulated process, with added action
perturbations for state-action coverage (a narrow expert-only dataset starves
offline RL of the counterfactuals it needs to improve on the expert).

A dataset is a set of (obs, action, reward, next_obs, done) transitions plus the
physical states (for diagnostics / dataset-safety reporting). It is built ONCE,
saved to .npz, and thereafter only read — the agent never steps the env during
offline pretraining.

  generate_dataset(scenario, expert, n_episodes, seed, ...) -> dict of arrays
  save_dataset / load_dataset                                -> .npz round-trip
  dataset_to_buffer(data)                                    -> models.ReplayBuffer
"""

import json
import os

import numpy as np

from constraints import constraint_metrics, violation_magnitudes
from models import ReplayBuffer
from scenarios import SCENARIOS, make_env_for
from schema import ExpertKind, Scenario


def generate_dataset(
    scenario: Scenario | str,
    expert,
    *,
    n_episodes: int = 200,
    seed: int = 0,
    perturb_frac: float = 0.4,
    perturb_std: float = 0.25,
    expert_kind: ExpertKind | None = None,
) -> dict:
    """
    Roll the expert over `n_episodes` episodes of the simulated process, logging
    every transition. On a fraction `perturb_frac` of steps the executed action is
    the expert's plus clipped Gaussian noise (std `perturb_std`, in normalised
    action space) — this injects the off-expert coverage offline RL needs while
    keeping the data centred on safe, expert-like behaviour.

    Returns a dict of stacked arrays:
      obs [M, obs_dim], actions [M, action_dim], rewards [M], next_obs [M, obs_dim],
      dones [M], states [M, n_states]   (M = n_episodes * n_steps)
    plus a JSON-able 'meta' dict (scenario, expert kind, sizes, dataset safety).
    """
    scenario = Scenario(scenario)
    cfg = SCENARIOS[str(scenario)]
    n_steps = cfg["n_steps"]
    constraint_spec = cfg.get("constraint_spec", [])
    env = make_env_for(str(scenario))
    rng = np.random.default_rng(seed)

    obs_l, act_l, rew_l, nobs_l, done_l, st_l = [], [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        if hasattr(expert, "reset"):
            expert.reset()
        done = False
        step = 0
        while not done and step < n_steps:
            a_expert, _ = expert.predict(obs)
            a_expert = np.asarray(a_expert, dtype=np.float32)
            if rng.random() < perturb_frac:
                a = a_expert + rng.normal(0.0, perturb_std, a_expert.shape).astype(np.float32)
                a = np.clip(a, -1.0, 1.0)
            else:
                a = a_expert
            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)

            obs_l.append(np.asarray(obs, dtype=np.float32))
            act_l.append(a.astype(np.float32))
            rew_l.append(np.float32(reward))
            nobs_l.append(np.asarray(next_obs, dtype=np.float32))
            done_l.append(np.float32(done))
            st_l.append(env.state.copy().astype(np.float32))

            obs = next_obs
            step += 1

    states = np.asarray(st_l, dtype=np.float32)
    # Dataset safety: how often the data-collection policy itself crossed a
    # constraint (the expert+perturbation should be largely safe — that is the
    # whole premise of pretraining on it).
    viol = violation_magnitudes(states, constraint_spec)
    if constraint_spec:
        vrate = float((viol > 0).any(axis=-1).mean())
        vmax = float(viol.max())
    else:
        vrate, vmax = 0.0, 0.0

    meta = {
        "scenario": str(scenario),
        "expert_kind": str(expert_kind) if expert_kind else None,
        "n_episodes": n_episodes, "n_steps": n_steps,
        "n_transitions": len(obs_l), "seed": seed,
        "perturb_frac": perturb_frac, "perturb_std": perturb_std,
        "dataset_viol_rate": vrate, "dataset_viol_max": vmax,
        "n_con": len(constraint_spec),
    }
    return {
        "obs": np.asarray(obs_l, dtype=np.float32),
        "actions": np.asarray(act_l, dtype=np.float32),
        "rewards": np.asarray(rew_l, dtype=np.float32),
        "next_obs": np.asarray(nobs_l, dtype=np.float32),
        "dones": np.asarray(done_l, dtype=np.float32),
        "states": states,
        "meta": meta,
    }


def save_dataset(data: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        obs=data["obs"], actions=data["actions"], rewards=data["rewards"],
        next_obs=data["next_obs"], dones=data["dones"], states=data["states"],
        meta=np.array(json.dumps(data["meta"])),
    )
    return path


def load_dataset(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {
        "obs": z["obs"], "actions": z["actions"], "rewards": z["rewards"],
        "next_obs": z["next_obs"], "dones": z["dones"], "states": z["states"],
        "meta": json.loads(str(z["meta"])),
    }


def dataset_to_buffer(data: dict, capacity: int | None = None) -> ReplayBuffer:
    """Load a dataset dict into a ReplayBuffer for offline pretraining.

    Capacity defaults to max(dataset size, 1e6) — generous HEADROOM above the
    dataset so that offline→online fine-tuning, which appends online transitions
    to this same buffer, does NOT evict the offline expert demonstrations (a
    bounded deque would otherwise overwrite them once online steps exceed the
    dataset size, causing the policy to forget the expert and degrade off the
    guarded distribution)."""
    n = len(data["obs"])
    buf = ReplayBuffer(capacity if capacity is not None else max(n, 1_000_000))
    buf.add_many(data["obs"], data["actions"], data["rewards"],
                 data["next_obs"], data["dones"])
    return buf


def dataset_cache_key(scenario: str, seed: int, expert_kind, n_episodes: int,
                      perturb_frac: float, perturb_std: float) -> str:
    """Filename uniquely determined by everything that affects the dataset."""
    ek = str(expert_kind) if expert_kind else "none"
    return (f"{scenario}__seed{seed}__ep{n_episodes}__{ek}"
            f"__pf{perturb_frac:g}_ps{perturb_std:g}.npz")


def get_or_make_dataset(scenario, expert, *, seed: int, n_episodes: int,
                        expert_kind=None, perturb_frac: float = 0.4,
                        perturb_std: float = 0.25,
                        cache_dir: str = "outputs/cache/datasets") -> dict:
    """
    Return the offline dataset for (scenario, seed, expert, episodes, perturb...),
    loading it from `cache_dir` if a matching one was generated before, otherwise
    generating (the expensive NMPC-heavy step) and caching it. This is what lets
    the offline and offline-to-online conditions at the same seed share one dataset
    instead of each re-running the expert.
    """
    key = dataset_cache_key(str(scenario), seed, expert_kind, n_episodes,
                            perturb_frac, perturb_std)
    path = os.path.join(cache_dir, key)
    if os.path.exists(path):
        print(f"  [dataset cache] hit -> {path}")
        return load_dataset(path)
    print("  [dataset cache] miss; generating (expert + perturbations)...")
    data = generate_dataset(scenario, expert, n_episodes=n_episodes, seed=seed,
                            perturb_frac=perturb_frac, perturb_std=perturb_std,
                            expert_kind=expert_kind)
    save_dataset(data, path)
    print(f"  [dataset cache] saved -> {path}")
    return data


def describe_dataset(data: dict) -> str:
    m = data["meta"]
    return (f"dataset[{m['scenario']}] expert={m['expert_kind']} "
            f"transitions={m['n_transitions']:,} "
            f"({m['n_episodes']}x{m['n_steps']})  "
            f"viol_rate={m['dataset_viol_rate']*100:.2f}%  vmax={m['dataset_viol_max']:.3g}")
