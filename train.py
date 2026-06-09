"""
train.py
========
Train an RL agent on a PC-Gym process-control environment — standard or
shadow-mode — and the shared episode-based training loop.

Programmatic API (no CLI):
  - `train_condition(condition, scenario, total_steps, seed, ...)` — train one
    `schema.Condition` (standard or shadow) on one (scenario, seed). The Condition
    is the single source of the agent's hyperparameters, the run-label slug, and
    the persisted RunSpec/run.json.
  - `train_model(agent, ...)` — the lower-level loop, if you already hold an agent.
  - `run_episode` (in evaluate.py) is the shared per-episode runner.

Shadow switching (Condition.shadow + switching_mode):
  qvalue  — execute agent action if Q(s, a_agent) > Q(s, a_baseline)
  agent   — agent outputs a control-authority probability; act if > eta_agent
            (optional L1 regularisation toward the baseline via lambda_reg)

Outputs: outputs/models/<scenario>/<run_label>[/seed<k>]/ — best.pt + run.json +
training_log.npz (+ takeover.png/npz for shadow runs).
"""

import json
import os

import numpy as np
import torch
from tqdm import tqdm

from constraints import violation_magnitudes
from evaluate import run_episode
from models import get_shadow_model, get_standard_model, TD3SwitchCritic
from scenarios import SCENARIOS, make_env_for
from schema import Condition, Device, RunSpec, Scenario
from util import device_label, resolve_device


def _print_train_header(scenario, model_desc, device, total_steps, seed):
    print(f"\n{'='*60}")
    print(f"  Scenario : {scenario}")
    print(f"  Model    : {model_desc}")
    print(f"  Device   : {device_label(device)}")
    print(f"  Steps    : {total_steps:,}  |  seed={seed}")
    print(f"{'='*60}\n")


def train_condition(
    condition:   Condition,
    scenario:    Scenario,
    total_steps: int,
    seed:        int,
    *,
    eval_freq:   int  = 1_000,
    checkpoint_freq: int = 0,
    device:      Device = Device.CPU,
    output_dir:  str  = "outputs/models",
    per_seed_dir: bool = False,
):
    """
    Train one Condition on one (scenario, seed). The Condition is the SINGLE source
    of the agent's hyperparameters, the run-label slug, and the RunSpec — there is
    no string round-trip. Returns the trained agent.
    """
    cfg = SCENARIOS[scenario]
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = resolve_device(device)
    state_dim, action_dim = cfg["state_dim"], cfg["action_dim"]

    if condition.shadow:
        extra = {}
        if condition.is_td3_qvalue:
            extra["switch_critic"] = (condition.switch_critic or TD3SwitchCritic.Q1).value
        agent = get_shadow_model(condition.algorithm.value)(
            state_dim=state_dim, action_dim=action_dim, switching_mode=condition.mode,
            eta_agent=condition.eta_agent, lambda_reg=condition.lambda_reg, device=dev, **extra,
        )
        model_desc = (f"Shadow {condition.algorithm.value.upper()} "
                      f"(mode={condition.mode.value}, lambda={condition.lambda_reg}, eta={condition.eta_agent})")
    else:
        agent = get_standard_model(condition.algorithm.value)(
            state_dim=state_dim, action_dim=action_dim, lambda_reg=condition.lambda_reg, device=dev,
        )
        model_desc = f"{condition.algorithm.value.upper()} (standard, no shadow)"

    run_spec = condition.to_run_spec(scenario, seed, total_steps)
    _print_train_header(scenario, model_desc, dev, total_steps, seed)
    return train_model(
        agent, run_spec, output_dir=output_dir, eval_freq=eval_freq,
        checkpoint_freq=checkpoint_freq, per_seed_dir=per_seed_dir,
    )


def train_model(
    agent,
    run_spec:    RunSpec,
    *,
    output_dir:  str  = "outputs/models",
    eval_freq:   int  = 1_000,
    checkpoint_freq: int = 0,
    n_eval:      int  = 5,
    per_seed_dir: bool = False,
):
    """
    Run the shared episode-based training loop for any ShadowDDPG-like agent
    (PureDDPG / PureTD3 / ShadowDDPG / ShadowTD3). All run identity (scenario,
    run-label, total steps, seed, shadow) comes from `run_spec` — the single
    source of truth. Saves best.pt whenever the periodic evaluation reward
    improves, plus a periodic snapshot epN.pt every `checkpoint_freq` episodes
    (0 disables the snapshots).

    `per_seed_dir` opts into a per-seed leaf directory (.../<run_label>/seed<k>/)
    so multi-seed orchestration does not overwrite checkpoints across seeds. It
    defaults to False to preserve the single-run layout (.../<run_label>/).
    """
    scenario    = run_spec.scenario
    run_label   = run_spec.run_label
    total_steps = run_spec.total_steps
    seed        = run_spec.seed
    is_shadow   = run_spec.shadow

    cfg       = SCENARIOS[scenario]
    n_steps   = cfg["n_steps"]
    constraint_spec = cfg.get("constraint_spec", [])
    save_path = os.path.join(output_dir, scenario, run_label)
    if per_seed_dir:
        save_path = os.path.join(save_path, f"seed{seed}")
    os.makedirs(save_path, exist_ok=True)

    # Structured run descriptor — the source of truth for this run's identity, so
    # nothing downstream needs to parse the directory slug (see schema.py).
    with open(os.path.join(save_path, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_spec.to_json(), f, indent=2)

    env      = make_env_for(scenario)
    eval_env = make_env_for(scenario)
    baseline = cfg["baseline_cls"]()

    episode        = 0
    best_reward    = -np.inf
    recent_rewards: list[float] = []
    next_eval      = eval_freq
    last_eval_tk: float | None = None

    # Behaviour-time log: one row PER TRAINING EPISODE for the policy that actually
    # ran on the "plant" (incl. exploration + warmup) — the basis for the C1
    # safety-during-training claim. Parallel arrays keyed by cumulative env step.
    beh = {"steps": [], "return": [], "viol_count": [], "viol_rate": [],
           "viol_max": [], "takeover": []}
    # Deployment log: one row per deterministic eval boundary (greedy, no explore).
    evl = {"steps": [], "return": [], "takeover": []}

    bar = tqdm(total=total_steps, unit="step", dynamic_ncols=True, desc=run_label)

    while agent.total_steps < total_steps:
        ep_seed      = episode + seed * 10_000
        steps_before = agent.total_steps

        reward, flags, beh_states = run_episode(
            env, agent, baseline, training=True, seed=ep_seed,
            n_steps=n_steps, collect_states=True)
        recent_rewards.append(reward)
        bar.update(agent.total_steps - steps_before)

        # Behaviour-time constraint violations on the trajectory that just ran
        # (physical env.state vs the scenario's constraint_spec; zero if none).
        bv = violation_magnitudes(beh_states, constraint_spec)
        any_v = (bv > 0).any(axis=-1) if bv.shape[-1] else np.zeros(bv.shape[0], dtype=bool)
        beh["steps"].append(int(agent.total_steps))
        beh["return"].append(float(reward))
        beh["viol_count"].append(int(any_v.sum()))
        beh["viol_rate"].append(float(any_v.mean()) if any_v.size else 0.0)
        beh["viol_max"].append(float(bv.max()) if bv.size else 0.0)
        beh["takeover"].append(float(np.mean(flags)) * 100.0)

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
            evl["steps"].append(int(agent.total_steps))
            evl["return"].append(eval_r)
            evl["takeover"].append(eval_tk)
            last_eval_tk = eval_tk
            if eval_r > best_reward:
                best_reward = eval_r
                agent.save(os.path.join(save_path, "best.pt"))
                tqdm.write(f"  [Save] step {agent.total_steps:,} - new best eval: {best_reward:.1f}")
            # Persist the behaviour-time log incrementally so a killed run keeps data.
            _save_training_log(save_path, scenario, run_label, seed, agent.warmup_steps,
                               total_steps, constraint_spec, beh, evl)
            if is_shadow:
                tqdm.write(f"  [Eval] step {agent.total_steps:,} | reward {eval_r:.1f} | "
                           f"deploy takeover {eval_tk:.1f}%")
                _plot_takeover(list(zip(beh["steps"], beh["takeover"])),
                               list(zip(evl["steps"], evl["takeover"])),
                               agent.warmup_steps, save_path, run_label)

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

    # Guarantee a best.pt exists even if no periodic eval ever improved on -inf
    # (e.g. total_steps < eval_freq): persist the final model so downstream
    # rollouts/orchestration never silently skip this run.
    best_path = os.path.join(save_path, "best.pt")
    if not os.path.exists(best_path):
        agent.save(best_path)

    # Final behaviour-time log (returns + violations + takeover over steps) for
    # EVERY run — this is the C1/C2/C3 training-curve data the analysis utility reads.
    log_path = _save_training_log(save_path, scenario, run_label, seed, agent.warmup_steps,
                                  total_steps, constraint_spec, beh, evl)
    tqdm.write(f"  Training log: {log_path}")

    # Shadow runs: also emit the takeover graph (training behaviour + deployment).
    if is_shadow:
        tk_png = _plot_takeover(list(zip(beh["steps"], beh["takeover"])),
                                list(zip(evl["steps"], evl["takeover"])),
                                agent.warmup_steps, save_path, run_label)
        np.savez(
            os.path.join(save_path, "takeover.npz"),
            train_steps=np.array(beh["steps"]), train_takeover=np.array(beh["takeover"]),
            eval_steps=np.array(evl["steps"]),  eval_takeover=np.array(evl["takeover"]),
        )
        if tk_png:
            tqdm.write(f"  Takeover graph: {tk_png}")

    tqdm.write(f"\n  Training complete.  Best eval reward: {best_reward:.1f}")
    tqdm.write(f"  Saved: {os.path.join(save_path, 'best.pt')}")
    return agent


def _save_training_log(save_path, scenario, run_label, seed, warmup, total_steps,
                       constraint_spec, beh, evl):
    """
    Serialise the behaviour-time training log to training_log.npz (written for
    every run, standard or shadow). Two parallel-array groups keyed by cumulative
    env step:
      behaviour (per training episode — the policy that ran on the plant):
        beh_steps, beh_return, beh_viol_count, beh_viol_rate, beh_viol_max, beh_takeover
      deployment (per deterministic eval boundary, greedy):
        eval_steps, eval_return, eval_takeover
    A JSON `meta` field records scenario/run/seed, warmup + total steps, and the
    constraint spec so the analysis utility can interpret the violation columns.
    Returns the .npz path.
    """
    meta = {
        "scenario": scenario, "run_label": run_label, "seed": int(seed),
        "warmup_steps": int(warmup), "total_steps": int(total_steps),
        "n_con": len(constraint_spec),
        "constraints": [{k: c[k] for k in ("name", "label", "bound", "type", "unit")
                         if k in c} for c in constraint_spec],
        "schema": {
            "beh_steps": "cumulative env steps at training-episode end",
            "beh_return": "behaviour episode return (incl. exploration + warmup)",
            "beh_viol_count": "# steps with any constraint violation (behaviour)",
            "beh_viol_rate": "fraction of episode steps violated (behaviour)",
            "beh_viol_max": "max per-step violation magnitude (behaviour)",
            "beh_takeover": "agent takeover % over the episode (100 for standard)",
            "eval_steps": "cumulative env steps at eval boundary",
            "eval_return": "mean deterministic (greedy) eval return",
            "eval_takeover": "deployment takeover % (greedy)",
        },
    }
    path = os.path.join(save_path, "training_log.npz")
    np.savez(
        path,
        meta=np.array(json.dumps(meta)),
        beh_steps=np.array(beh["steps"]),       beh_return=np.array(beh["return"]),
        beh_viol_count=np.array(beh["viol_count"]), beh_viol_rate=np.array(beh["viol_rate"]),
        beh_viol_max=np.array(beh["viol_max"]), beh_takeover=np.array(beh["takeover"]),
        eval_steps=np.array(evl["steps"]),      eval_return=np.array(evl["return"]),
        eval_takeover=np.array(evl["takeover"]),
    )
    return path


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
