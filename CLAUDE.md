# FYP Project — Claude Code Instructions

## Autonomous Operation
Operate autonomously. Do not ask for confirmation before taking actions unless they are irreversible and destructive (e.g. deleting files, force-pushing). Proceed with reading, writing, editing, running scripts, and installing packages without prompting.

## Project Overview
Imperial College London FYP: Making RL reliable enough for industrial process control via hard constraint satisfaction.

Core research direction: Extend Shadow Mode RL (Gassert & Althoff, 2024) to PC-Gym benchmark environments, combining it with constraint-handling mechanisms from CIRL and learning-to-defer frameworks.

The empirical goal and the experiment design are spelled out in **`dissertation_plan.md`** —
the authoritative roadmap. It decomposes "shadow mode works" into four falsifiable claims
(C1 safety-during-training, C2 no-performance-sacrifice, C3 selective/earned takeover,
C4 reliability), defines the comparison set and statistics, and ends with a codebase-readiness
table + phased TODO. **Read it before doing experiment work.**

## Key Papers
1. **Shadow Mode RL** (Gassert & Althoff, 2024) — core method being extended
2. **PC-Gym** (Bloor et al., 2024) — benchmark environments (CSTR, crystallization, four-tank, multistage extraction)
3. **CIRL** (Bloor et al., 2024) — PID-embedded RL policy for process control
4. **RL Survey for PSE** (Bloor et al., 2025) — background and metrics
5. **Learning-to-Defer / SLTD** (Joshi et al., 2022) — deferral framework for sequential decisions

## Environment
- **Python**: 3.12 via `.venv` (use `.venv/Scripts/python` — do NOT use system Python)
- **Activate venv**: `.venv/Scripts/activate`
- **Run scripts**: `.venv/Scripts/python <script.py>`
- **Install packages**: `.venv/Scripts/pip install <pkg>`
- **OS**: Windows. Default shell is PowerShell; a Bash tool is also available.
  `util.configure_utf8_output()` exists for non-ASCII stdout on Windows but is no longer
  auto-invoked (the CLIs were removed — see below); call it yourself in a driver if needed.

## Dependencies (requirements.txt)
tqdm, numpy, matplotlib, casadi, jax[cpu], equinox, diffrax, do-mpc[full], pcgym.
Also required at runtime: **torch** (the entire model core is PyTorch) and **gymnasium**
(pulled in transitively by pcgym). **`rliable`** (robust RL statistics) is installed in `.venv`
and to be wired into the analysis utility (Phase 4 — see `dissertation_plan.md` §6).

> History: `stable-baselines3` and `gymnasium` used to be explicit deps for an SB3 backend.
> The SB3 backend was removed in the "Human refactor" (commit `f386748`, 2026-06-08); the
> codebase is now a single custom PyTorch core. Do not reintroduce SB3 unless asked.

## PC-Gym API
```python
from pcgym import make_env

env_params = {
    'N': 100,           # number of time steps
    'tsim': 26,         # simulation time (minutes)
    'SP': {'Ca': [...]},  # setpoint schedule
    'o_space': {'low': ..., 'high': ...},
    'a_space': {'low': ..., 'high': ...},
    'x0': np.array([...]),
    'model': 'cstr',    # or 'multistage_extraction', 'crystallization', 'four_tank'
    'normalise_a': True,
    'normalise_o': True,
    'noise': True,
    'integration_method': 'casadi',
    'noise_percentage': 0.001,
    # NOW USED on all four scenarios (PC-Gym-native constraints, see scenarios.py):
    #   'constraints': {...}, 'cons_type': '<=' / '>=', 'done_on_cons_vio': False,
    #   'r_penalty': False  -> per-step magnitudes in info['cons_info']; oracle reads them too.
}

env = make_env(env_params)
obs, _ = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
# env.state — always in physical (unnormalised) units
```

## Codebase Architecture
A small set of shared **library modules** (no CLIs — invoked programmatically; the argparse
entry points were removed for simplicity). `schema.py` holds the typed vocabulary (StrEnums +
dataclasses); scenarios, models, and the episode runner are factored out so training and
evaluation share the same code paths. The model layer is **one custom PyTorch core** — there
is no second (SB3) backend.

**Design rules (enforced):** run metadata lives in **typed objects serialised to JSON**
(`schema.py`: `RunSpec`, `ModelSpec`, `MethodRecord`), never derived from directory slugs;
categorical values are **StrEnums** (`Scenario`, `Algorithm`, `SwitchingMode`, `TD3SwitchCritic`,
`MethodRole`, `Device`), not bare strings; type hints everywhere. Slugs only *locate* files.

- `scenarios.py` — central registry (`SCENARIOS` dict) of all four PC-Gym environments.
  **`env_params` are copied VERBATIM from PC-Gym's own paper training scripts**
  (`pc-gym_paper/train_policies/<env>/<env>_train.py` in github.com/MaximilianB2/pc-gym) —
  x0, o_space, a_space, SP schedules, tsim, N, noise, delta-u settings, and the custom OCP
  reward functions are exact copies, NOT reconstructions. **Do NOT "improve"/reconstruct these
  — copy from the source code (the docs pages disagree with the code; the code wins).** A
  verified verbatim copy is the whole point. The earlier hand-reconstructed configs were wrong
  (crystallization conc off ~1000× → NaN; multistage controlled the wrong variable; etc.).
  Each entry also carries `state_dim`, `action_dim`, `n_steps`, a `baseline_cls`, `plot_config`
  — these last two are **ours** (not PC-Gym), the baseline being a PID/PI safety-net for shadow
  mode that drives the SAME variable(s) PC-Gym controls. **All four train with DDPG + shadow DDPG.**
  - **Controlled variables / setup (per the verbatim source):** `cstr` Ca via Tc, a_space
    [295,302], SP 3-segment 0.85/0.9/0.87, N=60. `four_tank` controls **h3 & h4** (not h1/h2),
    a_space [0.1,0.1]–[10,10], o_space [0,0.6], noise 5%, tsim=1000 (slow). `multistage_extraction`
    controls **X5** (not X1), a_space low[5,10] high[500,1000] (L drives X5), SP 0.3/0.4/0.3.
    `crystallization` is **delta-u** (`a_delta=True`, `a_space_act`=[10,40]°C, `a_0`=39), controls
    **CV & Ln** (`state_dim=9`); x0 conc≈0.1586 + computed CV₀/Ln₀ (NOT 0.5/15). The NMPC oracle
    does **not** support delta-u → `run_rollouts` skips it for `crystallization`.
  - **Custom rewards** (copied verbatim, used via `env_params["custom_reward"]`): `sp_track_reward`
    (OCP setpoint tracking + R=0.1 Δu penalty) for cstr/four_tank/multistage; `cryst_oracle_reward`
    (CV/Ln tracking + R=0.01 Δu penalty) for crystallization.
  - **Constraints (Phase-1 safety layer) are ADDED on all four scenarios** — PC-Gym-native
    (`env_params` `constraints`/`cons_type`/`done_on_cons_vio=False`/`r_penalty=False`) so the
    env records `info["cons_info"]` and the do-mpc oracle enforces them as hard state bounds;
    also mirrored in a `constraint_spec` (list of dicts: `state_idx`/`bound`/`type`/`name`/`label`/
    `unit`) that the eval pipeline uses for its own `env.state`-based detection (see `constraints.py`).
    `cstr` is **verbatim PC-Gym** (T ∈ [321,327] K, from the constraint_showcase); `four_tank`
    (h3,h4 ≤ 0.6 m overflow), `multistage` (X5 ≤ 0.5 off-spec), `crystallization` (Conc ≥ 0.11
    over-depletion) are **OURS**, physically-calibrated so nominal control stays inside while
    exploration crosses them — full justification in `constraints_rationale.md`.

- `models.py` — all model definitions, one shared core; variants differ only in `decide_action`,
  a clean ablation set.
  - **Building blocks**: `Actor` (q-value mode → action vector; agent mode → `(action,
    decision_prob∈[0,1])`), `Critic` (single Q, used by ShadowDDPG), `CriticTwin` (twin Q1/Q2
    with `q_min` / `q1_only`, used by ShadowTD3), `ReplayBuffer` (stores executed action +
    agent action + baseline action per transition). Abstract base `_ShadowModel` holds the
    shared `decide_action` / `store` / `save` / `load` logic, plus `q_gap(obs, baseline_action)`
    → `Q(s, a_agent) − Q(s, a_baseline)` under the switching critic (q-value only; NaN for
    agent-mode / non-RL) — recorded per rollout step for the takeover-vs-advantage analysis (C3).
  - **Shadow agents** `ShadowDDPG` / `ShadowTD3`. Switching `mode`:
    - `qvalue` (`SwitchingMode.Q_VALUE`) — act if `Q(s, a_agent) > Q(s, a_baseline)` (Eq. 6;
      compares on the deterministic policy action, executes the noised one).
    - `agent` (`SwitchingMode.AGENT`) — the actor emits a control-authority probability; act if
      `> eta_agent` (Eq. 4). Optional `lambda_reg` adds an L1 reward penalty `−λ‖a_agent −
      a_baseline‖` toward the baseline (Eq. 5; **agent mode only** — shapes the reward/Q, not
      the actor loss). In agent mode the critic operates on the augmented action `(a^a,
      a^decision)` → `action_dim + 1` inputs.
    - `ShadowTD3` adds twin critics, delayed policy updates, target-policy smoothing, and a
      `switch_critic` choice (`TD3SwitchCritic`: `q1` actor-consistent default, or `qmin`
      conservative) used **only** for its q-value switching decision.
  - **Standard (no-shadow) agents** `DDPG(ShadowDDPG)` / `TD3(ShadowTD3)` — same core, but
    `decide_action` is overridden to **always execute the agent's own action** (switching off).
    These are THE fair "standard DDPG/TD3" baseline for the shadow ablation (identical learner,
    hyperparameters, PID-assisted warmup). Labelled `"DDPG"` / `"TD3"`.
  - **Enums / dispatch**: `StandardModels {ddpg, td3, ppo}`, `ShadowModels {ddpg, td3}`.
    `get_standard_model(name)` / `get_shadow_model(name)` return the class. **Note:** `ppo` is
    listed in `StandardModels` but **not implemented** (no PPO class; `get_standard_model`
    raises for it) — shadow switching needs a deterministic off-policy actor-critic.
  - **Save/load**: `agent.save(path)` writes a dict checkpoint via `_save_dict()` — keys: `type`
    (a `StandardModels`/`ShadowModels` enum), all hyperparameters, `state_dicts`, and `internal`
    (replay buffer, `total_steps`, takeover counts). Classmethod `Model.load(ckpt, device)`
    reconstructs from that dict. Checkpoint files: `best.pt` (best eval) + `epN.pt` (snapshots).
  - **`NMPCController`** — NMPC oracle (best-achievable reference / optimality-gap ceiling).
    Wraps PC-Gym's own do-mpc `oracle` (CasADi + IPOPT) so the prediction model is *exactly*
    the env dynamics; exposes `.predict(obs)` / `.reset()` like the PID baselines and runs true
    receding-horizon against the noisy env. **Setpoint-tracking only** — raises
    `NotImplementedError` on scenarios with disturbances or delta-u. Overrides PC-Gym's buggy
    `p_fun` with a scalar-safe, non-lagging setpoint lookup.

- `util.py` — `configure_utf8_output()`, `resolve_device("cpu"|"gpu")` (accepts a `Device`
  StrEnum too, since it's a `str`), `device_label()`.

- `schema.py` — **the shared typed vocabulary** (depends only on `models`; no cycles). StrEnums:
  `Scenario`, `Algorithm {ddpg,td3}`, `MethodRole {model,pid,nmpc_oracle}`, `Device {cpu,gpu}`
  (+ re-uses `SwitchingMode`/`TD3SwitchCritic` from `models`). Dataclasses (frozen, with
  `to_json`/`from_json` via an `asdict(dict_factory=…)` single pass):
  - `Condition` — a learner config (algorithm + shadow settings). The **single factory** for
    both the run-label slug (`.slug` via `run_label_for()`) and the metadata (`.to_run_spec()`);
    `__post_init__` rejects incoherent combos. Helpers `standard(...)` / `shadow(...)`.
  - `RunSpec` — a Condition bound to (scenario, seed, total_steps); written as `run.json`.
    `.artifact_stem` gives the rollout filename stem.
  - `ModelSpec` — checkpoint path + its `RunSpec` (the typed input to `run_rollouts`).
  - `MethodRecord` — one structured rollout-manifest entry (role + optional `RunSpec`).
  - Protocols `ReferenceController` / `ShadowAgent` (`RolloutController` union) replace `object`.

- `train.py` — standard OR shadow training, custom core. **No CLI** — programmatic API:
  - `train_condition(condition, scenario, total_steps, seed, *, eval_freq, checkpoint_freq,
    device, output_dir, per_seed_dir)` — trains one `Condition` on one (scenario, seed). The
    Condition is the single source of the agent hyperparameters, the slug, and the `RunSpec`.
  - `train_model(agent, run_spec, *, …)` — the inline episode-based loop. Periodic deterministic
    eval every `eval_freq` steps; saves `best.pt` on eval improvement (always guarantees a
    `best.pt` exists). Writes **`run.json`** (the RunSpec) + a behaviour-time **`training_log.npz`**
    per run (per-episode behaviour return + violation count/rate/max + takeover; per-eval
    deterministic return + takeover; JSON `meta`), plus `takeover.png`/`.npz` for shadow runs.
  - Run-label slugs (via `schema.run_label_for`): standard → `ddpg`/`td3`; shadow →
    `shadow_<algo>_<mode>`, `…_agent_reg<λ>`, `…_qvalue_<switch_critic>` (TD3 q-value).

- `evaluate.py` — runs models + references on a scenario and **serialises raw per-step rollouts**;
  no plotting/metrics (that is Phase 4). **No CLI.**
  - `run_episode(...)` — the shared per-episode runner, reused by `train.py`.
  - `run_rollouts(scenario, model_specs: list[ModelSpec], n_seeds, use_oracle, …)` runs PID +
    NMPC oracle + every given model, writing one `<run.artifact_stem>.npz` per model (+ `pid.npz`/
    `nmpc_oracle.npz`) and a `manifest.json` under `outputs/rollouts/<scenario>/<timestamp>/`.
    Method identity is carried by **`MethodRecord`** (role + `RunSpec`), in the manifest and each
    `.npz`'s `meta` — never parsed from a filename. Per-method arrays `[N, T, …]`: `states`
    (physical `env.state`), `obs`, `actions`, `actions_agent`, `actions_baseline`, `rewards`,
    `takeover` (1/0/NaN), **`q_gap`** (C3 advantage; NaN if N/A), **`violations`** `[N,T,n_con]`
    (from `env.state` vs `constraint_spec`). Manifest carries timing, `plot_config`, setpoints,
    `constraints`, and the array schema.

- `experiments.py` — **declarative config** (not execution). `GRIDS` registry of named
  `ExperimentGrid`s (envs × `Condition`s × seeds + budgets/rollout settings). Helpers:
  `iter_training_jobs`, `iter_model_refs`, `checkpoint_path`, `describe_grid`, `write_provenance`
  (snapshots resolved grid + git commit + lib versions). Grid building blocks live here; the
  `Condition` type itself lives in `schema.py`.

- `run_experiments.py` — **Phase-3 orchestrator** (programmatic, no CLI). `run_grid(grid, …)`
  writes provenance then runs `train_grid` (drives `train_condition`; resumable — skips existing
  `best.pt`; isolates per-job failures) and `rollout_grid` (builds `ModelSpec`s straight from the
  grid `Condition`s, calls `run_rollouts`). `override_grid(...)` subsets a grid for smoke runs;
  `Stage {all,train,rollouts}` selects stages.

- `constraints.py` — constraint-violation **detection** (`violation_magnitudes`, from `env.state`)
  + **metrics** (`constraint_metrics`: count/rate/magnitude/median+MAD, timing). Library only.

- `dissertation_plan.md` — the experiment plan and codebase-readiness assessment (see Project
  Overview). **The source of truth for what to build next.**
- `findings.md` — research notes (literature synthesis, pseudocode, extension designs).
- `TODO.txt` — current open research tasks.
- `examples/example_pcgym.py` — original CSTR + PPO + PID reference (standalone).
- `requirements.txt` — dependencies (see above).

### Output layout
- `outputs/models/<scenario>/<run_label>[/seed<k>]/` — `best.pt` (best-evaluating; `per_seed_dir`
  adds the `seed<k>` leaf for multi-seed runs) + `run.json` (the `RunSpec`) + `training_log.npz`
  (+ `takeover.png`/`.npz` for shadow runs). Snapshots `epN.pt` only if `checkpoint_freq > 0`
  (default 0).
- `outputs/rollouts/<scenario>/<timestamp>/` — `run_rollouts` output: one `.npz` per method
  (`<run_label>__seed<k>.npz`, `pid.npz`, `nmpc_oracle.npz`) + `manifest.json` with structured
  `MethodRecord`s. Raw per-step rollouts for the Phase-4 analysis utility.
- `outputs/experiments/<grid>/provenance.json` — resolved grid + git commit + library versions.
- `outputs/runs/`, `outputs/analysis/`, `runs/` — stale artifacts from the removed plotting
  subsystem / pre-refactor runs. Not written by current code.

### Typical workflow (programmatic — no CLI)
```python
from experiments import GRIDS
from run_experiments import run_grid, override_grid
from schema import Device

grid = GRIDS["phase1_cstr_fourtank"]                       # envs × conditions × seeds
grid = override_grid(grid, env_names=["cstr"], seeds=[0,1], steps=5000)  # smaller smoke grid
run_grid(grid, device=Device.CPU)                          # train + rollouts + provenance
# Phase-4 analysis utility (metrics/figures from the rollout .npz + training_log.npz): TO BE BUILT
```

## Metrics
Target metric set for evaluating each method across seeds:
- **Control/tracking**: median episodic return + MAD across seeds (robust to RL outliers —
  prefer median/MAD over mean/std); IAE/ISE per controlled output; overshoot, settling
  time, steady-state offset per setpoint segment.
- **Constraint/safety**: violation rate, count, max magnitude, time-to-recovery
  (requires `constraints` to be added to scenarios + `info["cons_info"]` captured — not yet done).
- **Shadow-specific**: agent takeover fraction (overall + over time), divergence from
  baseline (‖a_agent − a_baseline‖), Δ vs baseline return.
- **Sample efficiency**: steps-to-threshold, area under learning curve, asymptotic reward.
- **Robustness**: MAD/IQR/worst-case return across seeds.
- **Control effort**: total/mean |Δu|.
- **Optimality gap**: Δ = J(π*) − J(π_θ), π* = NMPC oracle (`models.NMPCController`, recorded in
  every rollout unless `use_oracle=False`). Normalise scores `[PID = 0, oracle = 1]` for cross-env
  aggregation. Use `rliable` (IQM, bootstrap CIs, P(Shadow > Pure)) for robust stats.

## Metrics Pipeline (current state)
Decoupled run → store → (rebuild) analyse:
- **Evaluation side (built).** `evaluate.run_rollouts()` writes raw per-step rollouts incl.
  `violations` (from `env.state` vs `constraint_spec`) and `q_gap`, with structured `MethodRecord`
  metadata. (Phase 2 ✅)
- **Training side (built).** `train_model` serialises a per-run `training_log.npz` (behaviour
  return + violation count/rate/max + takeover per episode; deterministic eval return + takeover
  per boundary) + `run.json`. (Phase 2 ✅)
- **Orchestration (built).** `run_experiments.run_grid()` loops conditions × seeds × envs
  (resumable, provenance, failure-isolated). (Phase 3 ✅)
- **Plotting / metrics utility (MISSING — the main build, Phase 4).** A new `analysis.py` should
  load the rollout `.npz` + `manifest.json` + per-run `training_log.npz` and emit metric tables
  (CSV) + figures (PNG): control (IAE/ISE, overshoot, settling, offset, median+MAD return), safety
  (C1), optimality gap (C2), learning curves, takeover analysis (C3), and `rliable` robust stats.
  See `dissertation_plan.md` Phase 4 and the "Removed subsystem" note below for the user's learned
  plot preferences.

## Removed: training-metrics & plotting subsystem (historical reference)
Deleted 2026-06-05 ("this isn't working… we can do better"); the model layer was then further
refactored 2026-06-08 (commit `f386748`, which also removed the SB3 backend, merged
`train_shadow.py` into `train.py --shadow`, and inlined `trainer.py` as `train_model`). The
deleted plotting/metrics code is the reference for what to rebuild **cleaner** — full requirements
live in `dissertation_plan.md`. Key user preferences to preserve when rebuilding:
- **Decouple** running from plotting: model run → write raw file → separate utility loads it and
  renders plots / computes metrics.
- **Robust stats** (median + MAD/IQR, not mean ± std) and the full metric families above.
- Training-reward y-axis **zoomed to the curves** (clip warmup/exploration dips); training-reward
  and eval-reward as **separate images**, not stacked panels; **fine** eval granularity
  (~every 1000 steps); ablation pairs (`DDPG`+`Shadow DDPG`, etc.) on shared y-scales.
- Quantify **stability + sample efficiency** specifically, comparing shadow vs same-core standard.

## Coding Conventions
- Add new environments to the `SCENARIOS` registry in `scenarios.py`; add new models to
  `models.py`; add new experiment grids/conditions to `experiments.py` (`GRIDS`, `Condition`).
  Keep the training loop in `train.py:train_model`.
- Reuse `run_episode` from `evaluate.py` rather than re-implementing rollouts/loops.
- **Typed metadata, no slugs.** New run/method metadata goes in `schema.py` dataclasses
  serialised to JSON; categorical values are StrEnums; type-hint everything. Never derive a run's
  identity by parsing a directory/file name — read the `RunSpec`/`MethodRecord`.
- **No CLIs.** Modules are libraries invoked programmatically (e.g. `run_experiments.run_grid`).
  Don't re-add argparse unless explicitly asked.
- **One backend.** All agents share the custom DDPG/TD3 core. Standard vs shadow is the fair
  ablation (`DDPG`/`TD3` vs `ShadowDDPG`/`ShadowTD3` — identical learner, switching off). Don't
  reintroduce SB3 unless explicitly asked.
- Use `env.state` (not `obs`) for physical state values when plotting/logging.
- Use `seed=` in `env.reset()` for reproducibility; training seeds episodes as
  `episode + seed * 10_000`.
- Prefer numpy over torch for non-NN computations.
- Use `do_mpc` + `casadi` for the NMPC oracle baseline (`models.NMPCController`).
