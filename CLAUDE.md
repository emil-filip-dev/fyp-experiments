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
- **OS**: Windows. Default shell is PowerShell; a Bash tool is also available. `util.configure_utf8_output()`
  is called at the top of each script's `main()` so non-ASCII logs survive stdout redirection on Windows.

## Dependencies (requirements.txt)
tqdm, numpy, matplotlib, casadi, jax[cpu], equinox, diffrax, do-mpc[full], pcgym.
Also required at runtime: **torch** (the entire model core is PyTorch) and **gymnasium**
(pulled in transitively by pcgym). **Not yet installed but planned** for the analysis utility:
`rliable` (robust RL statistics — see `dissertation_plan.md` Phase 0/§6).

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
    # NOT YET USED but supported by PC-Gym (prerequisite for safety metrics):
    #   'constraints': {...}, 'cons_type': '<=' / '>=', 'done_on_cons_vio': False
    #   -> per-step absolute violation magnitudes returned in info['cons_info']
}

env = make_env(env_params)
obs, _ = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
# env.state — always in physical (unnormalised) units
```

## Codebase Architecture
A small set of shared modules plus thin CLI scripts. Scenarios, models, and the episode
runner are factored out so `train.py` and `evaluate.py` share the same code paths. The model
layer is **one custom PyTorch core** — there is no longer a second (SB3) backend.

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
  - **Constraints (Phase-1 safety layer) are currently DROPPED.** `constraint_spec` is absent —
    the old bounds were calibrated for the old (wrong) operating points; they need fresh
    calibration against the verbatim operating points before re-adding. The infra
    (`constraints.py`, the eval-side capture) remains and treats absent `constraint_spec` as
    "no constraints". Re-calibrating them is a follow-up. See `constraints.py` / `dissertation_plan.md` §3.

- `models.py` — all model definitions, one shared core; variants differ only in `decide_action`,
  a clean ablation set.
  - **Building blocks**: `Actor` (q-value mode → action vector; agent mode → `(action,
    decision_prob∈[0,1])`), `Critic` (single Q, used by ShadowDDPG), `CriticTwin` (twin Q1/Q2
    with `q_min` / `q1_only`, used by ShadowTD3), `ReplayBuffer` (stores executed action +
    agent action + baseline action per transition). Abstract base `_ShadowModel` holds the
    shared `decide_action` / `store` / `save` / `load` logic.
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

- `util.py` — `configure_utf8_output()`, `resolve_device("cpu"|"gpu")`, `device_label()`.
  (These were relocated here from the deleted `trainer.py` in the refactor.)

- `train.py` — **single entry point for standard OR shadow training**, custom core only.
  `--shadow` flips switching on; standard and shadow share the same core/hyperparameters so they
  form a clean ablation. The **training loop lives inline here** as `train_model()` (formerly
  `trainer.py`): episode-based, periodic deterministic eval every `--eval-freq` steps (default
  1000), saves `best.pt` whenever eval reward improves, plus an `epN.pt` snapshot every
  `--checkpoint-freq` episodes (default 1000; `0` disables). **No behaviour-time metric logging**
  (violations / behaviour return / takeover over steps) — that subsystem was removed and is a
  required re-add for C1/C3/C4 (see `dissertation_plan.md` Phase 2).
  CLI: `--scenario --model {ddpg,td3} --shadow --mode {qvalue,agent} --steps --seed
  --lambda-reg --eta-agent --switch-critic {q1,qmin} --eval-freq --checkpoint-freq
  --output-dir --device {cpu,gpu}`.
  Run-label (output dir) names: standard → `ddpg` / `td3`; shadow → `shadow_<model>_<mode>`,
  `shadow_<model>_agent_reg<λ>` (agent mode + reg), `shadow_<model>_qvalue_<switch_critic>`
  (TD3 q-value).

- `evaluate.py` — runs models + reference controllers on a scenario and **serialises raw
  per-step rollouts** to disk; it does NOT plot or compute metrics (that utility is to be built).
  - Exports `run_episode` and `evaluate`, **reused by `train.py`** (the shared episode runner /
    deterministic evaluator).
  - `run_rollouts()` runs PID + NMPC oracle + every given/discovered model for `n_seeds` seeds and
    writes one `<slug>.npz` per method + a `manifest.json` under
    `outputs/rollouts/<scenario>/<timestamp>/`. Arrays per method (`[N, T, ...]`): `states`
    (physical `env.state`), `obs`, `actions`, `actions_agent`, `actions_baseline`, `rewards`,
    `takeover` (1=agent / 0=baseline / NaN=N/A). Manifest carries scenario timing, `plot_config`,
    setpoint schedule, method list, and the array schema.
  - Model loading is **inline** (`torch.load` → dispatch on `ckpt["type"]` via
    `get_shadow_model` / `get_standard_model`) — there is no separate `load_model` function and
    no SB3 adapters.
  - **Does NOT capture `info["cons_info"]`** yet — adding a per-step `violations` array is the
    other half of Phase 2.
  - CLI: `--scenario --models --n-seeds --no-oracle --mpc-horizon --output-dir --cpu`.

- `demo_takeover.py` — standalone demonstration (not part of the train/eval pipeline). Trains
  `ShadowDDPG` (q-value) via the *real* `run_episode` code path and plots the **deterministic
  (greedy) takeover fraction over training**, reproducing Gassert & Althoff Fig. 4 (high early
  takeover from a random critic, trending down as control is "earned"). Exists because `train.py`
  deliberately does not log/plot training metrics. Writes `outputs/takeover_<scenario>.png` + `.npz`.

- `dissertation_plan.md` — the experiment plan and codebase-readiness assessment (see Project
  Overview). **The source of truth for what to build next.**
- `findings.md` — research notes (literature synthesis, pseudocode, extension designs).
- `TODO.txt` — current open research tasks.
- `examples/example_pcgym.py` — original CSTR + PPO + PID reference (standalone).
- `requirements.txt` — dependencies (see above).

### Output layout
- `outputs/models/<scenario>/<run_label>/` — `best.pt` (best-evaluating) + periodic `epN.pt`
  snapshots. Run-label names as listed under `train.py` above. `evaluate.py` infers the human
  label from the loaded checkpoint's `label` property, not the dir name.
- `outputs/rollouts/<scenario>/<timestamp>/` — `evaluate.py` output: `<slug>.npz` per method +
  `manifest.json` (raw per-step rollouts for the to-be-built plotting/metrics utility).
- `outputs/takeover_<scenario>.png` / `.npz` — `demo_takeover.py` output.
- `outputs/runs/`, `outputs/analysis/`, `runs/` — stale artifacts from the removed plotting
  subsystem / pre-refactor runs. Not written by current scripts.

### Typical workflow
```bash
.venv/Scripts/python train.py    --scenario cstr --model ddpg                       # standard DDPG
.venv/Scripts/python train.py    --scenario cstr --model ddpg --shadow              # Shadow DDPG (qvalue)
.venv/Scripts/python train.py    --scenario cstr --model td3  --shadow --mode agent --lambda-reg 2.0
.venv/Scripts/python evaluate.py --scenario cstr --n-seeds 20                       # write rollouts (no plots)
.venv/Scripts/python demo_takeover.py --scenario cstr --steps 40000                 # takeover-trend demo plot
# plotting/metrics + multi-seed orchestration utilities: TO BE BUILT (dissertation_plan.md Phases 3–4)
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
  every rollout unless `--no-oracle`). Normalise scores `[PID = 0, oracle = 1]` for cross-env
  aggregation. Use `rliable` (IQM, bootstrap CIs, P(Shadow > Pure)) for robust stats.

## Metrics Pipeline (current state)
Decoupled run → store → (rebuild) analyse:
- **Evaluation side (built).** `evaluate.run_rollouts()` writes raw per-step rollouts to
  `outputs/rollouts/<scenario>/<timestamp>/`. No plotting/metrics in evaluate.py by design.
  **Gap:** does not yet capture `info["cons_info"]` (no `violations` array).
- **Training side (minimal).** `train.py`/`train_model` trains + saves best/snapshot checkpoints
  but does NOT serialise behaviour-time training metrics. **Gap:** a lightweight per-run
  `training_log.npz` (step, behaviour return, behaviour violations, takeover %, eval return) is a
  required re-add for the C1/C3/C4 claims.
- **Orchestration (missing).** `train.py` is single-run; a `run_experiments.py` looping
  conditions × seeds × envs is to be built (Phase 3).
- **Plotting / metrics utility (missing — the main build).** A new `analysis.py` should load the
  rollout `.npz` + `manifest.json` + per-run `training_log.npz` and emit the metric tables (CSV)
  and figures (PNG). See `dissertation_plan.md` Phase 4 for the exact metric/figure list and the
  "Removed subsystem" note below for the reference behaviour + the user's learned plot preferences.

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
  `models.py`. Keep training scripts thin — the training loop lives in `train.py:train_model`.
- Reuse `run_episode` / `evaluate` from `evaluate.py` rather than re-implementing rollouts/loops.
- **One backend.** All agents share the custom DDPG/TD3 core. Standard vs shadow is the fair
  ablation (`DDPG`/`TD3` vs `ShadowDDPG`/`ShadowTD3` — identical learner, switching off). Don't
  reintroduce SB3 unless explicitly asked.
- Use `env.state` (not `obs`) for physical state values when plotting/logging.
- Use `seed=` in `env.reset()` for reproducibility; training seeds episodes as
  `episode + seed * 10_000`.
- Prefer numpy over torch for non-NN computations.
- Use `do_mpc` + `casadi` for the NMPC oracle baseline (`models.NMPCController`).
