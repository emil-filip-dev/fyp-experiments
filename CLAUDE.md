# FYP Project — Claude Code Instructions

## Autonomous Operation
Operate autonomously. Do not ask for confirmation before taking actions unless they are irreversible and destructive (e.g. deleting files, force-pushing). Proceed with reading, writing, editing, running scripts, and installing packages without prompting.

## Project Overview
Imperial College London FYP: Making RL reliable enough for industrial process control via hard constraint satisfaction.

Core research direction: Extend Shadow Mode RL (Gassert & Althoff, 2024) to PC-Gym benchmark environments, combining it with constraint-handling mechanisms from CIRL and learning-to-defer frameworks.

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

## Dependencies (requirements.txt)
numpy, matplotlib, gymnasium, casadi, jax[cpu], equinox, diffrax, do-mpc[full], stable-baselines3, pcgym

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
}

env = make_env(env_params)
obs, _ = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
# env.state — always in physical (unnormalised) units
```

## Codebase Architecture
The project is organised as a small set of shared modules plus CLI training/evaluation
scripts. Scenarios, models, and the episode runner are factored out so train, train_shadow,
and evaluate all share the same code paths.

- `scenarios.py` — central registry (`SCENARIOS` dict) of all four PC-Gym environments.
  Each entry holds `env_params`, `state_dim`, `action_dim`, `n_steps`, a `baseline_cls`
  (PID/PI controller for that process), and `plot_config`. `make_env_for(scenario)` builds
  the env. **Add new scenarios here**, not in the training scripts.
- `models.py` — all model definitions. Two backends:
  - **Custom core** (default) — one shared PyTorch core (`Actor` / `Critic` / `_CriticTwin`
    / `ReplayBuffer`); variants differ only in `decide_action`, a clean ablation set:
    - Shadow agents `ShadowDDPG` / `ShadowTD3` via `create_shadow_agent()`. Modes: `qvalue`
      (act if Q(s,a_agent) > Q(s,a_baseline)) and `agent` (actor emits a control-authority
      probability, act if > `eta_agent`). Optional `lambda_reg` L1 penalty toward baseline.
    - Pure (no-shadow) `PureDDPG` / `PureTD3` via `create_pure_agent()` — same core, switching
      off. THE standard DDPG/TD3 baseline. Labelled "DDPG" / "TD3".
  - **SB3 backend** — Stable-Baselines3 DDPG/TD3, a separately-tuned off-the-shelf learner.
    `create_sb3_agent()` (plain, labelled "SB3 DDPG"/"SB3 TD3"); `create_shadow_sb3_agent()`
    (`ShadowSB3DDPG`/`ShadowSB3TD3` via `_ShadowSB3Mixin`, labelled "Shadow SB3 DDPG"). Shadow
    is added by overriding `_sample_action` so the *executed* (q-value-switched) action is
    stored in the replay buffer; the mixin tracks takeover counts and exposes `executed_action`
    for switched evaluation. **SB3 shadow supports qvalue switching only** (agent-decision mode
    needs a custom policy head). Assumes `normalise_a=True` (all scenarios).
  - `resolve_device()` / `device_label()`. `SHADOW_MODELS`, `PURE_MODELS`, `SB3_MODELS`.
- `trainer.py` — two shared training loops: `train_custom(agent, ...)` (episode-based, for the
  custom core) and `train_sb3(model, ...)` (drives SB3 `model.learn()` via a callback). Both
  periodically evaluate (every `eval_freq` steps, default 1000; `--eval-freq` on the trainers)
  and save the best checkpoint; SB3 shadow resets the PID baseline at episode boundaries. Also
  home to `configure_utf8_output()` (relocated from the deleted training_metrics.py).
  **These loops no longer serialise training metrics or plot curves** — see "Removed:
  training-metrics & plotting subsystem" below.
- `train.py` — train a STANDARD (no-shadow) agent. `--backend custom` (default) → `PureDDPG`/
  `PureTD3` saved to `<model>/` as `best.pt`, labelled "DDPG"/"TD3". `--backend sb3` → SB3
  DDPG/TD3 saved to `sb3_<model>/` as `best_model.zip`, labelled "SB3 DDPG". CLI:
  `--scenario --model {ddpg,td3} --backend {custom,sb3} --steps --seed --cpu`.
- `train_shadow.py` — train a shadow-mode agent. `--backend custom` (default) → `Shadow*`
  saved to `shadow_<model>_<mode>/`. `--backend sb3` → `ShadowSB3*` saved to
  `shadow_sb3_<model>/` (qvalue only). CLI adds `--mode {qvalue,agent} --lambda-reg --eta-agent`.
  For "shadow vs standard", compare within a backend (custom-shadow vs custom-DDPG, or
  SB3-shadow vs SB3 DDPG) — same core, switching off.
  (training_metrics.py, plot_training_metrics.py, training_analysis.py were REMOVED — see the
  "Removed: training-metrics & plotting subsystem" section for what they did.)
- `evaluate.py` — runs models on a scenario and SERIALISES raw per-step rollouts to disk; it
  does NOT plot or compute metrics (a separate plotting/metrics utility, to be built, consumes
  the rollout files). Exports `run_episode`, `evaluate` (reused by trainer.py), and
  `NMPCController` (do-mpc + IPOPT oracle on PC-Gym's `oracle`). `load_model` routes by dir
  name: `*.pt` → custom Shadow*/Pure*; `*.zip` → SB3 via `_SB3Adapter` (plain) or
  `_ShadowSB3Adapter` (q-value switching at eval). `run_rollouts()` runs PID + NMPC oracle +
  every model for `n_seeds` seeds and writes one `<method>.npz` per method (arrays: states,
  obs, actions, actions_agent, actions_baseline, rewards, takeover — each `[N, T, ...]`) plus a
  `manifest.json` (scenario timing, plot_config, setpoint schedule, method list, array schema)
  under `outputs/rollouts/<scenario>/<timestamp>/`. CLI: `--scenario --models --n-seeds
  --no-oracle --mpc-horizon --output-dir`.
- `examples/example_pcgym.py` — original CSTR + PPO + PID reference (standalone).
- `findings.md` — research notes (literature synthesis, pseudocode, extension designs).
- `TODO.txt` — current open research tasks.
- `requirements.txt` — all dependencies.

### Output layout
- `outputs/models/<scenario>/<method>/` — just the best checkpoint now (`best.pt` for custom,
  `best_model.zip` for SB3). Method dir names: `ddpg` / `td3` (custom standard),
  `shadow_<model>_<mode>`, `shadow_<model>_agent_reg<λ>`, `sb3_<model>` (SB3 standard),
  `shadow_sb3_<model>` (SB3 shadow). `evaluate.py` infers the human label from this dir name
  (e.g. "DDPG", "Shadow DDPG (Q-value)", "SB3 DDPG", "Shadow SB3 DDPG"). (Training no longer
  writes `training_metrics.npz` / `training_curves.png` — that subsystem was removed.)
- `outputs/rollouts/<scenario>/<timestamp>/` — `evaluate.py` output: `<method>.npz` per method
  + `manifest.json` (raw per-step rollouts for a future plotting/metrics utility).
- `outputs/runs/`, `outputs/analysis/` — stale artifacts from the removed plotting subsystem
  (results.txt/comparison.png; training/eval-reward figures). Not written by current scripts.
- `runs/` — older/legacy run artifacts (pre-refactor); not written by current scripts.

### Typical workflow
```bash
.venv/Scripts/python train.py        --scenario cstr --model ddpg                  # DDPG (custom)
.venv/Scripts/python train.py        --scenario cstr --model ddpg --backend sb3    # SB3 DDPG
.venv/Scripts/python train_shadow.py --scenario cstr --model ddpg --mode qvalue    # Shadow DDPG
.venv/Scripts/python evaluate.py     --scenario cstr --n-seeds 20   # write rollouts (no plots)
# plotting/metrics utility: TO BE REBUILT — consumes outputs/rollouts/<scenario>/<timestamp>/
```

## Metrics
Target metric set for evaluating each method across seeds:
- **Control/tracking**: median episodic return + MAD across seeds (robust to RL outliers —
  prefer median/MAD over mean/std); IAE/ISE per controlled output; overshoot, settling
  time, steady-state offset per setpoint segment.
- **Constraint/safety**: violation rate, count, max magnitude, time-to-recovery
  (requires `constraints` to be added to scenarios — not yet configured).
- **Shadow-specific**: agent takeover fraction (overall + over time), divergence from
  baseline (‖a_agent − a_baseline‖), Δ vs baseline return.
- **Sample efficiency**: steps-to-threshold, area under learning curve, asymptotic reward.
- **Robustness**: MAD/IQR/worst-case return across seeds.
- **Control effort**: total/mean |Δu|.
- **Optimality gap**: Δ = J(π*) − J(π_θ), π* = NMPC oracle via do-mpc + CasADi + IPOPT.
  The oracle is implemented as `NMPCController` in `evaluate.py` and is recorded in every
  rollout run (disable with `--no-oracle`).

## Metrics Pipeline (current state)
Decoupled run → store → (rebuild) analyse:
- **Evaluation side (built).** `evaluate.py` `run_rollouts()` writes raw per-step rollouts to
  `outputs/rollouts/<scenario>/<timestamp>/` (one `<method>.npz` per method + `manifest.json`).
  No plotting/metrics in evaluate.py by design.
- **Training side (stripped).** `trainer.py` still trains + saves the best checkpoint but no
  longer serialises training metrics or plots curves.
- **Plotting / metrics utility (TO BE REBUILT).** A new utility should load the rollout `.npz`
  + `manifest.json` and produce both the figures and the metric tables. The deleted code below
  is the reference for what to rebuild — but cleaner.

## Removed: training-metrics & plotting subsystem
Deleted 2026-06-05 at the user's request ("this isn't working… we can do better"). Recorded
here so the functionality and the user's requirements are not lost when rebuilding.

**Deleted files and what they did:**
- `training_metrics.py` — streamed training metrics to a binary `training_metrics.npz` during
  training (`save_training_metrics`, atomic temp-then-replace, snapshot every eval interval +
  final): per-episode reward (step, reward), periodic eval reward, agent-takeover % (shadow),
  + JSON metadata (scenario, run_label, model_type, seed, total_steps, backend, mode…).
  Provided `load_training_metrics`, `plot_training_curves` (2–3 panel per-run figure: training
  reward + MA, eval reward, shadow takeover %), `render_metrics_file`, and `configure_utf8_output`
  (this last one was KEPT — relocated to trainer.py).
- `plot_training_metrics.py` — CLI to load `training_metrics.npz` and render the per-run curves
  offline (no retraining); `--scenario` discovery, `--summary` numeric stats.
- `training_analysis.py` — quantified STABILITY & SAMPLE EFFICIENCY from the `.npz` files: a
  comparison table (final/best eval, steps-to-threshold, normalised AUC, max drawdown,
  converged std/CV, step volatility, collapses, worst episode, early reward, shadow takeover %)
  plus `--plot` (separate training-reward & eval-reward figures, y-axis zoomed to the curves),
  `--by-backend` (split into custom vs SB3 groups so each shares a y-scale), and `--csv`.
- `evaluate.py` plotting (also removed): `run_comparison` computed multi-seed mean/std reward
  per method and wrote `results.txt` + `comparison.png` (per-output setpoint-tracking panels +
  reward bar chart); `_record_trajectory`, `_write_results`, `_plot_comparison`. PID baseline +
  NMPC oracle were always included.

**What the user wanted from training metrics & plots (requirements to preserve):**
- Metrics families: the "## Metrics" list above (control/tracking median+MAD, IAE/ISE,
  overshoot/settling/offset; constraint/safety; shadow-specific takeover & divergence; sample
  efficiency: steps-to-threshold, AUC, asymptotic; robustness; control effort; NMPC optimality gap).
- Stability + sample-efficiency quantification specifically, comparing shadow vs the same-core
  standard (fair ablation), and comparing across backends.
- Plot preferences (learned through iteration): training-reward y-axis must be ZOOMED to the
  curves (raw warmup/exploration dips to ~−120 made lines unreadable — clip them off the
  bottom); training-reward and eval-reward must be SEPARATE images, not stacked panels; eval/
  data granularity should be fine (eval every ~1000 steps, not 10000); and the SB3 pair and the
  custom pair should each get their OWN graphs (separate y-scales) — `DDPG`+`Shadow DDPG` in one,
  `SB3 DDPG`+`Shadow SB3 DDPG` in another. Deployed-policy performance from the eval curve;
  learning-process stability from the dense training curve.
- Pipeline shape: model run → write raw outputs to a file → a separate utility loads that file
  and renders plots / computes metrics (do NOT couple running with plotting).

Enabling prerequisite for safety metrics: add a `constraints` block to each scenario in
`scenarios.py` and capture the per-step `info` violation data in the rollout writer.

## Coding Conventions
- Add new environments to the `SCENARIOS` registry in `scenarios.py`; add new models to
  `models.py`. Keep training scripts thin — the training loop lives in `trainer.py`.
- Reuse `run_episode` / `evaluate` from `evaluate.py` and `train_custom` / `train_sb3` from
  `trainer.py` rather than re-implementing rollouts/loops.
- Two backends, both first-class: the custom DDPG/TD3 core (Shadow*/Pure*, labelled "DDPG" /
  "Shadow DDPG") and Stable-Baselines3 (labelled "SB3 DDPG" / "Shadow SB3 DDPG"), via
  `--backend {custom,sb3}`. Compare shadow-vs-standard *within* a backend for a fair ablation.
  SB3 shadow is qvalue-only and assumes `normalise_a=True`.
- Use `env.state` (not `obs`) for physical state values when plotting/logging.
- Use `seed=` in `env.reset()` for reproducibility.
- Prefer numpy over torch for non-NN computations.
- Use `do_mpc` + `casadi` for the NMPC oracle baseline.
