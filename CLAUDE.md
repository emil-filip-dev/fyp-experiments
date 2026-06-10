# FYP Project — Claude Code Instructions

## Autonomous Operation
Operate autonomously. Do not ask for confirmation before taking actions unless they are irreversible and destructive (e.g. deleting files, force-pushing). Proceed with reading, writing, editing, running scripts, and installing packages without prompting.

## Project Overview
Imperial College London FYP: Making RL reliable enough for industrial process control via hard constraint satisfaction.

Core research direction (per **`project_proposal.md`** — the authoritative brief):
**OFFLINE** RL for process control. Pretrain an RL controller on historical + simulated
process data (no live exploration), introduce it in **shadow mode** alongside an established
expert (MPC), **graduate** its autonomy as confidence is earned, and test whether it can
finally **operate on its own**. (The earlier codebase implemented Gassert & Althoff's *online*
shadow mode, which contradicts the proposal; it was reframed to offline in 2026-06.)

The pipeline, claims, and experiment design are in **`dissertation_plan.md`** (reframed roadmap);
the engineering map is in **`PROGRAM_OVERVIEW.md`**. Claims: C1 offline-learnability,
C2 safe-introduction, C3 graduated-autonomy, C4 standalone-autonomy+reliability. **Read both
before doing experiment work.**

## Key Papers
1. **TD3+BC** (Fujimoto & Gu, 2021) — the offline RL algorithm used for pretraining
2. **Shadow Mode RL** (Gassert & Althoff, 2024) — source of the earned-takeover switching idea
   (their method is *online*; we apply the switch to an *offline-pretrained* agent)
3. **PC-Gym** (Bloor et al., 2024) — benchmark environments (CSTR, crystallization, four-tank, multistage extraction)
4. **CIRL** (Bloor et al., 2024) — PID-embedded RL policy for process control
5. **RL Survey for PSE** (Bloor et al., 2025) — background and metrics
6. **Learning-to-Defer / SLTD** (Joshi et al., 2022) — deferral framework for sequential decisions

## Environment
- **Python**: 3.12 via `.venv` (use `.venv/Scripts/python` — do NOT use system Python)
- **Activate venv**: `.venv/Scripts/activate`
- **Run scripts**: `.venv/Scripts/python <script.py>`
- **Install packages**: `.venv/Scripts/pip install <pkg>`
- **OS**: Windows. Default shell is PowerShell; a Bash tool is also available.
  `util.configure_utf8_output()` exists for non-ASCII stdout on Windows but is no longer
  auto-invoked (the CLIs were removed — see below); call it yourself in a driver if needed.
- **Long-running tasks → ALWAYS launch in a new visible WINDOWS TERMINAL window** (`wt`), running
  **cmd** (user preference). Do NOT use a PowerShell window: PS 5.1 red-wraps a native exe's
  stderr as `NativeCommandError`, and tqdm writes its progress bars to stderr, so a PS launch
  fills the window with spurious red errors. cmd shows stderr cleanly. Tee to a log via the
  repo's `tee.py` so the run is also monitorable:
  ```bat
  wt.exe new-tab --title "<title>" -d "<repo>" cmd /k ".venv\Scripts\python.exe -u driver.py 2>&1 | .venv\Scripts\python.exe tee.py run.log"
  ```
  (`-u` unbuffered; cmd's `2>&1` merges stderr as plain text — no error-wrapping; `tee.py` mirrors
  to `run.log`, which the agent tails for progress/completion since a detached window emits no
  harness completion notification; read the log with `Get-Content`/Read). Applies to full pipeline
  runs, multi-seed/grid runs, and any NMPC-heavy job. **Prefer CPU (`Device.CPU`) for long
  unattended runs** — a CPU process survives a Windows user-switch, whereas a CUDA process is
  killed when its session is switched/disconnected; the workload is NMPC-bound so the GPU buys
  little anyway.

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
A small set of shared **library modules** (no CLIs — invoked programmatically). **The
authoritative module map is `PROGRAM_OVERVIEW.md`** — read it. The pipeline is
`data.py` (offline dataset) → `pretrain.py` (offline TD3+BC/DDPG+BC) → `deploy.py`
(staged shadow→autonomous), orchestrated by `pipeline.py`. The model layer is
**one custom PyTorch core** — no SB3.

**Design rules (enforced):** run metadata lives in **typed objects serialised to JSON**
(`schema.py`: `RunSpec`, `ModelSpec`, `MethodRecord`), never derived from directory slugs;
categorical values are **StrEnums** (`Scenario`, `Algorithm`, `TrainingMode`, `ExpertKind`,
`DeploymentStage`, `MethodRole`, `Device`), not bare strings; type hints everywhere. Slugs
only *locate* files.

- `scenarios.py` — central registry (`SCENARIOS` dict) of all four PC-Gym environments.
  **`env_params` are copied VERBATIM from PC-Gym's own paper training scripts**
  (`pc-gym_paper/train_policies/<env>/<env>_train.py` in github.com/MaximilianB2/pc-gym) —
  x0, o_space, a_space, SP schedules, tsim, N, noise, delta-u settings, and the custom OCP
  reward functions are exact copies, NOT reconstructions. **Do NOT "improve"/reconstruct these
  — copy from the source code (the docs pages disagree with the code; the code wins).** A
  verified verbatim copy is the whole point. The earlier hand-reconstructed configs were wrong
  (crystallization conc off ~1000× → NaN; multistage controlled the wrong variable; etc.).
  Each entry also carries `state_dim`, `action_dim`, `n_steps`, a `baseline_cls`, `plot_config`
  — these last two are **ours** (not PC-Gym). The `baseline_cls` PID/PI drives the SAME
  variable(s) PC-Gym controls; it is the cheap reference floor and the **PID expert** for
  scenarios the NMPC cannot model (crystallization). **`scenarios.py` and `constraints.py` are
  untouched by the offline reframe — treat them as fixed.**
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

- `models.py` — agents + deployment + NMPC, one shared PyTorch core.
  - **Building blocks**: `Actor` (deterministic π(s)→action), `Critic` (single Q, DDPG),
    `CriticTwin` (Q1/Q2 with `q_min`/`q1_only`, TD3), `ReplayBuffer` (5-tuple (s,a,r,s',done);
    `add_many` for the static offline dataset). Abstract base `_BaseAgent` holds `act`,
    `q_value`, `q_gap(obs, expert_action)` → `Q(s,π(s)) − Q(s,a_expert)` (the earned-takeover
    signal, C3), `store`, `load_dataset`, `save`/`load`.
  - **Offline agents** `TD3Agent` / `DDPGAgent` — TD3+BC / DDPG+BC (Fujimoto & Gu 2021): actor
    loss mixes `−λ·Q(s,π(s))` with a BC term `‖π(s)−a_data‖²`, with `λ = bc_alpha/mean|Q|`.
    `bc_alpha=0` recovers plain TD3/DDPG (the online contrast). `update()` is one gradient step
    from the buffer — used identically for offline, offline→online, and online phases.
    `get_agent(algorithm)` returns the class. `total_updates` (grad steps) / `total_env_steps`.
  - **`DeploymentStage {shadow, autonomous}`** and **`ShadowController(agent,
    expert)`** — `decide(obs, stage, margin)` returns `(a_exec, used_agent, a_agent, a_expert)`:
    **shadow**→agent iff `q_gap > margin` else expert (the headline earned-takeover mode);
    autonomous→agent (optional expert safety fallback). Used for both staged evaluation and
    o2o data collection.
  - **Save/load**: `agent.save(path)` writes a dict via `_save_dict()` — keys `type` (an
    `AgentType`), hyperparameters, `state_dicts`, `internal`. `get_agent(type).load(ckpt, device)`
    reconstructs. Checkpoint file: `best.pt`.
  - **`NMPCController`** — do-mpc + IPOPT NMPC on the env's *exact* dynamics. It is BOTH the
    **MPC expert** (setpoint-tracking scenarios) and the optimality-gap ceiling. `.predict(obs)`/
    `.reset()`. Raises `NotImplementedError` on disturbances/delta-u (→ PID expert instead).
    Scalar-safe, non-lagging `p_fun`.

- `experts.py` — `make_expert(scenario)` → `(controller, ExpertKind)`: NMPC for setpoint-tracking
  scenarios, PID baseline fallback for delta-u/disturbance scenarios (crystallization).
  `expert_kind_for(scenario)` gives the kind without constructing.

- `util.py` — `configure_utf8_output()`, `resolve_device("cpu"|"gpu")`, `device_label()`.

- `schema.py` — **the shared typed vocabulary** (imports `AgentType`/`DeploymentStage` from
  `models`; no cycles). StrEnums: `Scenario`, `Algorithm`(=`AgentType` `{ddpg,td3}`),
  `TrainingMode {offline, offline_to_online, online_contrast}`, `ExpertKind {nmpc, pid}`,
  `MethodRole {model, pid, nmpc}`, `Device`. Frozen dataclasses with `to_json`/`from_json`:
  - `Condition` — a learner config (algorithm + training_mode + bc_alpha). **Single factory** for
    the slug (`.slug`) and metadata (`.to_run_spec()`). Helpers `offline(...)`,
    `offline_to_online(...)`, `online_contrast(...)`.
  - `RunSpec` — a Condition bound to (scenario, seed, offline/online budgets, expert_kind);
    `run.json`. `.artifact_stem` gives the rollout filename stem.
  - `ModelSpec` — checkpoint + `RunSpec` (input to `deploy.run_rollouts`).
  - `MethodRecord` — a manifest entry (role + optional `RunSpec` + `DeploymentStage`).

- `data.py` — offline **dataset generation** (the "historical + simulated data"):
  `generate_dataset(scenario, expert, …)` logs the expert + action perturbations across episodes;
  `save_dataset`/`load_dataset` (.npz); `dataset_to_buffer` → `ReplayBuffer`. Reports dataset
  safety (how often the data-collection policy itself violated).

- `pretrain.py` — training entry points (no CLI). `run_condition(condition, scenario, seed, …)`
  dispatches on `TrainingMode`:
  - `pretrain_offline` — TD3+BC/DDPG+BC gradient steps from the static buffer; periodic
    standalone (autonomous) eval; `best.pt` on improvement.
  - `finetune_o2o` — conservative offline→online from expert-guarded transitions, sweeping
    shadow→autonomous.
  - `train_online_contrast` — naive online RL from scratch (the unsafe foil).
  Writes `run.json`, `dataset.npz` (offline modes), and `training_log.npz` (mode-specific curve).

- `deploy.py` — staged deployment + raw rollout serialisation (no plotting). `evaluate_deploy(...)`
  is the cheap in-memory eval (used by `pretrain`). `run_rollouts(scenario, model_specs, stages,
  shadow_margins, …)` records PID + NMPC references + every model at each `DeploymentStage`,
  one `.npz` per method×stage + `manifest.json`. Per-method arrays `[N,T,…]`: `states`, `obs`,
  `actions`, `actions_agent`, `actions_expert`, `rewards`, `takeover` (1/0/NaN), `q_gap`,
  `divergence` (‖a_agent−a_expert‖), `violations`.

- `experiments.py` — **declarative config**. `GRIDS` registry of `ExperimentGrid`s (envs ×
  `Condition`s × seeds + budgets/stages/margins). `EnvSpec(name, offline_steps, online_steps)`.
  Helpers `iter_training_jobs`, `iter_model_refs`, `checkpoint_path`, `describe_grid`,
  `write_provenance`.

- `pipeline.py` — **Phase-3 orchestrator** (no CLI). `run_pipeline(grid, stage=Stage.{all,train,
  rollouts})` writes provenance then runs `train_grid` (drives `pretrain.run_condition`; resumable;
  failure-isolated) and `rollout_grid` (`deploy.run_rollouts` per scenario). `override_grid(...)`
  subsets a grid for smoke runs.

- `analysis.py` — **Phase-4 metrics + plotting** (no CLI). `analyse_rollout_dir(dir)` loads a
  rollout dir, aggregates model runs across seeds by (condition × stage), writes
  `metrics_summary.csv` (control / safety / optimality / takeover / divergence / effort) +
  figures (trajectories, return, normalized score, safety, takeover, box). `plot_training_curve`
  for learning curves. Robust stats (median/MAD/IQR) + `rliable` IQM/CI. `latest_rollout_dir`.
  **`plot_takeover_map(run_dir, grid_res, cell_px, sp_value)`** — the RL–MPC **takeover map**:
  a diverging ΔQ = Q(s,π_RL)−Q(s,a_MPC) heatmap over a 2D state-space slice (CSTR: Ca×T),
  orange where RL takes over / blue where MPC drives, rendered for every training snapshot to
  show the takeover boundary evolving (one PNG per snapshot + a combined `takeover_grid.png` →
  `<run_dir>/takeover_maps/`). Reads `snapshots/` saved by `pretrain` (offline: grad-step;
  o2o/online: env-step). Works for online-only runs too.

- `constraints.py` — constraint **detection** (`violation_magnitudes`, from `env.state`) +
  **metrics** (`constraint_metrics`). **Untouched** by the reframe. Library only.

- `dissertation_plan.md` — reframed plan + claims (C1–C4). `PROGRAM_OVERVIEW.md` — engineering map.
- `project_proposal.md` — the supervisor's brief (the source of truth for scope).
- `findings.md` — research notes. `TODO.txt` — open tasks.
- `examples/example_pcgym.py` — original CSTR + PPO + PID reference (standalone).
- `requirements.txt` — dependencies (see above).

### Output layout
- `outputs/models/<scenario>/<run_label>[/seed<k>]/` — `best.pt` + `run.json` (the `RunSpec`) +
  `training_log.npz` (mode-specific curve) + `dataset.npz` (offline modes) + `snapshots/`
  (periodic weights-only `snap_<phase>_<step>.pt` + `snapshots.json`, for the takeover-map viz) +
  `takeover_maps/` (generated by `analysis.plot_takeover_map`). Run-label slugs:
  `offline_ddpg_bc` (primary), `offline_td3_bc`, `o2o_ddpg`, `online_ddpg`.
- `outputs/rollouts/<scenario>/<timestamp>/` — `deploy.run_rollouts` output: `pid.npz`, `nmpc.npz`,
  and `<stem>__<stage>[_m<margin>].npz` per model×stage + `manifest.json` with structured
  `MethodRecord`s. Raw per-step rollouts for the Phase-4 analysis utility.
- `outputs/experiments/<grid>/provenance.json` — resolved grid + git commit + library versions.
- `outputs/cache/datasets/` — cached offline datasets keyed by (scenario, seed, expert,
  episodes, perturb), so offline + o2o conditions at one seed share a dataset instead of each
  re-running the expert. `outputs/cache/mpc_grids/` — cached MPC action-grids per (scenario,
  slice, sp, horizon, grid_res) so takeover maps don't recompute the same IPOPT solves. Both
  are pure caches — safe to delete; they just avoid re-running NMPC.
- `outputs/runs/`, `outputs/analysis/`, `runs/` — stale pre-reframe artifacts. Not written by current code.

### Typical workflow (programmatic — no CLI)
```python
from experiments import GRIDS
from pipeline import run_pipeline, override_grid
from schema import Device

grid = GRIDS["phase1_offline"]                             # envs × conditions × seeds
grid = override_grid(grid, env_names=["cstr"], seeds=[0,1],
                     offline_steps=5_000, online_steps=2_000)  # smaller smoke grid
run_pipeline(grid, device=Device.CPU)                      # dataset → pretrain → staged rollouts
from analysis import analyse_rollout_dir, latest_rollout_dir
analyse_rollout_dir(latest_rollout_dir("cstr"))            # metrics_summary.csv + figures
```
Or in one shot: `run_pipeline(grid)` now runs dataset → pretrain → rollouts → **analysis**.

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
- **Deployment side (built).** `deploy.run_rollouts()` writes raw per-stage rollouts incl.
  `violations`, `q_gap`, `divergence`, with structured `MethodRecord` metadata. (✅)
- **Training side (built).** `pretrain` serialises a mode-specific `training_log.npz` (offline:
  grad-step eval curve; o2o/online: behaviour-time return/violation/takeover) + `run.json`. (✅)
- **Orchestration (built).** `pipeline.run_pipeline()` loops conditions × seeds × envs through
  dataset → pretrain → staged rollouts → analysis (resumable, provenance, failure-isolated). (✅)
- **Plotting / metrics utility (built — Phase 4 ✅).** `analysis.py` loads a rollout dir
  (`.npz` + `manifest.json`) and emits `metrics_summary.csv` (control: IAE/ISE, overshoot,
  settling, offset, median+MAD return; safety: violation rate/count/max/first-step; optimality:
  normalized score [PID=0, NMPC=1] with `rliable` IQM+CI; takeover; divergence; control effort)
  plus figures (trajectories, return bar, normalized score, safety, takeover, return box).
  `plot_training_curve(run_dir)` renders the learning curve. Model runs are aggregated across
  seeds by (condition × stage); identity comes from `MethodRecord`, never the filename.
  `pipeline.run_pipeline(stage=Stage.ANALYSE)` runs it standalone.

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
- **Do NOT modify `scenarios.py` or `constraints.py`** — they are fixed (verbatim PC-Gym +
  calibrated constraints). Add new models to `models.py`, new grids/conditions to
  `experiments.py`. Keep training in `pretrain.py`, rollouts in `deploy.py`.
- Reuse `deploy._record_episode` / `evaluate_deploy` rather than re-implementing rollout loops.
- **Typed metadata, no slugs.** New run/method metadata goes in `schema.py` dataclasses serialised
  to JSON; categorical values are StrEnums; type-hint everything. Never parse identity from a
  filename — read the `RunSpec`/`MethodRecord`.
- **No CLIs.** Modules are libraries invoked programmatically (e.g. `pipeline.run_pipeline`).
- **One backend.** All agents share the custom TD3/DDPG core; `bc_alpha` toggles TD3+BC vs plain.
  No SB3 unless explicitly asked.
- **Offline ≠ online.** The headline method pretrains from a STATIC dataset (no env interaction).
  Online phases (o2o fine-tune, online contrast) are explicit and expert-guarded / clearly labelled.
- Use `env.state` (not `obs`) for physical state values when logging.
- Use `seed=` in `env.reset()`; training seeds episodes as `episode + seed * 10_000`.
- Prefer numpy over torch for non-NN computations.
- The NMPC (`models.NMPCController`, do-mpc + casadi) is the MPC expert AND optimality ceiling.
