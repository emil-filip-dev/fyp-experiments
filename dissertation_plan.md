# Dissertation Plan — Demonstrating Shadow Mode RL on PC-Gym

**Goal:** Show, with academic rigour, that Shadow Mode RL (Gassert & Althoff, 2024)
"works" on real-ish chemical-process-control environments (PC-Gym), as the empirical
core of the FYP on *making RL reliable enough for industrial process control via
hard constraint satisfaction*.

The central difficulty is that **"shadow mode works" is not a falsifiable claim.**
This plan decomposes it into testable sub-claims, defines the experiments and statistics
that would convince a skeptical examiner, and lists the concrete codebase work needed.

---

## 1. Turn "it works" into falsifiable claims

| # | Claim | Why it matters | Primary evidence |
|---|-------|----------------|------------------|
| C1 | **Safety during training.** Shadow mode incurs far fewer constraint violations / catastrophic actions *while learning on the system* than standard RL. | The unique selling point. In a simulator, everything else is obtainable by training offline — only training-time safety justifies the method. | Cumulative violations vs training step (behaviour policy), shadow vs pure. |
| C2 | **No performance sacrifice.** The deployed greedy combined policy matches/beats standard RL at equal env-step budget, and beats the baseline controller. | Shadow must not cost final performance, or the safety is worthless. | Deployed return, IAE/ISE, optimality gap vs NMPC oracle. |
| C3 | **Selective, earned takeover.** The agent takes control where it is genuinely better (transients, setpoint changes) and defers where the baseline is good. | Shows the *mechanism* behaves as theorised, not randomly. | Takeover fraction over training + over state/episode regions. |
| C4 | **Reliability.** Low across-seed variance; graceful degradation under noise / disturbance / unseen setpoints; no catastrophic collapses. | This is literally the thesis premise (reliability). | Robust dispersion stats, stress tests. |

C2 is *necessary*; C1, C3, C4 are what make the method *interesting*. The headline
result is **C1 + C2 together**: safety-during-training at no performance cost.

---

## 2. Comparison set (control conditions)

For every environment × seed, run the same conditions. Shadow-vs-standard must be a
**fair ablation**: identical core, hyperparameters, seeds, exploration noise, and
**equal env-step budget** — differing *only* in the switching decision. The codebase is
already built for this (one shared `_ShadowModel` core).

| Condition | Role | Code entry point |
|-----------|------|------------------|
| PID / PI baseline | safety-net floor | `scenarios.py` `baseline_cls` |
| NMPC oracle | best-achievable ceiling (optimality gap) | `models.NMPCController` |
| **Standard RL** (Pure DDPG / TD3) | the key ablation | `models.DDPG` / `models.TD3` |
| **Shadow RL (q-value)** | the paper's method | `models.ShadowDDPG` qvalue |
| Shadow ablations | agent-decision, reg, TD3 switch-critic | `--mode agent`, `--lambda-reg`, `--switch-critic` |

Normalise each environment's scores to `[PID = 0, NMPC oracle = 1]` so results can be
**aggregated across the PC-Gym suite** into a single suite-level claim.

---

## 3. Headline experiment — safety *during* training (C1)

This is the experiment that actually proves the point and the one most projects omit.
Measure cumulative harm incurred by the **behaviour policy** (what actually ran on the
"plant") as a function of training steps, for Shadow vs Pure:

- per-step **constraint-violation rate, count, max magnitude, time-to-recovery**;
- **cumulative training cost / regret** (area between behaviour return and baseline / oracle);
- number of **catastrophic episodes**.

**Expected result:** Pure RL accumulates violations early (random exploration on the live
plant); Shadow stays near-zero (the baseline catches dangerous proposals) — *at no cost to
final performance* (C2).

**Prerequisite (not yet in the codebase):**
1. Add a `constraints` block to each scenario (`scenarios.py`). PC-Gym supports this
   natively via `env_params["constraints"]`, `env_params["cons_type"]` (`"<="` / `">="`),
   and `env_params["done_on_cons_vio"]`; per-step absolute violation magnitudes are
   returned in `info["cons_info"]` (shape `[n_con, N, 1]`).
2. **Log behaviour-time metrics during training** (violations + behaviour return), not just
   the deterministic eval. The training-metrics subsystem was removed, so this needs a
   lightweight re-add. Without it, C1/C3/C4 cannot be measured.

---

## 4. Supporting experiments

- **Sample efficiency / learning curves (C2).** Deterministic eval every ~1000 steps,
  ≥10 seeds; plot **median + IQR band**. Report steps-to-threshold, area-under-curve,
  asymptotic reward. Tests the guided-exploration / jump-start benefit.
- **Final performance + optimality gap (C2).** Cross-seed deployed return, IAE/ISE per
  controlled output, overshoot / settling time / steady-state offset per setpoint segment,
  and **Δ = J(NMPC) − J(π)**. Plot PID, Pure, Shadow, oracle on one axis.
- **Takeover / mechanism analysis (C3).** Takeover fraction over training (the
  down-then-recover curve the paper predicts for q-value switching), and at convergence
  *where* it takes over — bucketed by episode phase (transient vs steady-state) and by
  proximity to setpoint changes. Correlate takeover with `Q(agent) − Q(baseline)` and with
  realised advantage. This is the analogue of the paper's "agent favoured behind the
  obstacle" figure.
- **Robustness (C4).** Sweep noise level, unseen setpoint schedules, perturbed initial
  conditions, disturbances; report worst-case / dispersion, not just the mean.

---

## 5. Ablations (what actually matters)

- switching **on/off** (Shadow vs Pure) — the core ablation;
- **q-value vs agent-decision** switching;
- **baseline quality**: well-tuned PID vs detuned PID vs CIRL (RQ2) — does shadow still
  help with a weak baseline, and does a stronger baseline raise the floor?;
- **randomised start on/off** (the paper shows it matters — reproduce);
- **λ sweep** for agent-mode reward regularisation (Eq. 5);
- (RQ3) uncertainty-aware switching, if implemented.

---

## 6. Statistical rigour

This is where projects gain or lose marks.

- **≥10 seeds, ideally 20+.** PC-Gym episodes are cheap; few-seed results are not credible.
- **Robust statistics, not mean ± std.** RL returns are heavy-tailed — report
  **median + IQR / MAD** and the full distribution.
- **Use `rliable`** (Agarwal et al., 2021): interquartile mean (IQM), stratified-bootstrap
  confidence intervals, performance profiles, and **probability of improvement**
  P(Shadow > Pure). Citing it signals awareness of RL-evaluation pitfalls.
- **Non-parametric tests + effect sizes** (Mann–Whitney U / Wilcoxon) rather than t-tests
  on a handful of non-normal samples.
- **Aggregate across environments** with the normalised `[PID=0, oracle=1]` scores.

---

## 7. Threats to validity (pre-empt these in the viva)

- **Strawman baselines.** Tune the PID reasonably *and* tune the standard-RL baseline at
  least as hard as Shadow. Don't beat a crippled competitor.
- **Equal budget.** Same total env steps for Shadow and Pure; shadow "wastes" steps on
  baseline execution and that must count against it.
- **No reading off training curves.** Report deployed greedy performance for "how good";
  use the dense behaviour curve only for learning-process / safety claims.
- **Generality.** ≥3 of the 4 PC-Gym environments, with different dynamics / timescales.
- **Reproducibility.** Fixed seeds, all hyperparameters in a config, pre-declare the
  steps-to-threshold cutoff so it cannot be cherry-picked. Record compute.

---

## 8. The defensible thesis sentence

Aim to be able to write, with statistics behind every clause:

> *Across N PC-Gym environments and M seeds, shadow-mode DDPG matched standard DDPG's
> deployed control performance (within X% of the NMPC optimum) while reducing constraint
> violations incurred during on-system training by Y× (stratified-bootstrap 95% CI …),
> with takeover concentrated in setpoint transients — demonstrating that the safety net
> does not cost final performance.*

---

## 9. References

**Core method & benchmark**
- Gassert, P. & Althoff, M. (2024). *Stepping Out of the Shadows: Reinforcement Learning in
  Shadow Mode.* arXiv:2410.23419 — https://arxiv.org/abs/2410.23419
- Bloor, M. et al. (2024). *PC-Gym: Benchmark Environments for Process Control Problems.*
  arXiv:2410.22093 — https://arxiv.org/abs/2410.22093
- Bloor, M. et al. (2024). *Control-Informed Reinforcement Learning for Chemical Processes
  (CIRL).* arXiv:2408.13566 — https://arxiv.org/abs/2408.13566
- Bloor, M. et al. (2025). *A Survey and Tutorial of Reinforcement Learning Methods in
  Process Systems Engineering.* arXiv:2510.24272 — https://arxiv.org/abs/2510.24272
- Joshi, S., Parbhoo, S. & Doshi-Velez, F. (2022). *Learning-to-Defer for Sequential Medical
  Decision-Making under Uncertainty (SLTD).* arXiv:2109.06312 — https://arxiv.org/abs/2109.06312

**RL algorithms**
- Lillicrap, T. et al. (2015). *Continuous Control with Deep Reinforcement Learning (DDPG).*
  arXiv:1509.02971 — https://arxiv.org/abs/1509.02971
- Fujimoto, S., van Hoof, H. & Meger, D. (2018). *Addressing Function Approximation Error in
  Actor-Critic Methods (TD3).* arXiv:1802.09477 — https://arxiv.org/abs/1802.09477

**Evaluation methodology**
- Agarwal, R. et al. (2021). *Deep Reinforcement Learning at the Edge of the Statistical
  Precipice (rliable).* NeurIPS. arXiv:2108.13264 — https://arxiv.org/abs/2108.13264

**Safe RL context**
- García, J. & Fernández, F. (2015). *A Comprehensive Survey on Safe Reinforcement Learning.*
  JMLR 16. — https://jmlr.org/papers/v16/garcia15a.html
- Dalal, G. et al. (2018). *Safe Exploration in Continuous Action Spaces.* arXiv:1801.08757
  — https://arxiv.org/abs/1801.08757
- Wabersich, K.P. & Zeilinger, M.N. (2021). *A Predictive Safety Filter for Learning-Based
  Control.* Automatica 129. — https://doi.org/10.1016/j.automatica.2021.109639
- Brunke, L. et al. (2022). *Safe Learning in Robotics.* Annual Review of Control, Robotics,
  and Autonomous Systems 5. — https://doi.org/10.1146/annurev-control-042920-020211

---

## 10. Codebase readiness assessment

| Component | Status | Work needed |
|-----------|--------|-------------|
| Scenarios / 4 PC-Gym envs (`scenarios.py`) | ✅ Ready | Add `constraints` / `cons_type` / `done_on_cons_vio` blocks. |
| Models: Pure/Shadow DDPG·TD3, qvalue + agent, reg, switch-critic (`models.py`) | ✅ Ready & paper-faithful (after recent fixes) | None for core experiments. |
| NMPC oracle (`models.NMPCController`) | ✅ Ready | None (note: setpoint-tracking only, no disturbances/delta-u). |
| Save / load + run-label dirs | ✅ Ready | None. |
| Training loop (`train.py`) | 🟡 Partial | Trains one seed; saves best via deterministic eval. **No behaviour-time logging** (violations / behaviour return over steps). Re-add a lightweight logger. |
| Rollout recorder (`evaluate.run_rollouts`) | 🟡 Partial | Writes `<method>.npz` + `manifest.json`. **Does not capture `info["cons_info"]`** — add a `violations` array. |
| Multi-seed / multi-condition orchestration | ❌ Missing | `train.py` is single-run. Add a small runner that loops conditions × seeds × envs. |
| Metrics / analysis utility | ❌ Missing | Biggest build. Consume rollouts + training logs → control metrics (IAE/ISE, overshoot, settling, offset), safety metrics, takeover analysis, optimality gap, learning curves. |
| Robust statistics (`rliable`) | ❌ Missing | `pip install rliable`; wire IQM + bootstrap CIs + probability-of-improvement. |

**Summary:** the *learning* machinery is done and paper-faithful. The gaps are all on the
**measurement** side — constraints, behaviour-time logging, an analysis/metrics utility,
and orchestration. None are large individually; the analysis utility is the main build.

---

## TODO

Scoped to be completable **by tomorrow with Claude's help**. Ordered so that if we run out
of time, the headline claim (C1 + C2) is still covered. `[core]` = required for the headline
result; `[stretch]` = strengthens the thesis if time allows.

### Phase 0 — Setup (≈15 min)
- [ ] `[core]` `pip install rliable` into `.venv`; confirm import.
- [ ] `[core]` Fix the experiment grid for the first pass: envs = {`cstr`, `four_tank`},
      conditions = {PID, NMPC oracle, Pure DDPG, Shadow DDPG qvalue}, seeds = 5 (scale to
      10+ later), budget = 50k steps (CSTR) for a complete pipeline run, then scale.
- [ ] `[core]` Create `experiments/` config (envs, conditions, seeds, steps) so runs are reproducible.

### Phase 1 — Constraints (≈30 min, `scenarios.py`)
- [ ] `[core]` Add `constraints`, `cons_type`, `done_on_cons_vio=False` to `cstr` (e.g. cap
      reactor temperature `T` and/or bound `Ca`) and `four_tank` (tank-level limits).
- [ ] `[core]` Verify `env.step` populates `info["cons_info"]` and that the NMPC oracle still
      builds (it already reads `env_params["constraints"]`).
- [ ] `[stretch]` Add constraints to `crystallization` and `multistage_extraction`.

### Phase 2 — Capture safety + behaviour-time metrics (≈1.5 h, `evaluate.py` + `train.py`)
- [ ] `[core]` Extend `_record_rollout` / `run_rollouts` to record a per-step `violations`
      array (from `info["cons_info"]`) into each `<method>.npz`; document it in the manifest
      `array_schema`.
- [ ] `[core]` Add a lightweight **behaviour-time logger** to the training loop: every
      `eval_freq`, append (step, behaviour return, behaviour violation count/magnitude,
      takeover fraction, deterministic eval return) to a `training_log.npz` per run.
- [ ] `[stretch]` Record `Q(agent) − Q(baseline)` per step in rollouts for the takeover-vs-advantage analysis.

### Phase 3 — Orchestration (≈30 min, new `run_experiments.py`)
- [ ] `[core]` Script that loops conditions × seeds × envs, calling the existing `train()` and
      then `run_rollouts()`; write everything under `outputs/` with consistent naming.
- [ ] `[core]` Support `--steps`, `--seeds`, `--envs`, `--conditions` and run training jobs in
      the background so the analysis utility can be built in parallel.

### Phase 4 — Metrics / analysis utility (≈2–3 h, new `analysis.py`)
- [ ] `[core]` Loader: read `outputs/rollouts/<env>/<ts>/*.npz` + `manifest.json` + per-run
      `training_log.npz`.
- [ ] `[core]` **Control metrics**: median + MAD deployed return, IAE/ISE per output,
      overshoot / settling time / steady-state offset per setpoint segment.
- [ ] `[core]` **Safety metrics (C1)**: violation rate / count / max magnitude /
      time-to-recovery; cumulative training-time violations vs step (Shadow vs Pure).
- [ ] `[core]` **Optimality gap (C2)**: Δ = J(NMPC) − J(π), normalised `[PID=0, oracle=1]`.
- [ ] `[core]` **Learning curves**: deterministic eval median + IQR band; steps-to-threshold, AUC.
- [ ] `[core]` **Takeover analysis (C3)**: takeover fraction over training, and bucketed by
      episode phase (transient vs steady-state).
- [ ] `[core]` **Robust stats**: integrate `rliable` IQM + stratified-bootstrap CIs +
      probability of improvement P(Shadow > Pure).
- [ ] `[core]` Emit result **tables (CSV) + figures (PNG)** ready to drop into the dissertation.

### Phase 5 — Run & generate results (≈1–2 h wall-clock, mostly background)
- [ ] `[core]` Train all conditions × seeds for `cstr` (and `four_tank` if time), background jobs.
- [ ] `[core]` Record rollouts (incl. PID + NMPC oracle) for every trained model.
- [ ] `[core]` Run `analysis.py`; produce the C1 safety-during-training figure, the C2
      learning-curve + optimality-gap figure/table, and the C3 takeover figure.
- [ ] `[core]` Fill in the thesis sentence (§8) with the actual numbers.

### Phase 6 — Ablations & stress tests (`[stretch]`, if time remains)
- [ ] `[stretch]` Agent-decision vs q-value switching on `cstr`.
- [ ] `[stretch]` Randomised-start on/off ablation (reproduce the paper's finding).
- [ ] `[stretch]` Baseline-quality ablation (tuned vs detuned PID).
- [ ] `[stretch]` Robustness sweep: noise level + unseen setpoint schedule.
- [ ] `[stretch]` Scale seeds to 10–20 and re-run the headline figures.

### Realistic outcome for "tomorrow"
A complete, reproducible pipeline (constraints → training with safety logging →
multi-seed orchestration → metrics/figures) plus a **first full result on CSTR** (5 seeds,
4 conditions) covering C1 + C2 + C3, with `four_tank` and the ablations as immediate
follow-ups. Scaling to more seeds/envs is then just compute time, not new code.
