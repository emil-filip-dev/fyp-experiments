# Dissertation Plan — Offline RL for Industrial Process Control

> **Reframed (2026-06) to match `project_proposal.md`.** The earlier plan described
> Gassert & Althoff's *online* shadow-mode method (the agent explores on the live
> plant during training). That contradicts the proposal, which is explicitly
> **offline**: the RL controller is *pretrained on historical and simulated process
> data*, then introduced alongside an established expert (MPC) in **shadow mode**,
> graduating to autonomy only as confidence is earned. This document is the
> authoritative roadmap for that offline pipeline.

---

## 1. The problem and the thesis

RL could improve efficiency and adaptability in process control, but it is almost
never deployed in safety-critical chemical plants for one reason: **learning
requires trial-and-error exploration, and random actions on a live plant damage
equipment, violate constraints, and ruin product.** Operators must be able to
trust the controller *from the very first action*, including during learning.

The expert (an MPC, PID, or operator heuristic) is **established but suboptimal**,
and is expensive to maintain (it needs an accurate, continually re-identified
plant model). The thesis:

> An RL controller can be **pretrained offline** on logged/simulated data — never
> touching the plant during learning — and then **introduced gradually alongside
> the expert**, taking control only where it has *earned* it, until it can operate
> on its own at least as well as the expert. This delivers RL's upside (model-free
> adaptation, cheap inference, improvement on a suboptimal expert) with **no unsafe
> learning phase**.

The expert is a **safety scaffold and a performance floor**, not a competitor. The
NMPC oracle (perfect model + online optimisation) is the **optimality ceiling** we
measure the gap to — not something the agent must beat.

### Three axes, kept distinct
- **offline vs online** — the agent learns from a *fixed dataset*, with no env
  interaction, during pretraining. (Optional conservative fine-tuning later is
  *expert-guarded* online, not free exploration.)
- **sim vs real** — PC-Gym is the *stand-in for the real plant*. Training online
  *in sim* would still be "online"; we deliberately do not.
- **model-free vs model-based** — the RL *learner* uses no dynamics model; the NMPC
  expert/oracle does. The model-free benefit shows as the agent approaching NMPC
  performance using only logged transitions.

---

## 2. The pipeline (what the program does)

```
   ┌─────────────┐   ┌──────────────┐   ┌────────────────────────────────┐
   │ 1. DATASET  │ → │ 2. OFFLINE   │ → │ 3. STAGED DEPLOYMENT            │
   │ expert +    │   │  PRETRAIN    │   │  shadow (earned takeover) → auto │
   │ perturb.    │   │  (TD3+BC)    │   │  (+ optional o2o fine-tune)     │
   └─────────────┘   └──────────────┘   └────────────────────────────────┘
```

1. **Dataset** (`data.py`) — log the expert (MPC/PID) operating the simulated
   process, with action perturbations for coverage. This is the "historical and
   simulated process data". Static; built once.
2. **Offline pretrain** (`pretrain.py`) — **DDPG+BC** (the PRIMARY model;
   behaviour-cloning-regularised DDPG, after Fujimoto & Gu 2021, and the algorithm
   Gassert & Althoff's shadow-mode paper uses), with **TD3+BC** kept as a robustness
   comparison. Trained purely from the static buffer. No env interaction.
3. **Staged deployment** (`deploy.py`) — introduce the agent alongside the expert:
   - **shadow** (headline) — the agent takes control wherever it has *earned* it,
     `Q(s,a_agent) − Q(s,a_expert) > margin`; the expert handles the rest. A smaller
     margin grants more authority.
   - **autonomous** — agent controls alone (expert as optional hard safety fallback).
   - **offline→online (optional)** — conservative fine-tuning from expert-guarded
     transitions across these stages (low exploration; the expert catches un-earned
     actions, so training-time safety is preserved).
4. **Naive online contrast** — a standard online RL run on the live plant (full
   exploration, no guard) whose high training-time violations are the **foil** that
   motivates the offline approach.

---

## 3. Falsifiable claims

| | Claim | Why it matters | Evidence |
|---|---|---|---|
| **C1** | **Offline learnability.** The agent, trained *only* on the static expert dataset, matches/approaches the expert at deployment — with **zero plant interaction during learning**. | The core feasibility claim: you can learn a competent controller without risky exploration. | Offline learning curve (`training_log.npz`); autonomous deployment return vs expert. |
| **C2** | **Safe introduction.** In shadow deployment, constraint violations stay near expert levels and far below the naive online contrast; takeover is **earned** (only where Q-gap > margin). | Safety during introduction is the whole point. | Shadow violation rate (`deploy` rollouts); training-time violation curve, offline/o2o **vs** online contrast (C1 foil). |
| **C3** | **Graduated autonomy.** As the margin relaxes / fine-tuning proceeds, agent takeover fraction rises while performance is maintained or improved and safety preserved. | The "gradually increasing autonomy" of the proposal. | Takeover fraction vs margin/step; return & violations per margin; Q-gap distributions. |
| **C4** | **Standalone autonomy + reliability.** The agent eventually operates *alone*, matching/beating the expert and within X% of the NMPC optimum, with bounded violations, consistently across seeds. | "...with the ultimate goal of the model operating on its own." | Autonomous return, IAE/ISE, optimality gap vs NMPC; robust cross-seed stats (median/MAD, `rliable`). |

---

## 4. Comparison set

Per scenario, recorded by `deploy.run_rollouts`:
- **PID baseline** — the cheap reference floor.
- **NMPC** — the expert (setpoint-tracking scenarios) **and** the optimality ceiling.
- **Offline DDPG+BC** (primary) and **Offline TD3+BC** (robustness comparison) —
  each deployed at the shadow (MPC-guarded) and autonomous (normal) stages.
- **O2O DDPG** — offline + conservative expert-guarded fine-tuning.
- **Online DDPG (contrast)** — the unsafe foil (plain online DDPG, no MPC, no BC).

Expert per scenario (`experts.py`): **NMPC** for `cstr`, `four_tank`,
`multistage_extraction`; **PID** for `crystallization` (delta-u → NMPC N/A).

Normalise scores `[PID = 0, NMPC = 1]` for cross-environment aggregation; robust
stats (median + MAD/IQR, `rliable` IQM + bootstrap CIs, P(offline > expert)).

---

## 5. Metrics

- **Control/tracking**: median return + MAD across seeds; IAE/ISE per output;
  overshoot, settling time, steady-state offset per setpoint segment.
- **Safety**: violation rate/count/max magnitude, time-to-recovery — **per stage**
  and **per training step** (offline/o2o vs online contrast).
- **Introduction-specific**: takeover fraction (per stage, over fine-tuning),
  divergence `‖a_agent − a_expert‖`, Δ-vs-expert return, Q-gap distribution.
- **Offline-learning**: asymptotic offline return, gradient-steps-to-threshold.
- **Optimality gap**: Δ = J(NMPC) − J(agent), normalised `[PID=0, NMPC=1]`.

---

## 6. Status & TODO

**Built (Phases 1–4):**
- [x] Scenarios + constraints (`scenarios.py`, `constraints.py`) — unchanged, verbatim PC-Gym.
- [x] Expert factory (`experts.py`) — NMPC where supported, PID fallback.
- [x] Offline dataset generation (`data.py`).
- [x] Offline agents TD3+BC / DDPG+BC + staged `ShadowController` (`models.py`).
- [x] Training: offline, offline→online, online contrast (`pretrain.py`).
- [x] Staged deployment + raw rollout serialisation (`deploy.py`).
- [x] Orchestration: grids + resumable runner + provenance (`experiments.py`, `pipeline.py`).
- [x] **Analysis utility (`analysis.py`)** — rollout dir → `metrics_summary.csv`
      (control IAE/ISE/overshoot/settling/offset/median+MAD; safety rate/count/max;
      optimality normalized score + `rliable` IQM/CI; takeover; divergence; effort)
      + figures (trajectories, return, normalized score, safety, takeover, box) +
      `plot_training_curve`. Decoupled run→store→analyse; robust stats; aggregates
      models across seeds by (condition × stage). Auto-run by `run_pipeline`.

**Remaining / next:**
- [ ] Scale seeds (5–10) and run all four scenarios; pick scenarios where the
      standalone agent is reckless vs the expert so the C2 *safety* benefit shows.
- [ ] Per-setpoint-segment tables and cross-environment aggregate score figure.

---

## 7. Key references
- **Shadow mode** — Gassert & Althoff (2024). *Stepping Out of the Shadows.*
  (Switching idea; note: their method is online — we adapt the **switching/earned-
  takeover** mechanism to an **offline-pretrained** agent per the proposal.)
- **TD3+BC** — Fujimoto & Gu (2021). *A Minimalist Approach to Offline RL.*
- **PC-Gym** — Bloor et al. (2024/2025). Benchmark environments.
- **CIRL** — Bloor et al. (2024). PID-embedded RL for process control.
- **Learning-to-defer / SLTD** — Joshi et al. (2022). Sequential deferral.
