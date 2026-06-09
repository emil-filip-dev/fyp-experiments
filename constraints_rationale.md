# Constraint Rationale — PC-Gym Scenarios

This document explains the constraints added to the PC-Gym scenarios in
`scenarios.py`, and in particular **why the three constraints we chose ourselves
(four_tank, multistage_extraction, crystallization) are appropriate and
reasonable**. It is a companion to `dissertation_plan.md` (Phase 1) and the
verbatim-copy doctrine in `CLAUDE.md`.

## Background: why only the CSTR was constrained out of the box

PC-Gym ships an explicit, environment-specific constraint for **exactly one**
environment — the CSTR (a reactor-temperature band) — in its
`pc-gym_paper/constraint_showcase/`. That showcase exists to demonstrate the
constraint *feature*; it is not a claim that the other processes are
unconstrained. We copied the CSTR constraint **verbatim**:

> `cons = {'T': [327, 321]}`, `cons_type = {'T': ['<=', '>=']}` → **321 K ≤ T ≤ 327 K**.

For four_tank, multistage_extraction, and crystallization, PC-Gym defines no
constraint anywhere (training scripts, docs, notebooks, tests). Because the
dissertation's **headline claim C1 — safety *during* training — is defined by
constraint violations**, an environment with no constraint cannot carry any C1
evidence. To get safety results across the suite rather than on the CSTR alone,
we added one physically-motivated constraint to each of the other three. These
are clearly marked **OURS (not verbatim PC-Gym)** in `scenarios.py`.

## Design principles (applied to all three)

Each added constraint had to satisfy five criteria:

1. **Real physical hazard.** It must correspond to a genuine failure mode of that
   unit operation — something a plant engineer would actually protect against
   (overflow, off-spec product, loss of process driving force), not an arbitrary
   numeric fence.
2. **On an exposed, bound-able state.** It must be a state the environment
   exposes and that PC-Gym's constraint API can bound — i.e. a key in the model's
   `info()["states"]`, so the native `constraints`/`cons_type` dict works.
3. **Envelope-calibrated.** The bound is placed so that **nominal control (the
   PID/PI baseline) stays inside it, while saturating/exploratory control crosses
   it.** This is the precise regime C1 needs: a safe reference policy exists
   (otherwise "safety" is unachievable and the metric is vacuous), yet untrained
   exploration is genuinely hazardous (otherwise there is nothing for shadow mode
   to prevent). We rejected bounds so tight the baseline violates constantly and
   bounds so loose nothing ever triggers.
4. **Task-compatible.** The verbatim setpoints must sit safely inside the bound,
   so the control task stays feasible and (for the setpoint-tracking
   environments) the NMPC oracle remains solvable with the bound as a hard state
   constraint.
5. **API-faithful implementation.** Added as **PC-Gym-native `env_params`**
   (`constraints`, `cons_type`, `done_on_cons_vio=False`, `r_penalty=False`) so
   the env records `info["cons_info"]` and the do-mpc oracle enforces it as hard
   state bounds; **mirrored in `constraint_spec`** so the eval pipeline's own
   `env.state`-based detector (`constraints.py`) measures it.

### How the bounds were calibrated

For each candidate variable we measured its operating envelope over several seeds
under two regimes:

- **Baseline** — the scenario's PID/PI safety-net controller (nominal control).
- **Reachable** — sustained extreme/corner actions held over an episode, the
  proxy for a saturating, untrained policy (uniform per-step random *understates*
  the hazard because it averages out; an untrained network saturates its outputs
  and holds them, which is what actually drives excursions early in training).

The bound was then set between the baseline peak and the reachable extreme.

---

## 1. four_tank — lower-tank overflow: `h3 ≤ 0.6`, `h4 ≤ 0.6`

**Process.** Four interconnected tanks (Johansson, 2000). Two pumps (`v1`, `v2`)
feed the tanks through a split ratio; the agent controls the two **lower** tank
levels `h3`, `h4` to setpoints (`h3`: 0.5→0.1, `h4`: 0.2→0.3). State order
`[h1, h2, h3, h4]` (+ SP slots).

**Hazard: overflow.** The single most obvious safety limit on a tank is that it
must not overflow. PC-Gym's observation space bounds every level to `[0, 0.6]`
— i.e. `0.6` is the top of the modelled operating range for each tank, which we
take as the safe maximum fill level. The tank dynamics are not internally clipped
to this (levels *can* integrate above it), so exceeding `0.6` is a genuine,
representable overflow event.

**Why `0.6` is the right number (calibration).**

| Tank | Baseline peak | Reachable (held max pump) | Bound |
|------|--------------:|--------------------------:|------:|
| h3 (controlled to 0.5) | 0.589 | **0.733** | 0.6 |
| h4 (controlled to 0.3) | 0.337 | 0.414 | 0.6 |

The baseline tracks `h3`=0.5 with overshoot to 0.589 — safely under 0.6 — while
sustained maximum pumping drives `h3` to **0.733**, a clear overflow. So nominal
control is safe and aggressive control overflows, exactly as required. `h4` never
approaches 0.6 under any regime; we still bound it because it is the *same
physical tank ceiling* applied to the other controlled tank (an honest limit that
simply is not binding here), keeping the constraint set symmetric and physically
complete.

**Appropriateness.** Tank overflow is the textbook safety constraint for level
control and is the canonical constraint used with the quadruple-tank benchmark.
The setpoint (0.5) sits comfortably below the bound, so the task and the NMPC
oracle remain feasible — verified: the oracle solves the full episode with `h3`
in 0.092–0.526, zero steps out of bound.

---

## 2. multistage_extraction — product off-spec: `X5 ≤ 0.5`

**Process.** A counter-current liquid–gas extraction column (Ingham et al.,
2007). `X_i` is the solute concentration in the **liquid** phase at stage `i`
(observation space `[0, 1]`); the agent manipulates liquid/gas flowrates (`L`,
`G`) to control the stage-5 liquid concentration `X5` to setpoints 0.3→0.4→0.3.
State order `[X1, Y1, …, X5, Y5]` (+ SP slot); `X5` is index 8.

**Hazard: off-spec product.** The purpose of an extraction column is to *remove*
solute from the liquid; an excessive solute concentration in the controlled
stream is an off-specification / poor-separation condition (and, at the limit,
indicative of column flooding from excessive flow). A maximum-purity ceiling on
the controlled output is therefore the natural process constraint.

**Why `0.5` is the right number (calibration).**

| Variable | Baseline peak | Reachable (extreme flows) | Setpoints | Bound |
|----------|--------------:|--------------------------:|----------:|------:|
| X5 | 0.474 | **0.585** | 0.3 / 0.4 | 0.5 |

The setpoints (max 0.4) and the PI baseline (peak 0.474) sit below 0.5, while
sustained extreme flows push `X5` to **0.585**. The bound is just above the
nominal operating maximum, so good tracking is in-spec and aggressive control
goes off-spec. Verified: the oracle solves the full episode with `X5` in
0.298–0.402, zero steps out of bound.

**Appropriateness.** Product-quality limits are the dominant constraint class in
separation processes. Placing the ceiling on the *controlled* output (rather than
an internal stage) keeps it interpretable and directly tied to the control
objective: "track the setpoint, but never let the product exceed the off-spec
ceiling."

---

## 3. crystallization — minimum solute concentration: `Conc ≥ 0.11`

**Process.** K₂SO₄ crystallization by the method of moments (de Moraes et al.,
2023). State `[Mu0, Mu1, Mu2, Mu3, Conc, CV, Ln]`; `Conc` (index 4) is the
solute concentration `c` in solution (observation space `[0, 0.5]`). The action
is a **delta-u** change in cooling temperature; the agent shapes the crystal-size
distribution (CV, Ln). Supersaturation is `S = c·10³ − C_eq(T)`, the
thermodynamic driving force for growth and nucleation.

**Hazard: over-depletion / loss of driving force.** Crystallization should be
operated within the metastable zone. Aggressive over-cooling crashes the
concentration as solute deposits onto crystals; if `c` falls too far the solution
approaches depletion — the supersaturation driving force collapses and the
process loses controllability (and yield/quality degrade). A **minimum**
concentration is therefore the relevant safety/operability limit. (Concentration
only falls under cooling here — `x0`=0.1586 is the maximum and the dynamics
deplete it — so a *lower* bound is the meaningful one; an upper bound would never
be active.)

**Why `0.11` is the right number (calibration).**

| Variable | Baseline min | Reachable (extreme cooling) | x0 | Bound |
|----------|-------------:|----------------------------:|---:|------:|
| Conc | 0.125 | **0.103** | 0.1586 | 0.11 |

The P-baseline holds `Conc` ≥ 0.125 while sustained extreme cooling drives it to
**0.103**; `0.11` sits between, so nominal control is safe and aggressive cooling
over-depletes. The starting concentration (0.1586) is well above the bound, so no
spurious violation at `t=0`.

**Implementation note.** Unlike the other two, crystallization uses a delta-u
action space, which PC-Gym's do-mpc oracle does not support — so `NMPCController`
/ `run_rollouts` skip the oracle for this scenario. The constraint is therefore
**recorded by the environment** (`info["cons_info"]`) and **measured by the eval
pipeline** (`constraint_spec` from `env.state`), but it is not imposed as an MPC
bound (there is no MPC here). This is a property of the scenario, not a gap in the
constraint.

**A note on units.** `Conc` is the model's solute-concentration state `c`
(`o_space [0, 0.5]`); from the supersaturation relation `S = c·10³ − C_eq` with
`C_eq ≈ 120` g/L at 25 °C, `c` is a *scaled* concentration rather than a verified
named unit, so we leave its unit label blank rather than assert one.

---

## Summary

| Scenario | Constraint | Hazard | Source | Baseline peak/min | Reachable | C1-active? | Oracle |
|----------|-----------|--------|--------|------------------:|----------:|-----------:|--------|
| cstr | 321 ≤ T ≤ 327 K | thermal runaway / quench band | **verbatim PC-Gym** | — | — | yes | enforced |
| four_tank | h3 ≤ 0.6, h4 ≤ 0.6 | tank overflow | ours | 0.589 / 0.337 | 0.733 | yes (h3) | enforced |
| multistage | X5 ≤ 0.5 | off-spec product | ours | 0.474 | 0.585 | yes | enforced |
| crystallization | Conc ≥ 0.11 | over-depletion | ours | 0.125 | 0.103 | yes | N/A (delta-u) |

Under the saturating-policy proxy (per-episode held random action, 20 seeds) the
baseline incurred **0%** violations on all three of our constrained scenarios,
while the proxy incurred 8% (four_tank), 10% (multistage) and 2% (crystallization)
— confirming each bound separates safe nominal control from hazardous exploration,
the separation the C1 experiment relies on.

## Honesty / threats to validity

- **These three bounds are ours, not PC-Gym's.** They are physically principled
  and envelope-calibrated, but a reviewer should know they were chosen by us; the
  CSTR band is the only verbatim-PC-Gym constraint.
- **The CSTR baseline is not violation-free (~10%).** Its verbatim operating
  point starts at `x0` T=330 K, *above* the verbatim [321, 327] band (PC-Gym's
  showcase started at 325 K, inside it), so the opening transient violates for
  every controller. We left both `x0` and the band verbatim; the C1 gradient
  still holds (baseline well below exploratory violation rates).
- **Constraint strength varies.** Crystallization produces the weakest signal
  (rare, small-magnitude violations), partly because its P-baseline is itself a
  weak controller; four_tank's overflow is genuine but only `h3` is near-binding.
  These are honest properties of the verbatim operating points, not tuning knobs
  we optimised for effect.
