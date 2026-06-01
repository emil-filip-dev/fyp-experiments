# FYP Research Findings and Development Log

**Project:** Making Reinforcement Learning Reliable for Industrial Process Control
**Student:** Imperial College London, 2025–2026
**Supervisor:** [assigned]
**Date:** April 2026

---

## 1. Project Overview and Research Objectives

This project sits at the intersection of control theory and machine learning, targeting a fundamental open problem: how can we safely deploy reinforcement learning agents in real industrial chemical processes, where safety constraints must be satisfied and random exploration can damage equipment or compromise product quality?

### Research Questions

1. **RQ1:** Can Shadow Mode RL (Gassert & Althoff, 2024) be adapted from simple 2D tasks to nonlinear chemical process control (CSTR, crystallisation)?
2. **RQ2:** Is a CIRL-style PID baseline better than a hand-tuned PID as the shadow mode baseline $\pi^b$?
3. **RQ3:** Can SLTD-inspired uncertainty-aware switching improve the quality of switching decisions?
4. **RQ4:** What constraint satisfaction guarantees can be provided for the combined shadow policy?

---

## 2. Literature Analysis and Synthesis

### 2.1 The Core Insight from Shadow Mode

The Shadow Mode paper identifies that the key barrier to on-system RL training is the cost of *bad actions* during early training, when the policy is essentially random. Shadow mode solves this by:

- Keeping the baseline in control whenever the agent is not confident it can do better
- Gradually increasing agent control as competence is demonstrated
- Providing guided exploration via baseline trajectories to interesting states

The Q-value switching criterion ($a^c = a^a$ if $Q(s,a^a) > Q(s,a^b)$) is particularly elegant because:
- No additional hyperparameter needed
- Naturally incorporates temporal credit assignment
- The agent "earns" control by learning good Q-values

### 2.2 The PC-Gym Optimality Gap as Ground Truth

A key insight from the PC-Gym paper is the *optimality gap* $\Delta = J(\pi^*) - J(\pi_\theta)$, where $\pi^*$ is the NMPC oracle. This gives us an absolute, interpretable performance ceiling. Standard RL achieves gaps of:
- CSTR: $\Delta_{\text{DDPG}} = 0.008$ (near-optimal)
- Crystallisation: $\Delta_{\text{PPO}} = 0.211$ (significant room)
- Four-Tank: $\Delta_{\text{SAC}} = 0.579$ (challenging)

The hypothesis is that shadow mode should *reduce the optimality gap* compared to standard RL, by avoiding catastrophic failure regions during training and getting guided exploration toward high-reward states.

### 2.3 Why CIRL is a Natural Shadow Mode Baseline

The CIRL paper demonstrates that embedding PID structure into the RL policy produces:
1. **Better starting point:** CIRL initialises at better-than-random reward (structured prior)
2. **Inherent disturbance rejection:** PID layer responds to errors even for unseen disturbances
3. **Interpretable switching:** When shadow mode switches, we can inspect CIRL's PID gains to understand *why* the baseline is handling the situation

Using CIRL as $\pi^b$ (instead of a static PID) makes the baseline adaptive, which should push the combined policy to a higher floor.

### 2.4 SLTD's Pre-emptive Deferral Insight

The most relevant insight from SLTD for process control is **pre-emptive deferral**: defer *before* the ML policy causes harm, not just when it fails. In process terms:
- A CSTR approaching a temperature constraint boundary should trigger deferral *before* violation, not after
- This maps naturally to constraint-aware switching: if a proposed action would push the state toward a constraint, prefer the baseline

SLTD achieves this by comparing long-term value estimates, exactly like Q-value switching. The difference is SLTD uses Bayesian dynamics models to quantify uncertainty in those estimates.

---

## 3. Implementation Architecture

### 3.1 Shadow Mode DDPG (`shadow_mode.py`)

The implementation faithfully reproduces the Gassert & Althoff algorithm with adaptations for the CSTR environment.

**Key design decisions:**

| Decision | Choice | Rationale |
|---|---|---|
| RL algorithm | DDPG | Required for Q-value switching; off-policy for replay |
| Baseline | PID Controller (normalised space) | Matches PC-Gym convention; always available |
| Network | 256-hidden MLP × 2 layers | Balances capacity vs. overfitting for low-dim CSTR |
| Buffer | 100k transitions | Off-policy; stores executed actions for critic training |
| Exploration | Gaussian noise on actor | Standard DDPG exploration |
| Replay start | 5000 warmup steps | Avoid degenerate early updates |

**Critical implementation detail (from paper, Section 4.1.2):**
> "The transition tuples $(s_t, a^c_t, r_t, s_{t+1})$ used for training the Q-function must contain the *actually executed action* $a^c$."

This is why `ReplayBuffer.push()` stores `action_exec` (the action actually applied to the environment), not the agent's proposed action. The agent's action is stored separately for actor training.

### 3.2 Two Switching Mechanisms

**Q-value switching (recommended):**
```python
q_agent    = critic(s, action_agent)
q_baseline = critic(s, action_baseline)
use_agent  = q_agent > q_baseline
```

This is preferred because:
- No extra hyperparameter ($\eta_{\text{agent}}$)
- Q-values are already computed during training
- Naturally accounts for long-term consequences

**Action-decision switching:**
```python
# Actor outputs extra dim
action_agent, decision_prob = actor(state)
use_agent = decision_prob > eta_agent  # η ∈ [0,1]
```

With optional regularisation:
```python
reg_loss = lambda_reg * ||action_agent - action_baseline||
```

### 3.3 Randomised Training Start

At the start of each episode, a random $t^{\text{train}}$ is sampled uniformly from $[0, 0.5 \times N]$. The baseline runs for this many steps before shadow mode activates. This:
1. Exposes the agent to states throughout the episode
2. Avoids always training from the same initial state
3. Provides baseline action demonstrations for states the agent hasn't visited

---

## 4. Novel Extensions

Beyond the base paper implementation, this FYP explores several novel directions:

### 4.1 Extension 1: Constraint-Aware Shadow Mode

**Motivation:** The original shadow mode makes no use of constraint information. In chemical processes, operating near constraint boundaries (e.g., maximum temperature) is particularly dangerous.

**Proposed modification:** Augment the switching criterion with a constraint safety margin:

$$a_t^c = \begin{cases} a_t^b & \text{if } g_i(\hat{x}_{t+1}) > -\epsilon \text{ for any } i \text{ (near constraint)} \\ a_t^a & \text{if } Q(s_t, a_t^a) > Q(s_t, a_t^b) \\ a_t^b & \text{otherwise} \end{cases}$$

where $\hat{x}_{t+1}$ is a one-step lookahead prediction and $\epsilon > 0$ is a safety margin. This creates a hierarchy: constraint safety overrides Q-value comparison.

For the CSTR, the relevant constraint is reactor temperature $T \in [321\text{K}, 327\text{K}]$.

**Implementation status:** Planned — requires one-step model prediction (can use the CasADi model embedded in PC-Gym).

### 4.2 Extension 2: CIRL as Shadow Mode Baseline

**Motivation:** The static PID baseline in shadow mode has fixed gains and cannot adapt to setpoint changes. CIRL's neural-network gain scheduler should provide a better baseline because:
- Gains are tuned to the current operating point
- Generalises across the setpoint trajectory
- Disturbance rejection is maintained

**Design:** Replace `PIDController` with a pre-trained CIRL agent:
```python
class CIRLBaseline:
    def __init__(self, model_path):
        self.model = CIRLAgent.load(model_path)

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)
```

Expected effect: higher floor on combined policy performance; shadow mode agent needs to work harder to "earn" control.

**Implementation status:** Requires training CIRL first (separate script); then drop-in replacement in `shadow_mode.py`.

### 4.3 Extension 3: Uncertainty-Quantified Switching

**Motivation:** Early in training, the Q-values are poorly estimated (random initialisation). Switching based on random Q-values leads to random (rather than exploratory) behaviour.

**Proposed modification (inspired by SLTD):** Use ensemble Q-networks to quantify epistemic uncertainty in Q-value estimates:

$$a_t^c = \begin{cases} a_t^a & \text{if } P(Q_\theta(s,a^a) > Q_\theta(s,a^b)) > \tau \\ a_t^b & \text{otherwise} \end{cases}$$

where the probability is computed from an ensemble of $K$ critic networks. During early training, all critics disagree → probability stays near 0.5 → baseline is preferred. As training progresses, critics converge → probability becomes more decisive.

**Implementation status:** Designed. Requires replacing single critic with $K=5$ ensemble.

### 4.4 Extension 4: Multi-Environment Shadow Mode

**Motivation:** The optimal switching policy may be environment-specific (different constraints, dynamics, timescales). A meta-learned switching policy could generalise across PC-Gym environments.

**Design:** Train shadow mode jointly on CSTR and four-tank system, sharing the critic architecture but with environment-specific observation encoders. The combined policy $\pi^c$ learns which environments it can handle vs. which need the baseline.

**Implementation status:** Future work direction.

---

## 5. Key Findings and Observations

### 5.1 Algorithm Design Findings

**Finding 1: Q-value switching is more robust than action-decision switching**

The action-decision mechanism (Section 4.1.1) requires the agent to simultaneously learn:
- The best action for each state
- Which states it should control vs. defer

This dual objective can lead to exploitation: the agent learns to always defer (decision → 0) to avoid the harder task of learning a good action policy. Regularisation ($\lambda > 0$) partially mitigates this but introduces another hyperparameter.

Q-value switching avoids this problem because the switching criterion emerges naturally from the Q-function, which is already being trained for the primary RL objective.

**Finding 2: Randomised training start is critical for state coverage**

Without randomised start times, the agent only trains from states that occur early in the episode (near initial conditions). The CSTR setpoint switches at step 50; without randomised starts, the agent rarely sees the behaviour near the second setpoint during early training.

**Finding 3: The baseline must cover the full state space**

If the baseline is poorly calibrated for certain regions of the state space, the agent will never be taken to those states (and thus never learn to handle them). The PID controller works well near the first setpoint but may struggle near the second — this is exactly the region where the RL agent should learn to do better.

### 5.2 Process Control Specific Insights

**Finding 4: CSTR is an ideal shadow mode testbed**

The CSTR exhibits a clear structure:
- Near initial conditions: both PID and RL work
- Near the obstacle (setpoint transition): PID overshoots; RL can do better
- At steady state: PID is near-optimal; RL should defer

This mirrors exactly the reach-avoid task structure in the original shadow mode paper, making CSTR the most natural process control analogue.

**Finding 5: The optimality gap interpretation for shadow mode**

For shadow mode, the relevant metric is not just the final optimality gap but also:
- **Safety during training:** total constraint violations during learning
- **Sample efficiency:** how quickly the gap reduces below the baseline gap
- **Takeover curve:** the fraction of steps the agent controls, over training

A well-functioning shadow mode should show: (a) agent takeover % gradually increases, (b) reward increases monotonically or near-monotonically, (c) no constraint violations even during early training.

---

## 6. Experimental Plan

### 6.1 Baseline Experiments

| Experiment | Config | Goal |
|---|---|---|
| Pure DDPG | Standard DDPG on CSTR | Comparison baseline |
| Pure PID | Static PID controller | Minimum performance floor |
| Shadow (Q-val) | Mode='qvalue', 200k steps | Core implementation |
| Shadow (Agent, λ=0) | Mode='agent', λ=0 | Action-decision without reg |
| Shadow (Agent, λ=2) | Mode='agent', λ=2 | With regularisation |
| NMPC Oracle | PC-Gym oracle | Performance ceiling |

### 6.2 Evaluation Metrics

For each method, report over 20 evaluation seeds:
1. **Mean episode reward** ± std
2. **Optimality gap** $\Delta = J(\pi^*) - J(\pi)$ (against NMPC)
3. **Agent takeover fraction** at the end of training
4. **Constraint violation rate** (when constraints are active)
5. **Training reward curve** — monotonicity and convergence speed

### 6.3 Hyperparameter Study

For shadow mode specifically, study sensitivity to:
- Training steps (100k, 200k, 500k)
- Noise standard deviation (0.05, 0.1, 0.2)
- Max t_train fraction (0.25, 0.5, 0.75)
- Regularisation λ (0, 0.5, 1, 2, 5) — agent-decision mode only

---

## 7. Running the Code

### Setup

```bash
# Activate virtual environment
.venv/Scripts/activate    # Windows

# Verify installation
.venv/Scripts/python -c "from pcgym import make_env; print('PC-Gym OK')"
```

### Training

```bash
# Train Shadow Mode with Q-value switching (recommended)
.venv/Scripts/python shadow_mode.py --mode qvalue --steps 200000

# Train with agent-decision switching + regularisation
.venv/Scripts/python shadow_mode.py --mode agent --steps 200000 --lambda-reg 2.0

# Run full comparison of all methods
.venv/Scripts/python shadow_mode.py --mode compare --steps 100000

# Evaluate a saved checkpoint
.venv/Scripts/python shadow_mode.py --eval-only --checkpoint runs/shadow_qvalue/best.pt
```

### PC-Gym Example

```bash
# Run the basic CSTR + PPO + PID example
.venv/Scripts/python example_pcgym.py
```

---

## 8. Next Steps and Open Questions

### Immediate Next Steps

1. **Run shadow_mode.py** on CSTR and collect training curves
2. **Compare reward curves** — does shadow mode outperform pure PID and approach NMPC?
3. **Check takeover curves** — does agent takeover % increase monotonically?
4. **Test constraint scenarios** — add PC-Gym constraints and measure violation rate

### Medium-Term Directions

5. **CIRL baseline:** Pre-train CIRL agent, use as $\pi^b$ in shadow mode
6. **Constraint-aware switching:** Implement lookahead constraint check
7. **Multi-seed evaluation:** 5+ random seeds for statistical robustness
8. **Harder environments:** Extend to crystallisation (hardest PC-Gym env, $\Delta = 0.211$)

### Long-Term Directions

9. **Ensemble switching:** Uncertainty-aware deferral à la SLTD
10. **Theoretical analysis:** Safety guarantee for constraint satisfaction during training
11. **Transfer:** Can a shadow mode policy trained on CSTR transfer to multistage extraction?

### Open Research Questions

- **Q:** What makes a "good" baseline for shadow mode in process control? Is higher baseline performance always better, or does a challenging baseline push the agent to learn more?
- **Q:** How does the optimality gap evolve *during training* in shadow mode vs. standard RL? Is the training trajectory safer (monotonically decreasing gap)?
- **Q:** Can the Q-value switching threshold be learned (meta-learned) rather than fixed?
- **Q:** For highly nonlinear processes (crystallisation), does shadow mode avoid the training instabilities that standard RL suffers?

---

## 9. Connections to FYP Thesis Structure

This project naturally structures into the following thesis chapters:

1. **Background:** MDP theory, RL algorithms, process control basics, NMPC
2. **Literature Review:** Shadow mode, CIRL, SLTD, constrained RL in PSE
3. **Methodology:** Shadow mode algorithm, adaptations for chemical processes, PC-Gym integration
4. **Experiments:** CSTR baseline; constraint scenarios; CIRL baseline comparison
5. **Extensions:** Constraint-aware switching; uncertainty quantification
6. **Discussion and Conclusions:** Safety, reliability, practical deployment considerations

---

## 10. Hard Constraint Satisfaction and Safe Handover in RL

*Research summary — April 2026*

This section surveys approaches to ensuring RL agents never violate system constraints, including methods that hand control back to a safe/conventional controller when constraint violation is imminent. These are directly relevant to RQ4 (what constraint satisfaction guarantees can be provided?) and Extension 1 (constraint-aware shadow mode).

---

### 10.1 Problem Statement

Standard RL optimises expected cumulative reward with no guarantee on individual trajectory constraint satisfaction. For industrial chemical processes this is unacceptable: a temperature excursion in a CSTR, even momentarily, can damage the reactor or ruin the batch. Five distinct families of approaches have been developed to address this.

---

### 10.2 Approach 1 — Constrained MDPs + Lagrangian Methods / CPO

**Concept:** Reformulate state/input constraints as additional cost functions within a Constrained Markov Decision Process (CMDP). The constrained optimisation problem is either relaxed via a Lagrange multiplier (dual ascent), or solved directly by Constrained Policy Optimization (CPO), which extends TRPO with a trust-region constraint on expected cumulative cost.

**Guarantee:** Constraint satisfaction *in expectation* over trajectories — individual trajectories can still violate. CVaR-constrained variants (CVaR-CPO) address tail risk by constraining the conditional value-at-risk of cost, but still do not provide per-step hard guarantees.

**Relevance to FYP:** This is the natural soft-constraint baseline. Stable-baselines3 has drop-in Lagrangian wrappers. Useful for demonstrating *why* soft methods are insufficient for industrial control, motivating harder approaches.

**Sources:**
- [Constrained Policy Optimization — Achiam et al., 2017](https://www.ri.cmu.edu/app/uploads/2017/11/1705.10528.pdf)
- [Survey of Constraint Formulations in Safe RL — IJCAI 2024](https://www.ijcai.org/proceedings/2024/0913.pdf)
- [Empirical Study of Lagrangian Methods in Safe RL, 2025](https://arxiv.org/html/2510.17564v1)

---

### 10.3 Approach 2 — Control Barrier Functions (CBFs)

**Concept:** Define a safe set $\mathcal{S} = \{x : h(x) \geq 0\}$ via a scalar function $h$. A Control Barrier Function enforces the invariance condition $\dot{h}(x,u) \geq -\alpha h(x)$, guaranteeing the system can never exit $\mathcal{S}$. At each timestep, the RL agent proposes an action which is *projected* onto the CBF-admissible set via a QP:

$$u^* = \arg\min_{u} \|u - u_{\text{RL}}\|^2 \quad \text{s.t.} \quad \dot{h}(x,u) \geq -\alpha h(x)$$

The RL policy trains freely; the CBF acts as a safety filter on top, correcting actions only when necessary.

**Guarantee:** Forward invariance of $\mathcal{S}$ — hard constraint satisfaction on every step, given a known differentiable model.

**Limitation:** Requires an explicit differentiable model and a manually designed $h(x)$. For nonlinear chemical processes this is non-trivial but tractable with CasADi. Learned CBFs (neural barrier certificates) relax the manual design requirement at the cost of probabilistic rather than deterministic guarantees.

**Relevance to FYP:** The CasADi models embedded in PC-Gym can provide $\dot{h}$ analytically. Defining $h(x) = T_{\max} - T$ (temperature margin) for the CSTR is straightforward. Would combine naturally with the shadow mode switching logic.

**Sources:**
- [End-to-End Safe RL through Barrier Functions — Cheng et al., 2019](https://arxiv.org/abs/1903.08792)
- [Safe RL Using Robust Control Barrier Functions](https://www.semanticscholar.org/paper/22e586f420b596e398072b07b1d4a08c2a067c04)
- [Review: Safe RL Using Lyapunov and Barrier Functions, 2025](https://arxiv.org/html/2508.09128v1)
- [Safe RL Using MPC with Probabilistic CBF — IEEE 2024](https://ieeexplore.ieee.org/document/10644734/)

---

### 10.4 Approach 3 — Predictive Safety Filter (PSF)

**Concept:** A wrapper around any RL policy that checks, before each action is applied, whether the proposed action leads to a feasible MPC problem over a finite horizon $N$. If the MPC problem is feasible (i.e. there exists a safe recovery trajectory from the resulting state), the RL action is applied. If infeasible, the MPC provides a minimally corrected safe action instead.

```
RL proposes u_RL
  → Solve MPC feasibility check over N steps
  → If feasible:    apply u_RL
  → If infeasible:  apply u_MPC (minimal correction to restore feasibility)
```

**Guarantee:** Hard constraint satisfaction on every step, as long as the initial state is feasible and the MPC model is accurate.

**Relevance to FYP:** This is the most natural fit. The NMPC oracle already available via do-mpc + CasADi *is* the MPC component. The PSF can be implemented as a thin wrapper around any RL policy: check feasibility, override if needed. The *intervention rate* (fraction of steps where MPC overrides RL) is a direct measure of how often the RL agent needs help — a useful metric alongside the optimality gap. As the RL agent improves, the intervention rate should decrease.

**This directly answers the "handover" question:** the PSF detects ahead of time when the RL action would lead to constraint violation and hands over to MPC for that step.

**Sources:**
- [Predictive Safety Filter for Learning-Based Control — Wabersich & Zeilinger, 2021](https://www.sciencedirect.com/science/article/abs/pii/S0005109821001175)
- [PSF for RL — Safe Exploration of Nonlinear Dynamical Systems](https://www.researchgate.net/publication/329641554_Safe_exploration_of_nonlinear_dynamical_systems_A_predictive_safety_filter_for_reinforcement_learning)
- [Predictive Safety Shield for Dyna-Q RL, 2024](https://arxiv.org/html/2511.21531)
- [Safety Assessment in RL via MPC, 2025](https://arxiv.org/html/2510.20955)

---

### 10.5 Approach 4 — Model Predictive Shielding (MPS)

**Concept:** A runtime monitor (the "shield") watches the RL policy's proposed actions. When an action would move the system outside a defined safe set (where a backup policy is known to maintain safety), the shield overrides with the backup. Model Predictive Shielding adds an MPC-based forward simulation to make this check over multiple steps rather than one, enabling earlier intervention.

Dynamic MPS (NeurIPS 2024) further adapts the shield online using the RL agent's own long-horizon predictions, reducing unnecessary interventions. Adaptive Robust MPS (2025) adds robustness to model uncertainty — directly relevant for chemical processes with imperfect models.

**Distinction from PSF:** PSF modifies the action minimally to restore feasibility; MPS switches entirely to a backup policy for the remainder of the unsafe period.

**Relevance to FYP:** The NMPC oracle serves as the backup policy. Adaptive Robust MPS is particularly relevant given model-plant mismatch in real chemical processes.

**Sources:**
- [Dynamic Model Predictive Shielding — NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/b589d92785e39486e978fa273d0dc343-Paper-Conference.pdf)
- [Dynamic MPS (preprint)](https://arxiv.org/html/2405.13863v1)
- [Adaptive Robust MPC Shielding for Process Control — 2025](https://www.sciencedirect.com/science/article/pii/S0098135425005241)

---

### 10.6 Approach 5 — Recovery RL

**Concept:** Two separate policies are trained: a *task policy* that maximises reward with no constraint awareness, and a *recovery policy* that takes over when constraint violation is imminent. A *safety critic* $Q_{\text{safety}}(s, a)$ estimates the probability of future constraint violation; when this exceeds a threshold $\tau$, the recovery policy takes control.

```
safety_q = Q_safety(state, task_action)
if safety_q > τ:
    action = recovery_policy(state)   # safety takes over
else:
    action = task_policy(state)       # normal operation
```

Offline data from constraint-violating episodes pre-trains the safety critic before policy learning begins, avoiding dangerous exploration from scratch.

**Guarantee:** Probabilistic — the safety critic may under-estimate risk in novel states. Softer than PSF/MPS but requires no explicit model.

**Relevance to FYP:** Structurally the reverse of Shadow Mode. Shadow Mode: RL shadows a controller and steps in when it's confident. Recovery RL: a safe controller shadows RL and steps in when RL is unsafe. A **bidirectional shadow mode** combining both directions is a natural FYP contribution.

**Sources:**
- [Recovery RL — Berkeley, 2021 (arXiv)](https://arxiv.org/pdf/2010.15920)
- [Recovery RL — IEEE RA-L](https://ieeexplore.ieee.org/document/9392290/)
- [Learning to Recover for Safe RL, 2023](https://arxiv.org/html/2309.11907)

---

### 10.7 Synthesis: Connecting the Approaches to this FYP

Shadow Mode (Gassert & Althoff, 2024) solves the *entry* direction: safe transition from a conventional controller *into* RL as competence is demonstrated. The research question for this FYP is the *exit* direction: safe transition back to a conventional controller when constraints are at risk.

| Approach | Hard guarantee | Model needed | Fits existing tooling | Complexity |
|---|---|---|---|---|
| Lagrangian / CPO | No (expectation) | No | High (SB3) | Low |
| CBF + RL | Yes (if model exact) | Yes (differentiable) | Medium (CasADi) | Medium |
| Predictive Safety Filter | Yes | Yes (MPC) | High (do-mpc exists) | Medium |
| MPS | Yes | Yes (MPC) | High (do-mpc exists) | Medium |
| Recovery RL | No (probabilistic) | No | Medium | Medium |

**Recommended path for RQ4 / Extension 1:**

1. Implement a PSF wrapper using the existing do-mpc + CasADi NMPC oracle as the feasibility checker.
2. Evaluate on CSTR: track *intervention rate* alongside reward and optimality gap.
3. Compare against Lagrangian-PPO as the soft-constraint baseline.
4. Frame the combined system as a bidirectional shadow mode: Shadow Mode handles RL take-over; PSF handles RL hand-back.

The PSF is the most natural choice because: (a) the NMPC oracle already exists in the codebase, (b) it provides provably hard guarantees, (c) the intervention rate is an interpretable metric that directly quantifies how much the RL agent needs safety assistance over training.

---

### 10.8 Additional Reading

- [Enforcing Hard Constraints with Soft Barriers — ICML 2023](https://proceedings.mlr.press/v202/wang23as/wang23as.pdf)
- [Safe RL Baselines Repository (GitHub)](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baselines)
- [Shadow Mode RL — Gassert & Althoff, 2024 (arXiv)](https://arxiv.org/abs/2410.23419)

---

```
Algorithm: Shadow Mode DDPG Training
======================================
Input:  Environment E, Baseline π^b, Max steps T_max
Output: Trained agent π^a, Combined policy π^c

Initialise: Actor π^a_θ, Critics Q_φ, Replay buffer B
Initialise target networks: π^a_θ', Q_φ'

for step = 1 to T_max do

  if new episode:
    obs ← E.reset()
    t_train ← Uniform(0, 0.5 × N_steps)  // randomised start

  a^b ← π^b(obs)                          // baseline always acts

  if step_in_episode < t_train:
    a^exec ← a^b                          // follow baseline during warmup
    a^agent ← a^b                         // record baseline as "agent action"
  else:
    a^agent ← π^a_θ(obs) + ε             // agent proposes action + noise

    if mode == 'qvalue':
      if Q_φ(obs, a^agent) > Q_φ(obs, a^b):
        a^exec ← a^agent                  // agent takes control
      else:
        a^exec ← a^b                      // baseline retains control

    elif mode == 'agent':
      (a^agent, decision) ← π^a_θ(obs)
      if decision > η_agent:
        a^exec ← a^agent
      else:
        a^exec ← a^b

  next_obs, r ← E.step(a^exec)
  B.push(obs, a^exec, r, next_obs, done, a^agent, a^b)

  // DDPG update
  (s, a_e, r, s', d, a_ag, a_bl) ← B.sample(batch_size)

  // Critic update
  a_next ← π^a_θ'(s')
  y      ← r + γ(1-d) Q_φ'(s', a_next)
  L_Q   ← MSE(Q_φ(s, a_e), y)
  Update φ via ∇L_Q

  // Actor update
  L_π ← -E[Q_φ(s, π^a_θ(s))]
  if mode == 'agent' and λ > 0:
    L_π ← L_π + λ ||π^a_θ(s) - a_bl||   // regularisation (Eq. 5)
  Update θ via ∇L_π

  // Soft target update
  φ' ← τφ + (1-τ)φ'
  θ' ← τθ + (1-τ)θ'

return π^a_θ, π^c (combined policy using trained agent + baseline)
```

---

## Appendix B: CSTR Model Equations

The PC-Gym CSTR implements the standard two-state reactor:

$$\frac{dC_A}{dt} = \frac{q}{V}(C_{Af} - C_A) - k C_A e^{-E_A/(RT)}$$

$$\frac{dT}{dt} = \frac{q}{V}(T_f - T) - \frac{\Delta H_R}{\rho C_p} k C_A e^{-E_A/(RT)} + \frac{UA}{\rho C_p V}(T_c - T)$$

**Control problem:** Track a setpoint schedule for $C_A$ by manipulating $T_c$ (cooling water temperature).

**Normalised observations** (what the RL agent sees):
- $o_0 = 2(C_A - 0.7)/(1.0 - 0.7) - 1$ (normalised concentration)
- $o_1 = 2(T - 300)/(350 - 300) - 1$ (normalised temperature)
- $o_2 = 2(C_A^* - 0.8)/(0.9 - 0.8) - 1$ (normalised setpoint)

**Reward function** (PC-Gym default with scale):
$$r_t = -1000 \cdot |\bar{C}_A^* - \bar{C}_A|_1 - (\bar{u}_t - \bar{u}_{t-1})^T R (\bar{u}_t - \bar{u}_{t-1})$$
