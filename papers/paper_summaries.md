# Research Paper Summaries

Comprehensive summaries of all five core papers for the FYP project:
*"Making Reinforcement Learning Reliable for Industrial Process Control"*

---

## Table of Contents

1. [Stepping Out of the Shadows: RL in Shadow Mode](#1-stepping-out-of-the-shadows-rl-in-shadow-mode)
2. [PC-Gym: Benchmark Environments for Process Control Problems](#2-pc-gym-benchmark-environments-for-process-control-problems)
3. [Control-Informed Reinforcement Learning for Chemical Processes (CIRL)](#3-control-informed-reinforcement-learning-for-chemical-processes-cirl)
4. [A Survey and Tutorial of RL Methods in Process Systems Engineering](#4-a-survey-and-tutorial-of-rl-methods-in-process-systems-engineering)
5. [Learning-to-Defer for Sequential Medical Decision-Making (SLTD)](#5-learning-to-defer-for-sequential-medical-decision-making-sltd)

---

## 1. Stepping Out of the Shadows: RL in Shadow Mode

**Authors:** Philipp Gassert, Matthias Althoff
**Affiliation:** Technical University of Munich, Munich Center for Machine Learning
**arXiv:** 2410.23419v1 [cs.LG], October 2024
**Keywords:** reinforcement learning, agent-expert combination, exploration strategies, imitation

---

### 1.1 Motivation and Problem

Reinforcement learning faces a fundamental barrier in cyber-physical systems (robotics, process automation, power systems): training cannot be accelerated in simulation because either no high-fidelity simulator exists, or the sim-to-real gap is prohibitive. Training directly on the physical system is expensive and dangerous — random RL actions can cause equipment damage. Offline RL methods avoid this but are limited by expert trajectory coverage and do not allow exploration.

**Shadow Mode Training** is the proposed solution: train an RL agent on the real system, but always with a functioning *baseline* controller as a safety net. The agent "trains in the shadow" — observing and occasionally taking control — so the system operates safely throughout training.

---

### 1.2 Problem Formulation

The setting is a **finite-horizon, discrete-time MDP** $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, d^0, p, r, T \rangle$ with:

- States $s_t \in \mathcal{S}$, actions $a_t \in \mathcal{A}$
- Initial distribution $d^0$, transition $p(s_{t+1}|s_t, a_t)$, reward $r_t = r(s_t, a_t)$
- Task performance: expected cumulative reward

$$J(\pi) = \mathbb{E}_{\tau \sim \pi, p}\left[\sum_{t=0}^{T} r_t\right] \tag{1}$$

Two policies are defined:
- $\pi^a$: the **RL agent** policy (learnable, initially random)
- $\pi^b$: the **baseline** policy (fixed, suboptimal but reasonable)

The **combined policy** $\pi^c$ selects between them at each step. The goal is:

$$\pi^c_* = \arg\max_{\pi^c, \pi^a} \mathbb{E}_{\tau \sim \pi^c(\pi^a, \pi^b)}\left[\sum_{t=0}^{T} r_t\right] \tag{3}$$

---

### 1.3 The Shadow Mode Algorithm

At each timestep, both the RL agent and the baseline independently compute their actions:
- $a_t^a = \pi^a(s_t)$ — RL agent's proposed action
- $a_t^b = \pi^b(s_t)$ — baseline's proposed action

Only one action $a_t^c$ is executed. Two mechanisms decide which:

#### 4.1.1 Control Authority by Agent (Action-Based Switching)

The RL agent outputs an *augmented action* $a_t^{a,\text{total}} = (a_t^a, a_t^{\text{decision}})$, where $a_t^{\text{decision}} \in [0,1]$ is the probability of asserting control. A threshold $\eta_{\text{agent}} \in [0,1]$ determines switching:

$$a_t^c = \begin{cases} a_t^a & \text{if } a_t^{\text{decision}} > \eta_{\text{agent}} \\ a_t^b & \text{otherwise} \end{cases} \tag{4}$$

An optional **regularisation penalty** discourages excessive deviation from the baseline:

$$r_t^{\text{reg}} = -\lambda \|a_t^a - a_t^b\| \tag{5}$$

#### 4.1.2 Control Authority by Q-Value (Value-Based Switching)

Using the critic's Q-function to compare actions:

$$a_t^c = \begin{cases} a_t^a & \text{if } Q(s_t, a_t^a) > Q(s_t, a_t^b) \\ a_t^b & \text{otherwise} \end{cases} \tag{6}$$

This requires no additional hyperparameter $\eta_{\text{agent}}$ but needs an architecture that provides $Q$-values (e.g., DDPG).

---

### 1.4 Exploration and Trajectory Generation

A key feature is **randomised training start times**: at the beginning of each episode, the combined agent follows $\pi^b$ for $t^{\text{train}}$ randomly chosen steps before allowing the RL agent to take over. This:
- Ensures exposure to diverse states throughout the episode
- Leverages Jump-Start RL concepts (Uchendu et al., 2023)
- Provides baseline actions as guidance in low-density regions of state space

---

### 1.5 Experimental Validation

**Environment:** A 2D continuous reach-avoid task.
- Agent starts top-left, goal is in right half ($x > 5$), 95% chance of a line-segment obstacle
- Action space $\mathcal{A} = [-1,1]^2$, episode horizon $T = 100$
- Baseline: direct greedy movement toward goal (optimal without obstacles, fails with them)

**Reward functions tested:**
- Sparse: $r_{\text{sparse}} = 500$ on reaching goal, $-1$ per step otherwise
- Dense: additionally rewards progress toward goal and penalises collisions

$$r_{\text{dense}} = r_{\text{reach\_goal}} + 2\cdot\left(\|x_{t-1}^{\text{agent}} - x^{\text{goal}}\| - \|x_t^{\text{agent}} - x^{\text{goal}}\|\right) - 2\cdot\mathbf{1}[\text{collision}] - 1 \tag{10}$$

**RL algorithm:** DDPG (off-policy actor-critic, replay buffer, Q-function available)

**Key results:**
- Standard DDPG (no shadow mode) completely fails to learn with sparse rewards; learns poorly with dense
- Shadow mode with action-based switching + dense reward achieves >90% goal-reaching rate
- Regularisation ($\lambda = 2$) substantially improves performance and enables sparse-reward training
- Q-value switching also succeeds, relying on baseline in >90% of steps but outperforming $\pi^b$
- Decision heatmap shows agent correctly asserts control near the obstacle region only

---

### 1.6 Key Contributions and Limitations

**Contributions:**
1. Novel shadow mode training paradigm enabling on-system RL training
2. Two principled switching mechanisms (action-based and Q-value-based)
3. Randomised training start for state space coverage
4. Demonstrated efficacy on a hard sparse-reward task where standard RL fails

**Limitations / Open Questions:**
- Validated only on a simple 2D task; not yet on real cyber-physical systems
- Does not address formal safety guarantees (action authority is probabilistic, not certified)
- The baseline must cover the full state space reasonably well
- Currently model-free; no theoretical convergence guarantees

---

## 2. PC-Gym: Benchmark Environments for Process Control Problems

**Authors:** Maximilian Bloor, José Torraca, Ilya Orson Sandoval, Akhil Ahmed, Martha White, Mehmet Mercangöz, Calvin Tsay, Ehecatl Antonio Del Rio Chanona, Max Mowbray
**Affiliation:** Imperial College London, University of Alberta, Universidade Federal do Rio de Janeiro
**arXiv:** 2410.22093v2 [eess.SY], October 2024

---

### 2.1 Overview

PC-Gym is an open-source Python library providing Gymnasium-compatible environments for RL in chemical process control. It addresses a gap: existing RL frameworks (Gymnasium, Brax, MuJoCo) do not provide process-control-specific functionality such as NMPC oracles, constraint tracking, and domain-specific disturbance models.

**Key features:**
- Four benchmark chemical process environments
- Nonlinear Model Predictive Control (NMPC) oracle for performance benchmarking
- Disturbance generation and constraint handling
- Customisable reward functions
- Standardised metrics: median expected return, MAD, optimality gap $\Delta$

---

### 2.2 Theoretical Background

#### Dynamical System Model

$$X_{t+1} \sim p(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) \tag{1}$$

with nonlinear state-space representation:
$$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t)$$

where $\mathbf{w}_t \in \mathcal{W} \subseteq \mathbb{R}^{n_w}$ are disturbance/uncertainty parameters.

#### Constrained RL Problem

$$\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^T \gamma^t r_t(X_t, U_t)\right]$$
$$\text{s.t.} \quad X_0 \sim p_0(\mathbf{x}_0), \quad X_{t+1} \sim p(\mathbf{x}_{t+1}|\mathbf{x}_t, \mathbf{u}_t)$$
$$\mathbb{P}[g_i(X_t) \leq 0] \geq 1 - \alpha_{i,t}, \quad i \in \{1,\ldots,n_g\}$$

where $g_i(\cdot)$ are inequality constraints and $\alpha_{i,t}$ is the allowed violation probability.

#### Performance Metrics

- **Median expected return** $\tilde{J}_\pi$: median over multiple seeds
- **MAD:** $\text{MAD}(J(\pi)) = \text{median}(|J(\pi) - \tilde{J}(\pi)|)$ — measures reproducibility
- **Optimality gap:** $\Delta(\pi_\theta) = J(\pi^*) - J(\pi_\theta)$, where $\pi^*$ is the NMPC oracle

---

### 2.3 NMPC Oracle

The oracle solves a finite-horizon optimal control problem at each step:

$$\min_{\mathbf{u}_t, \ldots, \mathbf{u}_{t+N-1}} J(\mathbf{x}, \mathbf{u}) = \sum_{k=t}^{t+N}(\bar{\mathbf{x}}_k - \bar{\mathbf{x}}^*)^\top \mathbf{Q}(\bar{\mathbf{x}}_k - \bar{\mathbf{x}}^*) + \sum_{k=t}^{t+N-1}(\bar{\mathbf{u}}_k - \bar{\mathbf{u}}_{k-1})^\top \mathbf{R}(\bar{\mathbf{u}}_k - \bar{\mathbf{u}}_{k-1})$$

subject to $\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k)$, $\mathbf{g}(\mathbf{x}_k) \leq 0$.

Implemented using **do-mpc** (CasADi + IPOPT), with orthogonal collocation for ODE integration.

---

### 2.4 Benchmark Environments

#### 4.1 Continuously Stirred Tank Reactor (CSTR)

A nonlinear, two-state system with irreversible reaction $A \to B$:

$$\frac{dC_A}{dt} = \frac{q}{V}(C_{Af} - C_A) - k C_A e^{-E_A/RT} \tag{13}$$
$$\frac{dT}{dt} = \frac{q}{V}(T_f - T) - \frac{\Delta H_R}{\rho C_p} k C_A e^{-E_A/RT} + \frac{UA}{\rho C_p V}(T_c - T) \tag{14}$$

- **States:** $C_A$ (mol/L), $T$ (K)
- **Control input:** $T_c$ (cooling water temperature, K)
- **Setpoint tracking:** track $C_A$ setpoint
- **Nonlinearity:** Arrhenius kinetics in exponential terms
- **Challenge:** nonlinear, coupled, potentially unstable

Results: DDPG achieves optimality gap $\Delta = 0.008$ (best among SAC, PPO, DDPG).

#### 4.3 Multistage Extraction Column

Counter-current liquid-gas extraction with 5 stages, 10 ODE states. For stage $n$:

$$\frac{dX_n}{dt} = \frac{1}{V_l}(L(X_{n-1} - X_n) - Q_n) \tag{15}$$
$$\frac{dY_n}{dt} = \frac{1}{V_g}(G(Y_{n+1} - Y_n) + Q_n) \tag{16}$$

where $Q_n = K_{la}(X_n - X_{n,eq})V_l$ is the mass transfer rate.

- Control variables: liquid flowrate $L$ and gas flowrate $G$
- SAC achieves best optimality gap $\Delta = 0.002$

#### 4.4 Crystallisation Reactor

Population balance model for $\text{K}_2\text{SO}_4$ crystallisation using method of moments:

$$\frac{d\mu_0}{dt} = B_0, \quad \frac{d\mu_1}{dt} = G_\infty(\alpha\mu_0 + b\mu_1 \times 10^{-4}) \times 10^4 \tag{19-20}$$
$$\frac{dc}{dt} = -0.5\rho\alpha G_\infty(\alpha\mu_2 \times 10^{-8} + b\mu_3 \times 10^{-12}) \tag{23}$$

Nucleation rate $B_0$ and growth rate $G_\infty$ depend on supersaturation $S$ and temperature $T$ (control variable) via Arrhenius expressions. PPO achieves best $\Delta = 0.211$.

#### 4.5 Four-Tank System

Four interconnected water tanks with nonlinear level dynamics:

$$\frac{dh_1}{dt} = -\frac{a_1}{A_1}\sqrt{2g h_1} + \frac{a_3}{A_1}\sqrt{2g h_3} + \frac{\gamma_1 k_1}{A_1} v_1 \tag{28}$$

(similar equations for tanks 2, 3, 4). Control inputs are pump voltages $v_1, v_2$. SAC achieves $\Delta = 0.579$ (all algorithms struggle; this is the hardest environment).

---

### 2.5 Disturbance and Constraint Handling

**Disturbances:** User-defined time-indexed disturbance schedules injected as additional model inputs. Example (CSTR inlet temperature):

$$T_{in} = \begin{cases} 350\text{ K} & \text{if } i < 20 \text{ or } i > 40 \\ \mathcal{U}(330, 370) & \text{if } 20 \leq i \leq 40 \end{cases}$$

**Constraint reward shaping:**

$$r_{\text{con}}(\mathbf{x}_t, \mathbf{u}_t) = r(\mathbf{x}_t, \mathbf{u}_t) - \lambda \sum_{i=0}^{n_g} \max(0, g_i(\mathbf{x}_t, \mathbf{u}_t)) \tag{34}$$

Results on CSTR temperature constraint ($321\text{ K} \leq T \leq 327\text{ K}$): empirical violation probability drops from 0.0727 (unconstrained) to 0.0003 (shaped reward) vs. 0.0000 (oracle).

---

### 2.6 Overall Performance Table

| Environment | Best $\Delta$ | Best Algorithm |
|---|---|---|
| CSTR | 0.008 | DDPG |
| Multistage Extraction | 0.002 | SAC |
| Crystallisation | 0.211 | PPO |
| Four Tank | 0.579 | SAC |

---

## 3. Control-Informed Reinforcement Learning for Chemical Processes (CIRL)

**Authors:** Maximilian Bloor, Akhil Ahmed, Niki Kotecha, Mehmet Mercangöz, Calvin Tsay, Ehecatl Antonio Del Rio Chanona
**Affiliation:** Imperial College London (Sargent Centre for PSE)
**arXiv:** 2408.13566v2 [eess.SY], August 2024

---

### 3.1 Motivation

Standard model-free deep RL treats the policy as a black box mapping states to actions, discarding centuries of control-engineering knowledge. PID controllers are ubiquitous in industry — they enforce error correction, integral action, and derivative damping by design. CIRL embeds PID structure directly into the RL policy network, getting the best of both:
- PID's structural stability, disturbance rejection, interpretability
- RL's adaptive gain scheduling and generalisation

---

### 3.2 CIRL Agent Architecture

The CIRL agent consists of:
1. **Deep neural network:** takes state $\mathbf{s}_t$ as input, outputs PID gains $K_{p,t}$, $\tau_{i,t}$, $\tau_{d,t}$ (one set per controlled variable)
2. **PID controller layer:** uses the learned gains and error signal $e_t = x_t^* - x_t$ to compute control action $\mathbf{u}_t$

The state includes $N_t = 2$ timesteps of history: $\mathbf{s}_t = [\mathbf{x}_{t\ldots t-N_t}, \mathbf{x}^*_{t\ldots t-N_t}]$

The **velocity-form PID** (avoids windup from sudden gain changes) for controller $k$:

$$\Delta u_t^{(k)} = K_{p,t}^{(k)} \Delta e_t^{(k)} + \frac{K_{p,t}^{(k)}}{\tau_{i,t}^{(k)}} e_t^{(k)} \Delta t + K_{p,t}^{(k)} \tau_{d,t}^{(k)} \frac{\Delta^2 e_t^{(k)}}{\Delta t} \tag{10}$$

where $\Delta e_t^{(k)} = e_t^{(k)} - e_{t-1}^{(k)}$ and $\Delta^2 e_t^{(k)} = \Delta e_t^{(k)} - 2e_{t-1}^{(k)} + e_{t-2}^{(k)}$.

---

### 3.3 Reward Function

Squared tracking error (analogous to MPC objective):

$$r_t = -(e_t^T Q e_t + \mathbf{u}_t^T R \mathbf{u}_t) \tag{11}$$

where $Q \in \mathbb{R}^{n_x \times n_x}$ penalises setpoint deviation and $R \in \mathbb{R}^{n_u \times n_u}$ penalises control effort.

Objective (cumulative reward):

$$J(\boldsymbol{\theta}) = \mathbb{E}_{\pi_{\boldsymbol{\theta}}}\left[\sum_{t=0}^T -\left(e_t^T Q e_t + \mathbf{u}_t^T R \mathbf{u}_t\right)\right] \tag{12}$$

---

### 3.4 Policy Optimisation

Unlike typical RL which uses gradient-based methods (PPO, SAC), CIRL uses a **hybrid evolutionary strategy: random search + Particle Swarm Optimization (PSO)**:

1. Initialise $N = 30$ random policy parameter vectors
2. PSO update equations for particle $i$:

$$\mathbf{v}_i^{t+1} = w\mathbf{v}_i^t + c_1 r_1 (\mathbf{p}_i^t - \boldsymbol{\theta}_i^t) + c_2 r_2 (\mathbf{g}^t - \boldsymbol{\theta}_i^t) \tag{13}$$
$$\boldsymbol{\theta}_i^{t+1} = \boldsymbol{\theta}_i^t + \mathbf{v}_i^{t+1} \tag{14}$$

where $w = 0.6$ (inertia), $c_1 = c_2 = 1$ (cognitive/social), $\mathbf{g}^t$ is global best.

Evolutionary methods are chosen because: (a) small network (16 neurons × 3 layers) makes gradient-free methods competitive, (b) PID layer is non-differentiable w.r.t. network outputs in velocity form, (c) robustness against local optima.

---

### 3.5 CSTR Case Study

A more complex CSTR than PC-Gym: reaction $A \to B \to C$, two controlled variables ($C_B$, $V$), two PID controllers. State vector includes history: $\mathbf{s}_t = [\mathbf{x}_t, \mathbf{x}_{t-1}, \mathbf{x}_{t-2}, \mathbf{x}^*_t, \mathbf{x}^*_{t-1}, \mathbf{x}^*_{t-2}]$.

**Training:** 9 setpoints spanning operating space. Training time ~10 min on i7-1355U + RTX A500.

**Results (Normal Operation Test):**
| Method | Test Reward |
|---|---|
| Pure RL | -2.08 |
| **CIRL** | **-1.33** |
| Static PID | -1.77 |

**Results (High Operating Point — challenging nonlinear regime):**
| Method | Test Reward |
|---|---|
| CIRL (Initial Training) | -4.04 |
| **CIRL (Extended Training)** | **-2.07** |
| Static PID | -6.81 |

**Disturbance rejection** (unmeasured step change in $C_{A,in}$):
| Method | Test Reward |
|---|---|
| **CIRL** | **-1.38** |
| Pure RL | -1.76 |

CIRL's PID layer intrinsically rejects disturbances through error feedback — even for disturbances never seen during training.

---

### 3.6 Key Insights

1. **Sample efficiency:** CIRL initialises at better reward than pure RL because the PID layer provides a structured prior. Requires 128-neuron pure RL to match 16-neuron CIRL.
2. **Interpretability:** PID gains are directly observable, enabling engineers to inspect and trust the controller.
3. **Generalisation:** When tested outside the training distribution, CIRL adapts its gains while pure RL fails.
4. **Limitation:** At extreme operating points (nonlinear regime where PID gain must change sign), CIRL requires extended training to cover that region.

---

## 4. A Survey and Tutorial of RL Methods in Process Systems Engineering

**Authors:** Maximilian Bloor, Max Mowbray, Ehecatl Antonio del Rio Chanona, Calvin Tsay
**Affiliation:** Imperial College London
**arXiv:** 2510.24272v1 [eess.SY], October 2025

---

### 4.1 Purpose and Scope

This paper serves two goals: (1) an accessible tutorial on RL for PSE researchers, and (2) a structured survey of existing applications. It covers the mathematical foundations, algorithm families, and domain-specific challenges and opportunities in PSE.

---

### 4.2 Problem Formulation

**Stochastic Control Problem (PSE framing):**

$$\min_\pi \mathbb{E}_{\tau \sim p_\pi}\left[\sum_{t=0}^{T-1} \phi_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{x}_{t+1})\right]$$

subject to:
- $X_{t+1} \sim p(\mathbf{x}_{t+1}|\mathbf{x}_t, \mathbf{u}_t)$ — stochastic dynamics
- $\mathbf{u}_t \in \mathbb{U}(\mathbf{x}_t)$ — state-dependent control constraints
- $\mathbb{P}[\mathbf{x}_t \in \mathbb{X}_t] \geq 1 - \delta_t$ — chance constraints on states

The cost $\phi_t$ encodes setpoint deviation, energy use, safety penalties, etc.

**MDP formalism:** $\mathcal{M} = \langle \mathcal{X}, \mathcal{U}, p, \phi, \gamma \rangle$. The Markov assumption holds for ODEs without time delays (which is most PSE problems).

**Bellman optimality equations:**

$$V^*(\mathbf{x}_t) = \min_\mathbf{u} \mathbb{E}_{\mathbf{x}_{t+1} \sim p(\cdot|\mathbf{x}_t, \mathbf{u}_t)}\left[\phi(\mathbf{x}_t, \mathbf{u}_t, \mathbf{x}_{t+1}) + \gamma V^*(\mathbf{x}_{t+1})\right] \tag{7}$$

$$Q^*(\mathbf{x}_t, \mathbf{u}_t) = \mathbb{E}_{\mathbf{x}_{t+1} \sim p(\cdot|\mathbf{x}_t, \mathbf{u}_t)}\left[\phi(\mathbf{x}_t, \mathbf{u}_t, \mathbf{x}_{t+1}) + \gamma \min_{\mathbf{u}_{t+1}} Q^*(\mathbf{x}_{t+1}, \mathbf{u}_{t+1})\right]$$

---

### 4.3 Algorithm Families Covered

#### Model-Free Methods

| Algorithm | Type | Key Property |
|---|---|---|
| Q-learning | Off-policy TD | Tabular; convergence guarantees |
| DQN | Off-policy | Neural Q-function; experience replay |
| DDPG | Off-policy AC | Continuous actions; deterministic policy |
| SAC | Off-policy AC | Maximum entropy; automatic temperature |
| PPO | On-policy AC | Clipped surrogate; stable updates |
| TD3 | Off-policy AC | Twin critics; delayed updates |

**DQN loss:**
$$L(\omega) = \mathbb{E}_{(\mathbf{x}_t, \mathbf{u}_t, \phi_t, \mathbf{x}_{t+1}) \sim \mathcal{D}}\left[\left(\phi_t + \gamma \min_{\mathbf{u}_{t+1}} Q^\pi_{\omega'}(\mathbf{x}_{t+1}, \mathbf{u}_{t+1}) - Q^\pi_\omega(\mathbf{x}_t, \mathbf{u}_t)\right)^2\right] \tag{14}$$

**PPO clipped objective:**
$$L^{CLIP}(\theta_k) = \min\left(\frac{\pi_{\theta_k}}{\pi_{\theta_{k-1}}}\hat{A}_t, \text{clip}\left(\frac{\pi_{\theta_k}}{\pi_{\theta_{k-1}}}, 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right) \tag{9 in PC-Gym}$$

#### Distributional RL

Models the full return distribution $Z^\pi(\mathbf{x}_t, \mathbf{u}_t) = \sum_{k=t}^\infty \gamma^{k-t}\phi_{k+1}$ (a random variable) rather than just its expectation. Enables risk-aware control through CVaR:

$$Q(\mathbf{x}, \mathbf{u}) = \mathbb{E}[Z(\mathbf{x}, \mathbf{u})]$$

Particularly relevant for PSE where tail-risk (rare but catastrophic constraint violations) matters more than average performance.

#### Model-Based RL (MBRL)

Learn a dynamics model $\hat{f}(\mathbf{x}, \mathbf{u})$ from data, use it for planning or policy improvement. Relevant when data efficiency is critical (expensive experiments). Key challenge: model uncertainty and compound errors in multi-step rollouts.

#### Constrained RL

Extends the RL framework to include hard or chance constraints. Key approaches:
- **Lagrangian methods:** $\max_\pi \min_\lambda [J(\pi) - \lambda \cdot c(\pi)]$ where $c(\pi)$ is constraint violation
- **Safe policy optimisation:** CPO, PCPO, FOCOPS
- **Reward shaping:** $r^{\text{safe}} = r - \lambda \sum_i \max(0, g_i(\mathbf{x}))$

#### Goal-Conditioned RL

Learns a single policy conditioned on the setpoint $\mathbf{x}^*$, enabling generalisation across operating points without retraining. Directly relevant for plants with varying setpoint schedules.

---

### 4.4 PSE Application Survey

| Domain | Challenge | Representative Work |
|---|---|---|
| Batch process control | Finite horizon; high stakes end-point | DDPG on polymerisation (Ma et al.) |
| Regulatory control | Continuous; PID replacement | CIRL (Bloor et al.) |
| Setpoint scheduling | Economic objectives; long horizon | RL + Economic MPC (Zhang & Li) |
| Supply chain management | Decentralised; partial info | MARL with GNNs (Kotecha et al.) |

---

### 4.5 Key Open Research Directions (from paper)

1. **Safe RL at deployment:** Training constraints are insufficient for deployment safety
2. **Online RL on real systems:** Exploration risk; shadow mode is one avenue
3. **Sim-to-real transfer:** High-fidelity plant models remain scarce
4. **Goal-conditioned control:** Single policy covering full operating space
5. **Distributional/risk-aware control:** Tail constraint satisfaction
6. **Multi-agent and hierarchical:** Plant-wide control across units

---

## 5. Learning-to-Defer for Sequential Medical Decision-Making (SLTD)

**Authors:** Shalmali Joshi, Sonali Parbhoo, Finale Doshi-Velez
**Affiliation:** Harvard University (SEAS), Imperial College London
**arXiv:** 2109.06312v2 [cs.LG], December 2022

---

### 5.1 Motivation

When an ML policy $\pi_{\text{tar}}$ may perform poorly in certain states (e.g., due to distribution shift, non-stationarity, or data scarcity), it can be beneficial to *defer* to a domain expert $\pi_0$ in those states. Existing learning-to-defer methods are **myopic** — they defer based only on immediate outcomes, ignoring long-term consequences and dynamics.

**SLTD** proposes a fully sequential, model-based approach:
- Defer when $\pi_{\text{tar}}$ is unlikely to improve long-term outcomes vs. expert
- Account for non-stationarity in the environment dynamics
- Defer *pre-emptively* — before the ML policy causes harm
- Decompose uncertainty at deferral to interpret the decision

---

### 5.2 Problem Setup

The environment is a **non-stationary sequence of MDPs** $\mathcal{M}^* = \{\mathcal{M}_t^*\}_t$, with:
- $\mathcal{S}$: state space, $\mathcal{A}$: action space
- Dynamics $p(s'|s,a)$ change over time (non-stationarity)
- Fixed target policy $\pi_{\text{tar}}$ (ML policy)
- Expert behaviour policy $\pi_0$

**Deferral action $\bot$:** Augment action space to $\mathcal{A}_\bot = \mathcal{A} \cup \{\bot\}$. At each step, SLTD decides whether to execute $\pi_{\text{tar}}$ or defer to $\pi_0$.

**Deferral policy definition (Definition 1):** Let $\pi_{t,\text{tar}}$ be the mixture policy where $\pi_{\text{tar}}$ is deployed at time $t$ and $\pi_{\text{mix}}$ otherwise. Deferral incurs a cost $c > 0$ (representing human intervention cost). The deferral policy:

$$g_{\pi_{\text{tar}}}(s, t) \triangleq \mathbf{1}\left[P\left(V^\mathcal{M}_{\pi_{\text{tar}},\text{mix}(t_+)}(s) < V^\mathcal{M}_{\pi_{0(t)},\text{mix}(t_+)}(s) - c\right) > \tau\right]$$

Defer if the *probability* that the expert does better than the ML policy (by at least $c$) exceeds threshold $\tau$.

---

### 5.3 The SLTD Algorithm

```
Algorithm 1: Sequential Learning to Defer
Input: D* (batch data), π₀ (expert), π_tar (ML target policy)
Estimate posterior distributions {M_t} over dynamics for each t
Initialize: g(s,t) = 0 for all s,t

for n in BOOTSTRAPS(D*):
  Sample K MDPs from posterior for each time t
  for t in {T, T-1, ..., 1}:
    for s in S:
      Compute V^M_{π_tar,mix(t+)} and V^M_{π₀,mix(t+)} for each M
      Update g̃(s,t) ≈ fraction of MDPs where target < expert - c

return g(s,t) = 1[g̃(s,t) > τ] for all s,t
```

**Bayesian dynamics estimation:**
- Discrete states: Dirichlet prior over transitions, conjugate posterior
- Continuous states: Normal-Gamma prior, Gaussian posterior

$$p(\boldsymbol{\theta}|\mathcal{D}^*) \propto p(\mathcal{D}^*|\boldsymbol{\theta}) p(\boldsymbol{\theta})$$

---

### 5.4 Uncertainty Decomposition at Deferral

At deferral time $t_d$, the total outcome variance decomposes as:

$$\underbrace{\text{Var}(r_T|s_{t_d}, \mathcal{D})}_{\text{Total Uncertainty}} = \underbrace{\mathbb{E}_{\mu_{t_d}}[\text{Var}(r_T|\mu_{t_d}, s_{t_d}, \mathcal{D})]}_{\text{Irreducible/Aleatoric}} + \underbrace{\text{Var}_{\mu_{t_d}}(\mathbb{E}[r_T|\mu_{t_d}, s_{t_d}, \mathcal{D}])}_{\text{Epistemic/Modeling}} \tag{9}$$

- **Aleatoric:** inherent stochasticity of environment; cannot be reduced by data
- **Epistemic:** model uncertainty; can be reduced by collecting more data
- High epistemic uncertainty → flag for targeted data collection

---

### 5.5 Experiments

**Datasets:**
1. **Synthetic:** 8-state MDP with known non-stationary dynamics (dynamics flip at $t = 5$). SLTD correctly identifies pre-emptive deferral regions.
2. **Diabetes simulator (T1DMS):** Type-1 Diabetes, 13 glucose states, 25 drug combinations, 24-hour episodes with aging patient dynamics. SLTD: value 36.9 vs. target policy 13.2.
3. **HIV real-world data:** 32,960 patients, drug resistance evolution as non-stationarity. SLTD achieves best trade-off between deferral frequency and long-term outcome.

**Key result (Table 1):** SLTD achieves value 8.029 (Synthetic), 36.931 (Diabetes) vs. $\pi_{\text{tar}}$ alone at −1.80 and 13.202 respectively, while deferring to the expert only ~51% and ~40% of steps.

---

### 5.6 Connection to This FYP

| SLTD concept | Shadow Mode adaptation |
|---|---|
| Defer to expert when $\pi_{\text{tar}}$ likely worse | Switch to baseline when Q-value lower |
| Long-term outcome modeling | Q-function estimates cumulative reward |
| Pre-emptive deferral | Randomised training start |
| Non-stationary dynamics | Process disturbances and setpoint changes |
| Uncertainty quantification | Confidence-based switching threshold |

The key synergy: SLTD's *pre-emptive* deferral logic based on long-term value estimates can sharpen the switching criterion in shadow mode from binary Q-value comparison to a probabilistic, uncertainty-aware decision.

---

## Cross-Paper Synthesis: The FYP Research Landscape

### Core Thesis
The common thread across all five papers is the challenge of making RL decisions reliable when:
1. The environment is safety-critical (chemical processes, medical decisions)
2. A suboptimal but reasonable baseline exists (PID controller, clinical protocol)
3. Direct RL exploration is risky or expensive (real systems, patients)
4. Hard constraints must be satisfied (temperature bounds, drug toxicity)

### How the Papers Connect

```
PC-Gym (environments + NMPC oracle)
    ↓
    [CSTR, Multistage, Crystallisation, Four-Tank]
    ↓
CIRL (embed PID structure → safer exploration, interpretable gains)
    ↓
Shadow Mode (train on real system with PID/CIRL as baseline)
    ↓
SLTD (probabilistic, pre-emptive deferral; uncertainty-aware switching)
    ↓
FYP Novel Contribution: Shadow Mode RL on PC-Gym environments
    with CIRL baseline + SLTD-inspired switching
```

### Open Questions This FYP Addresses
1. Does shadow mode generalise from the simple 2D task to nonlinear chemical process control?
2. Is CIRL a better baseline for shadow mode than a hand-tuned PID (due to its interpretable gains)?
3. Can SLTD-style uncertainty quantification improve the switching criterion in shadow mode?
4. What are the constraint satisfaction properties of the combined system?
