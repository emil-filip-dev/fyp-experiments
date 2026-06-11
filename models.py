"""
models.py
=========
RL agents and controllers for the OFFLINE process-control pipeline.

The project (see project_proposal.md / dissertation_plan.md) trains agents
**offline** on a static dataset of expert (+ perturbed) trajectories, then
introduces them alongside the expert through a staged
``shadow -> autonomous`` deployment. The same agent core supports an
optional *conservative offline-to-online* fine-tuning phase (learning only from
expert-guarded transitions) and a *naive online* contrast baseline (the unsafe
foil the offline method is meant to avoid).

Contents
--------
  Building blocks : Actor (deterministic policy), Critic, CriticTwin, ReplayBuffer
  Offline agents  : TD3Agent (twin critics, TD3+BC) and DDPGAgent (single critic,
                    DDPG+BC). Both expose the SAME interface (act / q / q_gap /
                    update / store) and support offline, offline-to-online, and
                    naive-online use. get_agent(algorithm) returns the class.
  Deployment      : DeploymentStage (shadow/autonomous) and
                    ShadowController, which wraps (agent, expert) and decides
                    per stage whether the agent has *earned* control — i.e. whether
                    Q(s, a_agent) - Q(s, a_expert) clears a (relaxing) margin.
  Expert / oracle : NMPCController — do-mpc + IPOPT NMPC on the env's exact
                    dynamics. It is BOTH the MPC expert (setpoint-tracking
                    scenarios) and the optimality ceiling.

TD3+BC follows Fujimoto & Gu (2021): the deterministic-policy gradient is mixed
with a behaviour-cloning term toward the dataset action, with the policy-gradient
side normalised by the mean |Q| so a single coefficient (`bc_alpha`) balances the
two regardless of reward scale. Setting bc_alpha=0 recovers plain TD3/DDPG (the
naive online contrast).
"""
import abc
import copy
import enum
import io
import os
from collections import deque
from contextlib import contextmanager, redirect_stdout
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Categorical types
# ---------------------------------------------------------------------------

class AgentType(enum.StrEnum):
    """Off-policy learner family used for offline pretraining."""
    DDPG = "ddpg"
    TD3 = "td3"


class DeploymentStage(enum.StrEnum):
    """
    The staged introduction of the offline-pretrained agent alongside the expert
    (project_proposal.md). Ordered from least to most agent authority:

      shadow      — the agent runs alongside the expert and TAKES OVER wherever it
                    has *earned* it: Q(s, a_agent) - Q(s, a_expert) > margin; the
                    expert handles the remaining steps. This is the headline
                    deployment mode (earned, selective takeover). A smaller margin
                    grants more authority.
      autonomous  — the agent controls alone (the expert remains only as an
                    optional hard safety fallback).
    """
    SHADOW = "shadow"
    AUTONOMOUS = "autonomous"


def get_agent(algorithm: str | AgentType):
    """Return the agent class for an algorithm name ('ddpg' / 'td3')."""
    match algorithm:
        case AgentType.DDPG.value | AgentType.DDPG: return DDPGAgent
        case AgentType.TD3.value | AgentType.TD3: return TD3Agent
        case _: raise ValueError(f"Unknown algorithm {algorithm!r} (use 'ddpg' or 'td3').")


# ---------------------------------------------------------------------------
# Neural network building blocks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Deterministic policy network π(s) -> action ∈ [-1, 1]^action_dim (tanh)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state))


class Critic(nn.Module):
    """Single Q-network Q(s, a) — used by DDPGAgent."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class CriticTwin(nn.Module):
    """Twin Q-networks Q1, Q2 — used by TD3Agent. min(Q1,Q2) curbs overestimation."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = state_dim + action_dim

        def block():
            return nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = block()
        self.q2 = block()

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=-1))

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class ReplayBuffer:
    """
    Uniform replay buffer of (s, a, r, s', done) transitions. Used as both the
    STATIC offline dataset (filled once via add_many, never appended to) and the
    growing buffer for offline-to-online fine-tuning / naive-online training.
    """

    def __init__(self, capacity: int = 1_000_000):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buf.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def add_many(self, states, actions, rewards, next_states, dones) -> None:
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.push(s, a, r, ns, d)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        s, a, r, ns, d = zip(*(self.buf[i] for i in idx))
        return (
            torch.from_numpy(np.stack(s)),
            torch.from_numpy(np.stack(a)),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1),
            torch.from_numpy(np.stack(ns)),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# Offline agents (TD3+BC / DDPG+BC) — shared base
# ---------------------------------------------------------------------------

class _BaseAgent(abc.ABC):
    """
    Shared logic for the offline actor-critic agents. The behaviour-cloning
    coefficient `bc_alpha` (Fujimoto & Gu 2021) anchors the policy to the dataset
    actions during offline pretraining and to expert-guarded actions during
    conservative fine-tuning; bc_alpha=0 recovers a plain online learner (the
    naive contrast baseline).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        hidden: int = 256,
        batch_size: int = 256,
        bc_alpha: float = 2.5,
        expl_noise: float = 0.1,
        buffer_size: int = 1_000_000,
        device: torch.device = torch.device("cpu"),
        **_ignored,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.hidden = hidden
        self.batch_size = batch_size
        self.bc_alpha = bc_alpha
        self.expl_noise = expl_noise
        self.device = device

        self.buffer = ReplayBuffer(buffer_size)
        self.total_updates = 0     # gradient steps taken (offline + online)
        self.total_env_steps = 0   # env interactions (online phases only)
        self._build()

    # --- subclass hooks ---------------------------------------------------
    @abc.abstractmethod
    def _build(self) -> None: ...

    @property
    @abc.abstractmethod
    def algorithm(self) -> AgentType: ...

    @property
    @abc.abstractmethod
    def label(self) -> str: ...

    @abc.abstractmethod
    def q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Scalar switching-critic estimate used for the Q-gap takeover decision."""

    @abc.abstractmethod
    def update(self):
        """One gradient step from the buffer. Returns a dict of scalars or None."""

    @abc.abstractmethod
    def _save_dict(self) -> dict: ...

    @abc.abstractmethod
    def _load_state_dicts(self, sd: dict) -> None: ...

    # --- shared inference -------------------------------------------------
    def _to_t(self, x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device).unsqueeze(0)

    def act(self, obs: np.ndarray, explore: bool = False) -> np.ndarray:
        """Deterministic policy action; optional clipped-Gaussian exploration noise."""
        with torch.no_grad():
            a = self.actor(self._to_t(obs)).cpu().numpy()[0]
        if explore:
            a = a + np.random.normal(0.0, self.expl_noise, self.action_dim).astype(np.float32)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def q_value(self, obs: np.ndarray, action: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.q(self._to_t(obs), self._to_t(action)).item())

    def q_gap(self, obs: np.ndarray, expert_action: np.ndarray) -> float:
        """
        Q(s, a_agent) - Q(s, a_expert) for the deterministic agent action. > margin
        means the critic rates the agent's action above the expert's; this is the
        'earned takeover' signal recorded per step (claim C3).
        """
        with torch.no_grad():
            s = self._to_t(obs)
            a_agent = self.actor(s)
            a_exp = self._to_t(expert_action)
            return float(self.q(s, a_agent).item() - self.q(s, a_exp).item())

    # --- buffer / persistence --------------------------------------------
    def store(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def load_dataset(self, buffer: ReplayBuffer) -> None:
        """Adopt a pre-filled static dataset buffer for offline pretraining."""
        self.buffer = buffer

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._save_dict(), path)

    @classmethod
    def load(cls, ckpt: dict, device: torch.device = torch.device("cpu")) -> Self:
        # Pass only hyperparameters to the constructor — not the persisted payload
        # keys (type / state_dicts / internal), which are restored explicitly below.
        skip = {"type", "state_dicts", "internal"}
        hparams = {k: v for k, v in ckpt.items() if k not in skip}
        model = cls(**hparams, device=device)
        model._load_state_dicts(ckpt["state_dicts"])
        for k, v in ckpt.get("internal", {}).items():
            setattr(model, k, v)
        return model

    def _soft_update(self, net, target) -> None:
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)


class TD3Agent(_BaseAgent):
    """
    TD3 with optional behaviour cloning (TD3+BC). Twin critics, delayed policy
    updates, and target-policy smoothing; the actor objective adds a BC term
    toward the dataset action, normalised by mean |Q| (Fujimoto & Gu 2021).
    """

    def __init__(self, *args, policy_delay: int = 2, target_noise: float = 0.2,
                 noise_clip: float = 0.5, **kwargs):
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        super().__init__(*args, **kwargs)

    def _build(self) -> None:
        sd, ad, h = self.state_dim, self.action_dim, self.hidden
        self.actor = Actor(sd, ad, h).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = CriticTwin(sd, ad, h).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    @property
    def algorithm(self) -> AgentType:
        return AgentType.TD3

    @property
    def label(self) -> str:
        return "TD3+BC" if self.bc_alpha > 0 else "TD3"

    def q(self, state, action):
        return self.critic.q1_only(state, action)

    def update(self):
        if len(self.buffer) == 0:
            return None   # sample() draws with replacement, so a non-empty buffer
                          # smaller than batch_size is fine (offline datasets may be tiny)
        s, a, r, ns, d = (t.to(self.device) for t in self.buffer.sample(self.batch_size))

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_target(ns) + noise).clamp(-1.0, 1.0)
            q_target = r + self.gamma * (1 - d) * self.critic_target.q_min(ns, a2)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        self.total_updates += 1
        out = {"critic_loss": float(critic_loss.item())}
        if self.total_updates % self.policy_delay == 0:
            pi = self.actor(s)
            q_pi = self.critic.q1_only(s, pi)
            if self.bc_alpha > 0:
                lam = self.bc_alpha / (q_pi.abs().mean().detach() + 1e-8)
                actor_loss = -lam * q_pi.mean() + F.mse_loss(pi, a)
            else:
                actor_loss = -q_pi.mean()
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)
            out["actor_loss"] = float(actor_loss.item())
        return out

    def _save_dict(self) -> dict:
        return {
            "type": AgentType.TD3,
            "state_dim": self.state_dim, "action_dim": self.action_dim,
            "gamma": self.gamma, "tau": self.tau, "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic, "hidden": self.hidden, "batch_size": self.batch_size,
            "bc_alpha": self.bc_alpha, "expl_noise": self.expl_noise,
            "policy_delay": self.policy_delay, "target_noise": self.target_noise,
            "noise_clip": self.noise_clip,
            "state_dicts": {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
            },
            "internal": {"total_updates": self.total_updates,
                         "total_env_steps": self.total_env_steps},
        }

    def _load_state_dicts(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.actor_target.load_state_dict(sd["actor_target"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.opt_actor.load_state_dict(sd["opt_actor"])
        self.opt_critic.load_state_dict(sd["opt_critic"])


class DDPGAgent(_BaseAgent):
    """DDPG with optional behaviour cloning (DDPG+BC) — single critic, no delay."""

    def _build(self) -> None:
        sd, ad, h = self.state_dim, self.action_dim, self.hidden
        self.actor = Actor(sd, ad, h).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(sd, ad, h).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    @property
    def algorithm(self) -> AgentType:
        return AgentType.DDPG

    @property
    def label(self) -> str:
        return "DDPG+BC" if self.bc_alpha > 0 else "DDPG"

    def q(self, state, action):
        return self.critic(state, action)

    def update(self):
        if len(self.buffer) == 0:
            return None   # sample() draws with replacement, so a non-empty buffer
                          # smaller than batch_size is fine (offline datasets may be tiny)
        s, a, r, ns, d = (t.to(self.device) for t in self.buffer.sample(self.batch_size))

        with torch.no_grad():
            q_target = r + self.gamma * (1 - d) * self.critic_target(ns, self.actor_target(ns))
        critic_loss = F.mse_loss(self.critic(s, a), q_target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        pi = self.actor(s)
        q_pi = self.critic(s, pi)
        if self.bc_alpha > 0:
            lam = self.bc_alpha / (q_pi.abs().mean().detach() + 1e-8)
            actor_loss = -lam * q_pi.mean() + F.mse_loss(pi, a)
        else:
            actor_loss = -q_pi.mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)
        self.total_updates += 1
        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def _save_dict(self) -> dict:
        return {
            "type": AgentType.DDPG,
            "state_dim": self.state_dim, "action_dim": self.action_dim,
            "gamma": self.gamma, "tau": self.tau, "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic, "hidden": self.hidden, "batch_size": self.batch_size,
            "bc_alpha": self.bc_alpha, "expl_noise": self.expl_noise,
            "state_dicts": {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
            },
            "internal": {"total_updates": self.total_updates,
                         "total_env_steps": self.total_env_steps},
        }

    def _load_state_dicts(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.actor_target.load_state_dict(sd["actor_target"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.opt_actor.load_state_dict(sd["opt_actor"])
        self.opt_critic.load_state_dict(sd["opt_critic"])


# ---------------------------------------------------------------------------
# Staged deployment: agent + expert with earned, relaxing takeover
# ---------------------------------------------------------------------------

class ShadowController:
    """
    Wraps an offline-pretrained `agent` and an `expert` controller and decides, at
    each step, which action is applied — implementing the proposal's staged
    introduction (shadow -> autonomous).

    decide(obs, stage, margin) returns (a_exec, used_agent, a_agent, a_expert):
      SHADOW     : a_exec = a_agent iff q_gap(obs, a_expert) > margin, else a_expert.
                   The agent takes over wherever its critic rates its action above
                   the expert's. A larger margin demands the agent be *clearly*
                   better (conservative); a smaller margin grants more authority.
      AUTONOMOUS : a_exec = a_agent. If `safety_fallback` is set, the expert is
                   re-asserted whenever q_gap < fallback_margin (a hard guard).

    The controller is used both as the EVALUATION protocol (frozen agent) and as
    the data-collection policy for conservative offline-to-online fine-tuning
    (the executed transitions are stored and learned from).
    """

    def __init__(self, agent, expert, *, safety_fallback: bool = True,
                 fallback_margin: float = 0.0):
        self.agent = agent
        self.expert = expert
        self.safety_fallback = safety_fallback
        self.fallback_margin = fallback_margin

    def reset(self) -> None:
        if hasattr(self.expert, "reset"):
            self.expert.reset()

    def decide(self, obs, stage: DeploymentStage, margin: float = 0.0,
               explore: bool = False):
        a_expert, _ = self.expert.predict(obs)
        a_expert = np.asarray(a_expert, dtype=np.float32)
        a_agent = self.agent.act(obs, explore=explore)

        if stage is DeploymentStage.SHADOW:
            used = self.agent.q_gap(obs, a_expert) > margin
            return (a_agent if used else a_expert), used, a_agent, a_expert

        # AUTONOMOUS
        if self.safety_fallback and self.agent.q_gap(obs, a_expert) < self.fallback_margin:
            return a_expert, False, a_agent, a_expert
        return a_agent, True, a_agent, a_expert


# ---------------------------------------------------------------------------
# NMPC Controller — do-mpc receding-horizon controller.
# Serves as BOTH the MPC expert (setpoint-tracking scenarios) and the optimality
# ceiling. Verbatim from the previous oracle implementation (unchanged behaviour).
# ---------------------------------------------------------------------------

@contextmanager
def _suppressed_fds(enabled: bool = True):
    """Redirect C-level stdout(1)/stderr(2) to os.devnull for the duration. This is
    what silences native CasADi/IPOPT solver warnings (e.g. 'nlp_g failed: NaN
    detected for output g') — they are emitted from C++ and bypass a Python-level
    redirect_stdout. No-op when `enabled` is False (pass quiet=False to see them)."""
    if not enabled:
        yield
        return
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved1, saved2 = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1); os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved1, 1); os.dup2(saved2, 2)
        os.close(devnull); os.close(saved1); os.close(saved2)


class NMPCController:
    """
    Nonlinear MPC on a PC-Gym scenario, built on PC-Gym's own do-mpc `oracle`
    (CasADi + IPOPT) so the prediction model is *exactly* the environment's
    dynamics. Exposes .predict(obs) / .reset() like the PID baselines and runs in
    true receding-horizon fashion against the real (noisy) environment.

    Setpoint-tracking scenarios only — raises NotImplementedError on scenarios
    with disturbances or delta-u actions (e.g. crystallization), where the PID
    baseline is used as the expert instead.
    """

    def __init__(self, cfg: dict, horizon: int = 20, quiet: bool = True):
        from pcgym import make_env
        from pcgym.oracle import oracle

        # quiet=True suppresses the native CasADi/IPOPT solver chatter (incl. the
        # benign 'NaN detected' line-search warnings). Pass quiet=False to debug.
        self._quiet = quiet
        env_params = copy.deepcopy(cfg["env_params"])
        with _suppressed_fds(self._quiet):
            self._oracle = oracle(make_env, env_params, MPC_params={"N": horizon})

        self._nx = self._oracle.env.Nx_oracle
        self._a_low = np.asarray(env_params["a_space"]["low"], dtype=np.float64)
        self._a_high = np.asarray(env_params["a_space"]["high"], dtype=np.float64)
        self._o_low = np.asarray(env_params["o_space"]["low"], dtype=np.float64)
        self._o_high = np.asarray(env_params["o_space"]["high"], dtype=np.float64)
        self._normalise_o = env_params.get("normalise_o", True)
        self._x0 = np.asarray(cfg["env_params"]["x0"][:self._nx], dtype=np.float64)

        with redirect_stdout(io.StringIO()), _suppressed_fds(self._quiet):
            self._mpc, _ = self._oracle.setup_mpc()

        if self._oracle.has_disturbances or self._oracle.use_delta_u:
            raise NotImplementedError(
                "NMPCController supports setpoint-tracking scenarios only "
                "(no disturbances or delta-u)."
            )
        self._mpc.p_fun = self._make_p_fun()
        self.reset()

    def _make_p_fun(self):
        sp_dict = self._oracle.env_params["SP"]
        sp_arrays = [np.asarray(v, dtype=float) for v in sp_dict.values()]
        dt = self._oracle.env.dt
        get_p_template = self._mpc.get_p_template

        def p_fun(t_now):
            t = float(np.asarray(t_now).flatten()[0])
            k = int(round(t / dt))
            p_template = get_p_template(1)
            sp_vals = [arr[min(k, len(arr) - 1)] for arr in sp_arrays]
            p_template["_p", 0, "SP"] = np.array(sp_vals).reshape(-1, 1)
            return p_template

        return p_fun

    def reset(self):
        self._mpc.reset_history()
        self._mpc.x0 = self._x0
        with redirect_stdout(io.StringIO()), _suppressed_fds(self._quiet):
            self._mpc.set_initial_guess()

    def predict(self, obs, deterministic: bool = True):
        obs = np.asarray(obs, dtype=np.float64)
        if self._normalise_o:
            phys = (obs + 1.0) / 2.0 * (self._o_high - self._o_low) + self._o_low
        else:
            phys = obs
        x0 = phys[:self._nx].reshape(-1, 1)
        with redirect_stdout(io.StringIO()), _suppressed_fds(self._quiet):
            u = np.asarray(self._mpc.make_step(x0)).flatten()
        u_norm = 2.0 * (u - self._a_low) / (self._a_high - self._a_low) - 1.0
        return np.clip(u_norm, -1.0, 1.0).astype(np.float32), None
