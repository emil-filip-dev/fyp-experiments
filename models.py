"""
models.py
=========
All RL model definitions for this project. Every agent shares one custom core
(Actor / Critic / replay buffer); the variants differ only in how decide_action
behaves, which makes them a clean ablation set:

Standard / non-shadow models
-----------------------------------------------
  ddpg - DDPG
  td3  - TD3
  ppo  - PPO

Shadow models
----------------------------------------
  ddpg - ShadowDDPG  (single critic, Q-value or agent-decision switching)
  td3  - ShadowTD3   (twin critics, delayed updates, target policy smoothing)

These are the fair "standard DDPG/TD3" baselines for shadow mode: identical
core, hyperparameters, and PID-assisted exploration, differing ONLY in the
switching decision. (PPO is not supported - shadow switching needs a
deterministic off-policy actor-critic.)
"""
import abc
import copy
import enum
import io
import os
from abc import abstractmethod
from collections import deque
from contextlib import redirect_stdout
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardModels(enum.StrEnum):
    DDPG = "ddpg"
    TD3 = "td3"
    PPO = "ppo"

class ShadowModels(enum.StrEnum):
    DDPG = "ddpg"
    TD3 = "td3"

class SwitchingMode(enum.StrEnum):
    Q_VALUE = "qvalue"
    AGENT = "agent"

class TD3SwitchCritic(enum.StrEnum):
    """Which twin-critic estimate ShadowTD3 uses for Q-value switching."""
    Q_MIN = "qmin"   # min(Q1, Q2) — conservative
    Q1    = "q1"     # Q1 only — consistent with the actor's objective

def get_standard_model(model_name: str | StandardModels):
    match model_name:
        case StandardModels.DDPG.value | StandardModels.DDPG: return DDPG
        case StandardModels.TD3.value | StandardModels.TD3: return TD3
        case _: raise ValueError(f"Standard model '{model_name}' does not exist.")

def get_shadow_model(model_name: str | ShadowModels):
    match model_name:
        case ShadowModels.DDPG.value | ShadowModels.DDPG: return ShadowDDPG
        case ShadowModels.TD3.value | ShadowModels.TD3: return ShadowTD3
        case _: raise ValueError(f"Shadow model '{model_name}' does not exist.")


# ---------------------------------------------------------------------------
# Neural network building blocks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Deterministic policy network π^a.

    Q-value mode  : outputs action vector of shape (action_dim,).
    Agent-decision: outputs (action, decision_prob) where decision_prob ∈ [0,1]
                    controls how often the agent takes over from the baseline.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden: int = 256,
            switching_mode: SwitchingMode = SwitchingMode.Q_VALUE
    ):
        super().__init__()
        self.switching_mode = switching_mode
        out_dim = action_dim + 1 if switching_mode is SwitchingMode.AGENT else action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, state: torch.Tensor):
        out = self.net(state)
        if self.switching_mode is SwitchingMode.AGENT:
            return torch.tanh(out[..., :-1]), torch.sigmoid(out[..., -1:])
        else:
            return torch.tanh(out)


class Critic(nn.Module):
    """Single Q-network Q(s, a) — used by ShadowDDPG."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class CriticTwin(nn.Module):
    """
    Twin Q-networks Q1, Q2 - used by ShadowTD3.
    min(Q1, Q2) reduces overestimation and gives more conservative switching.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    """Uniform experience replay buffer for off-policy learning."""

    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(
            self,
            state,
            action_exec,
            reward,
            next_state,
            done,
            action_agent=None,
            action_baseline=None
    ):
        self.buf.append((
            np.array(state,           dtype=np.float32),
            np.array(action_exec,     dtype=np.float32), float(reward),
            np.array(next_state,      dtype=np.float32), float(done),
            np.array(action_agent,    dtype=np.float32) if action_agent is not None else None,
            np.array(action_baseline, dtype=np.float32) if action_baseline is not None else None,
        ))

    def sample(self, batch_size: int):
        idx   = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        s, a, r, ns, d, aa, ab = zip(*batch)
        return (
            torch.FloatTensor(np.stack(s)),
            torch.FloatTensor(np.stack(a)),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(np.stack(ns)),
            torch.FloatTensor(d).unsqueeze(1),
            torch.FloatTensor(np.stack(aa)) if aa[0] is not None else None,
            torch.FloatTensor(np.stack(ab)) if ab[0] is not None else None,
        )

    def __len__(self) -> int:
        return len(self.buf)


class _ShadowModel(abc.ABC):

    def __init__(
            self,
            state_dim: int = 3,
            action_dim: int = 1,
            switching_mode: SwitchingMode = SwitchingMode.Q_VALUE,
            gamma: float = 0.99,
            tau: float = 0.005,
            buffer_size: int = 100_000,
            batch_size: int = 256,
            eta_agent: float = 0.5,
            lambda_reg: float = 0.0,
            noise_std: float = 0.1,
            warmup_steps: int = 5_000,
            max_t_train_frac: float = 0.5,
            device: torch.device = torch.device("cpu"),
            **kwargs
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mode = switching_mode
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.eta_agent = eta_agent
        self.lambda_reg = lambda_reg
        self.noise_std = noise_std
        self.warmup_steps = warmup_steps
        self.max_t_train_frac = max_t_train_frac
        self.device = device

        self.buffer = ReplayBuffer(buffer_size)
        self.total_steps = 0
        self.agent_takeover_count = 0
        self.baseline_count = 0

    @property
    @abstractmethod
    def label(self) -> str:
        ...

    @property
    @abstractmethod
    def _actor(self):
        ...

    @property
    @abstractmethod
    def _critic(self):
        ...

    def _get_agent_action(self, state: np.ndarray):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            match self.mode:
                case SwitchingMode.AGENT:
                    action, decision = self._actor(s)
                    return action.cpu().numpy()[0], float(decision.cpu().numpy()[0][0])
                case _:
                    action = self._actor(s)
                    return action.cpu().numpy()[0], None

    def _actor_action(self, actor, x: torch.Tensor) -> torch.Tensor:
        """
        Actor output as a single action tensor for the critic. In agent-decision
        mode the policy emits (a^a, a^decision); the critic operates on the
        augmented action, so concatenate the two heads.
        """
        out = actor(x)
        if self.mode is SwitchingMode.AGENT:
            return torch.cat(out, dim=-1)
        return out

    def _baseline_record_action(self, baseline_action: np.ndarray) -> np.ndarray:
        """
        Agent action stored in the replay buffer when the baseline is executed by
        force (warmup). Agent-decision mode stores the augmented action
        (a^b, decision=0) so the critic's input width stays constant.
        """
        a = np.asarray(baseline_action, dtype=np.float32)
        if self.mode is SwitchingMode.AGENT:
            return np.concatenate([a, np.zeros(1, dtype=np.float32)])
        return a.copy()

    def decide_action(
            self,
            obs: np.ndarray,
            baseline_action: np.ndarray,
            training: bool = True,
            force_baseline: bool = False,
    ):
        # Warmup / randomised start: execute the baseline, but store a shape-
        # consistent agent action so agent-decision buffers stack cleanly.
        if force_baseline:
            return baseline_action, False, self._baseline_record_action(baseline_action)

        noise = (
            np.random.normal(0, self.noise_std, self.action_dim).astype(np.float32)
            if training else np.zeros(self.action_dim, dtype=np.float32)
        )
        action_det, decision_prob = self._get_agent_action(obs)
        action_noisy = np.clip(action_det + noise, -1.0, 1.0)

        match self.mode:
            case SwitchingMode.Q_VALUE:
                # Eq. (6): compare on the NOISY action that is actually executed
                # (a^c = a^a + exploration noise), so the switch evaluates exactly
                # the action we are about to apply to the system.
                use_agent = self._decide_qvalue(
                    torch.FloatTensor(obs).unsqueeze(0).to(self.device),
                    torch.FloatTensor(action_noisy).unsqueeze(0).to(self.device),
                    torch.FloatTensor(baseline_action).unsqueeze(0).to(self.device)
                )
                a_agent = action_noisy
            case _:
                # Eq. (4): explore the decision too, then store the augmented
                # behaviour action (a^a, a^decision) so the critic can learn the
                # value of the decision and train the decision head.
                decision = (
                    float(np.clip(decision_prob + np.random.normal(0, self.noise_std), 0.0, 1.0))
                    if training else decision_prob
                )
                use_agent = self._decide_agent(decision)
                a_agent = np.concatenate(
                    [action_noisy, np.array([decision], dtype=np.float32)]
                )

        if use_agent:
            self.agent_takeover_count += 1
            return action_noisy, True, a_agent
        else:
            self.baseline_count += 1
            return baseline_action, False, a_agent

    @abstractmethod
    def _decide_qvalue(self, state, action_agent, action_baseline) -> bool:
        ...

    @abstractmethod
    def _decide_agent(self, decision_prob) -> bool:
        ...

    def store(
            self, state, action_exec, reward, next_state, done, action_agent, action_baseline
    ):
        self.buffer.push(
            state, action_exec, reward, next_state, done, action_agent, action_baseline
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._save_dict(), path)

    @abstractmethod
    def _save_dict(self) -> dict:
        ...

    @classmethod
    def load(cls, ckpt: dict, device: torch.device = torch.device("cpu")) -> Self:
        model = cls(**ckpt, device=device)
        sd = ckpt["state_dicts"]
        model._load_state_dicts(sd)
        intern = ckpt["internal"]
        for k, v in intern.items():
            model.__setattr__(k, v)
        return model

    @abstractmethod
    def _load_state_dicts(self, sd: dict):
        ...


class ShadowDDPG(_ShadowModel):
    """
    DDPG agent trained in shadow mode.

    At each step the agent proposes an action alongside the baseline controller.
    A switching criterion decides which action is actually applied:

      qvalue  — execute agent action if Q(s, a_agent) > Q(s, a_baseline)
      agent   — agent outputs a control-authority probability;
                execute agent action if that probability > eta_agent

    The replay buffer stores the *actually executed* action so the critic
    learns value estimates consistent with the deployed mixed policy.
    """

    def __init__(
        self,
        state_dim:        int   = 3,
        action_dim:       int   = 1,
        switching_mode:   SwitchingMode = SwitchingMode.Q_VALUE,
        gamma:            float = 0.99,
        tau:              float = 0.005,
        lr_actor:         float = 1e-4,
        lr_critic:        float = 3e-4,
        hidden:           int   = 256,
        buffer_size:      int   = 100_000,
        batch_size:       int   = 256,
        eta_agent:        float = 0.5,
        lambda_reg:       float = 0.0,
        noise_std:        float = 0.1,
        warmup_steps:     int   = 5_000,
        max_t_train_frac: float = 0.5,
        device:           torch.device = torch.device("cpu"),
        **kwargs
    ):
        super().__init__(
            state_dim, action_dim, switching_mode, gamma, tau, buffer_size, batch_size, eta_agent, lambda_reg,
            noise_std, warmup_steps, max_t_train_frac, device, **kwargs
        )

        self.hidden = hidden
        # Agent-decision mode augments the action with a^decision, so the critic
        # operates on (a^a, a^decision) -> action_dim + 1 inputs.
        q_action_dim = action_dim + 1 if switching_mode is SwitchingMode.AGENT else action_dim
        self.actor        = Actor(state_dim, action_dim, hidden, switching_mode).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden, switching_mode).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = Critic(state_dim, q_action_dim, hidden).to(self.device)
        self.critic_target = Critic(state_dim, q_action_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    @property
    def _actor(self):
        return self.actor

    @property
    def _critic(self):
        return self.critic

    @property
    def label(self) -> str:
        match self.mode:
            case SwitchingMode.AGENT: mode_label = " (Agent)"
            case SwitchingMode.Q_VALUE: mode_label = " (Q Value)"
            case _: mode_label = ""
        return f"Shadow DDPG{mode_label}"

    def _decide_qvalue(self, state, action_agent, action_baseline) -> bool:
        # Eq. (6): take over when the single online critic rates the agent's
        # action above the baseline's.
        with torch.no_grad():
            q_agent    = self.critic(state, action_agent).item()
            q_baseline = self.critic(state, action_baseline).item()
        return q_agent > q_baseline

    def _decide_agent(self, decision_prob) -> bool:
        # Eq. (4): take over when the agent's control-authority probability
        # exceeds the threshold.
        return decision_prob > self.eta_agent

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a_exec, r, ns, d, a_agent, a_baseline = self.buffer.sample(self.batch_size)
        s      = s.to(self.device)
        a_exec = a_exec.to(self.device)
        r      = r.to(self.device)
        ns     = ns.to(self.device)
        d      = d.to(self.device)
        if a_agent is not None:
            a_agent = a_agent.to(self.device)

        # In agent-decision mode the critic learns Q over the augmented behaviour
        # action (a^a, a^decision); in q-value mode it learns Q over the executed
        # action a^c (which may be the baseline).
        a_critic = a_agent if self.mode is SwitchingMode.AGENT else a_exec

        # Eq. (5): reward penalty r^reg = -lambda * ||a^a - a^b|| regularising the
        # agent toward the baseline. Agent-decision switching only; it shapes the
        # reward (hence the Q-function), not the actor loss directly.
        if self.mode is SwitchingMode.AGENT and self.lambda_reg > 0 and a_agent is not None:
            r = r - self.lambda_reg * torch.norm(
                a_agent[:, :self.action_dim] - a_baseline.to(self.device), dim=-1, keepdim=True
            )

        with torch.no_grad():
            a_next   = self._actor_action(self.actor_target, ns)
            q_target = r + self.gamma * (1 - d) * self.critic_target(ns, a_next)

        critic_loss = F.mse_loss(self.critic(s, a_critic), q_target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        a_cur      = self._actor_action(self.actor, s)
        actor_loss = -self.critic(s, a_cur).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _save_dict(self) -> dict:
        return {
            "type": ShadowModels.DDPG,
            "switching_mode": self.mode,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "eta_agent": self.eta_agent,
            "lambda_reg": self.lambda_reg,
            "noise_std": self.noise_std,
            "warmup_steps": self.warmup_steps,
            "max_t_train_frac": self.max_t_train_frac,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden": self.hidden,
            "state_dicts": {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
            },
            "internal": {
                "buffer": self.buffer,
                "total_steps": self.total_steps,
                "agent_takeover_count": self.agent_takeover_count,
                "baseline_count": self.baseline_count
            }
        }

    def _load_state_dicts(self, sd: dict):
        self.actor.load_state_dict(sd["actor"])
        self.actor_target.load_state_dict(sd.get("actor_target", sd["actor"]))
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd.get("critic_target", sd["critic"]))
        self.opt_actor.load_state_dict(sd["opt_actor"])
        self.opt_critic.load_state_dict(sd["opt_critic"])


class ShadowTD3(_ShadowModel):
    """
    TD3 agent trained in shadow mode.

    Three improvements over ShadowDDPG:
      1. Twin critics — min(Q1, Q2) for target and switching, reducing
         Q-value overestimation and making takeover decisions more conservative.
      2. Delayed policy updates — actor updated every policy_delay critic steps.
      3. Target policy smoothing — clipped noise on target actions prevents
         the critic from exploiting sharp Q-function peaks.

    Interface identical to ShadowDDPG so it works with run_episode() unchanged.
    """

    def __init__(
        self,
        state_dim:          int   = 3,
        action_dim:         int   = 1,
        switching_mode:     SwitchingMode = SwitchingMode.Q_VALUE,
        gamma:              float = 0.99,
        tau:                float = 0.005,
        lr_actor:           float = 1e-4,
        lr_critic:          float = 3e-4,
        hidden:             int   = 256,
        buffer_size:        int   = 100_000,
        batch_size:         int   = 256,
        eta_agent:          float = 0.5,
        lambda_reg:         float = 0.0,
        noise_std:          float = 0.1,
        warmup_steps:       int   = 5_000,
        policy_delay:       int   = 2,
        target_noise_std:   float = 0.2,
        target_noise_clip:  float = 0.5,
        switch_critic:      TD3SwitchCritic = TD3SwitchCritic.Q1,
        max_t_train_frac:   float = 0.5,
        device:             torch.device = torch.device("cpu"),
        **kwargs
    ):
        super().__init__(
            state_dim, action_dim, switching_mode, gamma, tau, buffer_size, batch_size, eta_agent, lambda_reg,
            noise_std, warmup_steps, max_t_train_frac, device, **kwargs
        )

        self.policy_delay      = policy_delay
        self.target_noise_std  = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.switch_critic     = TD3SwitchCritic(switch_critic)

        self.hidden = hidden
        # Agent-decision mode augments the action with a^decision, so the critic
        # operates on (a^a, a^decision) -> action_dim + 1 inputs.
        q_action_dim = action_dim + 1 if switching_mode is SwitchingMode.AGENT else action_dim
        self.actor        = Actor(state_dim, action_dim, hidden, switching_mode).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden, switching_mode).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = CriticTwin(state_dim, q_action_dim, hidden).to(self.device)
        self.critic_target = CriticTwin(state_dim, q_action_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self._update_count = 0

    @property
    def _actor(self):
        return self.actor

    @property
    def _critic(self):
        return self.critic

    @property
    def label(self) -> str:
        match self.mode:
            case SwitchingMode.AGENT: mode_label = " (Agent)"
            case SwitchingMode.Q_VALUE: mode_label = f" (Q Value, {self.switch_critic.value})"
            case _: mode_label = ""
        return f"Shadow TD3{mode_label}"

    def _decide_qvalue(self, state, action_agent, action_baseline) -> bool:
        # Eq. (6): take over when the chosen twin-critic estimate rates the
        # agent's action above the baseline's. q1 is consistent with the actor's
        # objective; qmin is the conservative min(Q1, Q2).
        match self.switch_critic:
            case TD3SwitchCritic.Q_MIN: q = self.critic.q_min
            case _:                     q = self.critic.q1_only
        with torch.no_grad():
            q_agent    = q(state, action_agent).item()
            q_baseline = q(state, action_baseline).item()
        return q_agent > q_baseline

    def _decide_agent(self, decision_prob) -> bool:
        # Eq. (4): take over when the agent's control-authority probability
        # exceeds the threshold.
        return decision_prob > self.eta_agent

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a_exec, r, ns, d, a_agent, a_baseline = self.buffer.sample(self.batch_size)
        s      = s.to(self.device)
        a_exec = a_exec.to(self.device)
        r      = r.to(self.device)
        ns     = ns.to(self.device)
        d      = d.to(self.device)
        if a_agent is not None:
            a_agent = a_agent.to(self.device)

        # In agent-decision mode the critic learns Q over the augmented behaviour
        # action (a^a, a^decision); in q-value mode it learns Q over the executed
        # action a^c (which may be the baseline).
        a_critic = a_agent if self.mode is SwitchingMode.AGENT else a_exec

        # Eq. (5): reward penalty r^reg = -lambda * ||a^a - a^b|| regularising the
        # agent toward the baseline. Agent-decision switching only; it shapes the
        # reward (hence the Q-function), not the actor loss directly.
        if self.mode is SwitchingMode.AGENT and self.lambda_reg > 0 and a_agent is not None:
            r = r - self.lambda_reg * torch.norm(
                a_agent[:, :self.action_dim] - a_baseline.to(self.device), dim=-1, keepdim=True
            )

        with torch.no_grad():
            a_next    = self._actor_action(self.actor_target, ns)
            smoothing = torch.clamp(
                torch.randn_like(a_next) * self.target_noise_std,
                -self.target_noise_clip, self.target_noise_clip,
            )
            a_next = a_next + smoothing
            if self.mode is SwitchingMode.AGENT:
                # action dims live in [-1, 1] (tanh); the decision dim lives in [0, 1].
                a_next = torch.cat([
                    a_next[:, :self.action_dim].clamp(-1.0, 1.0),
                    a_next[:, self.action_dim:].clamp(0.0, 1.0),
                ], dim=-1)
            else:
                a_next = a_next.clamp(-1.0, 1.0)
            q_target = r + self.gamma * (1 - d) * self.critic_target.q_min(ns, a_next)

        q1, q2      = self.critic(s, a_critic)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        self._update_count += 1
        if self._update_count % self.policy_delay != 0:
            return

        a_cur      = self._actor_action(self.actor, s)
        actor_loss = -self.critic.q1_only(s, a_cur).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _save_dict(self):
        return {
            "type": ShadowModels.TD3,
            "switching_mode": self.mode,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "eta_agent": self.eta_agent,
            "lambda_reg": self.lambda_reg,
            "noise_std": self.noise_std,
            "warmup_steps": self.warmup_steps,
            "policy_delay": self.policy_delay,
            "target_noise_std": self.target_noise_std,
            "target_noise_clip": self.target_noise_clip,
            "switch_critic": self.switch_critic,
            "max_t_train_frac": self.max_t_train_frac,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden": self.hidden,
            "state_dicts": {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_q1": self.critic.q1.state_dict(),
                "critic_q2": self.critic.q2.state_dict(),
                "critic_target_q1": self.critic_target.q1.state_dict(),
                "critic_target_q2": self.critic_target.q2.state_dict(),
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
            },
            "internal": {
                "buffer": self.buffer,
                "total_steps": self.total_steps,
                "_update_count": self._update_count,
                "agent_takeover_count": self.agent_takeover_count,
                "baseline_count": self.baseline_count
            }
        }

    def _load_state_dicts(self, sd: dict):
        self.actor.load_state_dict(sd["actor"])
        self.actor_target.load_state_dict(sd.get("actor_target", sd["actor"]))
        self.critic.q1.load_state_dict(sd["critic_q1"])
        self.critic.q2.load_state_dict(sd["critic_q2"])
        self.critic_target.q1.load_state_dict(sd.get("critic_target_q1", sd["critic_q1"]))
        self.critic_target.q2.load_state_dict(sd.get("critic_target_q2", sd["critic_q2"]))
        self.opt_actor.load_state_dict(sd["opt_actor"])
        self.opt_critic.load_state_dict(sd["opt_critic"])


# ---------------------------------------------------------------------------
# Pure (no-shadow) custom agents — the fair ablation baseline
# ---------------------------------------------------------------------------
# These share the ShadowDDPG / ShadowTD3 core exactly but disable shadow-mode
# switching: the agent's own action is always executed. They are the correct
# "standard DDPG/TD3" baseline for measuring the *effect of shadow switching*,
# since the learner is held identical.

class DDPG(ShadowDDPG):
    """Standard DDPG: always executes the agent's action, no shadow switching."""

    def __init__(
            self,
            state_dim: int = 3,
            action_dim: int = 1,
            gamma: float = 0.99,
            tau: float = 0.005,
            lr_actor: float = 1e-4,
            lr_critic: float = 3e-4,
            hidden: int = 256,
            buffer_size: int = 100_000,
            batch_size: int = 256,
            noise_std: float = 0.1,
            warmup_steps: int = 5_000,
            max_t_train_frac: float = 0.5,
            device: torch.device = torch.device("cpu"),
            **kwargs
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            tau=tau,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            hidden=hidden,
            buffer_size=buffer_size,
            batch_size=batch_size,
            noise_std=noise_std,
            warmup_steps=warmup_steps,
            max_t_train_frac=max_t_train_frac,
            device=device,
            **kwargs
        )

    @property
    def label(self) -> str:
        return f"DDPG"

    def decide_action(self, obs, baseline_action, training: bool = True, force_baseline: bool = False):
        if force_baseline:
            return baseline_action, False, np.asarray(baseline_action, dtype=np.float32).copy()
        noise = (
            np.random.normal(0, self.noise_std, self.action_dim).astype(np.float32)
            if training else np.zeros(self.action_dim, dtype=np.float32)
        )
        action_agent, _ = self._get_agent_action(obs)
        action_noisy    = np.clip(action_agent + noise, -1.0, 1.0)
        self.agent_takeover_count += 1
        return action_noisy, True, action_noisy

    def _save_dict(self) -> dict:
        d = super()._save_dict()
        d["type"] = StandardModels.DDPG
        return d


class TD3(ShadowTD3):
    """Standard TD3: always executes the agent's action, no shadow switching."""

    def __init__(
        self,
        state_dim:          int   = 3,
        action_dim:         int   = 1,
        gamma:              float = 0.99,
        tau:                float = 0.005,
        lr_actor:           float = 1e-4,
        lr_critic:          float = 3e-4,
        hidden:             int   = 256,
        buffer_size:        int   = 100_000,
        batch_size:         int   = 256,
        noise_std:          float = 0.1,
        warmup_steps:       int   = 5_000,
        policy_delay:       int   = 2,
        target_noise_std:   float = 0.2,
        target_noise_clip:  float = 0.5,
        max_t_train_frac:   float = 0.5,
        device:             torch.device = torch.device("cpu"),
        **kwargs
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            tau=tau,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            hidden=hidden,
            buffer_size=buffer_size,
            batch_size=batch_size,
            noise_std=noise_std,
            warmup_steps=warmup_steps,
            policy_delay=policy_delay,
            target_noise_std=target_noise_std,
            target_noise_clip=target_noise_clip,
            max_t_train_frac=max_t_train_frac,
            device=device,
            **kwargs
        )

    @property
    def label(self) -> str:
        return f"TD3"

    def decide_action(self, obs, baseline_action, training: bool = True, force_baseline: bool = False):
        if force_baseline:
            return baseline_action, False, np.asarray(baseline_action, dtype=np.float32).copy()
        noise = (
            np.random.normal(0, self.noise_std, self.action_dim).astype(np.float32)
            if training else np.zeros(self.action_dim, dtype=np.float32)
        )
        action_agent, _ = self._get_agent_action(obs)
        action_noisy    = np.clip(action_agent + noise, -1.0, 1.0)
        self.agent_takeover_count += 1
        return action_noisy, True, action_noisy

    def _save_dict(self):
        d = super()._save_dict()
        d["type"] = StandardModels.TD3
        return d


# ---------------------------------------------------------------------------
# NMPC Controller oracle - receding-horizon do-mpc controller (best-achievable reference)
# ---------------------------------------------------------------------------

class NMPCController:
    """
    Nonlinear MPC oracle for a PC-Gym scenario.

    Built on PC-Gym's own do-mpc `oracle` (CasADi + IPOPT) so the prediction
    model is *exactly* the environment's dynamics — the "best achievable"
    reference. Exposes .predict(obs) / .reset() like the PID baselines and runs
    in true receding-horizon fashion against the real (noisy) environment.
    """

    def __init__(self, cfg: dict, horizon: int = 20):
        # Imported lazily so --no-oracle runs never pay the do-mpc import cost.
        from pcgym import make_env
        from pcgym.oracle import oracle

        # oracle mutates env_params in place, so hand it a private copy.
        env_params = copy.deepcopy(cfg["env_params"])
        self._oracle = oracle(make_env, env_params, MPC_params={"N": horizon})

        self._nx          = self._oracle.env.Nx_oracle
        self._a_low       = np.asarray(env_params["a_space"]["low"],  dtype=np.float64)
        self._a_high      = np.asarray(env_params["a_space"]["high"], dtype=np.float64)
        self._o_low       = np.asarray(env_params["o_space"]["low"],  dtype=np.float64)
        self._o_high      = np.asarray(env_params["o_space"]["high"], dtype=np.float64)
        self._normalise_o = env_params.get("normalise_o", True)
        self._x0          = np.asarray(cfg["env_params"]["x0"][:self._nx], dtype=np.float64)

        # Build the NLP once; reset() re-initialises it cheaply each episode.
        with redirect_stdout(io.StringIO()):
            self._mpc, _ = self._oracle.setup_mpc()

        # PC-Gym's shipped p_fun does `int(t_now/dt - 1)`, which crashes on modern
        # numpy (do-mpc passes t_now as a 1-D array) and lags the setpoint by one
        # step. make_step looks up self.p_fun dynamically, so we override it with a
        # scalar-safe version that tracks the *current* setpoint. Only setpoint
        # scheduling is supported (no disturbances / delta-u).
        if self._oracle.has_disturbances or self._oracle.use_delta_u:
            raise NotImplementedError(
                "NMPCController supports setpoint-tracking scenarios only "
                "(no disturbances or delta-u)."
            )
        self._mpc.p_fun = self._make_p_fun()
        self.reset()

    def _make_p_fun(self):
        """Build a scalar-safe do-mpc parameter function for setpoint tracking."""
        sp_dict        = self._oracle.env_params["SP"]
        sp_arrays      = [np.asarray(v, dtype=float) for v in sp_dict.values()]
        dt             = self._oracle.env.dt
        get_p_template = self._mpc.get_p_template

        def p_fun(t_now):
            t = float(np.asarray(t_now).flatten()[0])
            k = int(round(t / dt))                       # current step index
            p_template = get_p_template(1)
            sp_vals = [arr[min(k, len(arr) - 1)] for arr in sp_arrays]
            p_template["_p", 0, "SP"] = np.array(sp_vals).reshape(-1, 1)
            return p_template

        return p_fun

    def reset(self):
        """Restart for a new episode: reset the MPC clock and initial guess."""
        self._mpc.reset_history()            # resets internal t0 -> 0 (SP indexing)
        self._mpc.x0 = self._x0
        with redirect_stdout(io.StringIO()):
            self._mpc.set_initial_guess()

    def predict(self, obs, deterministic: bool = True):
        obs = np.asarray(obs, dtype=np.float64)
        if self._normalise_o:
            phys = (obs + 1.0) / 2.0 * (self._o_high - self._o_low) + self._o_low
        else:
            phys = obs
        x0 = phys[:self._nx].reshape(-1, 1)        # physical model state (no setpoints)

        with redirect_stdout(io.StringIO()):
            u = np.asarray(self._mpc.make_step(x0)).flatten()   # physical input(s)

        # Re-normalise to the env's [-1, 1] action space.
        u_norm = 2.0 * (u - self._a_low) / (self._a_high - self._a_low) - 1.0
        return np.clip(u_norm, -1.0, 1.0).astype(np.float32), None
