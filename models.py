"""
models.py
=========
All RL model definitions for this project. Every agent shares one custom core
(Actor / Critic / replay buffer); the variants differ only in how decide_action
behaves, which makes them a clean ablation set:

Shadow models  (used by train_shadow.py)
----------------------------------------
  ddpg  — ShadowDDPG  (single critic, Q-value or agent-decision switching)
  td3   — ShadowTD3   (twin critics, delayed updates, target policy smoothing)

Standard / no-shadow models  (used by train.py)
-----------------------------------------------
  ddpg  — PureDDPG    (ShadowDDPG with switching disabled — agent always acts)
  td3   — PureTD3     (ShadowTD3  with switching disabled)

These are the fair "standard DDPG/TD3" baselines for shadow mode: identical
core, hyperparameters, and PID-assisted exploration, differing ONLY in the
switching decision. (PPO is not supported — shadow switching needs a
deterministic off-policy actor-critic.)

Stable-Baselines3 backend  (labelled "SB3 DDPG" / "Shadow SB3 DDPG")
-------------------------------------------------------------------
An alternative, separately-tuned off-the-shelf implementation kept alongside the
custom core (create_sb3_agent / create_shadow_sb3_agent). Shadow mode is added to
SB3 by overriding _sample_action so the *executed* (Q-value-switched) action is
what gets stored in the replay buffer. Lets us test whether shadow switching
helps a strong, well-tuned learner. Only q-value switching is supported on SB3.

Device helpers
--------------
  resolve_device(force_cpu) -> torch.device
  device_label(device)      -> str
"""

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DDPG as SB3DDPG
from stable_baselines3 import TD3 as SB3TD3
from stable_baselines3.common.noise import NormalActionNoise


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def resolve_device(force_cpu: bool = False) -> torch.device:
    """
    Return the compute device to use for training/inference.
    Priority: CUDA GPU → CPU.  Pass force_cpu=True (or --cpu) to skip the GPU.
    """
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    """Human-readable device description for log headers."""
    if device.type == "cuda":
        return f"GPU — {torch.cuda.get_device_name(0)}"
    return "CPU"


# ---------------------------------------------------------------------------
# Available model names
# ---------------------------------------------------------------------------

SHADOW_MODELS = ["ddpg", "td3"]
PURE_MODELS   = ["ddpg", "td3"]   # standard (no-shadow) custom agents
SB3_MODELS    = ["ddpg", "td3"]   # Stable-Baselines3 backend (normal + shadow)


# ---------------------------------------------------------------------------
# Neural network building blocks  (shared by ShadowDDPG and ShadowTD3)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Deterministic policy network π^a.

    Q-value mode  : outputs action vector of shape (action_dim,).
    Agent-decision: outputs (action, decision_prob) where decision_prob ∈ [0,1]
                    controls how often the agent takes over from the baseline.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256,
                 agent_decision: bool = False):
        super().__init__()
        self.agent_decision = agent_decision
        out_dim = action_dim + 1 if agent_decision else action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, state: torch.Tensor):
        out = self.net(state)
        if self.agent_decision:
            return torch.tanh(out[..., :-1]), torch.sigmoid(out[..., -1:])
        return torch.tanh(out)


class Critic(nn.Module):
    """Single Q-network Q(s, a) — used by ShadowDDPG."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class _CriticTwin(nn.Module):
    """
    Twin Q-networks Q1, Q2 — used by ShadowTD3.
    min(Q1, Q2) reduces overestimation and gives more conservative switching.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
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

    def push(self, state, action_exec, reward, next_state, done,
             action_agent=None, action_baseline=None):
        self.buf.append((
            np.array(state,           dtype=np.float32),
            np.array(action_exec,     dtype=np.float32),
            float(reward),
            np.array(next_state,      dtype=np.float32),
            float(done),
            np.array(action_agent,    dtype=np.float32) if action_agent    is not None else None,
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


# ---------------------------------------------------------------------------
# ShadowDDPG — single-critic shadow agent  (Gassert & Althoff, 2024)
# ---------------------------------------------------------------------------

class ShadowDDPG:
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
        mode:             str   = "qvalue",
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
        device:           str   = "cpu",
    ):
        self.mode             = mode
        self.gamma            = gamma
        self.tau              = tau
        self.batch_size       = batch_size
        self.eta_agent        = eta_agent
        self.lambda_reg       = lambda_reg
        self.noise_std        = noise_std
        self.warmup_steps     = warmup_steps
        self.max_t_train_frac = max_t_train_frac
        self.action_dim       = action_dim
        self.device           = torch.device(device)

        agent_decision = (mode == "agent")

        self.actor        = Actor(state_dim, action_dim, hidden, agent_decision).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden, agent_decision).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = Critic(state_dim, action_dim, hidden).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer               = ReplayBuffer(buffer_size)
        self.total_steps          = 0
        self.agent_takeover_count = 0
        self.baseline_count       = 0

    def _get_agent_action(self, state: np.ndarray):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.mode == "agent":
                action, decision = self.actor(s)
                return action.cpu().numpy()[0], float(decision.cpu().numpy()[0][0])
            action = self.actor(s)
            return action.cpu().numpy()[0], None

    def decide_action(self, obs: np.ndarray, baseline_action: np.ndarray,
                      training: bool = True):
        noise = (
            np.random.normal(0, self.noise_std, self.action_dim).astype(np.float32)
            if training else np.zeros(self.action_dim, dtype=np.float32)
        )
        action_agent, decision_prob = self._get_agent_action(obs)
        action_noisy = np.clip(action_agent + noise, -1.0, 1.0)

        if self.mode == "qvalue":
            s    = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            a_ag = torch.FloatTensor(action_noisy).unsqueeze(0).to(self.device)
            a_bl = torch.FloatTensor(baseline_action).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_agent    = self.critic(s, a_ag).item()
                q_baseline = self.critic(s, a_bl).item()
            used_agent = q_agent > q_baseline
        else:
            used_agent = decision_prob > self.eta_agent

        if used_agent:
            self.agent_takeover_count += 1
            return action_noisy, True, action_noisy
        else:
            self.baseline_count += 1
            return baseline_action, False, action_noisy

    def store(self, state, action_exec, reward, next_state, done,
              action_agent, action_baseline):
        self.buffer.push(state, action_exec, reward, next_state, done,
                         action_agent, action_baseline)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a_exec, r, ns, d, _, a_baseline = self.buffer.sample(self.batch_size)
        s      = s.to(self.device)
        a_exec = a_exec.to(self.device)
        r      = r.to(self.device)
        ns     = ns.to(self.device)
        d      = d.to(self.device)

        with torch.no_grad():
            a_next   = (self.actor_target(ns)[0] if self.mode == "agent"
                        else self.actor_target(ns))
            q_target = r + self.gamma * (1 - d) * self.critic_target(ns, a_next)

        critic_loss = F.mse_loss(self.critic(s, a_exec), q_target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        a_cur      = (self.actor(s)[0] if self.mode == "agent" else self.actor(s))
        actor_loss = -self.critic(s, a_cur).mean()
        if self.lambda_reg > 0 and a_baseline is not None:
            actor_loss = actor_loss + self.lambda_reg * F.l1_loss(
                a_cur, a_baseline.to(self.device)
            )
        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "actor":         self.actor.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]))
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt.get("critic_target", ckpt["critic"]))


# ---------------------------------------------------------------------------
# ShadowTD3 — twin-critic shadow agent
# ---------------------------------------------------------------------------

class ShadowTD3:
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
        mode:               str   = "qvalue",
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
        max_t_train_frac:   float = 0.5,
        device:             str   = "cpu",
    ):
        self.mode              = mode
        self.gamma             = gamma
        self.tau               = tau
        self.batch_size        = batch_size
        self.eta_agent         = eta_agent
        self.lambda_reg        = lambda_reg
        self.noise_std         = noise_std
        self.warmup_steps      = warmup_steps
        self.policy_delay      = policy_delay
        self.target_noise_std  = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.max_t_train_frac  = max_t_train_frac
        self.action_dim        = action_dim
        self.device            = torch.device(device)

        agent_decision = (mode == "agent")

        self.actor        = Actor(state_dim, action_dim, hidden, agent_decision).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden, agent_decision).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = _CriticTwin(state_dim, action_dim, hidden).to(self.device)
        self.critic_target = _CriticTwin(state_dim, action_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer               = ReplayBuffer(buffer_size)
        self.total_steps          = 0
        self._update_count        = 0
        self.agent_takeover_count = 0
        self.baseline_count       = 0

    def _get_agent_action(self, state: np.ndarray):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.mode == "agent":
                action, decision = self.actor(s)
                return action.cpu().numpy()[0], float(decision.cpu().numpy()[0][0])
            action = self.actor(s)
            return action.cpu().numpy()[0], None

    def decide_action(self, obs: np.ndarray, baseline_action: np.ndarray,
                      training: bool = True):
        noise = (
            np.random.normal(0, self.noise_std, self.action_dim).astype(np.float32)
            if training else np.zeros(self.action_dim, dtype=np.float32)
        )
        action_agent, decision_prob = self._get_agent_action(obs)
        action_noisy = np.clip(action_agent + noise, -1.0, 1.0)

        if self.mode == "qvalue":
            s    = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            a_ag = torch.FloatTensor(action_noisy).unsqueeze(0).to(self.device)
            a_bl = torch.FloatTensor(baseline_action).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_agent    = self.critic.q_min(s, a_ag).item()
                q_baseline = self.critic.q_min(s, a_bl).item()
            used_agent = q_agent > q_baseline
        else:
            used_agent = decision_prob > self.eta_agent

        if used_agent:
            self.agent_takeover_count += 1
            return action_noisy, True, action_noisy
        else:
            self.baseline_count += 1
            return baseline_action, False, action_noisy

    def store(self, state, action_exec, reward, next_state, done,
              action_agent, action_baseline):
        self.buffer.push(state, action_exec, reward, next_state, done,
                         action_agent, action_baseline)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a_exec, r, ns, d, _, a_baseline = self.buffer.sample(self.batch_size)
        s      = s.to(self.device)
        a_exec = a_exec.to(self.device)
        r      = r.to(self.device)
        ns     = ns.to(self.device)
        d      = d.to(self.device)

        with torch.no_grad():
            a_next = (self.actor_target(ns)[0] if self.mode == "agent"
                      else self.actor_target(ns))
            smoothing = torch.clamp(
                torch.randn_like(a_next) * self.target_noise_std,
                -self.target_noise_clip, self.target_noise_clip,
            )
            a_next   = torch.clamp(a_next + smoothing, -1.0, 1.0)
            q_target = r + self.gamma * (1 - d) * self.critic_target.q_min(ns, a_next)

        q1, q2      = self.critic(s, a_exec)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        self._update_count += 1
        if self._update_count % self.policy_delay != 0:
            return

        a_cur      = (self.actor(s)[0] if self.mode == "agent" else self.actor(s))
        actor_loss = -self.critic.q1_only(s, a_cur).mean()
        if self.lambda_reg > 0 and a_baseline is not None:
            actor_loss = actor_loss + self.lambda_reg * F.l1_loss(
                a_cur, a_baseline.to(self.device)
            )
        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "actor":            self.actor.state_dict(),
            "actor_target":     self.actor_target.state_dict(),
            "critic_q1":        self.critic.q1.state_dict(),
            "critic_q2":        self.critic.q2.state_dict(),
            "critic_target_q1": self.critic_target.q1.state_dict(),
            "critic_target_q2": self.critic_target.q2.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]))
        self.critic.q1.load_state_dict(ckpt["critic_q1"])
        self.critic.q2.load_state_dict(ckpt["critic_q2"])
        self.critic_target.q1.load_state_dict(ckpt.get("critic_target_q1", ckpt["critic_q1"]))
        self.critic_target.q2.load_state_dict(ckpt.get("critic_target_q2", ckpt["critic_q2"]))


# ---------------------------------------------------------------------------
# Pure (no-shadow) custom agents — the fair ablation baseline
# ---------------------------------------------------------------------------
# These share the ShadowDDPG / ShadowTD3 core exactly but disable shadow-mode
# switching: the agent's own action is always executed. They are the correct
# "standard DDPG/TD3" baseline for measuring the *effect of shadow switching*,
# since the learner is held identical (unlike the SB3 agents in train.py, which
# are a different, separately-tuned implementation).

class PureDDPG(ShadowDDPG):
    """Standard DDPG: always executes the agent's action, no shadow switching."""

    def decide_action(self, obs, baseline_action, training: bool = True):
        noise = (
            np.random.normal(0, self.noise_std, self.action_dim).astype(np.float32)
            if training else np.zeros(self.action_dim, dtype=np.float32)
        )
        action_agent, _ = self._get_agent_action(obs)
        action_noisy    = np.clip(action_agent + noise, -1.0, 1.0)
        self.agent_takeover_count += 1
        return action_noisy, True, action_noisy


class PureTD3(ShadowTD3):
    """Standard TD3: always executes the agent's action, no shadow switching."""

    def decide_action(self, obs, baseline_action, training: bool = True):
        noise = (
            np.random.normal(0, self.noise_std, self.action_dim).astype(np.float32)
            if training else np.zeros(self.action_dim, dtype=np.float32)
        )
        action_agent, _ = self._get_agent_action(obs)
        action_noisy    = np.clip(action_agent + noise, -1.0, 1.0)
        self.agent_takeover_count += 1
        return action_noisy, True, action_noisy


def create_pure_agent(
    model_type:  str,
    state_dim:   int,
    action_dim:  int,
    device:      torch.device | None = None,
):
    """Return an untrained no-shadow custom agent (PureDDPG or PureTD3) — the
    ablation baseline that shares the shadow agents' core."""
    dev_str = str(device) if device is not None else "cpu"
    kwargs  = dict(state_dim=state_dim, action_dim=action_dim, mode="qvalue",
                   device=dev_str)
    return PureTD3(**kwargs) if model_type == "td3" else PureDDPG(**kwargs)


# ---------------------------------------------------------------------------
# Shadow agent factory  — used by train_shadow.py
# ---------------------------------------------------------------------------

def create_shadow_agent(
    model_type:  str,
    state_dim:   int,
    action_dim:  int,
    mode:        str             = "qvalue",
    lambda_reg:  float           = 0.0,
    eta_agent:   float           = 0.5,
    device:      torch.device | None = None,
):
    """Return an untrained shadow-mode agent (ShadowDDPG or ShadowTD3)."""
    dev_str = str(device) if device is not None else "cpu"
    kwargs  = dict(state_dim=state_dim, action_dim=action_dim, mode=mode,
                   lambda_reg=lambda_reg, eta_agent=eta_agent, device=dev_str)
    return ShadowTD3(**kwargs) if model_type == "td3" else ShadowDDPG(**kwargs)


# ---------------------------------------------------------------------------
# Stable-Baselines3 backend  — "SB3 DDPG" / "Shadow SB3 DDPG"
# ---------------------------------------------------------------------------
# A separately-tuned off-the-shelf learner kept alongside the custom core. Shadow
# mode is added by overriding _sample_action so the *executed* (switched) action
# is stored in the replay buffer, exactly like the custom ShadowDDPG.
#
# Assumes a normalised action space ([-1, 1]) — all project scenarios use
# normalise_a=True, so SB3's scaled action space equals the env action space and
# the PID baselines (which output in [-1, 1]) drop in without rescaling.

def _sb3_kwargs(env, seed, device):
    dev_str   = str(device) if device is not None else "auto"
    n_actions = env.action_space.shape[0]
    noise     = NormalActionNoise(mean=np.zeros(n_actions),
                                  sigma=0.1 * np.ones(n_actions))
    return dict(policy="MlpPolicy", env=env, seed=seed, verbose=0, device=dev_str,
                learning_rate=1e-3, batch_size=256, gamma=0.99, tau=0.005,
                action_noise=noise)


class _ShadowSB3Mixin:
    """
    Adds shadow-mode Q-value switching to an SB3 off-policy actor-critic.

    During data collection the executed action is the baseline's unless the
    critic prefers the agent's (Q(s, a_agent) > Q(s, a_baseline)). The executed
    action is what is stored in the replay buffer, so the critic learns the value
    of the deployed mixed policy. Switching activates only after the warmup
    (learning_starts), when the critic is usable.
    """

    baseline = None

    def set_baseline(self, baseline):
        self.baseline     = baseline
        self._n_decisions = 0      # number of switching decisions (for takeover %)
        self._n_agent     = 0      # how many chose the agent's action
        return self

    def reset_baseline(self):
        if self.baseline is not None:
            self.baseline.reset()

    def _q1(self, obs_2d, act_2d):
        """min-batch Q1(obs, action) as a 1-D numpy array; actions are scaled [-1,1]."""
        obs_t = self.policy.obs_to_tensor(np.asarray(obs_2d, dtype=np.float32))[0]
        act_t = torch.as_tensor(np.asarray(act_2d, dtype=np.float32), device=self.device)
        with torch.no_grad():
            return self.critic.q1_forward(obs_t, act_t).cpu().numpy().reshape(-1)

    def _switch(self, obs_2d, agent_scaled_2d):
        """Return (executed_scaled_2d, used_agent_bool_array) for a batch of obs."""
        n      = agent_scaled_2d.shape[0]
        a_base = np.array([np.asarray(self.baseline.predict(obs_2d[i])[0], dtype=np.float32)
                           for i in range(n)])
        use_agent = self._q1(obs_2d, agent_scaled_2d) > self._q1(obs_2d, a_base)
        executed  = np.where(use_agent[:, None], agent_scaled_2d, a_base).astype(np.float32)
        return executed, use_agent

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)
        if self.baseline is None or self.num_timesteps < learning_starts:
            return action, buffer_action
        executed, use_agent = self._switch(self._last_obs, buffer_action)
        self._n_decisions += int(use_agent.size)
        self._n_agent     += int(use_agent.sum())
        return self.policy.unscale_action(executed), executed

    def executed_action(self, obs, baseline):
        """Deterministic switched action for a single obs (used at evaluation)."""
        a_agent, _ = self.predict(obs, deterministic=True)
        a_base     = np.asarray(baseline.predict(obs)[0], dtype=np.float32)
        q_agent    = self._q1(obs[None], a_agent[None])[0]
        q_base     = self._q1(obs[None], a_base[None])[0]
        return a_agent if q_agent > q_base else a_base


class ShadowSB3DDPG(_ShadowSB3Mixin, SB3DDPG):
    """SB3 DDPG with shadow-mode Q-value switching."""


class ShadowSB3TD3(_ShadowSB3Mixin, SB3TD3):
    """SB3 TD3 with shadow-mode Q-value switching."""


def create_sb3_agent(model_type: str, env, seed: int = 42,
                     device: torch.device | None = None):
    """Return an untrained plain SB3 agent (DDPG or TD3) — the 'SB3 DDPG' baseline."""
    cls = SB3TD3 if model_type == "td3" else SB3DDPG
    return cls(**_sb3_kwargs(env, seed, device))


def create_shadow_sb3_agent(model_type: str, env, baseline, seed: int = 42,
                           device: torch.device | None = None):
    """Return an untrained shadow-mode SB3 agent with the PID baseline attached."""
    cls = ShadowSB3TD3 if model_type == "td3" else ShadowSB3DDPG
    return cls(**_sb3_kwargs(env, seed, device)).set_baseline(baseline)
