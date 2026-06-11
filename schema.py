"""
schema.py
=========
The shared, typed vocabulary of the OFFLINE process-control pipeline. Run
metadata lives in STRUCTURED, TYPED objects (StrEnums + dataclasses) serialised
to JSON — never encoded into, nor parsed back out of, directory/file slugs.

Sources of truth for a run's identity:
  - Condition    : a learner configuration (algorithm + training mode + BC strength
                   + expert), declared in experiments.py grids; produces both its
                   pretraining kwargs and a RunSpec — so the two never drift.
  - RunSpec      : a Condition bound to (scenario, seed, budgets); written as
                   run.json beside each checkpoint.
  - ModelSpec    : the typed input to deploy.run_rollouts (checkpoint + RunSpec).
  - MethodRecord : one structured entry per method (and deployment stage) in a
                   rollout manifest, consumed by the analysis utility — which reads
                   these fields and never inspects a filename.

The pipeline (project_proposal.md):
  offline pretrain  ->  shadow (earned takeover alongside the expert)  ->  autonomous
"""

import enum
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from models import AgentType, DeploymentStage


# ---------------------------------------------------------------------------
# Categorical types (StrEnum — never bare strings at the boundaries)
# ---------------------------------------------------------------------------

class Scenario(enum.StrEnum):
    """The PC-Gym scenarios in the SCENARIOS registry."""
    CSTR = "cstr"
    FOUR_TANK = "four_tank"
    MULTISTAGE_EXTRACTION = "multistage_extraction"
    CRYSTALLIZATION = "crystallization"


# Algorithm is an alias of the agent family defined in models.py.
Algorithm = AgentType


class TrainingMode(enum.StrEnum):
    """
    How an agent is produced:
      offline           — pretrain purely on the static expert(+perturbed) dataset
                          (no env interaction). The headline method.
      offline_to_online — offline pretrain, then conservative fine-tuning from
                          expert-GUARDED online transitions across the staged
                          (shadow -> autonomous) deployment.
      online_contrast   — naive online RL trained on the live plant from scratch
                          (the unsafe foil the offline method is meant to avoid).
      online_shadow     — online RL from scratch but expert-GUARDED every step
                          (G&A-style earned-takeover gate, no offline pretrain).
                          Shares online_contrast's from-scratch init; the ONLY
                          difference is the guard. The fair shadow-vs-standard
                          learning ablation (stability / speed / early trajectory).
    """
    OFFLINE = "offline"
    OFFLINE_TO_ONLINE = "offline_to_online"
    ONLINE_CONTRAST = "online_contrast"
    ONLINE_SHADOW = "online_shadow"


class ExpertKind(enum.StrEnum):
    """Which expert a scenario deploys the agent alongside."""
    NMPC = "nmpc"   # do-mpc NMPC — setpoint-tracking scenarios
    PID = "pid"     # PID/PI baseline — delta-u / disturbance scenarios (e.g. crystallization)


class MethodRole(enum.StrEnum):
    """Role a method plays in a rollout."""
    MODEL = "model"          # a learned agent (at some DeploymentStage)
    PID = "pid"              # PID/PI baseline controller
    NMPC = "nmpc"            # NMPC (expert and/or optimality ceiling)


class Device(enum.StrEnum):
    AUTO = "auto"   # use the GPU when one is available, else CPU
    CPU = "cpu"
    GPU = "gpu"


def _opt_enum(enum_cls: type[enum.StrEnum], value: Any) -> Any:
    return enum_cls(value) if value else None


def _json_factory(items: list[tuple[str, Any]]) -> dict[str, Any]:
    return {k: (v.value if isinstance(v, enum.Enum) else v) for k, v in items}


# ---------------------------------------------------------------------------
# Controller protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class ReferenceController(Protocol):
    """A non-learned controller (PID / NMPC): predict() + optional reset()."""
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]: ...


@runtime_checkable
class Agent(Protocol):
    """A learned offline agent: deterministic action, switching-critic Q-gap."""
    def act(self, obs: np.ndarray, explore: bool = False) -> np.ndarray: ...
    def q_gap(self, obs: np.ndarray, expert_action: np.ndarray) -> float: ...


# ---------------------------------------------------------------------------
# Run-label (directory slug) — human-readable convenience ONLY
# ---------------------------------------------------------------------------

def run_label_for(algorithm: Algorithm, mode: TrainingMode, bc: bool) -> str:
    """
    Canonical output-directory name for a training condition — the SINGLE source
    of slug naming. Semantic identity lives in RunSpec/run.json, NOT in this string.

      offline (BC)        -> "offline_<algo>_bc"      e.g. "offline_td3_bc"
      offline_to_online   -> "o2o_<algo>"
      online_contrast     -> "online_<algo>"
    """
    algo = algorithm.value
    match mode:
        case TrainingMode.OFFLINE:
            return f"offline_{algo}_bc" if bc else f"offline_{algo}"
        case TrainingMode.OFFLINE_TO_ONLINE:
            return f"o2o_{algo}"
        case TrainingMode.ONLINE_CONTRAST:
            return f"online_{algo}"
        case TrainingMode.ONLINE_SHADOW:
            return f"online_shadow_{algo}"


# ---------------------------------------------------------------------------
# Condition — a learner configuration (the grid building block)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """
    One training configuration (run once per seed). Produces both its agent
    kwargs and a RunSpec from the SAME fields, so config and metadata cannot
    disagree. Use the offline()/offline_to_online()/online_contrast() factories.
    """
    label: str
    algorithm: Algorithm
    training_mode: TrainingMode
    bc_alpha: float = 2.5

    _ONLINE_FROM_SCRATCH = (TrainingMode.ONLINE_CONTRAST, TrainingMode.ONLINE_SHADOW)

    @property
    def uses_bc(self) -> bool:
        return self.bc_alpha > 0.0 and self.training_mode not in self._ONLINE_FROM_SCRATCH

    @property
    def slug(self) -> str:
        return run_label_for(self.algorithm, self.training_mode, self.uses_bc)

    def agent_kwargs(self) -> dict[str, Any]:
        """Hyperparameters forwarded to the agent constructor."""
        bc = 0.0 if self.training_mode in self._ONLINE_FROM_SCRATCH else self.bc_alpha
        return {"bc_alpha": bc}

    def to_run_spec(self, scenario: Scenario, seed: int, *, offline_steps: int,
                    online_steps: int, expert_kind: ExpertKind) -> "RunSpec":
        return RunSpec(
            scenario=scenario, algorithm=self.algorithm, training_mode=self.training_mode,
            bc_alpha=self.bc_alpha, offline_steps=offline_steps, online_steps=online_steps,
            seed=seed, expert_kind=expert_kind,
            condition_label=self.label, run_label=self.slug,
        )


def offline(algorithm: Algorithm, bc_alpha: float = 2.5, label: str | None = None) -> Condition:
    """Offline (TD3+BC / DDPG+BC) condition — the headline method."""
    tag = "+BC" if bc_alpha > 0 else ""
    return Condition(label=label or f"Offline {algorithm.value.upper()}{tag}",
                     algorithm=algorithm, training_mode=TrainingMode.OFFLINE, bc_alpha=bc_alpha)


def offline_to_online(algorithm: Algorithm, bc_alpha: float = 1.0,
                      label: str | None = None) -> Condition:
    """Offline pretrain then conservative, expert-guarded online fine-tuning."""
    return Condition(label=label or f"O2O {algorithm.value.upper()}",
                     algorithm=algorithm, training_mode=TrainingMode.OFFLINE_TO_ONLINE,
                     bc_alpha=bc_alpha)


def online_contrast(algorithm: Algorithm, label: str | None = None) -> Condition:
    """Naive online RL on the live plant (the unsafe contrast baseline)."""
    return Condition(label=label or f"Online {algorithm.value.upper()} (contrast)",
                     algorithm=algorithm, training_mode=TrainingMode.ONLINE_CONTRAST, bc_alpha=0.0)


def online_shadow(algorithm: Algorithm, label: str | None = None) -> Condition:
    """Online RL from scratch, expert-guarded every step (G&A earned-takeover gate,
    no offline pretrain). Same from-scratch init as online_contrast; the guard is
    the only difference — the fair shadow-vs-standard learning ablation."""
    return Condition(label=label or f"Online shadow {algorithm.value.upper()}",
                     algorithm=algorithm, training_mode=TrainingMode.ONLINE_SHADOW, bc_alpha=0.0)


# ---------------------------------------------------------------------------
# RunSpec — a Condition bound to (scenario, seed, budgets)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """Everything needed to identify one trained model. Serialised to run.json."""
    scenario: Scenario
    algorithm: Algorithm
    training_mode: TrainingMode
    bc_alpha: float
    offline_steps: int
    online_steps: int
    seed: int
    expert_kind: ExpertKind
    condition_label: str
    run_label: str

    @property
    def artifact_stem(self) -> str:
        return f"{self.run_label}__seed{self.seed}"

    def to_json(self) -> dict[str, Any]:
        return asdict(self, dict_factory=_json_factory)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "RunSpec":
        return cls(
            scenario=Scenario(d["scenario"]),
            algorithm=Algorithm(d["algorithm"]),
            training_mode=TrainingMode(d["training_mode"]),
            bc_alpha=float(d["bc_alpha"]),
            offline_steps=int(d["offline_steps"]),
            online_steps=int(d.get("online_steps", 0)),
            seed=int(d["seed"]),
            expert_kind=ExpertKind(d["expert_kind"]),
            condition_label=str(d["condition_label"]),
            run_label=str(d["run_label"]),
        )


# ---------------------------------------------------------------------------
# ModelSpec — typed input to run_rollouts (checkpoint + its RunSpec)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """A trained checkpoint to roll out, paired with its structured metadata."""
    checkpoint: str
    run: RunSpec


# ---------------------------------------------------------------------------
# MethodRecord — one structured manifest entry per rolled-out method + stage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodRecord:
    """
    A method's entry in a rollout manifest. `run` is populated for MODEL methods
    and None for reference controllers (PID / NMPC). `stage` records the
    deployment stage for MODEL rollouts (None for references).
    """
    role: MethodRole
    label: str
    npz_file: str
    scenario: Scenario
    run: RunSpec | None = None
    stage: DeploymentStage | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self, dict_factory=_json_factory)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "MethodRecord":
        run = d.get("run")
        return cls(
            role=MethodRole(d["role"]),
            label=str(d["label"]),
            npz_file=str(d["npz_file"]),
            scenario=Scenario(d["scenario"]),
            run=RunSpec.from_json(run) if run else None,
            stage=_opt_enum(DeploymentStage, d.get("stage")),
        )
