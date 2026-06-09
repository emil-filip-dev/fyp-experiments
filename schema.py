"""
schema.py
=========
The shared, typed vocabulary of the experiment pipeline. Run metadata lives in
STRUCTURED, TYPED objects (StrEnums + dataclasses) serialised to JSON — never
encoded into, nor parsed back out of, directory/file slugs. Slugs (run-label dir
names) remain ONLY a human-readable convenience.

Sources of truth for a run's identity:
  - Condition    : a learner configuration (algorithm + shadow settings), declared
                   in experiments.py grids; produces both its train() kwargs and a
                   RunSpec — so the two never drift.
  - RunSpec      : a Condition bound to (scenario, seed, total_steps); written as
                   run.json beside each checkpoint.
  - ModelSpec    : the typed input to evaluate.run_rollouts (checkpoint + RunSpec).
  - MethodRecord : one structured entry per method in a rollout manifest, consumed
                   by the analysis utility (Phase 4) — which reads these fields and
                   never inspects a filename.

StrEnum members are `str` subclasses; `_to_jsonable` converts them (and nested
dataclasses) to plain JSON, and we parse them back explicitly on load.
"""

import enum
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from models import SwitchingMode, TD3SwitchCritic


# ---------------------------------------------------------------------------
# Categorical types (StrEnum — never bare strings at the boundaries)
# ---------------------------------------------------------------------------

class Scenario(enum.StrEnum):
    """The PC-Gym scenarios in the SCENARIOS registry."""
    CSTR = "cstr"
    FOUR_TANK = "four_tank"
    MULTISTAGE_EXTRACTION = "multistage_extraction"
    CRYSTALLIZATION = "crystallization"


class Algorithm(enum.StrEnum):
    """The off-policy learner family. Orthogonal to shadow-mode switching."""
    DDPG = "ddpg"
    TD3 = "td3"


class MethodRole(enum.StrEnum):
    """Role a method plays in a rollout: a learned model, or a reference controller."""
    MODEL = "model"
    PID = "pid"
    NMPC_ORACLE = "nmpc_oracle"


class Device(enum.StrEnum):
    CPU = "cpu"
    GPU = "gpu"


def _opt_enum(enum_cls: type[enum.StrEnum], value: Any) -> Any:
    """Parse an optional StrEnum field from JSON (None/'' -> None)."""
    return enum_cls(value) if value else None


def _json_factory(items: list[tuple[str, Any]]) -> dict[str, Any]:
    """
    `asdict` dict_factory that converts StrEnum values to their `.value`. asdict
    already recurses nested dataclasses (applying this factory at each level), so a
    single pass yields plain JSON — no separate re-walk, no json.dumps reliance.
    """
    return {k: (v.value if isinstance(v, enum.Enum) else v) for k, v in items}


# ---------------------------------------------------------------------------
# Controller protocols (replace `object` for the heterogeneous rollout actors)
# ---------------------------------------------------------------------------

@runtime_checkable
class ReferenceController(Protocol):
    """A non-learned controller (PID / NMPC oracle): predict() + optional reset()."""
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]: ...


@runtime_checkable
class ShadowAgent(Protocol):
    """A learned shadow/standard agent: shadow-aware decision + q-value advantage."""
    def decide_action(self, obs: np.ndarray, baseline_action: np.ndarray,
                      training: bool = True, force_baseline: bool = False
                      ) -> tuple[np.ndarray, bool, np.ndarray]: ...
    def q_gap(self, obs: np.ndarray, baseline_action: np.ndarray) -> float: ...


# A rollout actor is one or the other; _record_rollout dispatches on decide_action.
RolloutController = ReferenceController | ShadowAgent


# ---------------------------------------------------------------------------
# Run-label (directory slug) — human-readable convenience ONLY
# ---------------------------------------------------------------------------

def run_label_for(
    algorithm: Algorithm,
    *,
    shadow: bool = False,
    mode: SwitchingMode = SwitchingMode.Q_VALUE,
    lambda_reg: float = 0.0,
    switch_critic: TD3SwitchCritic = TD3SwitchCritic.Q1,
) -> str:
    """
    Canonical output-directory name for a training condition — the SINGLE source
    of slug naming, so the producer (train) and consumers (orchestrator/analysis)
    cannot drift. Semantic identity lives in RunSpec/run.json, NOT in this string.

      standard            -> "<algo>"                         e.g. "ddpg"
      shadow agent + reg  -> "shadow_<algo>_agent_reg<lambda>"
      shadow td3 qvalue   -> "shadow_<algo>_qvalue_<switch_critic>"
      shadow (other)      -> "shadow_<algo>_<mode>"
    """
    algo = algorithm.value
    if not shadow:
        return algo
    if mode is SwitchingMode.AGENT and lambda_reg > 0.0:
        return f"shadow_{algo}_agent_reg{lambda_reg}"
    if algorithm is Algorithm.TD3 and mode is SwitchingMode.Q_VALUE:
        return f"shadow_{algo}_{mode.value}_{switch_critic.value}"
    return f"shadow_{algo}_{mode.value}"


# ---------------------------------------------------------------------------
# Condition — a learner configuration (the grid building block)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """
    One *learned* training configuration (run once per seed). Fully typed: the
    shadow-only fields are None/ignored for standard runs. Produces both its
    train() kwargs and a RunSpec from the SAME fields, so config and metadata
    cannot disagree. Use the `standard()` / `shadow()` factories for readability.
    """
    label: str
    algorithm: Algorithm
    shadow: bool = False
    switching_mode: SwitchingMode | None = None
    lambda_reg: float = 0.0
    eta_agent: float = 0.5
    switch_critic: TD3SwitchCritic | None = None

    def __post_init__(self) -> None:
        # Reject incoherent combinations rather than silently ignoring them.
        if not self.shadow and (self.switching_mode is not None or self.switch_critic is not None):
            raise ValueError("standard (non-shadow) Condition must not set switching_mode/switch_critic")
        if self.switch_critic is not None and not self.is_td3_qvalue:
            raise ValueError("switch_critic only applies to shadow TD3 q-value switching")

    @property
    def mode(self) -> SwitchingMode:
        """Effective switching mode (defaults to Q-value for shadow runs)."""
        return self.switching_mode or SwitchingMode.Q_VALUE

    @property
    def is_td3_qvalue(self) -> bool:
        return self.shadow and self.algorithm is Algorithm.TD3 and self.mode is SwitchingMode.Q_VALUE

    @property
    def slug(self) -> str:
        return run_label_for(
            self.algorithm, shadow=self.shadow, mode=self.mode,
            lambda_reg=self.lambda_reg,
            switch_critic=self.switch_critic or TD3SwitchCritic.Q1,
        )

    def train_kwargs(self) -> dict[str, Any]:
        """Kwargs forwarded to train.train() (strings at that CLI-style boundary)."""
        kw: dict[str, Any] = {"model_type": self.algorithm.value, "shadow": self.shadow}
        if self.shadow:
            kw["mode"] = self.mode.value
            kw["lambda_reg"] = self.lambda_reg
            kw["eta_agent"] = self.eta_agent
            if self.is_td3_qvalue:
                kw["switch_critic"] = (self.switch_critic or TD3SwitchCritic.Q1).value
        return kw

    def to_run_spec(self, scenario: Scenario, seed: int, total_steps: int) -> "RunSpec":
        """Bind this condition to a (scenario, seed, budget) — the only RunSpec factory."""
        # switch_critic is only meaningful for TD3 q-value switching; record the
        # SAME effective value the slug/train_kwargs use, so the grid-side and
        # train-side RunSpecs cannot disagree.
        effective_switch = (self.switch_critic or TD3SwitchCritic.Q1) if self.is_td3_qvalue else None
        return RunSpec(
            scenario=scenario, algorithm=self.algorithm, shadow=self.shadow,
            total_steps=total_steps, seed=seed,
            condition_label=self.label, run_label=self.slug,
            switching_mode=self.mode if self.shadow else None,
            lambda_reg=self.lambda_reg, eta_agent=self.eta_agent,
            switch_critic=effective_switch,
        )


def standard(algorithm: Algorithm, label: str | None = None) -> Condition:
    """A standard (no-shadow) condition."""
    return Condition(label=label or algorithm.value.upper(), algorithm=algorithm, shadow=False)


def shadow(
    algorithm: Algorithm,
    mode: SwitchingMode = SwitchingMode.Q_VALUE,
    *,
    lambda_reg: float = 0.0,
    eta_agent: float = 0.5,
    switch_critic: TD3SwitchCritic | None = None,
    label: str | None = None,
) -> Condition:
    """A shadow-mode condition (switching_mode always set)."""
    auto = f"Shadow {algorithm.value.upper()} ({mode.value})"
    return Condition(
        label=label or auto, algorithm=algorithm, shadow=True, switching_mode=mode,
        lambda_reg=lambda_reg, eta_agent=eta_agent, switch_critic=switch_critic,
    )


# ---------------------------------------------------------------------------
# RunSpec — a Condition bound to (scenario, seed, budget)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """
    Everything needed to identify one trained model, with no reference to its
    directory name. Serialised to run.json beside the checkpoint. Built only via
    Condition.to_run_spec() or from_json()/from_checkpoint().
    """
    scenario: Scenario
    algorithm: Algorithm
    shadow: bool
    total_steps: int
    seed: int
    condition_label: str               # human label for tables/plots
    run_label: str                     # directory slug (convenience only)
    switching_mode: SwitchingMode | None = None
    lambda_reg: float = 0.0
    eta_agent: float = 0.5
    switch_critic: TD3SwitchCritic | None = None

    @property
    def artifact_stem(self) -> str:
        """Unique, human-readable file stem for this run's rollout (object -> name)."""
        return f"{self.run_label}__seed{self.seed}"

    def to_json(self) -> dict[str, Any]:
        return asdict(self, dict_factory=_json_factory)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "RunSpec":
        return cls(
            scenario=Scenario(d["scenario"]),
            algorithm=Algorithm(d["algorithm"]),
            shadow=bool(d["shadow"]),
            total_steps=int(d["total_steps"]),
            seed=int(d["seed"]),
            condition_label=str(d["condition_label"]),
            run_label=str(d["run_label"]),
            switching_mode=_opt_enum(SwitchingMode, d.get("switching_mode")),
            lambda_reg=float(d.get("lambda_reg", 0.0)),
            eta_agent=float(d.get("eta_agent", 0.5)),
            switch_critic=_opt_enum(TD3SwitchCritic, d.get("switch_critic")),
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
# MethodRecord — one structured manifest entry per rolled-out method
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodRecord:
    """
    A method's entry in a rollout manifest. `run` is populated for MODEL methods
    and None for reference controllers (PID / NMPC oracle). Phase 4 groups and
    aggregates on these fields — it never parses `npz_file`.
    """
    role: MethodRole
    label: str
    npz_file: str
    scenario: Scenario
    run: RunSpec | None = None

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
        )
