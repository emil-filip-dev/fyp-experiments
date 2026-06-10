"""
experts.py
==========
The *expert* controller a scenario deploys the RL agent alongside (the
"established expert such as MPC" of project_proposal.md).

  - Setpoint-tracking scenarios (cstr, four_tank, multistage_extraction) use the
    do-mpc NMPC (models.NMPCController) — a genuine MPC expert AND the optimality
    ceiling.
  - Scenarios the NMPC cannot model (crystallization: delta-u action) fall back to
    the scenario's PID/PI baseline as the expert.

Both expert kinds implement the ReferenceController protocol (predict / reset), so
the rest of the pipeline treats them uniformly.

make_expert(scenario, ...) -> (controller, ExpertKind)
expert_kind_for(scenario)  -> ExpertKind          (no construction; for metadata)
"""

from scenarios import SCENARIOS
from schema import ExpertKind, Scenario


def expert_kind_for(scenario: Scenario | str) -> ExpertKind:
    """Which expert a scenario uses, without constructing it (delta-u/disturbance
    scenarios cannot use the NMPC and fall back to PID)."""
    cfg = SCENARIOS[str(scenario)]
    ep = cfg["env_params"]
    if ep.get("a_delta") or ep.get("disturbance_bounds") or ep.get("disturbances"):
        return ExpertKind.PID
    return ExpertKind.NMPC


def make_expert(scenario: Scenario | str, *, mpc_horizon: int = 20):
    """
    Build the expert controller for a scenario.

    Returns (controller, ExpertKind). Tries the NMPC for setpoint-tracking
    scenarios; if construction raises NotImplementedError (disturbances / delta-u)
    it falls back to the scenario's PID/PI baseline.
    """
    cfg = SCENARIOS[str(scenario)]
    kind = expert_kind_for(scenario)
    if kind is ExpertKind.NMPC:
        from models import NMPCController
        try:
            return NMPCController(cfg, horizon=mpc_horizon), ExpertKind.NMPC
        except NotImplementedError:
            pass
    return cfg["baseline_cls"](), ExpertKind.PID
