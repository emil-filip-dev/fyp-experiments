"""
scenarios.py
============
PC-Gym scenario definitions shared across the project.

Exports
-------
SCENARIOS : dict
    Registry keyed by scenario name.  Each entry contains:
      env_params   — passed directly to pcgym.make_env()
      state_dim    — observation dimension (states + setpoint dims)
      action_dim   — action dimension
      n_steps      — episode length in timesteps
      baseline_cls — callable() -> baseline controller
      plot_config  — list of dicts describing which physical states to plot:
                       state_idx : index into env.state
                       sp_idx    : index into env.state for the corresponding setpoint
                       label     : axis label string
                       unit      : physical unit string

make_env_for(scenario) -> gym.Env
"""

import numpy as np
from pcgym import make_env


# ---------------------------------------------------------------------------
# Baseline controllers
# ---------------------------------------------------------------------------

class CSTRBaseline:
    """
    PID for CSTR Ca tracking in normalised obs space.
    obs: [Ca_norm, T_norm, Ca_sp_norm]
    """
    def __init__(self, kp=-2.0, ki=-0.3, kd=-0.1):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral = 0.0
        self._prev_err = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0

    def predict(self, obs, deterministic=True):
        err   = obs[2] - obs[0]
        deriv = err - self._prev_err
        self._prev_err = err
        u = self.kp * err + self.ki * self._integral + self.kd * deriv
        if -1.0 < u < 1.0:
            self._integral += err
        return np.array([np.clip(u, -1.0, 1.0)], dtype=np.float32), None


class FourTankBaseline:
    """
    Decoupled PID for four-tank system in normalised obs space.
    obs: [h1, h2, h3, h4, h1_sp, h2_sp]
    """
    def __init__(self, kp=5.0, ki=0.15, kd=0.3):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integrals = np.zeros(2)
        self._prev_errs = np.zeros(2)

    def reset(self):
        self._integrals = np.zeros(2)
        self._prev_errs = np.zeros(2)

    def predict(self, obs, deterministic=True):
        errs   = np.array([obs[4] - obs[0], obs[5] - obs[1]])
        derivs = errs - self._prev_errs
        self._prev_errs = errs.copy()
        u = self.kp * errs + self.ki * self._integrals + self.kd * derivs
        for i in range(2):
            if -1.0 < u[i] < 1.0:
                self._integrals[i] += errs[i]
        return np.clip(u, -1.0, 1.0).astype(np.float32), None


class MultistageExtractionBaseline:
    """
    PI controller for multistage extraction X1 tracking in normalised obs space.
    obs: [X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X1_sp]
    """
    def __init__(self, kp=3.0, ki=0.2):
        self.kp, self.ki = kp, ki
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def predict(self, obs, deterministic=True):
        err = obs[10] - obs[0]
        self._integral = np.clip(self._integral + err, -5.0, 5.0)
        u = float(np.clip(self.kp * err + self.ki * self._integral, -1.0, 1.0))
        return np.array([u, u], dtype=np.float32), None


class CrystallizationBaseline:
    """
    PI controller for crystallisation concentration tracking in normalised obs space.
    obs: [mu0, mu1, mu2, mu3, Conc, CV, Ln, Conc_sp]
    """
    def __init__(self, kp=2.0, ki=0.1):
        self.kp, self.ki = kp, ki
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def predict(self, obs, deterministic=True):
        err = obs[4] - obs[7]
        self._integral = np.clip(self._integral + err, -5.0, 5.0)
        u = float(np.clip(-(self.kp * err + self.ki * self._integral), -1.0, 1.0))
        return np.array([u], dtype=np.float32), None


# ---------------------------------------------------------------------------
# Environment configurations
# ---------------------------------------------------------------------------

def _cstr_config():
    N = 60
    return {
        "env_params": {
            "N":    N,
            "tsim": 25,
            "SP":   {"Ca": [0.85] * (N // 2) + [0.90] * (N // 2)},
            "o_space": {
                "low":  np.array([0.70, 300.0, 0.80], dtype=np.float32),
                "high": np.array([1.00, 350.0, 0.90], dtype=np.float32),
            },
            "a_space": {
                "low":  np.array([295.0], dtype=np.float32),
                "high": np.array([302.0], dtype=np.float32),
            },
            "x0":             np.array([0.80, 330.0, 0.85]),
            "model":          "cstr",
            "r_scale":        {"Ca": 1e3},
            "normalise_a":    True,
            "normalise_o":    True,
            "noise":          True,
            "integration_method": "casadi",
            "noise_percentage":   0.001,
        },
        "state_dim":    3,
        "action_dim":   1,
        "n_steps":      N,
        "baseline_cls": CSTRBaseline,
        "plot_config": [
            {"state_idx": 0, "sp_idx": 2, "label": "Ca", "unit": "mol/L"},
        ],
    }


def _four_tank_config():
    N = 100
    return {
        "env_params": {
            "N":    N,
            "tsim": 20.0,
            "SP": {
                "h1": [0.14] * (N // 2) + [0.20] * (N // 2),
                "h2": [0.20] * (N // 2) + [0.14] * (N // 2),
            },
            "o_space": {
                "low":  np.array([0.01] * 6, dtype=np.float32),
                "high": np.array([0.80] * 6, dtype=np.float32),
            },
            "a_space": {
                "low":  np.array([0.5,  0.5],  dtype=np.float32),
                "high": np.array([10.0, 10.0], dtype=np.float32),
            },
            "x0": np.array([0.12, 0.12, 0.30, 0.15, 0.14, 0.20]),
            "model":          "four_tank",
            "r_scale":        {"h1": 1e3, "h2": 1e3},
            "normalise_a":    True,
            "normalise_o":    True,
            "noise":          True,
            "integration_method": "casadi",
            "noise_percentage":   0.002,
        },
        "state_dim":    6,
        "action_dim":   2,
        "n_steps":      N,
        "baseline_cls": FourTankBaseline,
        "plot_config": [
            {"state_idx": 0, "sp_idx": 4, "label": "h1", "unit": "m"},
            {"state_idx": 1, "sp_idx": 5, "label": "h2", "unit": "m"},
        ],
    }


def _multistage_extraction_config():
    # States: [X1,Y1,X2,Y2,X3,Y3,X4,Y4,X5,Y5] (10) + X1 setpoint = 11-dim obs
    # Actions: [L, G] (liquid and gas flow rates, m3/hr)
    N = 100
    return {
        "env_params": {
            "N":    N,
            "tsim": 10.0,
            "SP": {"X1": [0.08] * (N // 2) + [0.12] * (N // 2)},
            "o_space": {
                "low":  np.array([0.0] * 11, dtype=np.float32),
                "high": np.array([0.8] * 10 + [0.4], dtype=np.float32),
            },
            "a_space": {
                "low":  np.array([1.0, 1.0], dtype=np.float32),
                "high": np.array([9.0, 9.0], dtype=np.float32),
            },
            "x0": np.array([
                0.50, 0.18,
                0.45, 0.14,
                0.38, 0.10,
                0.30, 0.07,
                0.20, 0.05,
                0.08,
            ]),
            "model":          "multistage_extraction",
            "r_scale":        {"X1": 1e2},
            "normalise_a":    True,
            "normalise_o":    True,
            "noise":          True,
            "integration_method": "casadi",
            "noise_percentage":   0.001,
        },
        "state_dim":    11,
        "action_dim":   2,
        "n_steps":      N,
        "baseline_cls": MultistageExtractionBaseline,
        "plot_config": [
            {"state_idx": 0, "sp_idx": 10, "label": "X1", "unit": "mol/L"},
        ],
    }


def _crystallization_config():
    # States: [mu0,mu1,mu2,mu3,Conc,CV,Ln] (7) + Conc setpoint = 8-dim obs
    # Action: [T] temperature in °C
    N = 60
    return {
        "env_params": {
            "N":    N,
            "tsim": 60.0,
            "SP": {"Conc": [110.0] * (N // 2) + [95.0] * (N // 2)},
            "o_space": {
                "low":  np.array([0.0, 0.0, 0.0,  0.0,  80.0, 0.0,   0.0,  80.0], dtype=np.float32),
                "high": np.array([2e6, 2e8, 2e10, 2e12, 160.0, 2.0, 300.0, 140.0], dtype=np.float32),
            },
            "a_space": {
                "low":  np.array([0.0],  dtype=np.float32),
                "high": np.array([40.0], dtype=np.float32),
            },
            "x0": np.array([
                1e5, 1e7, 1e9, 1e11,
                130.0,
                0.5,
                100.0,
                110.0,
            ]),
            "model":          "crystallization",
            "r_scale":        {"Conc": 10.0},
            "normalise_a":    True,
            "normalise_o":    True,
            "noise":          True,
            "integration_method": "casadi",
            "noise_percentage":   0.001,
        },
        "state_dim":    8,
        "action_dim":   1,
        "n_steps":      N,
        "baseline_cls": CrystallizationBaseline,
        "plot_config": [
            {"state_idx": 4, "sp_idx": 7, "label": "Conc", "unit": "g/kg"},
        ],
    }


SCENARIOS: dict = {
    "cstr":                  _cstr_config(),
    "four_tank":             _four_tank_config(),
    "multistage_extraction": _multistage_extraction_config(),
    "crystallization":       _crystallization_config(),
}


def make_env_for(scenario: str):
    return make_env(SCENARIOS[scenario]["env_params"])
