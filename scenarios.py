"""
scenarios.py
============
PC-Gym scenario definitions. The environment parameters are copied VERBATIM from
PC-Gym's own paper training scripts (github.com/MaximilianB2/pc-gym, under
`pc-gym_paper/train_policies/<env>/<env>_train.py`) — x0, o_space, a_space, SP
schedules, tsim, N, noise, delta-u settings, controlled variables, and the custom
OCP reward functions are exact copies of that source, NOT reconstructions.

Source files (branch `main`):
  cstr                  : train_policies/cstr/cstr_train.py
                          (+ cstr/custom_reward.py -> sp_track_reward)
  four_tank             : train_policies/four_tank/4tank_train.py
  multistage_extraction : train_policies/multistage_extraction/me_train.py
  crystallization       : train_policies/crystalisation/cryst_train.py

Each SCENARIOS entry also carries fields that are OURS (not part of PC-Gym), used
by this project's training/eval pipeline and shadow mode:
  state_dim, action_dim, n_steps, baseline_cls (a PID/PI safety-net controller
  that drives the SAME variable PC-Gym controls), plot_config.

NOTE: constraints (Phase-1 safety layer). All four are added as PC-Gym-NATIVE
env_params (constraints/cons_type/done_on_cons_vio=False/r_penalty=False) so the
env records info["cons_info"] and, for setpoint-tracking scenarios, the do-mpc
oracle (models.NMPCController) enforces them as hard state bounds. Each is ALSO
mirrored in `constraint_spec` for the eval pipeline's own env.state-based
violation detection (constraints.py / analysis.py).
  cstr      : reactor temperature 319 K <= T <= 331 K, from PC-Gym's constraints
              guide (cons={'T':[331,319]}, cons_type={'T':['<=','>=']}). NOT the
              constraint_showcase's tighter 321..327 band: that excludes the
              verbatim x0 (T0=330 K starts above 327) and forces even the NMPC
              oracle to violate, whereas 319..331 contains x0 with ~4.5 K of
              headroom. Oracle-enforced.
  four_tank : lower-tank high-level h3 <= 0.55 m, h4 <= 0.55 m. OURS (PC-Gym
              defines none) — a high-level alarm 0.05 m below the 0.6 m tank top
              (the o_space ceiling), kept inside the observable range so the
              normalised observation does not saturate. Oracle-enforced.
  multistage_extraction : product off-spec X5 <= 0.5 mol/L. OURS. Oracle-enforced.
  crystallization : minimum liquor concentration Conc >= 0.11 kg/kg. OURS. Oracle
              N/A (delta-u action -> NMPCController/run_rollouts skip the oracle);
              constraint still recorded by the env and detected from env.state.
The three OURS bounds are calibrated against the measured operating envelope so
nominal (PID/PI baseline) control stays inside while aggressive/exploratory
control crosses them — the regime the C1 safety-during-training claim needs.

make_env_for(scenario) -> gym.Env
"""

import numpy as np
from pcgym import make_env


# ---------------------------------------------------------------------------
# Custom reward functions — copied VERBATIM from PC-Gym's training scripts.
# ---------------------------------------------------------------------------
# cstr / four_tank / multistage_extraction all use the same OCP setpoint-tracking
# reward with a control-move penalty R=0.1 (cstr imports it as `sp_track_reward`,
# 4tank_train.py and me_train.py define an identical inline `oracle_reward`).

def sp_track_reward(self, x, u, con):
    Sp_i = 0
    cost = 0
    R = 0.1
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

    for k in self.env_params["SP"]:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]

        o_space_low = self.env_params["o_space"]["low"][i]
        o_space_high = self.env_params["o_space"]["high"][i]

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        cost += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)

        Sp_i += 1
    u_normalized = (u - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    u_prev_norm =  (self.u_prev - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    self.u_prev = u

    # Add the control cost
    cost += np.sum(R * (u_normalized-u_prev_norm)**2)
    r = -cost
    try:
        return r[0]
    except Exception:
        return r


# crystallisation uses a CV/Ln-specific OCP reward with control-move penalty R=0.01
# (verbatim from cryst_train.py).
def cryst_oracle_reward(self, x, u, con):
    R = 0.01
    SP = self.SP
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

    # NaN-safety clamp (the ONLY deviation from the verbatim PC-Gym reward): under
    # measurement noise the radicand can dip below 0, and sqrt(negative) -> NaN.
    # Clamp it to >= 0 (a NaN radicand also fails ">0" and falls to 0.0).
    _cv_arg = x[2]*x[0]/(x[1]**2) - 1
    CV = _cv_arg**0.5 if _cv_arg > 0 else 0.0
    ln = x[1]/x[0] if x[0] != 0 else 0.0

    o_space_low = self.env_params["o_space"]["low"][[5,6]]
    o_space_high = self.env_params["o_space"]["high"][[5,6]]

    CV_normalized = (CV - o_space_low[0]) / (o_space_high[0] - o_space_low[0])
    Ln_normalized = (ln - o_space_low[1]) / (o_space_high[1] - o_space_low[1])
    SP_CV = (SP['CV'][self.t] - o_space_low[0]) / (o_space_high[0] - o_space_low[0])
    SP_Ln = (SP['Ln'][self.t] - o_space_low[1]) / (o_space_high[1] - o_space_low[1])

    r = -1*((SP_CV - CV_normalized)**2 + (SP_Ln - Ln_normalized)**2)

    u_normalized = (u - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    u_prev_norm =  (self.u_prev - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )

    r -= np.sum(R * (u_normalized-u_prev_norm)**2)
    self.u_prev = u

    return r


# ---------------------------------------------------------------------------
# Baseline (PID/PI) safety-net controllers — OURS, for shadow mode. Each drives
# the SAME variable(s) PC-Gym controls, working in the normalised obs space.
# ---------------------------------------------------------------------------

class CSTRBaseline:
    """
    PID on Ca (obs: [Ca, T, Ca_sp]); manipulates the cooling temperature Tc.
    Ca uses o_space [0.7, 1.0] but Ca_sp uses [0.8, 0.9] — different scales — so
    both are un-normalised to physical concentration before forming the error
    (otherwise the loop converges to a setpoint-dependent offset).
    """
    _CA_LO, _CA_HI = 0.7, 1.0
    _SP_LO, _SP_HI = 0.8, 0.9

    def __init__(self, kp=-30.0, ki=-4.0, kd=-1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral = 0.0
        self._prev_err = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0

    def predict(self, obs, deterministic=True):
        ca = (obs[0] + 1.0) / 2.0 * (self._CA_HI - self._CA_LO) + self._CA_LO
        ca_sp = (obs[2] + 1.0) / 2.0 * (self._SP_HI - self._SP_LO) + self._SP_LO
        err = ca_sp - ca
        deriv = err - self._prev_err
        self._prev_err = err
        u = self.kp * err + self.ki * self._integral + self.kd * deriv
        if -1.0 < u < 1.0:
            self._integral += err
        return np.array([np.clip(u, -1.0, 1.0)], dtype=np.float32), None


class FourTankBaseline:
    """
    Decoupled PI controlling h3 and h4 (the variables PC-Gym sets setpoints on).
    obs: [h1, h2, h3, h4, h3_sp, h4_sp]. Pump v2 (action[1]) drives h3, pump v1
    (action[0]) drives h4. States/SPs share o_space [0, 0.6], un-normalised here.
    """
    _LO, _HI = 0.0, 0.6

    def __init__(self, kp=15.0, ki=2.0):
        self.kp, self.ki = kp, ki
        self._i3 = 0.0
        self._i4 = 0.0

    def reset(self):
        self._i3 = 0.0
        self._i4 = 0.0

    def _phys(self, v):
        return (v + 1.0) / 2.0 * (self._HI - self._LO) + self._LO

    def predict(self, obs, deterministic=True):
        h3, h4 = self._phys(obs[2]), self._phys(obs[3])
        s3, s4 = self._phys(obs[4]), self._phys(obs[5])
        e3, e4 = s3 - h3, s4 - h4
        self._i3 = np.clip(self._i3 + e3, -1.0, 1.0)
        self._i4 = np.clip(self._i4 + e4, -1.0, 1.0)
        v2 = np.clip(self.kp * e3 + self.ki * self._i3, -1.0, 1.0)   # -> h3
        v1 = np.clip(self.kp * e4 + self.ki * self._i4, -1.0, 1.0)   # -> h4
        return np.array([v1, v2], dtype=np.float32), None


class MultistageExtractionBaseline:
    """
    PI controlling X5 (obs idx 8; the variable PC-Gym sets a setpoint on).
    obs: [X1,Y1,X2,Y2,X3,Y3,X4,Y4,X5,Y5, X5_sp]. Liquid flow L (action[0]) is the
    dominant input — higher L raises X5 (gas flow G has only a weak effect), so the
    PI drives L and G is held neutral. X5 uses o_space [0,1], X5_sp uses [0.3,0.4];
    both un-normalised before comparison.
    """
    def __init__(self, kp=5.0, ki=0.5):
        self.kp, self.ki = kp, ki
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def predict(self, obs, deterministic=True):
        x5 = (obs[8] + 1.0) / 2.0 * (1.0 - 0.0) + 0.0
        sp = (obs[10] + 1.0) / 2.0 * (0.4 - 0.3) + 0.3
        err = sp - x5
        self._integral = np.clip(self._integral + err, -3.0, 3.0)
        # X5 below setpoint -> raise liquid flow L to raise X5.
        L = float(np.clip(self.kp * err + self.ki * self._integral, -1.0, 1.0))
        G = 0.0   # neutral
        return np.array([L, G], dtype=np.float32), None


class CrystallizationBaseline:
    """
    Delta-u P-controller nudging the cooling temperature to track CV.
    obs: [mu0,mu1,mu2,mu3,Conc,CV,Ln, CV_sp, Ln_sp]. Lower T narrows the
    distribution (lowers CV), so dT ~ +kp*(CV_sp - CV). CV uses o_space [0,2],
    CV_sp uses [0.9,1.1]; both un-normalised before comparison.
    """
    _CV_LO, _CV_HI = 0.0, 2.0
    _CV_SP_LO, _CV_SP_HI = 0.9, 1.1

    def __init__(self, kp=3.0):
        self.kp = kp

    def reset(self):
        pass

    def predict(self, obs, deterministic=True):
        cv = (obs[5] + 1.0) / 2.0 * (self._CV_HI - self._CV_LO) + self._CV_LO
        cv_sp = (obs[7] + 1.0) / 2.0 * (self._CV_SP_HI - self._CV_SP_LO) + self._CV_SP_LO
        dT = float(np.clip(self.kp * (cv_sp - cv), -1.0, 1.0))
        return np.array([dT], dtype=np.float32), None


# ---------------------------------------------------------------------------
# Environment configurations — env_params copied VERBATIM (see module docstring).
# ---------------------------------------------------------------------------

def _cstr_config():
    # VERBATIM from pc-gym_paper/train_policies/cstr/cstr_train.py
    T = 26
    nsteps = 60
    SP = {
        'Ca': [0.85 for i in range(int(nsteps / 3))] + [0.9 for i in range(int(nsteps / 3))] + [0.87 for i in range(int(nsteps / 3))],
    }
    action_space = {
        'low': np.array([295]),
        'high': np.array([302]),
    }
    observation_space = {
        'low': np.array([0.7, 300, 0.8]),
        'high': np.array([1, 350, 0.9]),
    }
    r_scale = {'Ca': 1e3}
    # Reactor-temperature constraint from PC-Gym's own constraints guide, which
    # defines  cons = lambda x,u: [319 - T, T - 331]  i.e. 319 K <= T <= 331 K.
    # (An earlier version of this file used 321..327 — WRONG: that band is too tight,
    # it does not contain the verbatim x0 (T0=330 K starts OUTSIDE 327), and tracking
    # the Ca setpoints forces T to ~326.5 K, so even the NMPC oracle violated ~17%.
    # The documented 319..331 band contains x0 and gives the operating point ~4.5 K of
    # headroom -> the expert is genuinely safe, as a well-posed constraint requires.)
    # Passed PC-Gym-native so the env populates info["cons_info"] and the do-mpc oracle
    # imposes T as hard MPC state bounds. done_on_cons_vio=False (record, don't
    # terminate); r_penalty=False (reward unchanged).
    cons = {'T': [331, 319]}
    cons_type = {'T': ['<=', '>=']}
    env_params = {
        'N': nsteps,
        'tsim': T,
        'SP': SP,
        'o_space': observation_space,
        'a_space': action_space,
        'x0': np.array([0.8, 330, 0.8]),
        'r_scale': r_scale,
        'model': 'cstr',
        'normalise_a': True,
        'normalise_o': True,
        'noise': True,
        'integration_method': 'casadi',
        'noise_percentage': 0.001,
        'custom_reward': sp_track_reward,
        'constraints': cons,
        'cons_type': cons_type,
        'done_on_cons_vio': False,
        'r_penalty': False,
    }
    return {
        "env_params":   env_params,
        "state_dim":    3,
        "action_dim":   1,
        "n_steps":      nsteps,
        "baseline_cls": CSTRBaseline,
        "plot_config": [
            {"state_idx": 0, "sp_idx": 2, "label": "Ca", "unit": "mol/L"},
        ],
        # Mirrors the native env_params constraint above (PC-Gym docs: 319 K <= T <= 331 K).
        # State order for the cstr model is [Ca, T, Ca_SP], so T is physical-state index 1.
        "constraint_spec": [
            {"name": "T_max", "label": "Reactor temperature (upper)",
             "state_idx": 1, "bound": 331, "type": "<=", "unit": "K"},
            {"name": "T_min", "label": "Reactor temperature (lower)",
             "state_idx": 1, "bound": 319, "type": ">=", "unit": "K"},
        ],
    }


def _four_tank_config():
    # VERBATIM from pc-gym_paper/train_policies/four_tank/4tank_train.py
    T = 1000
    nsteps = 60
    SP = {
        'h3': [0.5 for i in range(int(nsteps / 2))] + [0.1 for i in range(int(nsteps / 2))],
        'h4': [0.2 for i in range(int(nsteps / 2))] + [0.3 for i in range(int(nsteps / 2))],
    }
    action_space = {
        'low': np.array([0.1, 0.1]),
        'high': np.array([10, 10]),
    }
    observation_space = {
        'low': np.array([0, ] * 6),
        'high': np.array([0.6] * 6),
    }
    # Constraint OURS (not verbatim PC-Gym — PC-Gym defines no four_tank constraint;
    # its docs describe 0.6 only as the tank HEIGHT / o_space ceiling). High-level
    # bound h3,h4 <= 0.55 m: a "high-level alarm" set 0.05 m below the 0.6 m tank top.
    # Set to 0.55 (NOT the 0.6 obs ceiling) so the constraint sits INSIDE the
    # observable range — at 0.6 the normalised observation saturates (h=0.6 -> obs=1),
    # leaving the agent blind to proximity/overshoot. Calibration: setpoints peak at
    # h3=0.5; the do-mpc oracle (which enforces this bound) and the offline agent stay
    # under 0.55, the sluggish PID overshoots to ~0.588, and unguarded exploration
    # drives h3 to ~0.68 — so nominal/guarded control is safe while reckless control
    # crosses, exactly what the C1 safety claim needs (and now perceivable to the agent).
    cons = {'h3': [0.55], 'h4': [0.55]}
    cons_type = {'h3': ['<='], 'h4': ['<=']}
    env_params = {
        'N': nsteps,
        'tsim': T,
        'SP': SP,
        'o_space': observation_space,
        'a_space': action_space,
        'x0': np.array([0.141, 0.112, 0.072, 0.42, SP['h3'][0], SP['h4'][0]]),
        'model': 'four_tank',
        'normalise_a': True,
        'normalise_o': True,
        'noise': True,
        'noise_percentage': 0.05,
        'custom_reward': sp_track_reward,
        'integration_method': 'casadi',
        'constraints': cons,
        'cons_type': cons_type,
        'done_on_cons_vio': False,
        'r_penalty': False,
    }
    return {
        "env_params":   env_params,
        "state_dim":    6,
        "action_dim":   2,
        "n_steps":      nsteps,
        "baseline_cls": FourTankBaseline,
        "plot_config": [
            {"state_idx": 2, "sp_idx": 4, "label": "h3", "unit": "m"},
            {"state_idx": 3, "sp_idx": 5, "label": "h4", "unit": "m"},
        ],
        # Mirrors the native env_params constraints above for the eval pipeline's
        # env.state-based detection. State order [h1,h2,h3,h4,h3_sp,h4_sp] -> h3=2, h4=3.
        "constraint_spec": [
            {"name": "h3_max", "label": "Tank 3 high-level",
             "state_idx": 2, "bound": 0.55, "type": "<=", "unit": "m"},
            {"name": "h4_max", "label": "Tank 4 high-level",
             "state_idx": 3, "bound": 0.55, "type": "<=", "unit": "m"},
        ],
    }


def _multistage_extraction_config():
    # VERBATIM from pc-gym_paper/train_policies/multistage_extraction/me_train.py
    T = 60
    nsteps = 60
    SP = {
        'X5': [0.3 for i in range(int(nsteps / 4))] + [0.4 for i in range(int(nsteps / 2))] + [0.3 for i in range(int(nsteps / 4))],
    }
    action_space = {
        'low': np.array([5, 10]),
        'high': np.array([500, 1000]),
    }
    observation_space = {
        'low': np.array([0] * 10 + [0.3]),
        'high': np.array([1] * 10 + [0.4]),
    }
    r_scale = {
        'X5': 1,
    }
    # Constraint OURS (not verbatim PC-Gym — PC-Gym defines no multistage constraint).
    # Physically motivated: the controlled product stream solute fraction X5 must not
    # exceed an off-spec ceiling of 0.5 mol/L (the setpoints are 0.3/0.4, well below).
    # Calibrated against the measured envelope: the PI baseline peaks at X5=0.474
    # (safe) while sustained extreme flows push X5 to 0.585 (off-spec) -> respected by
    # nominal control, crossed by aggressive/exploratory control. Passed PC-Gym-native
    # so the do-mpc oracle enforces X5 as a hard upper state bound.
    cons = {'X5': [0.5]}
    cons_type = {'X5': ['<=']}
    env_params = {
        'N': nsteps,
        'tsim': T,
        'SP': SP,
        'o_space': observation_space,
        'a_space': action_space,
        'dt': 1,
        'x0': np.array([0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1, 0.3]),
        'model': 'multistage_extraction',
        'r_scale': r_scale,
        'normalise_a': True,
        'normalise_o': True,
        'noise': True,
        'noise_percentage': 0.05,
        'integration_method': 'casadi',
        'custom_reward': sp_track_reward,
        'constraints': cons,
        'cons_type': cons_type,
        'done_on_cons_vio': False,
        'r_penalty': False,
    }
    return {
        "env_params":   env_params,
        "state_dim":    11,
        "action_dim":   2,
        "n_steps":      nsteps,
        "baseline_cls": MultistageExtractionBaseline,
        "plot_config": [
            {"state_idx": 8, "sp_idx": 10, "label": "X5", "unit": "mol/L"},
        ],
        # Mirrors the native env_params constraint above for env.state-based detection.
        # State order [X1,Y1,X2,Y2,X3,Y3,X4,Y4,X5,Y5,X5_sp] -> X5 = index 8.
        "constraint_spec": [
            {"name": "X5_max", "label": "Product solute fraction (off-spec)",
             "state_idx": 8, "bound": 0.5, "type": "<=", "unit": "mol/L"},
        ],
    }


def _crystallization_config():
    # VERBATIM from pc-gym_paper/train_policies/crystalisation/cryst_train.py
    T = 30
    nsteps = 30
    SP = {
        'CV': [1 for i in range(int(nsteps))],
        'Ln': [15 for i in range(int(nsteps))],
    }
    action_space = {
        'low': np.array([-1]),
        'high': np.array([1]),
    }
    action_space_act = {
        'low': np.array([10]),
        'high': np.array([40]),
    }
    lbMu0 = 0
    ubMu0 = 1e20
    lbMu1 = 0
    ubMu1 = 1e20
    lbMu2 = 0
    ubMu2 = 1e20
    lbMu3 = 0
    ubMu3 = 1e20
    lbC = 0
    ubC = 0.5
    observation_space = {
        'low': np.array([lbMu0, lbMu1, lbMu2, lbMu3, lbC, 0, 0, 0.9, 14]),
        'high': np.array([ubMu0, ubMu1, ubMu2, ubMu3, ubC, 2, 20, 1.1, 16]),
    }
    CV_0 = np.sqrt(1800863.24079725 * 1478.00986666666 / (22995.8230590611 ** 2) - 1)
    Ln_0 = 22995.8230590611 / (1478.00986666666 + 1e-6)
    # Constraint OURS (not verbatim PC-Gym — PC-Gym defines no crystallization
    # constraint). Physically motivated: maintain a MINIMUM solute concentration,
    # Conc >= 0.11 (the model's scaled concentration state c; o_space [0, 0.5]) —
    # aggressive over-cooling crashes the supersaturation S = c*1e3 - C_eq and
    # over-depletes the solution (toward dissolution / loss of driving force).
    # Calibrated against the measured envelope: the P baseline holds Conc >= 0.125
    # (x0 starts at 0.1586) while sustained extreme cooling drives Conc down to 0.103
    # (over-depleted) -> respected by nominal control, crossed by aggressive/
    # exploratory control. Native env_params so the env records info["cons_info"];
    # note the do-mpc oracle is N/A here (delta-u action, NMPCController/run_rollouts
    # skip the oracle for crystallization).
    cons = {'Conc': [0.11]}
    cons_type = {'Conc': ['>=']}
    env_params = {
        'N': nsteps,
        'tsim': T,
        'SP': SP,
        'o_space': observation_space,
        'a_space': action_space,
        'x0': np.array([1478.00986666666, 22995.8230590611, 1800863.24079725, 248516167.940593, 0.15861523304, CV_0, Ln_0, 1, 15]),
        'model': 'crystallization',
        'normalise_a': True,
        'normalise_o': True,
        'noise': True,
        'noise_percentage': 0.001,
        'integration_method': 'casadi',
        'a_0': 39,
        'a_delta': True,
        'a_space_act': action_space_act,
        'custom_reward': cryst_oracle_reward,
        'constraints': cons,
        'cons_type': cons_type,
        'done_on_cons_vio': False,
        'r_penalty': False,
    }
    return {
        "env_params":   env_params,
        "state_dim":    9,
        "action_dim":   1,
        "n_steps":      nsteps,
        "baseline_cls": CrystallizationBaseline,
        "plot_config": [
            {"state_idx": 5, "sp_idx": 7, "label": "CV", "unit": ""},
            {"state_idx": 6, "sp_idx": 8, "label": "Ln", "unit": "um"},
        ],
        # Mirrors the native env_params constraint above for env.state-based detection.
        # State order [Mu0,Mu1,Mu2,Mu3,Conc,CV,Ln,CV_sp,Ln_sp] -> Conc = index 4.
        "constraint_spec": [
            {"name": "Conc_min", "label": "Solute concentration (over-depletion)",
             "state_idx": 4, "bound": 0.11, "type": ">=", "unit": ""},
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
