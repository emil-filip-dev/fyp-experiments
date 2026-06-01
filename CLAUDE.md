# FYP Project — Claude Code Instructions

## Autonomous Operation
Operate autonomously. Do not ask for confirmation before taking actions unless they are irreversible and destructive (e.g. deleting files, force-pushing). Proceed with reading, writing, editing, running scripts, and installing packages without prompting.

## Project Overview
Imperial College London FYP: Making RL reliable enough for industrial process control via hard constraint satisfaction.

Core research direction: Extend Shadow Mode RL (Gassert & Althoff, 2024) to PC-Gym benchmark environments, combining it with constraint-handling mechanisms from CIRL and learning-to-defer frameworks.

## Key Papers
1. **Shadow Mode RL** (Gassert & Althoff, 2024) — core method being extended
2. **PC-Gym** (Bloor et al., 2024) — benchmark environments (CSTR, crystallization, four-tank, multistage extraction)
3. **CIRL** (Bloor et al., 2024) — PID-embedded RL policy for process control
4. **RL Survey for PSE** (Bloor et al., 2025) — background and metrics
5. **Learning-to-Defer / SLTD** (Joshi et al., 2022) — deferral framework for sequential decisions

## Environment
- **Python**: 3.12 via `.venv` (use `.venv/Scripts/python` — do NOT use system Python)
- **Activate venv**: `.venv/Scripts/activate`
- **Run scripts**: `.venv/Scripts/python <script.py>`
- **Install packages**: `.venv/Scripts/pip install <pkg>`

## Dependencies (requirements.txt)
numpy, matplotlib, gymnasium, casadi, jax[cpu], equinox, diffrax, do-mpc[full], stable-baselines3, pcgym

## PC-Gym API
```python
from pcgym import make_env

env_params = {
    'N': 100,           # number of time steps
    'tsim': 26,         # simulation time (minutes)
    'SP': {'Ca': [...]},  # setpoint schedule
    'o_space': {'low': ..., 'high': ...},
    'a_space': {'low': ..., 'high': ...},
    'x0': np.array([...]),
    'model': 'cstr',    # or 'multistage_extraction', 'crystallization', 'four_tank'
    'normalise_a': True,
    'normalise_o': True,
    'noise': True,
    'integration_method': 'casadi',
    'noise_percentage': 0.001,
}

env = make_env(env_params)
obs, _ = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
# env.state — always in physical (unnormalised) units
```

## Metrics
- Median expected return, MAD across seeds
- Optimality gap: Δ = J(π*) − J(π_θ), where π* is NMPC oracle
- NMPC oracle via do-mpc + CasADi + IPOPT

## Key Files
- `example_pcgym.py` — CSTR env + PPO training + PID baseline comparison; reference implementation
- `requirements.txt` — all dependencies
- `pcgym_example_results.png` — output plot from example

## Coding Conventions
- Keep scripts self-contained and runnable
- Use `env.state` (not `obs`) for physical state values when plotting/logging
- Use `seed=` in `env.reset()` for reproducibility
- Prefer numpy over torch for non-NN computations
- Use `stable_baselines3` for RL agents (PPO, SAC, TD3)
- Use `do_mpc` + `casadi` for NMPC baselines
