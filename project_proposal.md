# Offline reinforcement learning for industrial process control

## Description
Reinforcement learning (RL) offers the potential to improve efficiency and adaptability in process control, but its safe deployment in real industrial settings remains a major challenge. In particular, process operators must be able to trust the actions given by an RL algorithm, even during the initial learning phase. Our recent work introduced PC-Gym, a comprehensive benchmarking suite and development tool for process control algorithms in the chemical industry, including both RL and traditional model-based control strategies [Bloor et al., 2025].

This project explores an offline deployment framework where an RL controller is introduced alongside the existing control strategy of the process, comprising an established "expert" such as Model Predictive Control (MPC). For example, the RL agent is first trained on historical and simulated process data, then run in shadow mode to observe the system and generate candidate actions without influencing operations. As confidence in the learned policy grows, the RL controller can begin suggesting or testing actions under close supervision, gradually increasing its level of autonomy. This staged approach allows the RL system to build expertise while maintaining operational safety and reliability, with the ultimate goal of providing a practical pathway for integrating RL into critical process industries.

## References
Bloor, M., Torraca, J., Sandoval, I. O., Ahmed, A., White, M., Mercangöz, M., ... & Mowbray, M. (2025). PC-Gym: Benchmark environments for process control problems. Computers & Chemical Engineering, 109363.
