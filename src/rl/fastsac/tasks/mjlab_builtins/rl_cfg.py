"""FastSAC RL configs for mjlab built-in G1 tasks."""

from src.rl.fastsac.config import HolosomaFastSACRunnerCfg


def velocity_fastsac_runner_cfg() -> HolosomaFastSACRunnerCfg:
  return HolosomaFastSACRunnerCfg(
    experiment_name="g1_velocity_fastsac",
    max_iterations=50000,
    save_interval=1000,
    logging_interval=100,
    gamma=0.97,
    tau=0.125,
    buffer_size=1024,
    batch_size=8192,
    num_steps=1,
    num_updates=8,
    policy_frequency=4,
    learning_starts=10,
    num_atoms=101,
    v_min=-20.0,
    v_max=20.0,
    target_entropy_ratio=0.0,
    use_symmetry=False,
    use_tanh=True,
    obs_normalization=True,
    compile=False,
    amp=False,
  )


def tracking_fastsac_runner_cfg() -> HolosomaFastSACRunnerCfg:
  return HolosomaFastSACRunnerCfg(
    experiment_name="g1_tracking_fastsac",
    max_iterations=30001,
    save_interval=1000,
    logging_interval=100,
    gamma=0.99,
    tau=0.05,
    buffer_size=4096,
    batch_size=2048,
    num_updates=2,
    num_atoms=501,
    policy_frequency=4,
    use_symmetry=False,
    use_tanh=False,
    compile=False,
    amp=False,
  )


def tracking_nse_fastsac_runner_cfg() -> HolosomaFastSACRunnerCfg:
  cfg = tracking_fastsac_runner_cfg()
  cfg.experiment_name = "g1_tracking_fastsac_nse"
  return cfg
