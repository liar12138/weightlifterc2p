from dataclasses import dataclass, field
from typing import Literal, Tuple

@dataclass
class FlashSACActorCriticCfg:
  """Config for the FlashSAC actor and critic networks."""
  actor_num_blocks: int = 2
  actor_hidden_dim: int = 128
  actor_bc_alpha: float = 0.0
  actor_noise_zeta_mu: float = 2.0
  actor_noise_zeta_max: float = 16.0

  use_compile: bool = False
  compile_mode: str = "reduce-overhead"
  critic_num_blocks: int = 2
  critic_hidden_dim: int = 256
  critic_num_bins: int = 101
  critic_min_v: float = -5.0 
  critic_max_v: float = 5.0
  critic_target_update_tau: float = 0.01
  temp_initial_value :float = 0.01
  class_name: str = "FlashActorCritic" 

@dataclass
class FlashSACAlgorithmCfg:
  """Config for the FlashSAC algorithm."""
  buffer_max_length: int = 1_000_000
  buffer_min_length: int = 10_000
  sample_batch_size: int = 2048

  actor_update_period:int = 2
  asymmetric_observation : bool =True
  normalized_G_max = 5
  updates_per_interaction_step: int= 2
  normalize_reward: bool = True
  
  learning_rate_init: float = 3e-5
  learning_rate_peak: float = 3e-4
  learning_rate_end: float = 1.5e-4
  learning_rate_warmup_rate: float=1e-6
  learning_rate_decay_rate:float=1.0
  learning_rate_warmup_step: int = 1000 
  learning_rate_decay_step: int = 1000000
  
  #temp_initial_value: float = 0.01
  temp_target_sigma: float = 0.15
  temp_target_entropy: float | None = None 
  gamma: float = 0.99
  n_step: int = 3
  
  #use_compile: bool = True
  use_amp: bool = False
  class_name: str = "src.rl.flashsac.flash_rl.flash_sac.FlashSAC"
  rnd_cfg: dict | None = None

@dataclass
class FlashSACRunnerCfg:
  num_steps_per_env: int = 1
  """The number of steps per environment update."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"policy": ("policy",), "critic": ("critic",)},
  )
  save_interval: int = 2000
  """The number of iterations between saves."""
  log_interval: int = 200
  """The number of iterations between logs."""  
  run_name: str = ""
  """The run name. Default is empty string."""
  logger: Literal["wandb", "tensorboard"] = "tensorboard"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""
  wandb_tags: Tuple[str, ...] = ()
  """Tags for the wandb run. Default is empty tuple."""
  resume: bool = False
  """Whether to resume the experiment. Default is False."""
  
  load_run: str = ".*"
  """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
  load_flow_checkpoint: str = "model_.*.pt"
  clip_actions: float | None = None
  """The clipping range for action values. If None (default), no clipping is applied."""
  """Runner config for FlashSAC (can inherit common settings if you have a base class)."""
  seed: int = 6000
  device_type: str = 'cuda'#cuda 
  buffer_device_type: str = 'cuda'#cuda
  class_name: str = "src.tasks.rl.flash_sac_runner.FlashSACRunner"
  max_iterations: int = 300
  experiment_name: str = "flash_sac_exp"
  buffer_size : int = 1024
  normalized_G_max: float = 5.0
  num_env_steps: int=50_000_896
  policy: FlashSACActorCriticCfg = field(default_factory=FlashSACActorCriticCfg)
  algorithm: FlashSACAlgorithmCfg = field(default_factory=FlashSACAlgorithmCfg)
  load_optimizer: bool = True
  load_reward_normalizer: bool = True