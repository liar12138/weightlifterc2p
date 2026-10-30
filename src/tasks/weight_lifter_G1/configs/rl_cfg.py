"""FastSAC RL configuration for Unitree G1 Pick-A-Box task."""
from dataclasses import dataclass

from src.rl.fastsac.config import HolosomaFastSACRunnerCfg
from src.rl.fastsac.tasks.mjlab_builtins.rl_cfg import tracking_nse_fastsac_runner_cfg

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_pick_box_fastsac_runner_cfg() -> HolosomaFastSACRunnerCfg:
  """FastSAC runner config derived from NSE tracking (no state estimation)."""
  cfg = tracking_nse_fastsac_runner_cfg()
  cfg.experiment_name = "g1_o6_pullup_barbell_fastsac"
  cfg.max_iterations = 30000
  cfg.save_interval = 5000
  # Pick-box adds a free-body box; 2048 envs OOMs with default buffer_size=4096.
  cfg.buffer_size = 1024
  return cfg

def unitree_g1_cnn_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create CNN+GRU RL runner configuration for Unitree G1 velocity task.

  The actor and critic both use a CNN+GRU architecture:
  - **CNN**: 3-layer conv encoder processes 10-frame depth history as
    input channels, producing a compact spatial feature vector.
  - **GRU**: single-layer GRU (256 hidden) provides temporal memory
    across control steps.
  - **MLP head**: 3-layer MLP (512-256-128) maps the GRU latent to the
    output (29-dim Gaussian action distribution for actor, scalar value
    for critic).
  """
  # Shared CNN encoder configuration.
  # Depth input: [B, T=10, H=480, W=270] → conv → global-avg-pool → 64-dim.
  cnn_cfg = {
    "output_channels": (16, 32, 64),
    "kernel_size": (5, 3, 3),
    "stride": (2, 2, 2),
    "padding": "zeros",
    "activation": "elu",
    "global_pool": "avg",
    "flatten": True,
  }

  model_class = "src.networks.cnn_flow:CNNFlowModel"

  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name=model_class,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      class_name=model_class,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
    ),
    obs_groups={
      "actor": ("actor", "depth"),
      "critic": ("critic", "depth"),
    },
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_o6_pullup_cnn",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=10001,
  )


@dataclass
class CnnGruModelCfg(RslRlModelCfg):
  """Extended model config supporting CNN+GRU architecture.

  Adds recurrent-network parameters on top of the standard MLP/CNN config.
  These are passed through to :class:`CNNGRUModel` via ``**kwargs``.
  """

  rnn_type: str = "gru"
  """RNN type: ``"gru"`` or ``"lstm"``."""
  rnn_hidden_dim: int = 256
  """Hidden dimension of the RNN."""
  rnn_num_layers: int = 1
  """Number of RNN layers."""

def unitree_g1_cnn_gru_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create CNN+GRU RL runner configuration for Unitree G1 velocity task.

  The actor and critic both use a CNN+GRU architecture:
  - **CNN**: 3-layer conv encoder processes 10-frame depth history as
    input channels, producing a compact spatial feature vector.
  - **GRU**: single-layer GRU (256 hidden) provides temporal memory
    across control steps.
  - **MLP head**: 3-layer MLP (512-256-128) maps the GRU latent to the
    output (29-dim Gaussian action distribution for actor, scalar value
    for critic).
  """
  # Shared CNN encoder configuration.
  # Depth input: [B, T=10, H=480, W=270] → conv → global-avg-pool → 64-dim.
  cnn_cfg = {
    "output_channels": (16, 32, 64),
    "kernel_size": (5, 3, 3),
    "stride": (2, 2, 2),
    "padding": "zeros",
    "activation": "elu",
    "global_pool": "avg",
    "flatten": True,
  }

  model_class = "src.networks.cnn_gru_model:CNNGRUModel"

  return RslRlOnPolicyRunnerCfg(
    actor=CnnGruModelCfg(
      class_name=model_class,
      hidden_dims=(128, 64, 32),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      rnn_type="gru",
      rnn_hidden_dim=256,
      rnn_num_layers=1,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=CnnGruModelCfg(
      class_name=model_class,
      hidden_dims=(128, 64, 32),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      rnn_type="gru",
      rnn_hidden_dim=256,
      rnn_num_layers=1,
    ),
    obs_groups={
      "actor": ("actor", "depth_camera"),
      "critic": ("critic", "depth_camera"),
    },
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_o6_pullup_cnn",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=10001,
  )
