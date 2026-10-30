# Xuyang
"""CNN + GRU neural model for depth-camera-based velocity tracking.

This model combines a CNN encoder for depth image frames with a GRU for temporal
reasoning, followed by an MLP head.

Architecture::

    depth [B, T, H, W] ──→ CNN ──→ [B, D_cnn]
    proprio [B, D_prop] ──→ ObsNormalizer ──→ [B, D_prop]
                                                   │
                                        Concat → [B, D_total]
                                                   │
                                        GRU → [B, rnn_hidden_dim]
                                                   │
                                        MLP → output
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, RNN, HiddenState
import copy

class CNNFlowModel(MLPModel):
  """CNN + GRU neural model for depth-camera-based policy learning.

  This model:
  1. Encodes stacked depth frames through a CNN (T frames treated as channels).
  2. Concatenates CNN features with normalized proprioceptive observations.
  3. Passes the combined features through a GRU for temporal reasoning.
  4. Feeds the GRU output through an MLP head to produce the final output.

  The model is recurrent (``is_recurrent = True``) and supports both rollout-mode
  inference (single-step with internal hidden state) and batch-mode training
  (padded sequences with masks).
  """

  is_recurrent: bool = False
  """Whether the model contains a recurrent module."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    cnn_cfg: dict[str, Any] | None = None,
  ) -> None:
    """Initialize the CNN+GRU model.

    Args:
      obs: Observation dictionary from the environment.
      obs_groups: Mapping from observation set names to lists of observation groups.
      obs_set: Which observation set this model processes (e.g. "actor", "critic").
      output_dim: Dimension of the output (action dim for actor, 1 for critic).
      hidden_dims: Hidden dimensions of the MLP head.
      activation: Activation function name.
      obs_normalization: Whether to normalize 1D observations.
      distribution_cfg: Configuration for the output distribution (actor only).
      cnn_cfg: CNN encoder configuration dict. Passed to ``rsl_rl.modules.CNN``.
        Common keys: output_channels, kernel_size, stride, padding, activation,
        global_pool, flatten.
      rnn_type: RNN type, "gru" or "lstm".
      rnn_hidden_dim: Hidden dimension of the RNN.
      rnn_num_layers: Number of RNN layers.
    """
    if cnn_cfg is None:
      raise ValueError("cnn_cfg must be provided for CNNGRUModel.")

    # ── Phase 1: Discover observation group structure ──────────────────
    # Call _get_obs_dim first to identify 1D (proprio) vs 5D (depth) groups.
    # This sets self.obs_groups_2d, self.obs_dims_2d, self.obs_channels_2d.
    self._get_obs_dim(obs, obs_groups, obs_set)

    # ── Phase 2: Pre-build CNN encoder for depth (local, not on self) ──
    # We support one depth group. The temporal dimension (T) is the 2nd dim
    # and treated as input channels of the CNN.
    # IMPORTANT: must NOT assign to self before super().__init__() because
    # CNN is an nn.Module and PyTorch forbids module assignment before
    # Module.__init__(). Store locally and attach after parent init.
    if len(self.obs_groups_2d) != 1:
      raise ValueError(
        f"CNNGRUModel expects exactly one 2D (depth) observation group, "
        f"got {len(self.obs_groups_2d)}: {self.obs_groups_2d}"
      )

    _cnn = CNN(
      input_dim=self.obs_dims_2d[0],
      input_channels=self.obs_channels_2d[0],
      **cnn_cfg,
    )
    if _cnn.output_channels is not None:
      raise ValueError("CNN output must be flattened (set flatten=True in cnn_cfg).")
    self.cnn_latent_dim: int = int(_cnn.output_dim)

    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims,
      activation,
      obs_normalization,
      distribution_cfg,
    )

    # ── Phase 5: Attach CNN and build RNN ──────────────────────────────
    self.cnn = _cnn

  # ═══════════════════════════════════════════════════════════════════════
  # Observation dimension resolution
  # ═══════════════════════════════════════════════════════════════════════

  def _get_obs_dim(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ) -> tuple[list[str], int]:
    """Separate 1D (proprioceptive) and 5D (temporal-depth) observation groups.

    5D observations have shape ``[B, T, H, W, C]`` where T is the number of
    stacked depth frames (from ``history_length``).
    """
    active_obs_groups = obs_groups[obs_set]
    obs_dim_1d = 0
    obs_groups_1d: list[str] = []
    obs_dims_2d: list[tuple[int, int]] = []
    obs_channels_2d: list[int] = []
    obs_groups_2d: list[str] = []

    for obs_group in active_obs_groups:
      shape = obs[obs_group].shape
      if len(shape) == 2:  # [B, D] — 1D proprioceptive
        obs_groups_1d.append(obs_group)
        obs_dim_1d += shape[-1]
      elif len(shape) == 5:  # [B, T, H, W, C] — temporal stacked 2D
        obs_groups_2d.append(obs_group)
        obs_dims_2d.append(tuple(shape[2:4]))  # (H, W)
        obs_channels_2d.append(shape[1])  # T (num frames as channels)
      else:
        raise ValueError(
          f"Unsupported observation shape {shape} for group '{obs_group}'. "
          f"Expected 2D (B, D) or 5D (B, T, H, W, C)."
        )

    if not obs_groups_2d:
      raise ValueError(
        "CNNGRUModel requires at least one 5D (temporal-depth) observation group. "
        "Use MLPModel or RNNModel for 1D-only observations."
      )

    # Store 2D observation metadata for CNN construction
    self.obs_dims_2d = obs_dims_2d
    self.obs_channels_2d = obs_channels_2d
    self.obs_groups_2d = obs_groups_2d

    # Return 1D groups for parent MLPModel
    return obs_groups_1d, obs_dim_1d

  def _get_latent_dim(self) -> int:
    """Return latent dimension consumed by the MLP head (GRU output dim)."""
    return  self.obs_dim + self.cnn_latent_dim

  # ═══════════════════════════════════════════════════════════════════════
  # Forward pass
  # ═══════════════════════════════════════════════════════════════════════

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    """Build the model latent: 1D normalize → CNN encode depth → concat → GRU.

    Handles two modes transparently:
    - **Rollout** (inference): depth is ``[B, T, H, W]`` → 4D → CNN directly.
    - **Batch** (training): depth is ``[seq, batch, T, H, W]`` → 5D →
      the leading two dims are flattened before CNN and restored after.

    Returns:
      Latent tensor of shape ``[B, rnn_hidden_dim]`` (rollout) or
      ``[N, rnn_hidden_dim]`` (batch, after unpad).
    """
    # ── 1D proprioceptive: normalize and concatenate ──────────────────
    latent_1d = super().get_latent(obs)  # [B, D_prop] or [seq, batch, D_prop]

    # ── Depth: squeeze trailing channel, flatten seq+batch for CNN ────
    depth_group = self.obs_groups_2d[0]
    depth = obs[depth_group]  # [B, T, H, W, 1] or [seq, batch, T, H, W, 1]
    depth = depth.squeeze(-1)  # [B, T, H, W] or [seq, batch, T, H, W]

    if depth.dim() == 5:  # Batch / recurrent training mode
      seq_len, batch_size = depth.shape[:2]
      depth_flat = depth.reshape(-1, *depth.shape[2:])  # [seq*batch, T, H, W]
      latent_cnn = self.cnn(depth_flat)  # [seq*batch, D_cnn]
      latent_cnn = latent_cnn.reshape(seq_len, batch_size, -1)  # [seq, batch, D_cnn]
    else:  # Rollout / inference mode
      latent_cnn = self.cnn(depth)  # [B, D_cnn]

    # ── Concatenate and pass through GRU ──────────────────────────────
    latent = torch.cat([latent_1d, latent_cnn], dim=-1)  # [B, D_total] or [seq, batch, D_total]
    return latent

  # ═══════════════════════════════════════════════════════════════════════
  # Export placeholders (CNN+GRU ONNX export not yet implemented)
  # ═══════════════════════════════════════════════════════════════════════

  def as_jit(self) -> "nn.Module":
    """Not yet implemented — CNN+GRU JIT export requires custom handling."""
    raise NotImplementedError(
      "JIT export not yet implemented for CNNGRUModel. "
      "Use .pt checkpoint for deployment."
    )

  # def as_onnx(self, verbose: bool = False) -> "nn.Module":
  #   """Not yet implemented — CNN+GRU ONNX export requires custom handling."""
  #   raise NotImplementedError(
  #     "ONNX export not yet implemented for CNNGRUModel. "
  #     "Use .pt checkpoint for deployment."
  #   )
  def as_onnx(self, verbose: bool = False) -> nn.Module:
    """Return an ONNX-exportable deterministic CNN policy."""
    return _OnnxCNNFlowModel(self, verbose)
  
class _OnnxCNNFlowModel(nn.Module):
  """ONNX-exportable wrapper for CNNFlowModel.

  The exported policy accepts one flattened observation tensor:

      obs = concat(
          proprioception,       # [B, D_prop]
          flattened_depth,      # [B, T * H * W * 1]
      )

  It then restores the depth image to [B, T, H, W], applies the CNN,
  concatenates the CNN feature with normalized proprioception, and runs
  the MLP head.

  This single-input layout is compatible with the existing
  _OnnxMotionModel and export_policy_to_onnx implementation.
  """

  is_recurrent: bool = False

  def __init__(
    self,
    model: CNNFlowModel,
    verbose: bool = False,
  ) -> None:
    super().__init__()

    self.verbose = verbose

    # Copy modules so the export wrapper is independent of the training model.
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.cnn = copy.deepcopy(model.cnn)
    self.mlp = copy.deepcopy(model.mlp)

    # Export deterministic actor output rather than sampling from Gaussian.
    if model.distribution is not None:
      self.deterministic_output = (
        model.distribution.as_deterministic_output_module()
      )
    else:
      self.deterministic_output = nn.Identity()

    if len(model.obs_groups_2d) != 1:
      raise ValueError(
        "_OnnxCNNFlowModel currently supports exactly one depth group, "
        f"got {model.obs_groups_2d}."
      )

    # 1D proprioceptive observation dimension.
    # Actor: 95
    self.obs_dim_1d = int(model.obs_dim)

    # Depth metadata discovered during CNNFlowModel construction.
    # For your current configuration:
    #   frames = 10
    #   height = 96
    #   width  = 54
    self.depth_frames = int(model.obs_channels_2d[0])
    self.depth_height = int(model.obs_dims_2d[0][0])
    self.depth_width = int(model.obs_dims_2d[0][1])

    # The original observation has trailing channel C=1:
    # [B, T, H, W, 1]
    self.depth_channels = 1

    self.depth_size = (
      self.depth_frames
      * self.depth_height
      * self.depth_width
      * self.depth_channels
    )

    # Required by your existing export_policy_to_onnx():
    #
    #   obs = torch.zeros(1, model.policy.input_size)
    #
    # history_length=10:
    #   95 + 10*96*54 = 51935
    self.input_size = self.obs_dim_1d + self.depth_size

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    """Run deterministic CNN policy inference.

    Args:
      obs: Flattened tensor with shape
        [B, D_prop + T*H*W*1].

    Returns:
      Deterministic actions with shape [B, action_dim].
    """
    # Split proprioceptive and depth observations.
    obs_1d = obs[:, : self.obs_dim_1d]
    depth_flat = obs[:, self.obs_dim_1d :]

    # Restore:
    # [B, T*H*W] -> [B, T, H, W]
    #
    # The original trailing channel is 1, so it can be omitted directly.
    depth = depth_flat.reshape(
      -1,
      self.depth_frames,
      self.depth_height,
      self.depth_width,
    )

    # Same computation as CNNFlowModel.get_latent().
    latent_1d = self.obs_normalizer(obs_1d)
    latent_cnn = self.cnn(depth)

    latent = torch.cat(
      [latent_1d, latent_cnn],
      dim=-1,
    )

    mlp_output = self.mlp(latent)

    # For Gaussian actor, export the deterministic mean action.
    return self.deterministic_output(mlp_output)

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    """Return representative inputs for direct ONNX export."""
    return (
      torch.zeros(
        1,
        self.input_size,
        dtype=torch.float32,
      ),
    )

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]