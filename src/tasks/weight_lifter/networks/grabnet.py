
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.mlp_model import *
from src.networks.simple_cnn import DepthCNN

class GrabNet(MLPModel):

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims = (256, 256, 256), activation = "elu", obs_normalization = False, distribution_cfg = None):
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg)
        self.depth = DepthCNN()
        self.fake_depth = torch.nn.Parameter(torch.randn((32,)),requires_grad=True)

    def get_latent(self, obs, masks = None, hidden_state = None):
        latent = super().get_latent(obs, masks, hidden_state)
        depth = self.depth(obs['depth_camera'])
        depth = torch.where(obs['graping'],depth,self.fake_depth)
        return torch.cat([latent,depth],dim=-1)

    def _get_latent_dim(self):
        return super()._get_latent_dim() + 32

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the MLP model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Get MLP input latent
        latent = self.get_latent(obs, masks, hidden_state)
        # MLP forward pass
        mlp_output = self.mlp(latent) + obs['actor'][:,95:95 + 43]

        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
    
        return mlp_output

    def load_state_dict(self, state_dict, strict=True):
        """自动分流加载：
        - checkpoint 只有单个 MLP（无 expert./depth./fake_depth 前缀）→ 载入 self.expert
        - checkpoint 含全部权重 → 正常载入整个 ResMLP
        """
        has_full = any(
            k.startswith(('expert.', 'depth.', 'fake_depth'))
            for k in state_dict.keys()
        )
        if not has_full:
            return self.expert.load_state_dict(state_dict, strict)
        return super().load_state_dict(state_dict, strict)

    def as_onnx(self, verbose):
        return _OnnxMLPModel(self,verbose)

class _OnnxMLPModel(nn.Module):
    """Exportable ResMLP model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: GrabNet, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a ResMLP."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer).eval()
        self.mlp = copy.deepcopy(model.mlp)
        self.depth = copy.deepcopy(model.depth).eval()      # DepthCNN 里有 BN，导出要 eval()
        self.fake_depth = copy.deepcopy(model.fake_depth)   # 形状 (32,)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(
        self,
        x: torch.Tensor,           # (B, obs_dim) 未归一化的扁平 obs
        depth_camera: torch.Tensor,  # (B, 1, H, W) 训练里来自 g1 camera: (B,1,27,48)，已归一化 [0,1]
        graping: torch.Tensor,     # (B,) 或 (B,1) bool
    ) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        nx = self.obs_normalizer(x)                  # 与 super().get_latent 一致
        depth = self.depth(depth_camera)             # (B, 32)
        g = graping.bool().view(-1)                  # 确保一维 bool
        depth = torch.where(g.unsqueeze(-1), depth, self.fake_depth.unsqueeze(0))
        out = self.mlp(torch.cat([nx, depth], dim=-1)) + nx[:,95:95 + 43]
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (
            torch.zeros(1, self.input_size),          # obs
            torch.zeros(1, 1, 27, 48),                # depth_camera (B,1,H,W)，与 g1 camera 一致
            torch.ones(1, 1, dtype=torch.bool),       # graping
        )

    @property
    def input_names(self) -> list[str]:
        return ["obs", "depth_camera", "graping"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]