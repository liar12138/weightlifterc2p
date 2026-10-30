
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.mlp_model import *
from src.networks.simple_cnn import DepthCNN,DepthCNN2X

class ResMLP(MLPModel):

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims = (256, 256, 256), activation = "elu", obs_normalization = False, distribution_cfg = None):
        dcf = copy.deepcopy(distribution_cfg)
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg)
        self.expert = MLPModel(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, dcf)
        self.expert.load_state_dict(torch.load('/home/tcs/8t01/humanoid/weightlifter/src/tasks/weight_lifter_H2/networks/experts/model_14900.pt')['actor_state_dict'])
        for pm in self.expert.parameters():
            pm.requires_grad = True
        self.depth = DepthCNN()
        self.fake_depth = torch.nn.Parameter(torch.randn((32,)),requires_grad=True)
        self.layer_norm = torch.nn.LayerNorm(53)

    def get_latent(self, obs, masks = None, hidden_state = None):
        latent = super().get_latent(obs, masks, hidden_state)
        depth = self.depth(obs['depth_camera'])
        with torch.no_grad():
            eo = self.expert(obs,masks,hidden_state)
        depth = torch.where(obs['graping'],depth,self.fake_depth)
        return torch.cat([latent,depth],dim=-1), eo

    def _get_latent_dim(self):
        return super()._get_latent_dim() + 32 + 53 

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
        latent,expert_output = self.get_latent(obs, masks, hidden_state)
        # MLP forward pass
        mlp_output = self.mlp(torch.cat([latent, self.layer_norm(expert_output)],dim=-1)) + expert_output

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
        return _OnnxResMLPModel(self,verbose)

class ResMLP2X(ResMLP):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256, 256, 256), activation="elu", obs_normalization=False, distribution_cfg=None):
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg)
        del self.depth
        self.depth = DepthCNN2X()

class _OnnxResMLPModel(nn.Module):
    """Exportable ResMLP model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: ResMLP, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a ResMLP."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer).eval()
        self.mlp = copy.deepcopy(model.mlp)
        # 改成 ONNX 版扁平 obs 模型，不要深拷贝完整 MLPModel
        self.expert = model.expert.as_onnx(verbose).eval()   # _OnnxMLPModel
        self.layer_norm = copy.deepcopy(model.layer_norm).eval()   # 补上 LayerNorm(43)
        self.depth = copy.deepcopy(model.depth).eval()
        self.fake_depth = copy.deepcopy(model.fake_depth)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x, depth_camera, time_step):
        nx = self.obs_normalizer(x)
        eo = self.expert(x)                       # 现在 expert 接受裸 tensor 了
        depth = self.depth(depth_camera.div(6))
        graping = (time_step > 180) & (time_step < 310)
        g = graping.bool().view(-1)
        depth = torch.where(g.unsqueeze(-1), depth, self.fake_depth.unsqueeze(0))
        # 和训练一致：eo 先过 layer_norm 再 cat，残差加的仍是原始 eo
        out = self.mlp(torch.cat([nx, depth, self.layer_norm(eo)], dim=-1)) + eo
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (
            torch.zeros(1, self.input_size),          # obs
            torch.zeros(1, 1, 27, 48),                # depth_camera (B,1,H,W)，与 g1 camera 一致
            #torch.zeros(1, 1, 54, 96), 
            torch.ones(1, 1, dtype=torch.float),       # graping
        )

    @property
    def input_names(self) -> list[str]:
        return ["obs", "depth_camera", "time_step"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]