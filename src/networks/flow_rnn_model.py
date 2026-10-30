from rsl_rl.models.rnn_model import *
from rsl_rl.modules.distribution import Distribution
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.utils import resolve_callable, unpad_trajectories

class FlowRNN(RNNModel):

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims = (128, 128, 128), activation = "elu", obs_normalization = False, distribution_cfg = None, rnn_type = "lstm", rnn_hidden_dim = 256, rnn_num_layers = 1):
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg, rnn_type, rnn_hidden_dim, rnn_num_layers)
        del self.rnn
        depth_camera = obs['depth_camera'].flatten(-3)
        self.rnn = RNN(self.obs_dim + 128, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.depth_camera_encoder = nn.Sequential(nn.Linear(depth_camera.shape[-1],128))

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by passing normalized observation groups through the RNN."""
        # Extract and concatenate observation groups and normalize
        obs_norm = MLPModel.get_latent(self,obs)
        # Pass through the RNN
        latent = self.rnn(torch.cat([obs_norm,self.depth_camera_encoder(obs["depth_camera"].flatten(-3))],dim = -1), masks, hidden_state).squeeze(0)
        return latent

import torch
import torch.nn as nn

class LightFlow(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256, 256, 256),
                 activation="elu", obs_normalization=False, distribution_cfg=None):
        nn.Module.__init__(self)

        # Resolve observation groups and dimensions
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # 深度编码器：将 (batch, T, 48, 27) 映射为 (batch, T, 128)
        # 首先展平空间维度，然后通过线性层和激活函数
        act_fn = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid
        }.get(activation, nn.ELU)  # 默认使用 ELU

        self.depth_encoder = nn.Sequential(
            nn.Flatten(start_dim=-3),          # (batch, T, 48*27)
            nn.Linear(48 * 27, 768),
            act_fn(),
            nn.Linear(768,256),
            act_fn(),
            nn.Linear(256,128),
            act_fn(),
        )

        # RNN 模块：将深度序列 (batch, T, 128) 汇总为单个向量 (batch, 128)
        # 使用 GRU，取最后一个时间步的输出
        self.rnn = nn.GRU(input_size=128, hidden_size=128, batch_first=True, num_layers=3)

        self.low_proj = nn.Sequential(
            nn.Linear(128 * 3, 128),
            act_fn()
        )

        # MLP
        self.mlp = MLP(obs[obs_set].shape[-1] + 128 , mlp_output_dim, hidden_dims, activation)

        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def get_latent(self, obs, masks=None, hidden_state=None):
        # 1. 父类处理其他观测，返回形状为 (batch, D_other) 或 (batch, T, D_other)
        # Select and concatenate observations
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        # Normalize observations
        norm_obs = self.obs_normalizer(latent)

        # 2. 深度图像编码，输出 (batch, T, 128)
        depth_camera = self.depth_encoder(1.0 - obs["depth_camera"][:,::2])   # 注意 depth_encoder 内部已包含展平

        # 3. RNN 处理深度序列，输出所有时间步的隐藏状态 (batch, T, 128)
        rnn_output, _ = self.rnn(depth_camera, hidden_state)    # hidden_state 可为 None

        # 4. 取最后一个时间步的特征
        # 如果 norm_obs 包含时间维度，则取最后一个时间步；否则直接使用
        if norm_obs.dim() == 3:          # (batch, T, D)
            norm_obs = norm_obs[:, -1, :]
        rnn_last = rnn_output[:, -1, :]       # (batch, 128)

        prj = self.low_proj(torch.cat([depth_camera[:, -1, :], rnn_last , depth_camera[:, 0, :]],dim=-1))

        # 5. 拼接所有特征作为最终的潜在表示
        return torch.cat([norm_obs, prj], dim=-1)


class LightFrame(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256, 256, 256),
                 activation="elu", obs_normalization=False, distribution_cfg=None):
        nn.Module.__init__(self)

        # Resolve observation groups and dimensions
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # 深度编码器：将 (batch, T, 48, 27) 映射为 (batch, T, 128)
        # 首先展平空间维度，然后通过线性层和激活函数
        act_fn = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid
        }.get(activation, nn.ELU)  # 默认使用 ELU

        self.depth_encoder = nn.Sequential(
            nn.Flatten(start_dim=-3),          # (batch, T, 48*27)
            nn.Linear(48 * 27, 768),
            act_fn(),
            nn.Linear(768,256),
            act_fn(),
            nn.Linear(256,128),
            act_fn(),
        )

        self.low_proj = nn.Sequential(
            nn.Linear(128 * 3, 128),
            act_fn()
        )

        # MLP
        self.mlp = MLP(obs[obs_set].shape[-1] + 128 , mlp_output_dim, hidden_dims, activation)

        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def get_latent(self, obs, masks=None, hidden_state=None):
        # 1. 父类处理其他观测，返回形状为 (batch, D_other) 或 (batch, T, D_other)
        # Select and concatenate observations
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        # Normalize observations
        norm_obs = self.obs_normalizer(latent)

        # 2. 深度图像编码，输出 (batch, T, 128)
        depth_camera = self.depth_encoder(1.0 - obs["depth_camera"][:,[0,obs["depth_camera"].shape[1]//2,-1]])   # 注意 depth_encoder 内部已包含展平

        # 3. RNN 处理深度序列，输出所有时间步的隐藏状态 (batch, T, 128)
        # rnn_output, _ = self.rnn(depth_camera, hidden_state)    # hidden_state 可为 None

        # 4. 取最后一个时间步的特征
        # 如果 norm_obs 包含时间维度，则取最后一个时间步；否则直接使用
        # if norm_obs.dim() == 3:          # (batch, T, D)
        #     norm_obs = norm_obs[:, -1, :]
        # rnn_last = rnn_output[:, -1, :]       # (batch, 128)

        prj = self.low_proj(depth_camera.flatten(1))

        # 5. 拼接所有特征作为最终的潜在表示
        return torch.cat([norm_obs, prj], dim=-1)