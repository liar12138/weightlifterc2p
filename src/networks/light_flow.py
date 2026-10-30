from rsl_rl.models.mlp_model import *
from rsl_rl.modules import CNN, MLP
import torch.nn as nn
from typing import Any

def get_activation(name):
    if name == "elu":
        return nn.ELU()
    elif name == "relu":
        return nn.ReLU()
    # 可按需扩展
    else:
        raise ValueError(f"Unsupported activation: {name}")
    
class DepthCameraMLP(MLPModel):

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=..., activation="elu",
                obs_normalization=False, distribution_cfg=None, num_res_blocks=2):
        nn.Module.__init__(self)

        # 原有的解析
        self.obs_set = obs_set

        # 归一化器（沿用原有）
        self.com_obs_normalizer = EmpiricalNormalization(obs[obs_set].shape[-1])

        # 分布（不变）
        if distribution_cfg is not None:
            distribution_cfg = dict(distribution_cfg)
            dist_class = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        depth_flatten = obs["depth_camera"].flatten(1)
        depth_embedding_dim = 64

        self.depth_encoder = MLP(
            depth_flatten.shape[-1],
            depth_embedding_dim,
            hidden_dims[:-1],
            last_activation="elu",
        )
        
        # 头部网络（输入维度为 embed_dim）
        self.head = MLP(
            obs[obs_set].shape[-1] + depth_embedding_dim,
            mlp_output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )

        # 初始化分布专用权重
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.head)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs_norm = self.com_obs_normalizer(obs[self.obs_set])
        depth_video = obs["depth_camera"].flatten(1)
        depth_emb = self.depth_encoder(depth_video)

        # 5. 头部输出
        mlp_output = self.head(torch.cat([depth_emb,obs_norm],dim=-1))

        # 6. 分布处理（不变）
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        
        return mlp_output
    
    def update_normalization(self, obs: TensorDict):
        self.com_obs_normalizer.update(obs[self.obs_set])

class LightFlow(MLPModel):

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=...,
                 activation="elu", obs_normalization=False,
                 distribution_cfg=None, num_res_blocks=2):
        nn.Module.__init__(self)

        self.obs_set = obs_set
        self.com_obs_normalizer = EmpiricalNormalization(obs[obs_set].shape[-1])

        # ---- 新增：深度编码器输出维度和RNN ----
        depth_embedding_dim = 64
        # 计算深度展平后的特征维度（假设 obs["depth_camera"] 形状为 [B, T, H, W]）
        depth_flat_dim = obs["depth_camera"].flatten(2).shape[-1]   # H*W
        self.depth_encoder = MLP(
            depth_flat_dim,
            depth_embedding_dim,
            hidden_dims[:-1],
            last_activation="elu",
        )
        # 添加 RNN（GRU 为例，batch_first=True）
        self.rnn = nn.GRU(
            input_size=depth_embedding_dim,
            hidden_size=depth_embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        # ---- 修改头部网络输入维度：新增 a[:,-1], b, a[:,0] 三个 64 维向量 ----
        obs_dim = obs[obs_set].shape[-1]
        head_in_dim = obs_dim + 3 * depth_embedding_dim   # 原来是 obs_dim + depth_embedding_dim


        # 分布部分保持不变
        if distribution_cfg is not None:
            distribution_cfg = dict(distribution_cfg)
            dist_class = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        self.head = MLP(
            head_in_dim,
            mlp_output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.head)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        # 1. 归一化主观测（通常为当前时刻状态）
        obs_norm = self.com_obs_normalizer(obs[self.obs_set])   # [B, obs_dim]

        # 2. 深度图展平并编码
        #    obs["depth_camera"] 形状 [B, T, H, W] → flatten(2) → [B, T, H*W]
        depth_flat = obs["depth_camera"].flatten(2)             # [B, T, H*W]
        a = self.depth_encoder(depth_flat)                      # [B, T, 64]

        # 3. RNN 处理序列
        #    如果提供了 hidden_state 则使用，否则初始化为零
        if hidden_state is None:
            hidden_state = torch.zeros(1, a.size(0), a.size(-1), device=a.device)
        rnn_output, new_hidden = self.rnn(a, hidden_state)      # rnn_output: [B, T, 64], new_hidden: [1, B, 64]
        b = rnn_output[:, -1, :]                                # 取最后一个时间步输出 [B, 64]

        # 4. 拼接特征：a[:,-1] (最后时刻嵌入), b (RNN最后输出), a[:,0] (首时刻嵌入), obs_norm
        combined = torch.cat([a[:, -1, :], b, a[:, 0, :], obs_norm], dim=-1)  # [B, 3*64 + obs_dim]

        # 5. 头部网络
        mlp_output = self.head(combined)

        # 6. 分布处理（不变）
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)

        return mlp_output

    def update_normalization(self, obs: TensorDict):
        self.com_obs_normalizer.update(obs[self.obs_set])

class DepthVideoSkipFrameMLP(MLPModel):

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=...,
                 activation="elu", obs_normalization=False,
                 distribution_cfg=None, num_res_blocks=2):
        nn.Module.__init__(self)

        self.obs_set = obs_set
        self.com_obs_normalizer = EmpiricalNormalization(obs[obs_set].shape[-1])

        # ---- 新增：深度编码器输出维度和RNN ----
        depth_embedding_dim = 64
        # 计算深度展平后的特征维度（假设 obs["depth_camera"] 形状为 [B, T, H, W]）
        depth_flat_dim = obs["depth_camera"].flatten(2).shape[-1]   # H*W
        self.depth_encoder = MLP(
            depth_flat_dim,
            depth_embedding_dim,
            hidden_dims[:-1],
            last_activation="elu",
        )

        # ---- 修改头部网络输入维度：新增 a[:,-1], b, a[:,0] 三个 64 维向量 ----
        obs_dim = obs[obs_set].shape[-1]
        head_in_dim = obs_dim + 3 * depth_embedding_dim   # 原来是 obs_dim + depth_embedding_dim


        # 分布部分保持不变
        if distribution_cfg is not None:
            distribution_cfg = dict(distribution_cfg)
            dist_class = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim
            
        self.head = MLP(
            head_in_dim,
            mlp_output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.head)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        # 1. 归一化主观测（通常为当前时刻状态）
        obs_norm = self.com_obs_normalizer(obs[self.obs_set])   # [B, obs_dim]

        # 2. 深度图展平并编码
        #    obs["depth_camera"] 形状 [B, T, H, W] → flatten(2) → [B, T, H*W]
        depth_flat = obs["depth_camera"].flatten(2)             # [B, T, H*W]
        T = depth_flat.shape[1]
        middle = int(T/2)
        a = self.depth_encoder(depth_flat[:,[0,middle,-1]])                      # [B, T, 64]

        # 4. 拼接特征：a[:,-1] (最后时刻嵌入), b (RNN最后输出), a[:,0] (首时刻嵌入), obs_norm
        combined = torch.cat([a.flatten(1), obs_norm], dim=-1)  # [B, 3*64 + obs_dim]

        # 5. 头部网络
        mlp_output = self.head(combined)

        # 6. 分布处理（不变）
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)

        return mlp_output

    def update_normalization(self, obs: TensorDict):
        self.com_obs_normalizer.update(obs[self.obs_set])