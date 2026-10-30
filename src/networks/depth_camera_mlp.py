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

class SweetSpot_CNN(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        # 3x3 卷积，步长=2，padding=1：空间 48x27 -> 24x14，通道 1->64
        self.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # 再用 1x1 把通道从 64 提升到 output_dim（此时特征图已很小 24x14）
        self.conv1x1 = nn.Conv2d(32, output_dim, kernel_size=1, bias=False)
        # 最后压成向量
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # x: [4096, 1, 48, 27]
        x = self.conv(x)        # -> [4096, 64, 24, 14]  显存 ≈ 4096*64*24*14*4 = 352 MB
        x = torch.relu(x)
        x = self.conv1x1(x)     # -> [4096, 128, 24, 14] 显存 ≈ 4096*128*24*14*4 = 704 MB (峰值)
        x = torch.relu(x)
        x = self.gap(x)         # -> [4096, 128, 1, 1]
        return x.flatten(1)
    
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

        # depth_flatten = obs["depth_camera"].flatten(1)
        depth_embedding_dim = 128
        
        # self.depth_encoder = MLP(
        #     depth_flatten.shape[-1],
        #     depth_embedding_dim,
        #     hidden_dims[:-1],
        #     last_activation="elu",
        # )

        input_width, input_height = obs["depth_camera"].shape[2:]
        input_channels = 1

        self.depth_encoder = SweetSpot_CNN(depth_embedding_dim)

        # self.depth_encoder = CNN(
        #     input_dim=(input_height, input_width),          # (高, 宽)
        #     input_channels=input_channels,
        #     output_channels=(16, 32, 64),                 # 三个卷积层，最终输出通道为128
        #     kernel_size=3,                                 # 所有层卷积核大小均为3
        #     stride=1,                                      # 所有层步长为1
        #     padding='zeros',                               # 自动计算padding保持尺寸
        #     norm='none',                                   # 不使用归一化
        #     activation='elu',                              # 激活函数
        #     max_pool=(False, True, False),                 # 只在第二层后加最大池化（下采样）
        #     global_pool='avg',                             # 全局平均池化，压缩为1x1
        #     flatten=True                                   # 展平，得到128维
        # )

        self.rnn = RNN(self.obs_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
                
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
        depth_video = obs["depth_camera"]
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
