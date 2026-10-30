from rsl_rl.models.mlp_model import *

import torch.nn as nn

def get_activation(name):
    if name == "elu":
        return nn.ELU()
    elif name == "relu":
        return nn.ReLU()
    # 可按需扩展
    else:
        raise ValueError(f"Unsupported activation: {name}")

class ResidualBlock(nn.Module):
    """残差块：两个线性层 + 跳跃连接，保持维度不变"""
    def __init__(self, dim, activation="elu"):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = get_activation(activation)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out += residual
        out = self.act(out)   # 可选，增强非线性
        return out

class ResidualEncoder(nn.Module):
    """嵌入层 + 多个残差块，输出维度 = embed_dim"""
    def __init__(self, input_dim, embed_dim, activation="elu", num_res_blocks=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(embed_dim, activation) for _ in range(num_res_blocks)]
        )

    def forward(self, x):
        x = self.embedding(x)
        for block in self.res_blocks:
            x = block(x)
        return x

class FiLMLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, activation="elu"):
        super().__init__()
        # 从命令特征生成 gamma 和 beta，输出维度为 2 * embed_dim
        self.mlp = MLP(embed_dim, embed_dim * 2, [hidden_dim], activation)

    def forward(self, obs_feat, cmd):
        gamma_beta = self.mlp(cmd)          # [B, 2*embed_dim]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return gamma * obs_feat + beta      # 仿射变换

class TaskSpecExperts(MLPModel):

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims = ..., activation = "elu", obs_normalization = False, distribution_cfg = None):

        nn.Module.__init__(self)

        # Resolve observation groups and dimensions
        # self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_set = obs_set
        self.vec_obs_dim = obs['vcmd'].shape[-1]
        self.mot_obs_dim = obs['mcmd'].shape[-1]

        # Observation normalization
        self.vec_obs_normalizer = EmpiricalNormalization(self.vec_obs_dim)
        self.mot_obs_normalizer = EmpiricalNormalization(self.mot_obs_dim)
        self.com_obs_normalizer = EmpiricalNormalization(obs[obs_set].shape[-1])

        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        embed_dim = hidden_dims[-1]   # 特征维度
        num_res_blocks = 2            # 可根据需要调整

        self.vec_encoder = ResidualEncoder(self.vec_obs_dim, embed_dim, activation, num_res_blocks)
        self.mot_encoder = ResidualEncoder(self.mot_obs_dim, embed_dim, activation, num_res_blocks)
        self.com_encoder = ResidualEncoder(obs[obs_set].shape[-1], embed_dim * 2, activation, num_res_blocks)
        self.com_head = MLP(embed_dim * 2,embed_dim,(embed_dim * 2 // 3,))
        self.film = FiLMLayer(embed_dim, hidden_dims[0], activation)
        self.head = MLP(hidden_dims[-1], mlp_output_dim, hidden_dims , activation)

        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.head)

    def forward(self, obs, masks = None, hidden_state = None, stochastic_output = False):
        mot_ids = obs['task_flag'].squeeze(1) == 0
        vec_ids = obs['task_flag'].squeeze(1) == 1

        vec_cmd = self.vec_encoder(self.vec_obs_normalizer(obs['vcmd'][vec_ids]))

        cmd = torch.zeros((obs.size(0),vec_cmd.size(-1)),device=vec_cmd.device)

        if vec_ids.int().sum() > 0:
            cmd[vec_ids] = vec_cmd

        if mot_ids.int().sum() > 0:
            mot_cmd = self.mot_encoder(self.mot_obs_normalizer(obs['mcmd'][mot_ids]))
            cmd[mot_ids] = mot_cmd

        obs_feat = self.com_encoder(self.com_obs_normalizer(obs[self.obs_set]))
        
        fused_feat = self.film(self.com_head(obs_feat), cmd)

        mlp_output = self.head(fused_feat)

        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        
        return mlp_output
    
    def update_normalization(self, obs: TensorDict):
        self.vec_obs_normalizer.update(obs['vcmd'])
        self.mot_obs_normalizer.update(obs['mcmd'])
        self.com_obs_normalizer.update(obs[self.obs_set])


