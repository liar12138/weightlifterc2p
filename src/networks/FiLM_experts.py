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

class FiLMResidualBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, activation="elu"):
        super().__init__()
        # 由 cmd_emb 生成 gamma 和 beta
        self.film_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, embed_dim * 2)
        )
        # 主变换：两层线性层 + 激活，保持维度不变
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, embed_dim)
        )
        # 可选的 LayerNorm，稳定训练
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, obs_emb, cmd_emb):
        # 1. 生成调制参数
        gamma_beta = self.film_mlp(cmd_emb)          # [B, 2*embed_dim]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        # 2. FiLM 调制 obs_emb
        modulated = gamma * obs_emb + beta           # [B, embed_dim]

        # 3. 主变换（残差路径）
        delta = self.mlp(modulated)                  # [B, embed_dim]

        # 4. 残差连接 + 层归一化
        out = self.norm(obs_emb + delta)
        return out

class TaskSpecExperts(MLPModel):

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=..., activation="elu",
                obs_normalization=False, distribution_cfg=None, num_res_blocks=4):
        nn.Module.__init__(self)

        # 原有的解析
        self.obs_set = obs_set
        self.vec_obs_dim = obs['vcmd'].shape[-1]
        self.mot_obs_dim = obs['mcmd'].shape[-1]

        # 归一化器（沿用原有）
        self.vec_obs_normalizer = EmpiricalNormalization(self.vec_obs_dim)
        self.mot_obs_normalizer = EmpiricalNormalization(self.mot_obs_dim)
        self.com_obs_normalizer = EmpiricalNormalization(obs[obs_set].shape[-1])

        # 分布（不变）
        if distribution_cfg is not None:
            dist_class = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # 统一的嵌入维度（使用 hidden_dims[-1] 作为特征维度）
        embed_dim = hidden_dims[-2]

        # 嵌入层（将原始输入投影到 embed_dim）
        self.vec_embed = nn.Linear(self.vec_obs_dim, embed_dim)
        self.mot_embed = nn.Linear(self.mot_obs_dim, embed_dim)
        self.com_embed = nn.Linear(obs[obs_set].shape[-1], embed_dim)

        # FiLM 残差块堆叠（N 个）
        self.res_blocks = nn.ModuleList([
            FiLMResidualBlock(embed_dim, hidden_dims[0], activation)
            for _ in range(num_res_blocks)
        ])

        # 头部网络（输入维度为 embed_dim）
        self.head = MLP(embed_dim, mlp_output_dim, (hidden_dims[-1] // 2,), activation)

        # 初始化分布专用权重
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.head)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        # 获取 flag
        mot_ids = obs['task_flag'].squeeze(1) == 0
        vec_ids = obs['task_flag'].squeeze(1) == 1
        batch_size = obs['vcmd'].size(0)

        # 1. 归一化并嵌入命令
        vec_norm = self.vec_obs_normalizer(obs['vcmd'])
        mot_norm = self.mot_obs_normalizer(obs['mcmd'])

        vec_emb = self.vec_embed(vec_norm)   # [B, embed_dim]
        mot_emb = self.mot_embed(mot_norm)   # [B, embed_dim]

        # 2. 根据 flag 合成 cmd_emb
        cmd_emb = torch.zeros(batch_size, vec_emb.size(-1), device=vec_emb.device)
        if vec_ids.int().sum() > 0:
            cmd_emb[vec_ids] = vec_emb[vec_ids]
        if mot_ids.int().sum() > 0:
            cmd_emb[mot_ids] = mot_emb[mot_ids]

        # 3. 归一化并嵌入观测
        obs_norm = self.com_obs_normalizer(obs[self.obs_set])
        obs_emb = self.com_embed(obs_norm)   # [B, embed_dim]

        # 4. 通过多个 FiLM 残差块（每层都注入 cmd_emb）
        for block in self.res_blocks:
            obs_emb = block(obs_emb, cmd_emb)

        # 5. 头部输出
        mlp_output = self.head(obs_emb)

        # 6. 分布处理（不变）
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


