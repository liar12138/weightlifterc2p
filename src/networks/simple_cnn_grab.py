import torch
import torch.nn as nn

from rsl_rl.models.mlp_model import *
from rsl_rl.models.mlp_model import _OnnxMLPModel
class DepthCNN(nn.Module):
    """
    输入: (batch_size, 1, 48, 27)
    输出: (batch_size, 128)
    """
    def __init__(self, channels=64):
        super(DepthCNN, self).__init__()

        # 横向扫描：核 (3,27) 覆盖整个宽度，沿高度方向滑动
        self.conv_h = nn.Conv2d(
            in_channels=1,
            out_channels=channels,
            kernel_size=(3, 27),
            padding=(1, 0),   # 高度保持，宽度压缩为1
            stride=1
        )
        self.bn_h = nn.BatchNorm2d(channels)
        
        # 纵向扫描：核 (48,3) 覆盖整个高度，沿宽度方向滑动
        self.conv_v = nn.Conv2d(
            in_channels=1,
            out_channels=channels,
            kernel_size=(48, 3),
            padding=(0, 1),   # 宽度保持，高度压缩为1
            stride=1
        )
        self.bn_v = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，输出 (channels, 1, 1)
        
        # 将两个分支的特征拼接后线性映射到128维（可选，若直接拼接已是128维）
        self.fc = nn.Sequential(
            nn.Linear(2 * channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.LayerNorm(32)
        )
        
    def forward(self, x):
        # x: (B, 1, 48, 27)     
        if x.shape[-2:] == (27, 48):
            x = x.transpose(-2, -1)
        # 横向分支
        h = self.conv_h(x)          # (B, channels, 48, 1)
        h = self.bn_h(h)
        h = self.relu(h)
        h = self.gap(h)             # (B, channels, 1, 1)
        h = h.view(h.size(0), -1)   # (B, channels)
        
        # 纵向分支
        v = self.conv_v(x)          # (B, channels, 1, 27)
        v = self.bn_v(v)
        v = self.relu(v)
        v = self.gap(v)             # (B, channels, 1, 1)
        v = v.view(v.size(0), -1)   # (B, channels)
        
        # 拼接并输出
        combined = torch.cat([h, v], dim=1)   # (B, 2*channels)
        out = self.fc(combined)               # (B, 128)
        return out

class DepthCNNModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims = (256, 256, 256), activation = "elu", obs_normalization = False, distribution_cfg = None):
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

        self.cnn = DepthCNN()

        # MLP
        self.mlp = MLP(self._get_latent_dim() + 32, mlp_output_dim, hidden_dims, activation)

        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def get_latent(self, obs, masks = None, hidden_state = None):
        latent = super().get_latent(obs, masks, hidden_state)
        depth = self.cnn(obs["depth_camera"])
        return torch.cat([latent,depth],dim=-1)

    def as_onnx(self, verbose):
        return _OnnxCNNMLPModel(self,verbose)


class GrabLightModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims = (256, 256, 256), activation = "elu", obs_normalization = False, distribution_cfg = None):
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

        self.cnn = DepthCNN()
        self.fk_cnn = torch.nn.Parameter(torch.randn(32))   # 或 torch.zeros(32)

        # MLP
        self.mlp = MLP(self._get_latent_dim() + 32, mlp_output_dim, hidden_dims, activation)

        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def get_latent(self, obs, masks = None, hidden_state = None):
        latent = super().get_latent(obs, masks, hidden_state)
        depth_obs = obs["depth_camera"]
        if depth_obs.shape[-2:] == (27, 48):
            depth_obs = depth_obs.transpose(-2, -1)
        depth = self.cnn(depth_obs)
        is_grab = obs["is_grab"].squeeze(-1).bool()
        depth = depth.clone()
        depth[is_grab] = self.fk_cnn
        return torch.cat([latent,depth],dim=-1)

    def as_onnx(self, verbose):
        return _OnnxCNNMLPModel(self,verbose)


class DepthCNNIsGrabModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims = (256, 256, 256), activation = "elu", obs_normalization = False, distribution_cfg = None):
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

        self.cnn = DepthCNN()

        self.fk_cnn = torch.nn.Parameter(torch.randn(32))   # 或 torch.zeros(32)

        # MLP
        self.mlp_tracking = MLP(self._get_latent_dim() + 32, mlp_output_dim, hidden_dims, activation)

        self.mlp_grap = MLP(self._get_latent_dim() + 32, mlp_output_dim, hidden_dims, activation)

        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp_tracking)
            self.distribution.init_mlp_weights(self.mlp_grap)

    def get_latent(self, obs, masks = None, hidden_state = None):
        latent = super().get_latent(obs, masks, hidden_state)
        depth_obs = obs["depth_camera"]
        depth = self.cnn(depth_obs)
        # depth[obs["is_grab"].bool()] = self.fk_cnn
        is_grab = obs["is_grab"].squeeze(-1).bool()
        depth = depth.clone()
        depth[is_grab] = self.fk_cnn
        return torch.cat([latent,depth],dim=-1)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        # 1. 处理 padding（如有）
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        
        # 2. 获取公共 latent（其中已根据 is_grab 修改 depth）
        latent = self.get_latent(obs, masks, hidden_state)  # shape: (envnum, feat_dim)
        
        # 3. 获取 is_grab 掩码（确保为 bool 类型，并扩展维度以便索引）
        is_grab = obs["is_grab"].squeeze(-1).bool()  # shape: (envnum,)
        
        # 4. 分别计算两个 MLP 的输出
        output_tracking = self.mlp_tracking(latent)   # shape: (envnum, out_dim)
        output_grap = self.mlp_grap(latent)           # shape: (envnum, out_dim)
        
        # 5. 根据掩码选择对应输出
        # torch.where 支持广播，但需要条件为 bool 张量，且形状相同
        mlp_output = torch.where(is_grab.unsqueeze(-1), output_grap, output_tracking)
        # 或者使用索引赋值（但需注意梯度连续性）：
        # mlp_output = output_tracking.clone()
        # mlp_output[is_grab] = output_grap[is_grab]
        
        # 6. 若有分布，按需采样
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        mlp_output = torch.where(
            is_grab[:, None],
            output_grap,
            output_tracking,
        )

        return mlp_output

    def as_onnx(self, verbose):
        return _OnnxIsGrab(self,verbose)

class _OnnxCNNMLPModel(_OnnxMLPModel):
    def __init__(self, model, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an MLPModel."""
        super().__init__(model,verbose)
        self.cnn = copy.deepcopy(model.cnn)

    def forward(self, x: torch.Tensor,depth_camera: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        x = self.obs_normalizer(x)
        depth_camera = depth_camera[:,:,::10,::10].clamp(0, 6).div(6).transpose(-2, -1).flip(-1)
        depth = self.cnn(depth_camera)        
        out = self.mlp(torch.cat([x,depth],dim=-1))
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),torch.zeros(1, 1,270,480))

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs","depth_camera"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]

class _OnnxIsGrab(nn.Module):
    def __init__(self, model: DepthCNNIsGrabModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an MLPModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnn = copy.deepcopy(model.cnn)
        self.mlp_grap = copy.deepcopy(model.mlp_grap)
        self.mlp_tracking = copy.deepcopy(model.mlp_tracking)
        self.fk_cnn = copy.deepcopy(model.fk_cnn)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor, depth_camera: torch.Tensor, is_grab: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        depth_camera = depth_camera[:, :, ::10, ::10].clamp(0, 6).div(6)
        # 1. CNN 提取深度特征
        depth = self.cnn(depth_camera)  # (batch, 32)
        
        # 2. 根据 is_grab 替换为 fk_cnn
        is_grab = is_grab.bool().view(-1)  # 确保一维
        depth = torch.where(
            is_grab.unsqueeze(-1), 
            self.fk_cnn.unsqueeze(0).expand(depth.size(0), -1), 
            depth
        )
        
        # 3. 拼接 latent 和 depth 特征
        latent = torch.cat([x, depth], dim=-1)  # (batch, obs_dim + 32)
        
        # 4. 分别通过两个 MLP
        out_tracking = self.mlp_tracking(latent)
        out_grap = self.mlp_grap(latent)
        
        # 5. 根据 is_grab 选择输出
        out = torch.where(is_grab.unsqueeze(-1), out_grap, out_tracking)
        
        # 6. 若有分布则转为确定性输出
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),torch.zeros(1, 1,270,480),torch.zeros(1, 1))

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs","depth_camera","is_grab"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
