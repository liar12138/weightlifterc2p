import torch
import torch.nn as nn

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


class DepthCNN2X(nn.Module):
    """
    输入: (batch_size, 1, 54, 96)
    输出: (batch_size, 128)
    """
    def __init__(self, channels=64):
        super().__init__()
        
        self.conv_h = nn.Conv2d(
            in_channels=1,
            out_channels=channels,
            kernel_size=(3, 96),
            padding=(1, 0),   # 高度保持，宽度压缩为1
            stride=1
        )
        self.bn_h = nn.BatchNorm2d(channels)
        
        self.conv_v = nn.Conv2d(
            in_channels=1,
            out_channels=channels,
            kernel_size=(54, 3),
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
        # x: (B, 1, 54, 96)
        # 横向分支
        h = self.conv_h(x)          # (B, channels, 54, 1)
        h = self.bn_h(h)
        h = self.relu(h)
        h = self.gap(h)             # (B, channels, 1, 1)
        h = h.view(h.size(0), -1)   # (B, channels)
        
        # 纵向分支
        v = self.conv_v(x)          # (B, channels, 1, 96)
        v = self.bn_v(v)
        v = self.relu(v)
        v = self.gap(v)             # (B, channels, 1, 1)
        v = v.view(v.size(0), -1)   # (B, channels)
        
        # 拼接并输出
        combined = torch.cat([h, v], dim=1)   # (B, 2*channels)
        out = self.fc(combined)               # (B, 32)
        return out


if __name__ == "__main__":
    # 简单测试 DepthCNN2X 的输入输出
    net = DepthCNN2X()

    # 输入: (batch_size, 1, 54, 96)
    x = torch.randn(4, 1, 54, 96)
    print("输入形状:", tuple(x.shape))

    out = net(x)
    print("输出形状:", tuple(out.shape))


