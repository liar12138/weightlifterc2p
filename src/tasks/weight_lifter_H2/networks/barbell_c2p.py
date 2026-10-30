"""Depth-history camera-to-proprioception estimator for the H2 barbell task."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DepthHistoryEncoder(nn.Module):
    """CNN matching the project Parkour depth encoder."""

    def __init__(self, history_length: int, image_size: tuple[int, int], embedding_dim: int = 128) -> None:
        super().__init__()
        self.history_length = int(history_length)
        self.image_size = tuple(int(x) for x in image_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.history_length, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        with torch.no_grad():
            example = torch.zeros(1, self.history_length, *self.image_size)
            flattened_dim = int(self.cnn(example).flatten(1).shape[-1])
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 256),
            nn.ELU(),
            nn.Linear(256, embedding_dim),
            nn.ELU(),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim == 5:
            if depth.shape[-1] != 1:
                raise ValueError(f"Expected one depth channel, got {tuple(depth.shape)}")
            depth = depth[..., 0]
        if depth.ndim != 4:
            raise ValueError(f"Expected [B,T,H,W] or [B,T,H,W,1], got {tuple(depth.shape)}")
        expected = (self.history_length, *self.image_size)
        if tuple(depth.shape[1:]) != expected:
            raise ValueError(f"Expected depth tail {expected}, got {tuple(depth.shape[1:])}")
        return self.projection(self.cnn(depth))


class BarbellC2PModel(nn.Module):
    """Predict position, quaternion, axis and both hand-to-plate vectors (16-D)."""

    output_dim = 16

    def __init__(
        self,
        proprio_dim: int,
        depth_shape: tuple[int, ...],
        proprio_embedding_dim: int = 128,
        depth_embedding_dim: int = 128,
        hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        if len(depth_shape) == 3:
            history, height, width = depth_shape
        elif len(depth_shape) == 4 and depth_shape[-1] == 1:
            history, height, width, _ = depth_shape
        else:
            raise ValueError(f"Unsupported depth shape {depth_shape}")
        self.proprio_dim = int(proprio_dim)
        self.depth_shape = (int(history), int(height), int(width))
        self.proprio_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, proprio_embedding_dim),
            nn.LayerNorm(proprio_embedding_dim),
            nn.ELU(),
        )
        self.depth_encoder = DepthHistoryEncoder(
            self.depth_shape[0], self.depth_shape[1:], depth_embedding_dim
        )
        layers: list[nn.Module] = []
        input_dim = proprio_embedding_dim + depth_embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ELU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.output_dim))
        self.head = nn.Sequential(*layers)
        self.register_buffer(
            "target_scale",
            torch.tensor(
                [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0,
                 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                dtype=torch.float32,
            ),
        )

    def encode(self, proprio: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.proprio_encoder(proprio), self.depth_encoder(depth)), dim=-1))

    def normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        return target / self.target_scale

    def denormalize_output(self, output: torch.Tensor) -> torch.Tensor:
        return output * self.target_scale

    def forward(self, proprio: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.denormalize_output(self.encode(proprio, depth))

    def loss(self, proprio: torch.Tensor, depth: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.encode(proprio, depth), self.normalize_target(target))


__all__ = ["DepthHistoryEncoder", "BarbellC2PModel"]
