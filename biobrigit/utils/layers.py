"""
Layers module within the Brigit package.

Contains the architecture of layers used for building the models.

Copyright by Raúl Fernández Díaz
"""
import torch.nn as nn


class conv3D (nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        stride: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding
            )
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.activation(self.conv(x)))


class linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.activation(self.linear(x)))


if __name__ == '__main__':
    help(conv3D)
    help(linear)
