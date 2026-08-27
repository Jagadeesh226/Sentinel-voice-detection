import torch
import torch.nn as nn


class AcousticProjection(nn.Module):

    def __init__(
        self,
        input_dim=40,
        output_dim=128
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):

        return self.projection(x)


class WavLMProjection(nn.Module):

    def __init__(
        self,
        input_dim=768,
        output_dim=128
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):

        return self.projection(x)