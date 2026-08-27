import torch
import torch.nn as nn


class GlobalFeatureFusion(nn.Module):

    def __init__(
        self,
        temporal_dim=256,
        global_dim=11,
        output_dim=267
    ):
        super().__init__()

        self.temporal_dim = temporal_dim
        self.global_dim = global_dim
        self.output_dim = output_dim

    def forward(
        self,
        temporal_representation,
        global_features
    ):
        """
        temporal_representation:
            [batch, 256]

        global_features:
            [batch, 11]

        Returns:
            [batch, 267]
        """

        if global_features.shape[-1] != self.global_dim:

            raise ValueError(
                f"Expected {self.global_dim} "
                f"global features, got "
                f"{global_features.shape[-1]}"
            )

        fused = torch.cat(
            [
                temporal_representation,
                global_features
            ],
            dim=-1
        )

        return fused