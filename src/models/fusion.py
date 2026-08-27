import torch
import torch.nn as nn


class FeatureFusion(nn.Module):

    def __init__(
        self,
        acoustic_dim=128,
        wavlm_dim=128
    ):
        super().__init__()

        self.output_dim = (
            acoustic_dim + wavlm_dim
        )

    def forward(
        self,
        acoustic_features,
        wavlm_features
    ):
        """
        acoustic_features:
            [batch, time, 128]

        wavlm_features:
            [batch, time, 128]

        Returns:
            [batch, time, 256]
        """

        fused_features = torch.cat(
            [
                acoustic_features,
                wavlm_features
            ],
            dim=-1
        )

        return fused_features