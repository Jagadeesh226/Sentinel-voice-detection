import torch
import torch.nn as nn


class AttentionPooling(nn.Module):

    def __init__(self, input_dim=256):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x:
            [batch, time, 256]

        Returns:
            pooled:
                [batch, 256]

            attention_weights:
                [batch, time]
        """

        # Calculate attention score for every
        # temporal window

        scores = self.attention(x)

        # [batch, time, 1]
        scores = scores.squeeze(-1)

        # Convert scores to probabilities

        weights = torch.softmax(
            scores,
            dim=1
        )

        # [batch, time] → [batch, time, 1]

        weights = weights.unsqueeze(-1)

        # Weighted sum across time

        pooled = torch.sum(
            x * weights,
            dim=1
        )

        return pooled, weights.squeeze(-1)