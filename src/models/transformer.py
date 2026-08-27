import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model=256,
        max_len=100
    ):
        super().__init__()

        position = torch.arange(
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            )
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        sequence_length = x.size(1)

        x = x + self.pe[
            :, :sequence_length, :
        ]

        return x


class TemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dim=256,
        num_heads=4,
        num_layers=2,
        feedforward_dim=512,
        dropout=0.1
    ):
        super().__init__()

        self.positional_encoding = PositionalEncoding(
            d_model=input_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x,src_key_padding_mask=None):

        x = self.positional_encoding(x)

        x = self.transformer(x)

        return x

