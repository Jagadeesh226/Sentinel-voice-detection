import torch
import torch.nn as nn


class FatigueClassifier(nn.Module):

    def __init__(
        self,
        input_dim=267,
        hidden_dim1=128,
        hidden_dim2=64,
        num_classes=3,
        dropout=0.3
    ):
        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim1
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim1,
                hidden_dim2
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim2,
                num_classes
            )
        )

    def forward(self, x):

        logits = self.classifier(x)

        return logits