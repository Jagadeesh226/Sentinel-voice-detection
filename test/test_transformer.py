import torch

from src.models.transformer import TemporalTransformer


# -----------------------------------------
# Simulate temporal sequence
# -----------------------------------------

batch_size = 2
num_windows = 8
feature_dim = 256


sequence = torch.randn(
    batch_size,
    num_windows,
    feature_dim
)


# -----------------------------------------
# Transformer
# -----------------------------------------

transformer = TemporalTransformer(
    input_dim=256,
    num_heads=4,
    num_layers=2,
    feedforward_dim=512,
    dropout=0.1
)


output = transformer(
    sequence
)


# -----------------------------------------
# Results
# -----------------------------------------

print("\n==============================")
print("TRANSFORMER")
print("==============================")

print(
    "Input shape:",
    sequence.shape
)

print(
    "Output shape:",
    output.shape
)

print("==============================")