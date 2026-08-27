import torch

from src.models.attention import AttentionPooling


# -----------------------------------------
# Simulated Transformer output
# -----------------------------------------

batch_size = 2
num_windows = 8
feature_dim = 256


transformer_output = torch.randn(
    batch_size,
    num_windows,
    feature_dim
)


# -----------------------------------------
# Attention pooling
# -----------------------------------------

pooling = AttentionPooling(
    input_dim=256
)


pooled, weights = pooling(
    transformer_output
)


# -----------------------------------------
# Results
# -----------------------------------------

print("\n==============================")
print("ATTENTION POOLING")
print("==============================")

print(
    "Transformer output:",
    transformer_output.shape
)

print(
    "Pooled representation:",
    pooled.shape
)

print(
    "Attention weights:",
    weights.shape
)

print(
    "\nWeights for first recording:"
)

print(
    weights[0]
)

print(
    "\nSum of weights:",
    weights[0].sum().item()
)

print("==============================")