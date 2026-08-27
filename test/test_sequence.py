import torch

from src.features.sequence import build_temporal_sequence


# -----------------------------------------
# Simulate 8 temporal windows
# -----------------------------------------

num_windows = 8


acoustic_features = torch.randn(
    num_windows,
    128
)


wavlm_features = torch.randn(
    num_windows,
    128
)


# -----------------------------------------
# Build sequence
# -----------------------------------------

sequence = build_temporal_sequence(
    acoustic_features,
    wavlm_features
)


# -----------------------------------------
# Results
# -----------------------------------------

print("\n==============================")
print("TEMPORAL SEQUENCE")
print("==============================")

print(
    "Acoustic:",
    acoustic_features.shape
)

print(
    "WavLM:",
    wavlm_features.shape
)

print(
    "Sequence:",
    sequence.shape
)

print("==============================")