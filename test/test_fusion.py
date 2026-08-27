import torch

from src.models.fusion import FeatureFusion


# -----------------------------------------
# Simulated temporal features
# -----------------------------------------

batch_size = 2
num_windows = 5


acoustic_features = torch.randn(
    batch_size,
    num_windows,
    128
)


wavlm_features = torch.randn(
    batch_size,
    num_windows,
    128
)


# -----------------------------------------
# Fusion
# -----------------------------------------

fusion = FeatureFusion()

fused_features = fusion(
    acoustic_features,
    wavlm_features
)


# -----------------------------------------
# Results
# -----------------------------------------

print("\n==============================")
print("FEATURE FUSION")
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
    "Fused:",
    fused_features.shape
)

print("==============================")