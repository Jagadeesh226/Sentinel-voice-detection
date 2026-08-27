import torch

from src.models.global_fusion import GlobalFeatureFusion


# -----------------------------------------
# Simulated temporal representation
# -----------------------------------------

temporal_representation = torch.randn(
    2,
    256
)


# -----------------------------------------
# Simulated global features
# -----------------------------------------

global_features = torch.randn(
    2,
    11
)


# -----------------------------------------
# Fusion
# -----------------------------------------

fusion = GlobalFeatureFusion()

fused = fusion(
    temporal_representation,
    global_features
)


# -----------------------------------------
# Results
# -----------------------------------------

print("\n==============================")
print("GLOBAL FEATURE FUSION")
print("==============================")

print(
    "Temporal representation:",
    temporal_representation.shape
)

print(
    "Global features:",
    global_features.shape
)

print(
    "Final representation:",
    fused.shape
)

print("==============================")