import torch

from src.models.projection import (
    AcousticProjection,
    WavLMProjection
)


# -----------------------------------------
# Simulated acoustic feature
# -----------------------------------------

acoustic_features = torch.randn(
    1,
    40
)


# -----------------------------------------
# Simulated WavLM embedding
# -----------------------------------------

wavlm_features = torch.randn(
    1,
    768
)


# -----------------------------------------
# Create projection models
# -----------------------------------------

acoustic_projection = AcousticProjection()

wavlm_projection = WavLMProjection()


# -----------------------------------------
# Projection
# -----------------------------------------

acoustic_128 = acoustic_projection(
    acoustic_features
)

wavlm_128 = wavlm_projection(
    wavlm_features
)


# -----------------------------------------
# Results
# -----------------------------------------

print("\n==============================")
print("FEATURE PROJECTION")
print("==============================")

print(
    "Acoustic input:",
    acoustic_features.shape
)

print(
    "Acoustic output:",
    acoustic_128.shape
)

print()

print(
    "WavLM input:",
    wavlm_features.shape
)

print(
    "WavLM output:",
    wavlm_128.shape
)

print("==============================")