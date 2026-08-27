import torch

from src.models.classifier import FatigueClassifier


# -----------------------------------------
# Simulated fused representation
# -----------------------------------------

batch_size = 2

fused_features = torch.randn(
    batch_size,
    267
)


# -----------------------------------------
# Create classifier
# -----------------------------------------

classifier = FatigueClassifier()


# -----------------------------------------
# Forward pass
# -----------------------------------------

logits = classifier(
    fused_features
)


# -----------------------------------------
# Convert to probabilities
# -----------------------------------------

probabilities = torch.softmax(
    logits,
    dim=1
)


predictions = torch.argmax(
    probabilities,
    dim=1
)


# -----------------------------------------
# Results
# -----------------------------------------

print("\n==============================")
print("FATIGUE CLASSIFIER")
print("==============================")

print(
    "Input:",
    fused_features.shape
)

print(
    "Logits:",
    logits.shape
)

print(
    "Probabilities:",
    probabilities.shape
)

print(
    "\nProbabilities:"
)

print(probabilities)

print(
    "\nPredicted class:"
)

print(predictions)

print("==============================")