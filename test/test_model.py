import torch

from src.dataset import create_dataloader
from src.models.fatigue_model import FatigueModel


def main():

    print("\n")
    print("=" * 60)
    print("TESTING FATIGUE MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # Load one batch
    # --------------------------------------------------------

    loader = create_dataloader(
        split="train",
        batch_size=4,
        shuffle=True
    )

    batch = next(
        iter(loader)
    )

    acoustic = batch[
        "acoustic"
    ]

    wavlm = batch[
        "wavlm"
    ]

    global_features = batch[
        "global_features"
    ]

    attention_mask = batch[
        "attention_mask"
    ]

    labels = batch[
        "label"
    ]

    print("\nINPUTS")
    print(
        "Acoustic:",
        acoustic.shape
    )

    print(
        "WavLM:",
        wavlm.shape
    )

    print(
        "Global:",
        global_features.shape
    )

    print(
        "Mask:",
        attention_mask.shape
    )

    print(
        "Labels:",
        labels.shape
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FatigueModel(
        num_classes=3
    )

    model.eval()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            acoustic=acoustic,
            wavlm=wavlm,
            global_features=global_features,
            attention_mask=attention_mask
        )

    logits = output[
        "logits"
    ]

    pooled = output[
        "temporal_representation"
    ]

    attention = output[
        "attention_weights"
    ]

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print("\nOUTPUTS")

    print(
        "Logits:",
        logits.shape
    )

    print(
        "Temporal representation:",
        pooled.shape
    )

    print(
        "Attention weights:",
        attention.shape
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    predictions = torch.argmax(
        probabilities,
        dim=-1
    )

    print(
        "\nProbabilities:"
    )

    print(
        probabilities
    )

    print(
        "\nPredictions:"
    )

    print(
        predictions
    )

    print(
        "\nActual labels:"
    )

    print(
        labels
    )

    print("\n")
    print("=" * 60)
    print("MODEL FORWARD PASS SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()