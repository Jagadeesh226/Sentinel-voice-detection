import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.dataset import (
    create_dataloader,
    calculate_global_statistics
)

from src.models.fatigue_model import FatigueModel


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_PATH = (
    "models/best_fatigue_model_v3.pt"
)

CLASS_NAMES = [
    "Alert",
    "Mild Fatigue",
    "High Fatigue"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("SENTINEL FATIGUE MODEL EVALUATION")
    print("=" * 60)

    print(
        f"\nDevice: {DEVICE}"
    )

    # ========================================================
    # LOAD MODEL CHECKPOINT FIRST
    # ========================================================

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    # ========================================================
    # LOAD NORMALIZATION STATISTICS
    # ========================================================

    if (
        "global_mean" in checkpoint
        and
        "global_std" in checkpoint
    ):

        global_mean = checkpoint[
            "global_mean"
        ]

        global_std = checkpoint[
            "global_std"
        ]

        print(
            "\nUsing normalization statistics "
            "saved with the model."
        )

    else:

        print(
            "\nWARNING:"
            " Model does not contain saved "
            "normalization statistics."
        )

        print(
            "Calculating statistics from "
            "training data..."
        )

        global_mean, global_std = (
            calculate_global_statistics()
        )

    print(
        "\nGlobal mean:"
    )

    print(
        global_mean
    )

    print(
        "\nGlobal std:"
    )

    print(
        global_std
    )

    # ========================================================
    # TEST DATA
    # ========================================================

    test_loader = create_dataloader(
        split="test",
        batch_size=4,
        shuffle=False,
        global_mean=global_mean,
        global_std=global_std
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = FatigueModel(
        num_classes=3
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"\nLoaded model from:"
        f"\n{MODEL_PATH}"
    )

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "validation_loss" in checkpoint:

        print(
            f"Checkpoint validation loss: "
            f"{checkpoint['validation_loss']:.4f}"
        )

    if "validation_accuracy" in checkpoint:

        print(
            f"Checkpoint validation accuracy: "
            f"{checkpoint['validation_accuracy']:.4f}"
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    all_labels = []

    all_predictions = []

    total_loss = 0.0

    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():

        for batch in test_loader:

            acoustic = batch[
                "acoustic"
            ].to(DEVICE)

            wavlm = batch[
                "wavlm"
            ].to(DEVICE)

            global_features = batch[
                "global_features"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "label"
            ].to(DEVICE)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            output = model(
                acoustic=acoustic,
                wavlm=wavlm,
                global_features=global_features,
                attention_mask=attention_mask
            )

            logits = output[
                "logits"
            ]

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            total_samples += (
                labels.size(0)
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    test_loss = (
        total_loss /
        total_samples
    )

    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    # ========================================================
    # PRECISION / RECALL / F1
    # ========================================================

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=[0, 1, 2],
            zero_division=0
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2]
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"\nTest Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # ========================================================
    # PER-CLASS RESULTS
    # ========================================================

    print("\n")
    print("PER-CLASS RESULTS")
    print("-" * 60)

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"\n{class_name}"
        )

        print(
            f"  Precision: "
            f"{precision[i]:.4f}"
        )

        print(
            f"  Recall:    "
            f"{recall[i]:.4f}"
        )

        print(
            f"  F1-score:  "
            f"{f1[i]:.4f}"
        )

        print(
            f"  Samples:   "
            f"{support[i]}"
        )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("\n")
    print("CONFUSION MATRIX")
    print("-" * 60)

    print(
        "Rows    = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print()

    print(
        "                 Predicted"
    )

    print(
        "              Alert  Mild  High"
    )

    for i, row in enumerate(cm):

        print(
            f"Actual "
            f"{CLASS_NAMES[i]:<12}"
            f"{row[0]:>5}"
            f"{row[1]:>6}"
            f"{row[2]:>6}"
        )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n")
    print("CLASSIFICATION REPORT")
    print("-" * 60)

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # ========================================================
    # SAMPLE PREDICTIONS
    # ========================================================

    print("\n")
    print("SAMPLE PREDICTIONS")
    print("-" * 60)

    for i in range(
        len(all_labels)
    ):

        actual = CLASS_NAMES[
            all_labels[i]
        ]

        predicted = CLASS_NAMES[
            all_predictions[i]
        ]

        status = (
            "✓"
            if all_labels[i]
            == all_predictions[i]
            else "✗"
        )

        print(
            f"{status} "
            f"Actual: {actual:<15}"
            f"Predicted: {predicted}"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()