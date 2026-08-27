import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path

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

EPOCHS = 30

BATCH_SIZE = 4

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

MODEL_DIR = Path(
    "models"
)

MODEL_PATH = (
    MODEL_DIR /
    "best_fatigue_model_v3.pt"
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    for batch in loader:

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

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        output = model(
            acoustic=acoustic,
            wavlm=wavlm,
            global_features=global_features,
            attention_mask=attention_mask
        )

        logits = output[
            "logits"
        ]

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += labels.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for batch in loader:

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
            # Forward
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

            # ------------------------------------------------
            # Accuracy
            # ------------------------------------------------

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total += labels.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 60)

    print(
        "SENTINEL FATIGUE MODEL TRAINING"
    )

    print("=" * 60)

    print(
        f"\nDevice: {DEVICE}"
    )

    # ========================================================
    # GLOBAL FEATURE NORMALIZATION
    # ========================================================

    print(
        "\nCalculating global feature statistics..."
    )

    global_mean, global_std = (
        calculate_global_statistics()
    )

    print(
        "\nGlobal feature normalization:"
    )

    print(
        "Mean:"
    )

    print(
        global_mean
    )

    print(
        "\nStd:"
    )

    print(
        global_std
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    print(
        "\nCreating training DataLoader..."
    )

    train_loader = create_dataloader(
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=True,
        global_mean=global_mean,
        global_std=global_std
    )

    print(
        "\nCreating validation DataLoader..."
    )

    validation_loader = create_dataloader(
        split="validation",
        batch_size=BATCH_SIZE,
        shuffle=False,
        global_mean=global_mean,
        global_std=global_std
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nCreating FatigueModel..."
    )

    model = FatigueModel(
        num_classes=3
    )

    model = model.to(
        DEVICE
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    # Current training distribution:
    #
    # Alert         = 11
    # Mild Fatigue  = 9
    # High Fatigue  = 11
    #
    # The imbalance is small.
    #
    # Therefore equal weights are used.

    class_weights = torch.tensor(
        [
            1.0,
            1.0,
            1.0
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # LEARNING RATE SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3
        )
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_validation_loss = float(
        "inf"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        validation_loss, validation_accuracy = (
            validate(
                model,
                validation_loader,
                criterion
            )
        )

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # Print epoch
        # ----------------------------------------------------

        print(
            f"\nEpoch "
            f"{epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.7f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch,

                    "validation_loss":
                        validation_loss,

                    "validation_accuracy":
                        validation_accuracy,

                    "global_mean":
                        global_mean,

                    "global_std":
                        global_std
                },
                MODEL_PATH
            )

            print(
                "✓ Best model saved"
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")

    print("=" * 60)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 60)

    print(
        f"Best model:"
        f" {MODEL_PATH}"
    )

    print(
        f"Best validation loss:"
        f" {best_validation_loss:.4f}"
    )


if __name__ == "__main__":

    main()

