import torch
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.models.fatigue_model import FatigueModel
from src.personalization.baseline import SpeakerBaseline


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

METADATA_FILE = (
    "data/dataset_split.csv"
)

FEATURE_DIR = Path(
    "data/features"
)

BASELINE_DIR = (
    "data/baselines"
)

CLASS_NAMES = [
    "Alert",
    "Mild Fatigue",
    "High Fatigue"
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\nLoading fatigue model..."
    )

    model = FatigueModel(
        num_classes=3
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "✓ Model loaded"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint.get('validation_loss'):.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{checkpoint.get('validation_accuracy'):.4f}"
    )

    return model, checkpoint


# ============================================================
# LOAD FEATURE FILE
# ============================================================

def load_features(row):

    feature_file = (
        f"{row['source']}_"
        f"{row['pairing']}_"
        f"{row['speaker_id']}_"
        f"{row['class_name']}.pt"
    )

    feature_path = (
        FEATURE_DIR /
        feature_file
    )

    if not feature_path.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{feature_path}"
        )

    data = torch.load(
        feature_path,
        map_location="cpu",
        weights_only=False
    )

    acoustic = data[
        "acoustic_features"
    ].float()

    wavlm = data[
        "wavlm_features"
    ].float()

    global_features = data[
        "global_features"
    ].float()

    return (
        acoustic,
        wavlm,
        global_features
    )


# ============================================================
# PREDICT ONE SAMPLE
# ============================================================

def predict_sample(
    model,
    acoustic,
    wavlm,
    global_features,
    baseline,
    global_mean,
    global_std
):

    # --------------------------------------------------------
    # Current WPM
    # --------------------------------------------------------

    current_wpm = (
        global_features[10].item()
    )

    # --------------------------------------------------------
    # Relative speech rate
    # --------------------------------------------------------

    baseline_wpm = baseline[
        "wpm"
    ]

    if baseline_wpm <= 0:

        raise ValueError(
            "Baseline WPM must be greater than zero."
        )

    relative_speech_rate = (
        current_wpm /
        baseline_wpm
    )

    # --------------------------------------------------------
    # Append relative speech rate
    #
    # Original:
    # 11 features
    #
    # New:
    # 12 features
    # --------------------------------------------------------

    relative_tensor = torch.tensor(
        [relative_speech_rate],
        dtype=torch.float32
    )

    global_features = torch.cat(
        [
            global_features,
            relative_tensor
        ],
        dim=0
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------
    global_mean=global_mean.to(
        global_features.device
    )
    global_std=global_std.to(
        global_features.device
    )

    global_features = (
        global_features - global_mean
    ) / global_std

    # --------------------------------------------------------
    # Batch dimension
    # --------------------------------------------------------

    acoustic = acoustic.unsqueeze(
        0
    )

    wavlm = wavlm.unsqueeze(
        0
    )

    global_features = (
        global_features.unsqueeze(0)
    )

    sequence_length = (
        acoustic.shape[1]
    )

    attention_mask = torch.ones(
        1,
        sequence_length,
        dtype=torch.bool
    )

    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    acoustic = acoustic.to(
        DEVICE
    )

    wavlm = wavlm.to(
        DEVICE
    )

    global_features = (
        global_features.to(DEVICE)
    )

    attention_mask = (
        attention_mask.to(DEVICE)
    )

    # --------------------------------------------------------
    # Prediction
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

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]

        prediction = torch.argmax(
            probabilities
        ).item()

    confidence = (
        probabilities[prediction]
        .item()
    )

    return (
        prediction,
        probabilities.cpu().numpy(),
        confidence,
        relative_speech_rate,
        current_wpm,
        baseline_wpm
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("PERSONALIZED REAL-SPEAKER EVALUATION")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(
        METADATA_FILE
    )

    # --------------------------------------------------------
    # Select paired REAL recordings only
    # --------------------------------------------------------

    evaluation_df = metadata[
        (metadata["source"] == "real") &
        (metadata["pairing"] == "paired")
    ].copy()

    evaluation_df = (
        evaluation_df
        .sort_values(
            [
                "speaker_id",
                "label"
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"\nReal paired recordings: "
        f"{len(evaluation_df)}"
    )

    print(
        f"Speakers: "
        f"{evaluation_df['speaker_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Load normalization statistics
    # --------------------------------------------------------

    print(
        "\nLoading normalization statistics..."
    )

    global_mean = None
    global_std = None

    # --------------------------------------------------------
    # Load model checkpoint
    # --------------------------------------------------------

    model, checkpoint = load_model()

    # --------------------------------------------------------
    # Checkpoint normalization
    # --------------------------------------------------------

    if (
        "global_mean" not in checkpoint or
        "global_std" not in checkpoint
    ):

        raise RuntimeError(
            "Normalization statistics "
            "are missing from checkpoint."
        )

    global_mean = checkpoint[
        "global_mean"
    ].float()

    global_std = checkpoint[
        "global_std"
    ].float()

    # Prevent zero division

    global_std = torch.where(
        global_std < 1e-8,
        torch.ones_like(global_std),
        global_std
    )

    print(
        f"Normalization dimensions: "
        f"{global_mean.shape[0]}"
    )

    # --------------------------------------------------------
    # Baseline manager
    # --------------------------------------------------------

    baseline_manager = (
        SpeakerBaseline(
            baseline_dir=BASELINE_DIR
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    all_labels = []
    all_predictions = []

    # ========================================================
    # EVALUATE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SAMPLE RESULTS")
    print("=" * 70)

    for _, row in evaluation_df.iterrows():

        speaker_id = row[
            "speaker_id"
        ]

        actual_label = int(
            row["label"]
        )

        # ----------------------------------------------------
        # Load speaker baseline
        # ----------------------------------------------------

        baseline = (
            baseline_manager.load(
                speaker_id
            )
        )

        if baseline is None:

            print(
                f"\n⚠ No baseline for "
                f"{speaker_id}"
            )

            continue

        # ----------------------------------------------------
        # Load features
        # ----------------------------------------------------

        acoustic, wavlm, global_features = (
            load_features(row)
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        (
            prediction,
            probabilities,
            confidence,
            relative_rate,
            current_wpm,
            baseline_wpm
        ) = predict_sample(
            model=model,
            acoustic=acoustic,
            wavlm=wavlm,
            global_features=global_features,
            baseline=baseline,
            global_mean=global_mean,
            global_std=global_std
        )

        correct = (
            actual_label ==
            prediction
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results.append(
            {
                "speaker": speaker_id,
                "actual": CLASS_NAMES[
                    actual_label
                ],
                "predicted": CLASS_NAMES[
                    prediction
                ],
                "current_wpm": current_wpm,
                "baseline_wpm": baseline_wpm,
                "relative_rate": relative_rate,
                "confidence": confidence,
                "correct": correct
            }
        )

        all_labels.append(
            actual_label
        )

        all_predictions.append(
            prediction
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        status = (
            "✓"
            if correct
            else "✗"
        )

        print(
            f"\n{status} "
            f"{speaker_id:<12}"
            f"Actual: "
            f"{CLASS_NAMES[actual_label]:<15}"
            f"Predicted: "
            f"{CLASS_NAMES[prediction]}"
        )

        print(
            f"   WPM: "
            f"{current_wpm:.2f}"
            f" | Baseline: "
            f"{baseline_wpm:.2f}"
            f" | Relative: "
            f"{relative_rate:.4f}"
            f" | Confidence: "
            f"{confidence * 100:.2f}%"
        )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    if len(result_df) == 0:

        raise RuntimeError(
            "No samples were evaluated."
        )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=[0, 1, 2],
            zero_division=0
        )
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2]
    )

    print("\n")
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"\nSamples evaluated: "
        f"{len(all_labels)}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # ========================================================
    # PER CLASS
    # ========================================================

    print("\n")
    print("PER-CLASS RESULTS")
    print("-" * 70)

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
    print("-" * 70)

    print(
        "                 Predicted"
    )

    print(
        "              Alert  Mild  High"
    )

    for i, row_cm in enumerate(cm):

        print(
            f"Actual "
            f"{CLASS_NAMES[i]:<12}"
            f"{row_cm[0]:>5}"
            f"{row_cm[1]:>6}"
            f"{row_cm[2]:>6}"
        )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n")
    print("CLASSIFICATION REPORT")
    print("-" * 70)

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
    # PER SPEAKER
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PER-SPEAKER RESULTS")
    print("=" * 70)

    speaker_results = (
        result_df
        .groupby("speaker")
        .agg(
            samples=("correct", "count"),
            accuracy=("correct", "mean"),
            mean_relative_rate=(
                "relative_rate",
                "mean"
            )
        )
    )

    speaker_results[
        "accuracy"
    ] *= 100

    print(
        speaker_results.to_string(
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    # ========================================================
    # RELATIVE SPEECH RATE BY CLASS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RELATIVE SPEECH RATE BY CLASS")
    print("=" * 70)

    class_rate = (
        result_df
        .assign(
            class_name=result_df["actual"]
        )
        .groupby("class_name")
        ["relative_rate"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max"
            ]
        )
    )

    print(
        class_rate.to_string(
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_path = Path(
        "data/personalized_evaluation.csv"
    )

    result_df.to_csv(
        output_path,
        index=False
    )

    print("\n")
    print(
        f"✓ Detailed results saved to:"
        f"\n  {output_path}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "PERSONALIZED EVALUATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()