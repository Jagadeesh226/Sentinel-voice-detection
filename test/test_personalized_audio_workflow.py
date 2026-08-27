from test.predict_audio import (
    load_model,
    extract_features,
    predict
)

from src.personalization.workflow import (
    process_audio_for_speaker
)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    audio_path = (
        "data/dataset/real/paired/"
        "alert/real_spk006.wav"
    )

    speaker_id = (
        "audio_workflow_test"
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "PERSONALIZED AUDIO WORKFLOW TEST"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()

    print(
        "Loading fatigue model..."
    )

    model = load_model()

    print(
        "✓ Model loaded"
    )

    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    print()

    print(
        "Extracting audio features..."
    )

    features = extract_features(
        audio_path
    )

    print(
        "✓ Features extracted"
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print()

    print(
        "Running fatigue prediction..."
    )

    prediction, probabilities = predict(
        model,
        features
    )

    print(
        "✓ Prediction completed"
    )

    # --------------------------------------------------------
    # PERSONALIZED WORKFLOW
    # --------------------------------------------------------

    print()

    print(
        "Processing personalized workflow..."
    )

    result = (
        process_audio_for_speaker(
            speaker_id=speaker_id,
            features=features,
            prediction=prediction,
            probabilities=probabilities
        )
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "WORKFLOW RESULT"
    )

    print("=" * 60)

    print()

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    print()

    print("=" * 60)

    print(
        "TEST COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()