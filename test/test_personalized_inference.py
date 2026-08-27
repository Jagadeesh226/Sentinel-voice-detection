from test.predict_audio import (
    load_model
)

from src.personalization.inference import (
    analyze_speaker_audio
)


# ============================================================
# MAIN
# ============================================================

def main():

    audio_path = (
        "data/dataset/real/paired/"
        "alert/real_spk006.wav"
    )

    speaker_id = (
        "real_spk006"
    )

    print()

    print("=" * 60)

    print(
        "SENTINEL PERSONALIZED INFERENCE TEST"
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
    # ANALYZE AUDIO
    # --------------------------------------------------------

    result = analyze_speaker_audio(
        audio_path=audio_path,
        speaker_id=speaker_id,
        model=model
    )

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "FINAL ANALYSIS RESULT"
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