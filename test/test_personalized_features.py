from test.predict_audio import (
    extract_features
)

from src.personalization.feature_extractor import (
    extract_personalized_features
)


def main():

    audio_path = (
        "/Users/jagadeesh/Desktop/sentinel_voice/data/dataset/real/paired/alert/real_spk006.wav"
    )

    print()

    print("=" * 60)

    print(
        "PERSONALIZED FEATURE EXTRACTION TEST"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # EXTRACT COMPLETE FEATURES
    # --------------------------------------------------------

    features = extract_features(
        audio_path
    )

    # --------------------------------------------------------
    # EXTRACT PERSONALIZED FEATURES
    # --------------------------------------------------------

    personalized_features = (
        extract_personalized_features(
            features
        )
    )

    print()

    print(
        "PERSONALIZED FEATURES"
    )

    print()

    for name, value in (
        personalized_features.items()
    ):

        print(
            f"{name}: {value:.6f}"
        )

    print()

    print("=" * 60)

    print(
        "TEST COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()