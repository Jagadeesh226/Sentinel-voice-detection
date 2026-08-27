import sys
import torch

from src.personalization.baseline import SpeakerBaseline


BASELINE_DIR = "data/baselines"


def create_baseline(
    speaker_id,
    feature_files
):

    baseline_manager = SpeakerBaseline(
        baseline_dir=BASELINE_DIR
    )

    records = []

    print("\n")
    print("=" * 60)
    print("SPEAKER ENROLLMENT")
    print("=" * 60)

    print(
        f"\nSpeaker: {speaker_id}"
    )

    # --------------------------------------------------------
    # Load Alert recordings
    # --------------------------------------------------------

    for feature_file in feature_files:

        print(
            f"\nLoading: {feature_file}"
        )

        data = torch.load(
            feature_file,
            map_location="cpu",
            weights_only=False
        )

        global_features = data[
            "global_features"
        ].float()

        wps = global_features[
            9
        ].item()

        wpm = global_features[
            10
        ].item()

        records.append(
            {
                "wps": wps,
                "wpm": wpm
            }
        )

        print(
            f"  WPS: {wps:.4f}"
        )

        print(
            f"  WPM: {wpm:.2f}"
        )

    # --------------------------------------------------------
    # Calculate baseline
    # --------------------------------------------------------

    baseline = (
        baseline_manager.calculate_baseline(
            records
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    path = baseline_manager.save(
        speaker_id,
        baseline
    )

    print("\n")
    print("-" * 60)

    print(
        "BASELINE CREATED"
    )

    print(
        f"\nSpeaker: "
        f"{speaker_id}"
    )

    print(
        f"Baseline WPS: "
        f"{baseline['wps']:.4f}"
    )

    print(
        f"Baseline WPM: "
        f"{baseline['wpm']:.2f}"
    )

    print(
        f"Recordings: "
        f"{baseline['num_recordings']}"
    )

    print(
        f"\nSaved to:"
        f"\n{path}"
    )

    print("\n")
    print("=" * 60)
    print("ENROLLMENT COMPLETE")
    print("=" * 60)


def main():

    if len(sys.argv) < 3:

        print(
            "\nUsage:"
        )

        print(
            "\npython -m test.enroll_speaker "
            "SPEAKER_ID ALERT_FEATURE.pt"
        )

        print(
            "\nExample:"
        )

        print(
            "python -m test.enroll_speaker "
            "new_spk001 "
            "data/features/alert_recording.pt"
        )

        sys.exit(1)

    speaker_id = sys.argv[1]

    feature_files = sys.argv[2:]

    create_baseline(
        speaker_id,
        feature_files
    )


if __name__ == "__main__":
    main()

