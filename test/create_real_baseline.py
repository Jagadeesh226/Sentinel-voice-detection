import pandas as pd
import torch

from pathlib import Path

from src.personalization.baseline import (
    SpeakerBaseline
)


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_FILE = (
    "data/dataset_split.csv"
)

FEATURE_DIR = Path(
    "data/features"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("REAL SPEAKER BASELINE CREATION")
    print("=" * 60)

    metadata = pd.read_csv(
        METADATA_FILE
    )

    # --------------------------------------------------------
    # Only real + paired + Alert recordings
    # --------------------------------------------------------

    alert_df = metadata[
        (metadata["source"] == "real") &
        (metadata["pairing"] == "paired") &
        (metadata["class_name"] == "alert")
    ].copy()

    print(
        f"\nAlert recordings found: "
        f"{len(alert_df)}"
    )

    if len(alert_df) == 0:

        raise ValueError(
            "No real paired Alert recordings found."
        )

    baseline_manager = SpeakerBaseline()

    # ========================================================
    # CREATE BASELINE FOR EACH SPEAKER
    # ========================================================

    for speaker_id in sorted(
        alert_df["speaker_id"].unique()
    ):

        speaker_df = alert_df[
            alert_df["speaker_id"] == speaker_id
        ]

        feature_records = []

        print("\n")
        print("-" * 60)

        print(
            f"Speaker: {speaker_id}"
        )

        # ----------------------------------------------------
        # Read Alert feature files
        # ----------------------------------------------------

        for _, row in speaker_df.iterrows():

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

                print(
                    f"⚠ Missing feature file:"
                    f"\n  {feature_path}"
                )

                continue

            data = torch.load(
                feature_path,
                map_location="cpu",
                weights_only=False
            )

            global_features = data[
                "global_features"
            ].float()

            # ------------------------------------------------
            # Feature indices
            #
            # [9]  = Words per second
            # [10] = Words per minute
            # ------------------------------------------------

            wps = (
                global_features[9].item()
            )

            wpm = (
                global_features[10].item()
            )

            feature_records.append(
                {
                    "wps": wps,
                    "wpm": wpm
                }
            )

            print(
                f"  {feature_path.name}"
            )

            print(
                f"    WPS: {wps:.4f}"
            )

            print(
                f"    WPM: {wpm:.2f}"
            )

        # ----------------------------------------------------
        # Make sure we have data
        # ----------------------------------------------------

        if not feature_records:

            print(
                "⚠ No usable recordings "
                f"for {speaker_id}"
            )

            continue

        # ----------------------------------------------------
        # Calculate baseline
        # ----------------------------------------------------

        baseline = (
            baseline_manager.calculate_baseline(
                feature_records
            )
        )

        # ----------------------------------------------------
        # Save baseline
        # ----------------------------------------------------

        path = baseline_manager.save(
            speaker_id,
            baseline
        )

        print(
            "\n  BASELINE"
        )

        print(
            f"    WPS: "
            f"{baseline['wps']:.4f}"
        )

        print(
            f"    WPM: "
            f"{baseline['wpm']:.2f}"
        )

        print(
            f"    Recordings: "
            f"{baseline['num_recordings']}"
        )

        print(
            f"\n  ✓ Saved:"
            f"\n    {path}"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("BASELINE CREATION COMPLETE")
    print("=" * 60)

    print(
        "\nBaseline directory:"
        "\n  data/baselines/"
    )


if __name__ == "__main__":

    main()