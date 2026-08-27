import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_FILE = Path(
    "data/dataset_metadata.csv"
)

OUTPUT_FILE = Path(
    "data/dataset_split.csv"
)


RANDOM_STATE = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("========================================")
    print(" SPEAKER-AWARE DATASET SPLIT")
    print("========================================")

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    df = pd.read_csv(
        METADATA_FILE
    )

    # Keep only valid files

    df = df[
        df["valid"] == True
    ].copy()

    print(
        "\nTotal valid recordings:",
        len(df)
    )

    # --------------------------------------------------------
    # Get unique speakers
    # --------------------------------------------------------

    speakers = (
        df["speaker_id"]
        .unique()
        .tolist()
    )

    print(
        "Total speakers:",
        len(speakers)
    )

    # --------------------------------------------------------
    # Speaker-level split
    # --------------------------------------------------------

    train_speakers, temp_speakers = (
        train_test_split(
            speakers,
            test_size=0.30,
            random_state=RANDOM_STATE
        )
    )

    validation_speakers, test_speakers = (
        train_test_split(
            temp_speakers,
            test_size=0.50,
            random_state=RANDOM_STATE
        )
    )

    # --------------------------------------------------------
    # Assign split
    # --------------------------------------------------------

    def assign_split(speaker):

        if speaker in train_speakers:

            return "train"

        elif speaker in validation_speakers:

            return "validation"

        else:

            return "test"

    df["split"] = df[
        "speaker_id"
    ].apply(assign_split)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("========================================")
    print(" SPLIT SUMMARY")
    print("========================================")

    print("\nSpeakers:")

    print(
        "Train:",
        len(train_speakers)
    )

    print(
        "Validation:",
        len(validation_speakers)
    )

    print(
        "Test:",
        len(test_speakers)
    )

    print("\nRecordings:")

    print(
        df["split"].value_counts()
    )

    print("\nClass distribution:")

    print(
        pd.crosstab(
            df["split"],
            df["class_name"]
        )
    )

    print("\nSpeaker assignment:")

    for split, speaker_list in [
        ("train", train_speakers),
        ("validation", validation_speakers),
        ("test", test_speakers)
    ]:

        print(
            f"\n{split.upper()}:"
        )

        for speaker in sorted(
            speaker_list
        ):

            print(
                f"  {speaker}"
            )

    print(
        "\nSaved to:",
        OUTPUT_FILE
    )

    print("========================================")


if __name__ == "__main__":

    main()