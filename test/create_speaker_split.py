import pandas as pd
import random


INPUT_FILE = "data/dataset_split.csv"
OUTPUT_FILE = "data/dataset_split.csv"

RANDOM_SEED = 42


def create_speaker_groups(
    speakers,
    train_ratio=0.60,
    validation_ratio=0.20
):

    speakers = list(
        speakers
    )

    random.shuffle(
        speakers
    )

    total = len(speakers)

    train_count = max(
        1,
        round(
            total * train_ratio
        )
    )

    validation_count = max(
        1,
        round(
            total * validation_ratio
        )
    )

    # Make sure we don't exceed total
    if (
        train_count +
        validation_count
        >= total
    ):

        validation_count = max(
            1,
            total - train_count - 1
        )

    train_speakers = speakers[
        :train_count
    ]

    validation_speakers = speakers[
        train_count:
        train_count + validation_count
    ]

    test_speakers = speakers[
        train_count + validation_count:
    ]

    return (
        train_speakers,
        validation_speakers,
        test_speakers
    )


def main():

    print("\n")
    print("=" * 65)
    print("SPEAKER-INDEPENDENT DATASET SPLIT")
    print("=" * 65)

    random.seed(
        RANDOM_SEED
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["split"] = ""

    # ========================================================
    # REAL SPEAKERS
    # ========================================================

    real_speakers = (
        df[
            df["source"] == "real"
        ]["speaker_id"]
        .unique()
        .tolist()
    )

    (
        real_train,
        real_validation,
        real_test
    ) = create_speaker_groups(
        real_speakers
    )

    # ========================================================
    # SYNTHETIC SPEAKERS
    # ========================================================

    synthetic_speakers = (
        df[
            df["source"] == "synthetic"
        ]["speaker_id"]
        .unique()
        .tolist()
    )

    (
        synthetic_train,
        synthetic_validation,
        synthetic_test
    ) = create_speaker_groups(
        synthetic_speakers
    )

    # ========================================================
    # ASSIGN REAL SPLITS
    # ========================================================

    df.loc[
        (
            (df["source"] == "real") &
            (
                df["speaker_id"].isin(
                    real_train
                )
            )
        ),
        "split"
    ] = "train"

    df.loc[
        (
            (df["source"] == "real") &
            (
                df["speaker_id"].isin(
                    real_validation
                )
            )
        ),
        "split"
    ] = "validation"

    df.loc[
        (
            (df["source"] == "real") &
            (
                df["speaker_id"].isin(
                    real_test
                )
            )
        ),
        "split"
    ] = "test"

    # ========================================================
    # ASSIGN SYNTHETIC SPLITS
    # ========================================================

    df.loc[
        (
            (df["source"] == "synthetic") &
            (
                df["speaker_id"].isin(
                    synthetic_train
                )
            )
        ),
        "split"
    ] = "train"

    df.loc[
        (
            (df["source"] == "synthetic") &
            (
                df["speaker_id"].isin(
                    synthetic_validation
                )
            )
        ),
        "split"
    ] = "validation"

    df.loc[
        (
            (df["source"] == "synthetic") &
            (
                df["speaker_id"].isin(
                    synthetic_test
                )
            )
        ),
        "split"
    ] = "test"

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # PRINT SPLIT INFORMATION
    # ========================================================

    print("\nREAL SPEAKERS")

    print(
        "Train:",
        real_train
    )

    print(
        "Validation:",
        real_validation
    )

    print(
        "Test:",
        real_test
    )

    print("\nSYNTHETIC SPEAKERS")

    print(
        "Train:",
        synthetic_train
    )

    print(
        "Validation:",
        synthetic_validation
    )

    print(
        "Test:",
        synthetic_test
    )

    # ========================================================
    # RECORDING COUNTS
    # ========================================================

    print("\n")
    print("=" * 65)
    print("RECORDING COUNTS")
    print("=" * 65)

    print(
        df.groupby(
            ["split", "source", "class_name"]
        ).size()
    )

    # ========================================================
    # SPEAKER LEAKAGE CHECK
    # ========================================================

    print("\n")
    print("=" * 65)
    print("SPEAKER LEAKAGE CHECK")
    print("=" * 65)

    for source in [
        "real",
        "synthetic"
    ]:

        split_speakers = {}

        for split in [
            "train",
            "validation",
            "test"
        ]:

            split_speakers[
                split
            ] = set(
                df[
                    (
                        df["source"]
                        == source
                    )
                    &
                    (
                        df["split"]
                        == split
                    )
                ]["speaker_id"]
            )

        train_val = (
            split_speakers["train"]
            &
            split_speakers["validation"]
        )

        train_test = (
            split_speakers["train"]
            &
            split_speakers["test"]
        )

        val_test = (
            split_speakers["validation"]
            &
            split_speakers["test"]
        )

        print(
            f"\n{source.upper()}"
        )

        print(
            "Train ∩ Validation:",
            train_val
        )

        print(
            "Train ∩ Test:",
            train_test
        )

        print(
            "Validation ∩ Test:",
            val_test
        )

    print("\n")
    print("=" * 65)
    print("SPLIT COMPLETE")
    print("=" * 65)


if __name__ == "__main__":

    main()

    