import pandas as pd
import soundfile as sf

from pathlib import Path


DATASET_DIR = Path(
    "data/dataset"
)

OUTPUT_FILE = (
    "data/dataset_split.csv"
)


CLASS_LABELS = {
    "alert": 0,
    "mild_fatigue": 1,
    "high_fatigue": 2
}


def inspect_audio(file_path):

    try:

        info = sf.info(
            str(file_path)
        )

        duration = (
            info.frames /
            info.samplerate
        )

        return {
            "duration": round(
                duration,
                3
            ),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "sample_width": info.subtype,
            "valid": True
        }

    except Exception as e:

        print(
            f"ERROR: {file_path}"
        )

        print(e)

        return {
            "duration": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "sample_width": "",
            "valid": False
        }


def extract_speaker_id(
    file_path
):

    return file_path.stem


def scan_real():

    records = []

    real_dir = (
        DATASET_DIR /
        "real"
    )

    # --------------------------------------------------------
    # PAIRED REAL
    # --------------------------------------------------------

    paired_dir = (
        real_dir /
        "paired"
    )

    for class_name, label in CLASS_LABELS.items():

        class_dir = (
            paired_dir /
            class_name
        )

        if not class_dir.exists():
            continue

        for file_path in sorted(
            class_dir.glob("*.wav")
        ):

            info = inspect_audio(
                file_path
            )

            records.append({

                "file": str(
                    file_path.relative_to(
                        DATASET_DIR
                    )
                ),

                "speaker_id":
                    extract_speaker_id(
                        file_path
                    ),

                "source":
                    "real",

                "pairing":
                    "paired",

                "label":
                    label,

                "class_name":
                    class_name,

                **info
            })

    # --------------------------------------------------------
    # UNPAIRED REAL
    # --------------------------------------------------------

    unpaired_dir = (
        real_dir /
        "unpaired"
    )

    for class_name, label in CLASS_LABELS.items():

        class_dir = (
            unpaired_dir /
            class_name
        )

        if not class_dir.exists():
            continue

        for file_path in sorted(
            class_dir.glob("*.wav")
        ):

            info = inspect_audio(
                file_path
            )

            records.append({

                "file": str(
                    file_path.relative_to(
                        DATASET_DIR
                    )
                ),

                "speaker_id":
                    extract_speaker_id(
                        file_path
                    ),

                "source":
                    "real",

                "pairing":
                    "unpaired",

                "label":
                    label,

                "class_name":
                    class_name,

                **info
            })

    return records


def scan_synthetic():

    records = []

    synthetic_dir = (
        DATASET_DIR /
        "synthetic"
    )

    for class_name, label in CLASS_LABELS.items():

        class_dir = (
            synthetic_dir /
            class_name
        )

        if not class_dir.exists():
            continue

        for file_path in sorted(
            class_dir.glob("*.wav")
        ):

            info = inspect_audio(
                file_path
            )

            records.append({

                "file": str(
                    file_path.relative_to(
                        DATASET_DIR
                    )
                ),

                "speaker_id":
                    extract_speaker_id(
                        file_path
                    ),

                "source":
                    "synthetic",

                "pairing":
                    "paired",

                "label":
                    label,

                "class_name":
                    class_name,

                **info
            })

    return records


def main():

    print("\n")
    print("=" * 60)
    print("SENTINEL VOICE DATASET SCANNER")
    print("=" * 60)

    records = []

    records.extend(
        scan_real()
    )

    records.extend(
        scan_synthetic()
    )

    df = pd.DataFrame(
        records
    )

    if len(df) == 0:

        raise RuntimeError(
            "No WAV files found."
        )

    # --------------------------------------------------------
    # Initial split is empty.
    # We'll create speaker-independent
    # split separately.
    # --------------------------------------------------------

    df["split"] = ""

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"\nTotal recordings: "
        f"{len(df)}"
    )

    print("\nSource:")
    print(
        df["source"].value_counts()
    )

    print("\nPairing:")
    print(
        df["pairing"].value_counts()
    )

    print("\nClass:")
    print(
        df["class_name"].value_counts()
    )

    print("\nSource × Class:")
    print(
        pd.crosstab(
            df["source"],
            df["class_name"]
        )
    )

    print("\nPairing × Class:")
    print(
        pd.crosstab(
            df["pairing"],
            df["class_name"]
        )
    )

    print("\n")
    print("=" * 60)
    print(
        f"Metadata saved to "
        f"{OUTPUT_FILE}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()