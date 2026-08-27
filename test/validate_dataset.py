import csv
import wave
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("data/dataset")

OUTPUT_FILE = Path(
    "data/dataset_metadata.csv"
)


# Folder → numeric label

LABELS = {
    "alert": 0,
    "mid_fatigue": 1,
    "high_fatigue": 2
}


# ============================================================
# VALIDATE ONE WAV FILE
# ============================================================

def validate_wav(file_path):

    try:

        with wave.open(
            str(file_path),
            "rb"
        ) as wav:

            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()

            duration = (
                frames / sample_rate
                if sample_rate > 0
                else 0
            )

        valid = True
        problems = []

        # Check channels

        if channels != 1:

            valid = False

            problems.append(
                f"Expected mono, got {channels} channels"
            )

        # Check sample rate

        if sample_rate != 16000:

            valid = False

            problems.append(
                f"Expected 16000 Hz, got {sample_rate} Hz"
            )

        # Check duration

        if duration <= 0:

            valid = False

            problems.append(
                "Invalid duration"
            )

        return {
            "valid": valid,
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width": sample_width,
            "frames": frames,
            "duration": duration,
            "problems": problems
        }

    except Exception as e:

        return {
            "valid": False,
            "sample_rate": None,
            "channels": None,
            "sample_width": None,
            "frames": None,
            "duration": None,
            "problems": [
                str(e)
            ]
        }


# ============================================================
# DETERMINE SPEAKER ID
# ============================================================

def get_speaker_id(filename):

    """
    Expected examples:

        real_spk001.wav
        ai_spk001.wav
        spk001.wav

    We keep the filename prefix as the speaker ID.
    """

    stem = Path(filename).stem

    parts = stem.split("_")

    if len(parts) >= 2:

        return "_".join(parts[:2])

    return stem


# ============================================================
# DETERMINE SOURCE
# ============================================================

def get_source(filename):

    filename = filename.lower()

    if filename.startswith("real_"):

        return "real"

    if filename.startswith("ai_"):

        return "synthetic"

    return "unknown"


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("========================================")
    print(" SENTINEL DATASET VALIDATION")
    print("========================================")

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{DATASET_DIR}"
        )

    rows = []

    total_files = 0
    valid_files = 0
    invalid_files = 0

    class_counts = {
        "alert": 0,
        "mid_fatigue": 0,
        "high_fatigue": 0
    }

    # --------------------------------------------------------
    # Iterate through classes
    # --------------------------------------------------------

    for class_name, label in LABELS.items():

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():

            print(
                f"\nWARNING: Missing folder: "
                f"{class_dir}"
            )

            continue

        files = sorted(
            class_dir.glob("*.wav")
        )

        print(
            f"\n{class_name}: "
            f"{len(files)} files"
        )

        class_counts[class_name] = len(files)

        # ----------------------------------------------------
        # Validate files
        # ----------------------------------------------------

        for file_path in files:

            total_files += 1

            result = validate_wav(
                file_path
            )

            if result["valid"]:

                valid_files += 1

                status = "OK"

            else:

                invalid_files += 1

                status = "INVALID"

            print(
                f"  {file_path.name:<35}"
                f"{status}"
            )

            if not result["valid"]:

                for problem in result["problems"]:

                    print(
                        f"      → {problem}"
                    )

            speaker_id = get_speaker_id(
                file_path.name
            )

            source = get_source(
                file_path.name
            )

            rows.append({
                "file": str(
                    file_path.relative_to(
                        DATASET_DIR
                    )
                ),

                "speaker_id": speaker_id,

                "source": source,

                "label": label,

                "class_name": class_name,

                "duration": (
                    round(
                        result["duration"],
                        3
                    )
                    if result["duration"]
                    is not None
                    else None
                ),

                "sample_rate":
                    result["sample_rate"],

                "channels":
                    result["channels"],

                "sample_width":
                    result["sample_width"],

                "valid":
                    result["valid"]
            })

    # ========================================================
    # WRITE METADATA
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "file",
        "speaker_id",
        "source",
        "label",
        "class_name",
        "duration",
        "sample_rate",
        "channels",
        "sample_width",
        "valid"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("========================================")
    print(" DATASET SUMMARY")
    print("========================================")

    print(
        f"\nTotal files:   {total_files}"
    )

    print(
        f"Valid files:   {valid_files}"
    )

    print(
        f"Invalid files: {invalid_files}"
    )

    print("\nClass distribution:")

    for class_name, count in class_counts.items():

        print(
            f"  {class_name:<20} {count}"
        )

    print(
        "\nMetadata saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("========================================")


if __name__ == "__main__":

    main()