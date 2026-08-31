import sys
from pathlib import Path

import pandas as pd

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# EXISTING PIPELINE COMPONENTS
# ============================================================

from src.preprocessing.audio_loader import (
    load_audio
)

from src.preprocessing.normalization import (
    normalize_audio
)

from src.preprocessing.vad import (
    VoiceActivityDetector
)

from src.features.temporal import (
    extract_temporal_features
)

from src.features.speech_rate import (
    SpeechRateAnalyzer,
    calculate_speech_rate
)


# ============================================================
# PATHS
# ============================================================

HINDI_DATASET_DIR = (
    PROJECT_ROOT /
    "data" /
    "dataset" /
    "hindi"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "data" /
    "feature_analysis"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "hindi_global_features.csv"
)


# ============================================================
# GLOBAL FEATURE NAMES
# ============================================================

GLOBAL_FEATURE_NAMES = [

    "pause_count",

    "mean_pause_duration",

    "std_pause_duration",

    "max_pause_duration",

    "total_pause_duration",

    "speech_duration",

    "silence_duration",

    "speech_activity",

    "speech_segments",

    "wps",

    "wpm"
]


# ============================================================
# EXTRACT GLOBAL FEATURES
# ============================================================

def extract_hindi_global_features(
    audio_path
):

    print("\n")
    print("=" * 70)
    print(
        f"Processing: {audio_path.name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD AUDIO
    # --------------------------------------------------------

    waveform, sample_rate = (
        load_audio(
            str(audio_path)
        )
    )

    print(
        f"✓ Audio loaded | "
        f"Sample rate: {sample_rate}"
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    waveform = (
        normalize_audio(
            waveform
        )
    )

    print(
        "✓ Audio normalized"
    )

    # --------------------------------------------------------
    # VAD
    # --------------------------------------------------------

    vad = (
        VoiceActivityDetector()
    )

    speech_timestamps = (
        vad.detect(
            waveform
        )
    )

    print(
        f"✓ Speech segments: "
        f"{len(speech_timestamps)}"
    )

    if len(speech_timestamps) == 0:

        raise ValueError(
            "No speech detected."
        )

    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    total_audio_samples = (
        waveform.shape[1]
    )

    temporal_features = (
        extract_temporal_features(
            speech_timestamps=(
                speech_timestamps
            ),
            total_audio_samples=(
                total_audio_samples
            ),
            sample_rate=sample_rate
        )
    )

    print(
        f"✓ Temporal features: "
        f"{len(temporal_features)}"
    )

    if len(temporal_features) != 9:

        raise ValueError(
            "Expected 9 temporal features, "
            f"got {len(temporal_features)}"
        )

    # --------------------------------------------------------
    # SPEECH RATE
    # --------------------------------------------------------

    speech_rate_analyzer = (
        SpeechRateAnalyzer()
    )

    transcript = (
        speech_rate_analyzer.transcribe(
            waveform,
            sample_rate
        )
    )

    # --------------------------------------------------------
    # CALCULATE SPEECH DURATION
    # --------------------------------------------------------

    speech_duration = 0.0

    for segment in speech_timestamps:

        segment_duration = (
            segment["end"]
            -
            segment["start"]
        ) / sample_rate

        speech_duration += (
            segment_duration
        )

    # --------------------------------------------------------
    # WPS / WPM
    # --------------------------------------------------------

    words_per_second, words_per_minute = (
        calculate_speech_rate(
            transcript,
            speech_duration
        )
    )

    print(
        f"✓ WPS: "
        f"{words_per_second:.3f}"
    )

    print(
        f"✓ WPM: "
        f"{words_per_minute:.3f}"
    )

    # --------------------------------------------------------
    # COMBINE 11 GLOBAL FEATURES
    # --------------------------------------------------------

    global_features = [

        float(
            temporal_features[0]
        ),

        float(
            temporal_features[1]
        ),

        float(
            temporal_features[2]
        ),

        float(
            temporal_features[3]
        ),

        float(
            temporal_features[4]
        ),

        float(
            temporal_features[5]
        ),

        float(
            temporal_features[6]
        ),

        float(
            temporal_features[7]
        ),

        float(
            temporal_features[8]
        ),

        float(
            words_per_second
        ),

        float(
            words_per_minute
        )
    ]

    if len(global_features) != 11:

        raise ValueError(
            "Expected 11 global features, "
            f"got {len(global_features)}"
        )

    print(
        "✓ 11 global features extracted"
    )

    return global_features


# ============================================================
# PROCESS HINDI DATASET
# ============================================================

def process_dataset():

    if not HINDI_DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Hindi dataset not found:\n"
            f"{HINDI_DATASET_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    failed_files = []

    # --------------------------------------------------------
    # LABEL FOLDERS
    # --------------------------------------------------------

    label_folders = {

        "alert":"alert",

        "mild":"mild_fatigue",

        "high":"high_fatigue"
    }

    # --------------------------------------------------------
    # PROCESS EACH CLASS
    # --------------------------------------------------------

    for label,folder_name in label_folders.items():

        label_dir = (
            HINDI_DATASET_DIR /
            folder_name
        )

        if not label_dir.exists():

            print(
                f"\n⚠ Missing folder: "
                f"{label_dir}"
            )

            continue

        audio_files = sorted(

            [

                file

                for file in label_dir.iterdir()

                if file.is_file()
                and file.suffix.lower()
                in [
                    ".wav",
                    ".mp3",
                    ".flac",
                    ".m4a"
                ]

            ]

        )

        print("\n")
        print(
            "#" * 70
        )

        print(
            f"LABEL: {label.upper()}"
        )

        print(
            f"Files found: "
            f"{len(audio_files)}"
        )

        print(
            "#" * 70
        )

        # ----------------------------------------------------
        # PROCESS FILES
        # ----------------------------------------------------

        for audio_file in audio_files:

            try:

                global_features = (
                    extract_hindi_global_features(
                        audio_file
                    )
                )

                # ------------------------------------------------
                # SPEAKER ID
                #
                # Expected naming:
                #
                # vedika_hi_alert_01.wav
                #
                # Speaker ID becomes:
                #
                # vedika
                # ------------------------------------------------

                filename_parts = (
                    audio_file.stem.split("_")
                )

                if len(filename_parts) >= 3:

                    speaker_id = (
                        filename_parts[1]
                        + "_"
                        + filename_parts[2]
                    )

                else:

                    speaker_id = (
                        "unknown"
                    )

                # ------------------------------------------------
                # CREATE RECORD
                # ------------------------------------------------

                record = {

                    "filename": (
                        audio_file.name
                    ),

                    "speaker_id": (
                        speaker_id
                    ),

                    "language": "hindi",

                    "label": label
                }

                # ------------------------------------------------
                # ADD 11 FEATURES
                # ------------------------------------------------

                for index, feature_name in enumerate(
                    GLOBAL_FEATURE_NAMES
                ):

                    record[
                        feature_name
                    ] = global_features[
                        index
                    ]

                records.append(
                    record
                )

                print(
                    f"✓ Successfully processed: "
                    f"{audio_file.name}"
                )

            except Exception as error:

                failed_files.append({

                    "filename": (
                        audio_file.name
                    ),

                    "label": label,

                    "error": str(
                        error
                    )
                })

                print(
                    f"✗ Failed: "
                    f"{audio_file.name}"
                )

                print(
                    f"  Error: {error}"
                )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    if not records:

        raise RuntimeError(
            "No recordings were successfully processed."
        )

    dataframe = pd.DataFrame(
        records
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("HINDI FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nSuccessful recordings: "
        f"{len(records)}"
    )

    print(
        f"Failed recordings: "
        f"{len(failed_files)}"
    )

    print(
        f"\nOutput file:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nClass distribution:"
    )

    print(
        dataframe[
            "label"
        ].value_counts()
    )

    # ========================================================
    # FAILED FILE REPORT
    # ========================================================

    if failed_files:

        failed_file = (
            OUTPUT_DIR /
            "hindi_feature_extraction_failures.csv"
        )

        pd.DataFrame(
            failed_files
        ).to_csv(
            failed_file,
            index=False
        )

        print(
            f"\n⚠ Failure report:"
        )

        print(
            failed_file
        )

    print(
        "\nFeature columns:"
    )

    for index, feature_name in enumerate(
        GLOBAL_FEATURE_NAMES
    ):

        print(
            f"{index}: {feature_name}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_dataset()