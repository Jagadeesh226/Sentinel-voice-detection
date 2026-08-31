import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio
from src.preprocessing.vad import VoiceActivityDetector

from src.features.temporal import (
    extract_temporal_features
)

from src.features.speech_rate import (
    SpeechRateAnalyzer,
    calculate_speech_rate
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT /
    "data" /
    "dataset"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "data" /
    "feature_analysis"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "english_global_features.csv"
)


# ============================================================
# ENGLISH DATASET DIRECTORIES
# ============================================================

REAL_PAIRED_DIR = (
    DATASET_DIR /
    "real" /
    "paired"
)

REAL_UNPAIRED_DIR = (
    DATASET_DIR /
    "real" /
    "unpaired"
)

SYNTHETIC_DIR = (
    DATASET_DIR /
    "synthetic"
)


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_MAPPING = {

    "alert": "alert",

    "mild": "mild_fatigue",

    "mild_fatigue": "mild_fatigue",

    "high": "high_fatigue",

    "high_fatigue": "high_fatigue"
}


# ============================================================
# SUPPORTED AUDIO EXTENSIONS
# ============================================================

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a"
}


# ============================================================
# SPEAKER MAPPING
# ============================================================
#
# Existing real recordings may use:
#
#     real_spk001.wav
#
# while newer recordings may use:
#
#     rushabh_english_alert_01.wav
#
# This mapping keeps the same speaker identity consistent.
#
# Modify this dictionary according to your actual dataset.
# ============================================================

SPEAKER_MAP = {

    "spk001": "rushabh",
    "spk002": "shane",
    "spk003": "suyash",
    "spk004": "vedika"
}


# ============================================================
# EXTRACT SPEAKER ID
# ============================================================

def extract_speaker_id(
    audio_path
):

    filename = audio_path.stem.lower()

    parts = filename.split("_")


    # --------------------------------------------------------
    # real_spk001.wav
    # --------------------------------------------------------

    if (
        len(parts) >= 2
        and parts[0] == "real"
        and parts[1].startswith("spk")
    ):

        speaker_code = parts[1]

        return SPEAKER_MAP.get(
            speaker_code,
            speaker_code
        )


    # --------------------------------------------------------
    # rushabh_english_alert_01.wav
    # shane_english_alert_01.wav
    # --------------------------------------------------------

    if (
        len(parts) >= 2
        and parts[1] == "english"
    ):

        return parts[0]


    # --------------------------------------------------------
    # hindi-style / generic speaker naming
    #
    # This is mainly a fallback.
    # --------------------------------------------------------

    if (
        len(parts) >= 2
        and parts[0] == "spk"
    ):

        return (
            parts[0]
            + "_"
            + parts[1]
        )


    # --------------------------------------------------------
    # Unknown speaker
    # --------------------------------------------------------

    return "unknown"


# ============================================================
# DETERMINE LABEL
# ============================================================

def get_label(
    audio_path
):

    folder_name = (
        audio_path.parent.name.lower()
    )

    if folder_name in CLASS_MAPPING:

        return CLASS_MAPPING[
            folder_name
        ]

    return None


# ============================================================
# DETERMINE SOURCE TYPE
# ============================================================

def get_source_type(
    audio_path
):

    try:

        relative_path = (
            audio_path.relative_to(
                DATASET_DIR
            )
        )

    except ValueError:

        return "unknown"


    parts = [
        part.lower()
        for part in relative_path.parts
    ]


    if "synthetic" in parts:

        return "synthetic"

    if "paired" in parts:

        return "real_paired"

    if "unpaired" in parts:

        return "real_unpaired"


    return "unknown"


# ============================================================
# COLLECT AUDIO FILES
# ============================================================

def collect_audio_files():

    audio_files = []


    # --------------------------------------------------------
    # REAL / PAIRED
    # --------------------------------------------------------

    if REAL_PAIRED_DIR.exists():

        audio_files.extend(
            [
                path
                for path in REAL_PAIRED_DIR.rglob("*")
                if (
                    path.is_file()
                    and
                    path.suffix.lower()
                    in AUDIO_EXTENSIONS
                )
            ]
        )


    # --------------------------------------------------------
    # REAL / UNPAIRED
    # --------------------------------------------------------

    if REAL_UNPAIRED_DIR.exists():

        audio_files.extend(
            [
                path
                for path in REAL_UNPAIRED_DIR.rglob("*")
                if (
                    path.is_file()
                    and
                    path.suffix.lower()
                    in AUDIO_EXTENSIONS
                )
            ]
        )


    # --------------------------------------------------------
    # SYNTHETIC
    # --------------------------------------------------------

    if SYNTHETIC_DIR.exists():

        audio_files.extend(
            [
                path
                for path in SYNTHETIC_DIR.rglob("*")
                if (
                    path.is_file()
                    and
                    path.suffix.lower()
                    in AUDIO_EXTENSIONS
                )
            ]
        )


    return sorted(
        audio_files
    )


# ============================================================
# EXTRACT GLOBAL FEATURES FROM ONE RECORDING
# ============================================================

def extract_global_features(
    audio_path
):

    # --------------------------------------------------------
    # LOAD AUDIO
    # --------------------------------------------------------

    waveform, sample_rate = load_audio(
        str(audio_path)
    )


    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    waveform = normalize_audio(
        waveform
    )


    # --------------------------------------------------------
    # VAD
    # --------------------------------------------------------

    vad = VoiceActivityDetector()

    speech_timestamps = vad.detect(
        waveform
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
            speech_timestamps,
            total_audio_samples,
            sample_rate
        )
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
    # WPS + WPM
    # --------------------------------------------------------

    words_per_second, words_per_minute = (
        calculate_speech_rate(
            transcript,
            speech_duration
        )
    )


    # --------------------------------------------------------
    # CREATE 11 GLOBAL FEATURES
    # --------------------------------------------------------

    global_features = np.concatenate(
        [
            temporal_features,

            np.array(
                [
                    words_per_second,
                    words_per_minute
                ],
                dtype=np.float32
            )
        ]
    )


    if len(global_features) != 11:

        raise ValueError(
            "Expected 11 global features, "
            f"got {len(global_features)}"
        )


    return global_features


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ENGLISH GLOBAL FEATURE EXTRACTION")
    print("=" * 70)


    # --------------------------------------------------------
    # COLLECT FILES
    # --------------------------------------------------------

    audio_files = collect_audio_files()


    print()
    print(
        f"Total audio files found: "
        f"{len(audio_files)}"
    )


    if len(audio_files) == 0:

        print(
            "\nNo audio files found."
        )

        return


    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    records = []

    successful = 0
    failed = 0


    # --------------------------------------------------------
    # PROCESS RECORDINGS
    # --------------------------------------------------------

    for index, audio_path in enumerate(
        audio_files,
        start=1
    ):

        print()
        print("-" * 70)

        print(
            f"[{index}/{len(audio_files)}] "
            f"{audio_path.name}"
        )


        label = get_label(
            audio_path
        )


        if label is None:

            print(
                "⚠ Skipped: "
                "Unknown class folder."
            )

            failed += 1

            continue


        source_type = get_source_type(
            audio_path
        )


        speaker_id = extract_speaker_id(
            audio_path
        )


        try:

            features = (
                extract_global_features(
                    audio_path
                )
            )


            record = {

                "filename": (
                    audio_path.name
                ),

                "speaker_id": (
                    speaker_id
                ),

                "language": "english",

                "source": (
                    source_type
                ),

                "label": (
                    label
                ),

                "pause_count": (
                    float(features[0])
                ),

                "mean_pause_duration": (
                    float(features[1])
                ),

                "std_pause_duration": (
                    float(features[2])
                ),

                "max_pause_duration": (
                    float(features[3])
                ),

                "total_pause_duration": (
                    float(features[4])
                ),

                "speech_duration": (
                    float(features[5])
                ),

                "silence_duration": (
                    float(features[6])
                ),

                "speech_activity": (
                    float(features[7])
                ),

                "speech_segments": (
                    float(features[8])
                ),

                "wps": (
                    float(features[9])
                ),

                "wpm": (
                    float(features[10])
                )
            }


            records.append(
                record
            )

            successful += 1


            print(
                "✓ Success"
            )

            print(
                f"  Speaker: {speaker_id}"
            )

            print(
                f"  Source: {source_type}"
            )

            print(
                f"  Label: {label}"
            )

            print(
                f"  WPS: {features[9]:.3f}"
            )

            print(
                f"  WPM: {features[10]:.3f}"
            )


        except Exception as error:

            failed += 1

            print(
                f"✗ Failed: {error}"
            )


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    dataframe = pd.DataFrame(
        records
    )


    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ENGLISH FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Successful recordings: "
        f"{successful}"
    )

    print(
        f"Failed recordings: "
        f"{failed}"
    )

    print()
    print(
        "Output file:"
    )

    print(
        OUTPUT_FILE
    )


    if not dataframe.empty:

        print()
        print(
            "Class distribution:"
        )

        print(
            dataframe[
                "label"
            ].value_counts()
        )


        print()
        print(
            "Source distribution:"
        )

        print(
            dataframe[
                "source"
            ].value_counts()
        )


        print()
        print(
            "Feature columns:"
        )

        feature_columns = [
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


        for index, column in enumerate(
            feature_columns
        ):

            print(
                f"{index}: {column}"
            )


    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

