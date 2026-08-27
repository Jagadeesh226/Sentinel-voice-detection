import sys
from pathlib import Path

import torch
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio

from src.preprocessing.vad import VoiceActivityDetector

from src.speaker.verification import SpeakerVerifier

from src.segmentation.windows import (
    extract_speech_segments,
    create_overlapping_windows
)

from src.features.acoustic import (
    extract_acoustic_features
)

from src.features.wavlm import (
    WavLMFeatureExtractor
)

from src.features.temporal import (
    extract_temporal_features
)

from src.features.speech_rate import (
    SpeechRateAnalyzer,
    calculate_speech_rate
)


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_FILE = Path(
    "data/dataset_split.csv"
)

DATASET_DIR = Path(
    "data/dataset"
)

OUTPUT_DIR = Path(
    "data/features"
)

SAMPLE_RATE = 16000

WINDOW_SIZE = 1.0
HOP_SIZE = 0.5

SPEAKER_THRESHOLD = 0.25


# ============================================================
# PROCESS ONE AUDIO FILE
# ============================================================

def process_audio_file(
    file_path,
    verifier,
    vad,
    wavlm_extractor,
    speech_rate_analyzer
):

    print("\n")
    print("=" * 60)
    print("PROCESSING")
    print("=" * 60)

    print(
        f"File: {file_path}"
    )

    # ========================================================
    # LOAD AUDIO
    # ========================================================

    waveform, sample_rate = load_audio(
        str(file_path)
    )

    print(
        f"Audio shape: {waveform.shape}"
    )

    print(
        f"Sample rate: {sample_rate}"
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    waveform = normalize_audio(
        waveform
    )

    # ========================================================
    # VAD
    # ========================================================

    speech_timestamps = vad.detect(
        waveform
    )

    print(
        f"VAD speech segments: "
        f"{len(speech_timestamps)}"
    )

    if len(speech_timestamps) == 0:

        raise ValueError(
            "No speech detected."
        )

    # ========================================================
    # TARGET SPEAKER ENROLLMENT
    # ========================================================

    enrollment_embedding = (
        verifier.get_embedding(
            waveform
        )
    )

    # ========================================================
    # EXTRACT SPEECH SEGMENTS
    # ========================================================

    speech_segments = extract_speech_segments(
        waveform,
        speech_timestamps,
        sample_rate
    )

    target_segments = []

    for segment in speech_segments:

        segment_waveform = (
            segment["waveform"]
        )

        segment_embedding = (
            verifier.get_embedding(
                segment_waveform
            )
        )

        score = verifier.similarity(
            enrollment_embedding,
            segment_embedding
        )

        is_target = (
            score >= SPEAKER_THRESHOLD
        )

        print(
            f"Segment "
            f"{segment['start']:.2f}s → "
            f"{segment['end']:.2f}s | "
            f"speaker score={score:.4f} | "
            f"target={is_target}"
        )

        if is_target:

            target_segments.append(
                segment
            )

    print(
        f"\nTarget speaker segments: "
        f"{len(target_segments)}"
    )

    if len(target_segments) == 0:

        raise ValueError(
            "No target-speaker speech segments found."
        )

    # ========================================================
    # TEMPORAL WINDOWS
    # ========================================================

    windows = []

    for segment in target_segments:

        segment_windows = (
            create_overlapping_windows(
                segment["waveform"],
                window_size=WINDOW_SIZE,
                hop_size=HOP_SIZE,
                sample_rate=sample_rate
            )
        )

        windows.extend(
            segment_windows
        )

    print(
        f"Temporal windows: "
        f"{len(windows)}"
    )

    if len(windows) == 0:

        raise ValueError(
            "No complete 1-second windows created."
        )

    # ========================================================
    # ACOUSTIC + WAVLM FEATURES
    # ========================================================

    acoustic_features = []

    wavlm_features = []

    valid_windows = []

    for index, window_info in enumerate(
        windows
    ):

        window_waveform = (
            window_info["waveform"]
        )

        print(
            f"\nWindow "
            f"{index + 1}/{len(windows)}"
        )

        print(
            f"Time: "
            f"{window_info['start']:.2f}s → "
            f"{window_info['end']:.2f}s"
        )

        # ----------------------------------------------------
        # ACOUSTIC
        # ----------------------------------------------------

        try:

            acoustic = extract_acoustic_features(
                window_waveform,
                sample_rate
            )

            acoustic = torch.tensor(
                acoustic,
                dtype=torch.float32
            )

        except Exception as e:

            print(
                f"Acoustic extraction failed: {e}"
            )

            continue

        # ----------------------------------------------------
        # WAVLM
        # ----------------------------------------------------

        try:

            wavlm = (
                wavlm_extractor.extract_embedding(
                    window_waveform,
                    sample_rate
                )
            )

            wavlm = wavlm.squeeze()

            wavlm = wavlm.detach().cpu()

        except Exception as e:

            print(
                f"WavLM extraction failed: {e}"
            )

            continue

        # ----------------------------------------------------
        # CHECK DIMENSIONS
        # ----------------------------------------------------

        if acoustic.numel() != 40:

            print(
                f"WARNING: Expected acoustic "
                f"dimension 40, got "
                f"{acoustic.numel()}"
            )

        if wavlm.numel() != 768:

            print(
                f"WARNING: Expected WavLM "
                f"dimension 768, got "
                f"{wavlm.numel()}"
            )

        acoustic_features.append(
            acoustic
        )

        wavlm_features.append(
            wavlm
        )

        valid_windows.append(
            window_info
        )

    if len(acoustic_features) == 0:

        raise ValueError(
            "No valid acoustic/WavLM features extracted."
        )

    # ========================================================
    # STACK FEATURES
    # ========================================================

    acoustic_features = torch.stack(
        acoustic_features
    )

    wavlm_features = torch.stack(
        wavlm_features
    )

    print("\n")
    print("=" * 60)
    print("FEATURE SHAPES")
    print("=" * 60)

    print(
        "Acoustic:",
        acoustic_features.shape
    )

    print(
        "WavLM:",
        wavlm_features.shape
    )

    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================

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

    temporal_features = torch.tensor(
        temporal_features,
        dtype=torch.float32
    )

    print(
        "Temporal features:",
        temporal_features.shape
    )

    # ========================================================
    # SPEECH RATE
    # ========================================================

    print(
        "\nCalculating speech rate..."
    )

    transcript = speech_rate_analyzer.transcribe(
        waveform,
        sample_rate
    )

    print(
        "Transcript:"
    )

    print(
        transcript
    )

    # --------------------------------------------------------
    # Speech duration
    # --------------------------------------------------------

    speech_duration = 0.0

    for segment in speech_timestamps:

        segment_duration = (
            segment["end"]
            - segment["start"]
        ) / sample_rate

        speech_duration += (
            segment_duration
        )

    words_per_second, words_per_minute = (
        calculate_speech_rate(
            transcript,
            speech_duration
        )
    )

    speech_rate_features = torch.tensor(
        [
            words_per_second,
            words_per_minute
        ],
        dtype=torch.float32
    )

    print(
        f"Speech duration: "
        f"{speech_duration:.2f}s"
    )

    print(
        f"Words per second: "
        f"{words_per_second:.2f}"
    )

    print(
        f"Words per minute: "
        f"{words_per_minute:.2f}"
    )

    print(
        "Speech-rate features:",
        speech_rate_features.shape
    )

    # ========================================================
    # GLOBAL FEATURES
    # ========================================================

    global_features = torch.cat(
        [
            temporal_features,
            speech_rate_features
        ],
        dim=0
    )

    print(
        "Global features:",
        global_features.shape
    )

    if global_features.numel() != 11:

        raise ValueError(
            f"Expected 11 global features, "
            f"got {global_features.numel()}"
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "acoustic_features":
            acoustic_features,

        "wavlm_features":
            wavlm_features,

        "global_features":
            global_features,

        "windows":
            valid_windows,

        "num_target_segments":
            len(target_segments)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("SENTINEL DATASET FEATURE EXTRACTION")
    print("=" * 60)

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LOAD METADATA
    # ========================================================

    df = pd.read_csv(
        METADATA_FILE
    )

    df = df[
        df["valid"] == True
    ].copy()

    print(
        f"\nTotal recordings: {len(df)}"
    )

    # ========================================================
    # LOAD MODELS ONCE
    # ========================================================

    print(
        "\nLoading VAD..."
    )

    vad = VoiceActivityDetector()

    print(
        "\nLoading ECAPA speaker model..."
    )

    verifier = SpeakerVerifier()

    print(
        "\nLoading WavLM..."
    )

    wavlm_extractor = (
        WavLMFeatureExtractor()
    )

    print(
        "\nLoading Whisper..."
    )

    speech_rate_analyzer = (
        SpeechRateAnalyzer()
    )

    # ========================================================
    # PROCESS DATASET
    # ========================================================

    successful = 0
    failed = 0

    for index, row in df.iterrows():

        relative_file = Path(
            row["file"]
        )

        file_path = (
            DATASET_DIR /
            relative_file
        )

        print("\n")
        print("#" * 60)

        print(
            f"Recording "
            f"{index + 1}/{len(df)}"
        )

        print(
            f"Speaker: "
            f"{row['speaker_id']}"
        )

        print(
            f"Source: "
            f"{row['source']}"
        )

        print(
            f"Pairing: "
            f"{row['pairing']}"
        )

        print(
            f"Class: "
            f"{row['class_name']}"
        )

        print(
            f"Path: "
            f"{file_path}"
        )

        # ----------------------------------------------------
        # Check file
        # ----------------------------------------------------

        if not file_path.exists():

            print(
                f"ERROR: File does not exist:"
            )

            print(
                file_path
            )

            failed += 1

            continue

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        try:

            features = process_audio_file(
                file_path=file_path,
                verifier=verifier,
                vad=vad,
                wavlm_extractor=wavlm_extractor,
                speech_rate_analyzer=speech_rate_analyzer
            )

            # ------------------------------------------------
            # Output filename
            # ------------------------------------------------

            output_filename = (
                f"{row['source']}_"
                f"{row['pairing']}_"
                f"{row['speaker_id']}_"
                f"{row['class_name']}.pt"
            )

            output_path = (
                OUTPUT_DIR /
                output_filename
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            torch.save(
                {
                    "acoustic_features":
                        features[
                            "acoustic_features"
                        ],

                    "wavlm_features":
                        features[
                            "wavlm_features"
                        ],

                    "global_features":
                        features[
                            "global_features"
                        ],

                    "windows":
                        features[
                            "windows"
                        ],

                    "num_target_segments":
                        features[
                            "num_target_segments"
                        ],

                    "label":
                        int(row["label"]),

                    "class_name":
                        row["class_name"],

                    "speaker_id":
                        row["speaker_id"],

                    "source":
                        row["source"],

                    "pairing":
                        row["pairing"],

                    "original_file":
                        row["file"]
                },
                output_path
            )

            print(
                f"\nSaved features:"
            )

            print(
                output_path
            )

            successful += 1

        except Exception as e:

            print(
                "\nFEATURE EXTRACTION FAILED"
            )

            print(
                f"File: {file_path}"
            )

            print(
                f"Error: {e}"
            )

            failed += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Total recordings: "
        f"{len(df)}"
    )

    print(
        f"Successful: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        f"Feature directory: "
        f"{OUTPUT_DIR}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()