from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio
from src.preprocessing.vad import VoiceActivityDetector

from src.segmentation.windows import (
    extract_speech_segments,
    create_overlapping_windows
)

from src.features.wavlm import WavLMFeatureExtractor


TEST_FILE = "data/raw/test.wav"


# -----------------------------------------
# Load audio
# -----------------------------------------

audio, sample_rate = load_audio(
    TEST_FILE
)

audio = normalize_audio(
    audio
)


# -----------------------------------------
# VAD
# -----------------------------------------

vad = VoiceActivityDetector()

speech_timestamps = vad.detect(
    audio
)


# -----------------------------------------
# Extract speech segments
# -----------------------------------------

segments = extract_speech_segments(
    audio,
    speech_timestamps,
    sample_rate
)


# -----------------------------------------
# Find first valid segment
# -----------------------------------------

first_window = None

for segment in segments:

    duration = (
        segment["end"] -
        segment["start"]
    )

    if duration >= 1.0:

        windows = create_overlapping_windows(
            segment["waveform"],
            window_size=1.0,
            hop_size=0.5,
            sample_rate=sample_rate
        )

        if len(windows) > 0:

            first_window = windows[0]

            break


if first_window is None:

    raise RuntimeError(
        "No valid 1-second speech window found."
    )


# -----------------------------------------
# Load WavLM
# -----------------------------------------

extractor = WavLMFeatureExtractor()


# -----------------------------------------
# Extract embedding
# -----------------------------------------

embedding = extractor.extract_embedding(
    first_window["waveform"],
    sample_rate
)


# -----------------------------------------
# Results
# -----------------------------------------

print("\n==============================")
print("WAVLM FEATURES")
print("==============================")

print(
    "Embedding shape:",
    embedding.shape
)

print(
    "Embedding dimension:",
    embedding.shape[-1]
)

print("\nFirst 10 values:")

print(
    embedding[0][:10]
)

print("==============================")