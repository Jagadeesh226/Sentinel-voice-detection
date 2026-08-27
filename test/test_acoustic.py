from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio
from src.preprocessing.vad import VoiceActivityDetector

from src.segmentation.windows import (
    extract_speech_segments,
    create_overlapping_windows
)

from src.features.acoustic import extract_acoustic_features


TEST_FILE = "data/raw/test.wav"


# -----------------------------------------
# Load
# -----------------------------------------

audio, sample_rate = load_audio(TEST_FILE)

audio = normalize_audio(audio)


# -----------------------------------------
# VAD
# -----------------------------------------

vad = VoiceActivityDetector()

speech_timestamps = vad.detect(audio)


# -----------------------------------------
# Speech segments
# -----------------------------------------

segments = extract_speech_segments(
    audio,
    speech_timestamps,
    sample_rate
)


# -----------------------------------------
# Find first valid segment
# -----------------------------------------

for segment in segments:

    duration = segment["end"] - segment["start"]

    if duration >= 1.0:

        windows = create_overlapping_windows(
            segment["waveform"],
            window_size=1.0,
            hop_size=0.5,
            sample_rate=sample_rate
        )

        first_window = windows[0]

        break


# -----------------------------------------
# Acoustic features
# -----------------------------------------

features = extract_acoustic_features(
    first_window["waveform"],
    sample_rate
)


print("\n==============================")
print("ACOUSTIC FEATURES")
print("==============================")

print("Feature shape:", features.shape)
print("Feature dimension:", len(features))

print("\nFeatures:")
print(features)

print("==============================")