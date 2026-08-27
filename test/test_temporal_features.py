from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio
from src.preprocessing.vad import VoiceActivityDetector

from src.features.temporal import extract_temporal_features


TEST_FILE = "data/raw/test.wav"


# -----------------------------------------
# Load audio
# -----------------------------------------

audio, sample_rate = load_audio(TEST_FILE)

audio = normalize_audio(audio)


# -----------------------------------------
# VAD
# -----------------------------------------

vad = VoiceActivityDetector()

speech_timestamps = vad.detect(audio)


# -----------------------------------------
# Temporal features
# -----------------------------------------

features = extract_temporal_features(
    speech_timestamps,
    audio.shape[1],
    sample_rate
)


print("\n==============================")
print("TEMPORAL FEATURES")
print("==============================")

print("Feature dimension:", len(features))

print("\nFeatures:")

print(
    "Number of pauses:",
    features[0]
)

print(
    "Mean pause:",
    features[1]
)

print(
    "Pause std:",
    features[2]
)

print(
    "Maximum pause:",
    features[3]
)

print(
    "Total pause duration:",
    features[4]
)

print(
    "Total speech duration:",
    features[5]
)

print(
    "Total silence duration:",
    features[6]
)

print(
    "Speech ratio:",
    features[7]
)

print(
    "Number of speech segments:",
    features[8]
)

print("==============================")