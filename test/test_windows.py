from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio
from src.preprocessing.vad import VoiceActivityDetector

from src.segmentation.windows import (
    extract_speech_segments,
    create_overlapping_windows
)


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
# Extract speech
# -----------------------------------------

segments = extract_speech_segments(
    audio,
    speech_timestamps,
    sample_rate
)


print("\n==============================")
print("TEMPORAL SEGMENTATION")
print("==============================")


# Test the first sufficiently long segment

for i, segment in enumerate(segments, start=1):

    duration = segment["end"] - segment["start"]

    if duration < 1.0:
        continue

    print(
        f"\nOriginal segment {i}: "
        f"{segment['start']:.2f}s → "
        f"{segment['end']:.2f}s"
    )

    windows = create_overlapping_windows(
        segment["waveform"],
        window_size=1.0,
        hop_size=0.5,
        sample_rate=sample_rate
    )

    for j, window in enumerate(windows, start=1):

        print(
            f"Window {j}: "
            f"{window['start']:.2f}s → "
            f"{window['end']:.2f}s | "
            f"shape={window['waveform'].shape}"
        )

    