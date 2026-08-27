from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio
from src.preprocessing.vad import VoiceActivityDetector

from src.segmentation.windows import extract_speech_segments

from src.speaker.verification import SpeakerVerifier
from src.speaker.enrollment import create_enrollment_embedding


ENROLLMENT_FILE = "data/enrollment/worker_001.wav"
TEST_FILE = "data/raw/test.wav"

THRESHOLD = 0.25


# --------------------------------------------------
# 1. Load test audio
# --------------------------------------------------

audio, sample_rate = load_audio(TEST_FILE)

audio = normalize_audio(audio)


# --------------------------------------------------
# 2. VAD
# --------------------------------------------------

vad = VoiceActivityDetector()

speech_timestamps = vad.detect(audio)


print("\n==============================")
print("VAD")
print("==============================")

print(f"Speech segments detected: {len(speech_timestamps)}")


# --------------------------------------------------
# 3. Extract speech segments
# --------------------------------------------------

segments = extract_speech_segments(
    audio,
    speech_timestamps,
    sample_rate
)


# --------------------------------------------------
# 4. Load speaker model
# --------------------------------------------------

verifier = SpeakerVerifier()


# --------------------------------------------------
# 5. Create enrollment embedding
# --------------------------------------------------

print("\nCreating enrollment embedding...")

enrollment_embedding = create_enrollment_embedding(
    verifier,
    ENROLLMENT_FILE
)


# --------------------------------------------------
# 6. Verify every speech segment
# --------------------------------------------------

print("\n==============================")
print("SEGMENT SPEAKER VERIFICATION")
print("==============================")


target_segments = []

for i, segment in enumerate(segments, start=1):

    waveform = segment["waveform"]

    start = segment["start"]
    end = segment["end"]

    # Skip extremely short segments
    duration = end - start

    if duration < 0.5:
        print(
            f"Segment {i}: skipped "
            f"(too short: {duration:.2f}s)"
        )
        continue

    score, is_target = verifier.verify_segment(
        enrollment_embedding,
        waveform,
        threshold=THRESHOLD
    )

    print(
        f"Segment {i}: "
        f"{start:.2f}s → {end:.2f}s | "
        f"duration={duration:.2f}s | "
        f"score={score:.4f} | "
        f"target={is_target}"
    )

    if is_target:
        target_segments.append(segment)


# --------------------------------------------------
# 7. Summary
# --------------------------------------------------

print("\n==============================")
print("RESULT")
print("==============================")

print(
    f"Target speaker segments: "
    f"{len(target_segments)}"
)

total_target_duration = sum(
    segment["end"] - segment["start"]
    for segment in target_segments
)

print(
    f"Target speech duration: "
    f"{total_target_duration:.2f}s"
)

print("==============================")