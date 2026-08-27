from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio
from src.preprocessing.vad import VoiceActivityDetector


audio, sample_rate = load_audio("data/raw/test.wav")

audio = normalize_audio(audio)

vad = VoiceActivityDetector()

speech_segments = vad.detect(audio)

print("\n==============================")
print("VAD RESULTS")
print("==============================")

if not speech_segments:
    print("No speech detected.")

else:
    total_speech = 0

    for i, segment in enumerate(speech_segments, start=1):

        start = segment["start"] / sample_rate
        end = segment["end"] / sample_rate

        duration = end - start
        total_speech += duration

        print(
            f"Segment {i}: "
            f"{start:.2f}s → {end:.2f}s "
            f"({duration:.2f}s)"
        )

    print("------------------------------")
    print(f"Total speech: {total_speech:.2f}s")
    print(f"Total audio: {audio.shape[1] / sample_rate:.2f}s")
    print("==============================")