from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio
from src.preprocessing.vad import VoiceActivityDetector

from src.features.speech_rate import (
    SpeechRateAnalyzer,
    calculate_speech_rate
)


TEST_FILE = "data/raw/test.wav"


# -----------------------------------------
# 1. Load audio
# -----------------------------------------

audio, sample_rate = load_audio(TEST_FILE)

audio = normalize_audio(audio)


# -----------------------------------------
# 2. VAD
# -----------------------------------------

vad = VoiceActivityDetector()

speech_timestamps = vad.detect(audio)


# -----------------------------------------
# 3. Calculate total speech duration
# -----------------------------------------

total_speech_duration = 0.0

for segment in speech_timestamps:

    duration = (
        segment["end"] - segment["start"]
    ) / sample_rate

    total_speech_duration += duration


# -----------------------------------------
# 4. Load Whisper
# -----------------------------------------

analyzer = SpeechRateAnalyzer()


# -----------------------------------------
# 5. Transcribe
# -----------------------------------------

text = analyzer.transcribe(
    audio,
    sample_rate
)


# -----------------------------------------
# 6. Calculate speech rate
# -----------------------------------------

words_per_second, words_per_minute = (
    calculate_speech_rate(
        text,
        total_speech_duration
    )
)


# -----------------------------------------
# 7. Results
# -----------------------------------------

print("\n==============================")
print("SPEECH RATE")
print("==============================")

print("Transcript:")
print(text)

print(
    "\nSpeech duration:",
    round(total_speech_duration, 2),
    "seconds"
)

print(
    "Word count:",
    len(text.split())
)

print(
    "Words per second:",
    round(words_per_second, 2)
)

print(
    "Words per minute:",
    round(words_per_minute, 2)
)

print("==============================")