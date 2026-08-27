from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio

audio, sample_rate = load_audio("data/raw/test.wav")

audio = normalize_audio(audio)

print("Shape:", audio.shape)
print("Sample rate:", sample_rate)
print("Duration:", audio.shape[1] / sample_rate)
print("Minimum:", audio.min().item())
print("Maximum:", audio.max().item())