import torch

from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio


def create_enrollment_embedding(verifier, enrollment_file):

    audio, sample_rate = load_audio(enrollment_file)

    audio = normalize_audio(audio)

    embedding = verifier.get_embedding(audio)

    return embedding