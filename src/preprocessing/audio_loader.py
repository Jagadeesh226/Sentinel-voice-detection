import soundfile as sf
import torch
import torchaudio


TARGET_SAMPLE_RATE = 16000


def load_audio(file_path: str):
    waveform, sample_rate = sf.read(file_path, dtype="float32")

    # Convert numpy array → torch tensor
    waveform = torch.from_numpy(waveform)

    # Stereo → mono
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)

    # Add channel dimension
    waveform = waveform.unsqueeze(0)

    # Resample to 16 kHz
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE
        )
        waveform = resampler(waveform)

    return waveform, TARGET_SAMPLE_RATE