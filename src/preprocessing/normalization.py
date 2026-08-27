import torch


def normalize_audio(waveform: torch.Tensor):
    """
    Peak-normalize waveform to approximately [-1, 1].
    """

    peak = waveform.abs().max()

    if peak > 0:
        waveform = waveform / peak

    return waveform