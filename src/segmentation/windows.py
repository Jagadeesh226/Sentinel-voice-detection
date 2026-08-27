import torch


def extract_speech_segments(
    waveform,
    speech_timestamps,
    sample_rate=16000
):
    """
    Extract waveform corresponding to each
    VAD-detected speech region.
    """

    segments = []

    for timestamp in speech_timestamps:

        start_sample = timestamp["start"]
        end_sample = timestamp["end"]

        segment = waveform[:, start_sample:end_sample]

        segments.append({
            "waveform": segment,
            "start": start_sample / sample_rate,
            "end": end_sample / sample_rate
        })

    return segments


def create_overlapping_windows(
    waveform,
    window_size=1.0,
    hop_size=0.5,
    sample_rate=16000
):
    """
    Split a waveform into overlapping windows.

    window_size:
        Duration of each window in seconds.

    hop_size:
        Distance between consecutive windows.

    Returns:
        List of dictionaries containing waveform,
        start time and end time.
    """

    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)

    total_samples = waveform.shape[1]

    windows = []

    start = 0

    while start + window_samples <= total_samples:

        end = start + window_samples

        window = waveform[:, start:end]

        windows.append({
            "waveform": window,
            "start": start / sample_rate,
            "end": end / sample_rate
        })

        start += hop_samples

    return windows