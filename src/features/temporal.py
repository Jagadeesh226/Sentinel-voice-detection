import numpy as np


def extract_pause_features(
    speech_timestamps,
    sample_rate=16000
):
    """
    Extract pause-related features from VAD timestamps.

    speech_timestamps:
        List of dictionaries containing:
        start and end sample indices.
    """

    if len(speech_timestamps) < 2:
        return np.zeros(5, dtype=np.float32)

    pauses = []

    for i in range(len(speech_timestamps) - 1):

        current_end = speech_timestamps[i]["end"]
        next_start = speech_timestamps[i + 1]["start"]

        pause_duration = (
            next_start - current_end
        ) / sample_rate

        if pause_duration > 0:
            pauses.append(pause_duration)

    if len(pauses) == 0:
        return np.zeros(5, dtype=np.float32)

    pauses = np.array(pauses)

    return np.array([
        len(pauses),
        np.mean(pauses),
        np.std(pauses),
        np.max(pauses),
        np.sum(pauses)
    ], dtype=np.float32)

def extract_speech_activity_features(
    speech_timestamps,
    total_audio_samples,
    sample_rate=16000
):
    """
    Extract overall speech activity statistics.
    """

    total_audio_duration = (
        total_audio_samples / sample_rate
    )

    total_speech_duration = 0.0

    for segment in speech_timestamps:

        duration = (
            segment["end"] - segment["start"]
        ) / sample_rate

        total_speech_duration += duration

    if total_audio_duration > 0:
        speech_ratio = (
            total_speech_duration /
            total_audio_duration
        )
    else:
        speech_ratio = 0.0

    silence_duration = (
        total_audio_duration -
        total_speech_duration
    )

    return np.array([
        total_speech_duration,
        silence_duration,
        speech_ratio,
        len(speech_timestamps)
    ], dtype=np.float32)

def extract_temporal_features(
    speech_timestamps,
    total_audio_samples,
    sample_rate=16000
):
    """
    Extract all sequence-level temporal features.
    """

    pause_features = extract_pause_features(
        speech_timestamps,
        sample_rate
    )

    activity_features = extract_speech_activity_features(
        speech_timestamps,
        total_audio_samples,
        sample_rate
    )

    return np.concatenate([
        pause_features,
        activity_features
    ]).astype(np.float32)