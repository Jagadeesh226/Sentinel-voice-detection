import numpy as np
import librosa
import parselmouth

SAMPLE_RATE = 16000


def summarize_feature(feature):
    """
    Convert a [features, time] matrix
    into a fixed-size vector using mean and std.
    """

    mean = np.mean(feature, axis=1)
    std = np.std(feature, axis=1)

    return np.concatenate([mean, std])


def extract_mfcc(
    waveform,
    sample_rate=SAMPLE_RATE,
    n_mfcc=13
):
    """
    Extract MFCC features.

    Returns:
        numpy array of shape [n_mfcc, time]
    """

    audio = waveform.squeeze(0).cpu().numpy()

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=400,
        hop_length=160
    )

    return mfcc


def extract_pitch(
    waveform,
    sample_rate=SAMPLE_RATE
):
    """
    Extract fundamental frequency (F0).
    """

    audio = waveform.squeeze(0).cpu().numpy()

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        frame_length=400,
        hop_length=160
    )

    return f0, voiced_flag, voiced_prob


def summarize_pitch(f0):
    """
    Summarize valid F0 values.
    """

    valid_f0 = f0[~np.isnan(f0)]

    if len(valid_f0) == 0:
        return np.zeros(4, dtype=np.float32)

    return np.array([
        np.mean(valid_f0),
        np.std(valid_f0),
        np.min(valid_f0),
        np.max(valid_f0)
    ], dtype=np.float32)


def extract_energy(
    waveform,
    sample_rate=SAMPLE_RATE
):
    """
    Extract RMS energy statistics.
    """

    audio = waveform.squeeze(0).cpu().numpy()

    rms = librosa.feature.rms(
        y=audio,
        frame_length=400,
        hop_length=160
    )[0]

    return np.array([
        np.mean(rms),
        np.std(rms),
        np.min(rms),
        np.max(rms)
    ], dtype=np.float32)

def extract_acoustic_features(
    waveform,
    sample_rate=SAMPLE_RATE
):
    """
    Extract complete acoustic features.

    Current features:

    MFCC:
        26

    Pitch:
        4

    Energy:
        4

    Jitter:
        1

    Shimmer:
        1

    HNR:
        1

    Formants:
        3

    Total:
        40
    """

    # -----------------------------
    # MFCC
    # -----------------------------

    mfcc = extract_mfcc(
        waveform,
        sample_rate
    )

    mfcc_features = summarize_feature(mfcc)

    # -----------------------------
    # Pitch
    # -----------------------------

    f0, voiced_flag, voiced_prob = extract_pitch(
        waveform,
        sample_rate
    )

    pitch_features = summarize_pitch(f0)

    # -----------------------------
    # Energy
    # -----------------------------

    energy_features = extract_energy(
        waveform,
        sample_rate
    )

    # -----------------------------
    # Jitter
    # -----------------------------

    jitter = extract_jitter(
        waveform,
        sample_rate
    )

    # -----------------------------
    # Shimmer
    # -----------------------------

    shimmer = extract_shimmer(
        waveform,
        sample_rate
    )

    # -----------------------------
    # HNR
    # -----------------------------

    hnr = extract_hnr(
        waveform,
        sample_rate
    )

    # -----------------------------
    # Formants
    # -----------------------------

    formant_features = extract_formants(
        waveform,
        sample_rate
    )

    # -----------------------------
    # Combine
    # -----------------------------

    features = np.concatenate([
        mfcc_features,
        pitch_features,
        energy_features,
        np.array([jitter], dtype=np.float32),
        np.array([shimmer], dtype=np.float32),
        np.array([hnr], dtype=np.float32),
        formant_features
    ])

    return features.astype(np.float32)

def extract_jitter(waveform, sample_rate=SAMPLE_RATE):
    """
    Extract local jitter from the speech signal.
    """

    audio = waveform.squeeze(0).cpu().numpy()

    sound = parselmouth.Sound(
        audio,
        sampling_frequency=sample_rate
    )

    point_process = parselmouth.praat.call(
        sound,
        "To PointProcess (periodic, cc)",
        75,
        500
    )

    jitter = parselmouth.praat.call(
        point_process,
        "Get jitter (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3
    )

    if np.isnan(jitter):
        return 0.0

    return float(jitter)

def extract_shimmer(waveform, sample_rate=SAMPLE_RATE):
    """
    Extract local shimmer from the speech signal.
    """

    audio = waveform.squeeze(0).cpu().numpy()

    sound = parselmouth.Sound(
        audio,
        sampling_frequency=sample_rate
    )

    point_process = parselmouth.praat.call(
        sound,
        "To PointProcess (periodic, cc)",
        75,
        500
    )

    shimmer = parselmouth.praat.call(
        [sound, point_process],
        "Get shimmer (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6
    )

    if np.isnan(shimmer):
        return 0.0

    return float(shimmer)

def extract_hnr(waveform, sample_rate=SAMPLE_RATE):
    """
    Extract Harmonics-to-Noise Ratio.
    """

    audio = waveform.squeeze(0).cpu().numpy()

    sound = parselmouth.Sound(
        audio,
        sampling_frequency=sample_rate
    )

    harmonicity = sound.to_harmonicity_cc(
        time_step=0.01,
        minimum_pitch=75
    )

    hnr = parselmouth.praat.call(
        harmonicity,
        "Get mean",
        0,
        0
    )

    if np.isnan(hnr):
        return 0.0

    return float(hnr)

def extract_formants(waveform, sample_rate=SAMPLE_RATE):
    """
    Extract mean F1, F2 and F3.
    """

    audio = waveform.squeeze(0).cpu().numpy()

    sound = parselmouth.Sound(
        audio,
        sampling_frequency=sample_rate
    )

    formant = sound.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5,
        maximum_formant=5500,
        window_length=0.025,
        pre_emphasis_from=50
    )

    duration = sound.get_total_duration()

    f1_values = []
    f2_values = []
    f3_values = []

    times = np.arange(
        0,
        duration,
        0.01
    )

    for time in times:

        f1 = formant.get_value_at_time(
            1,
            time
        )

        f2 = formant.get_value_at_time(
            2,
            time
        )

        f3 = formant.get_value_at_time(
            3,
            time
        )

        if not np.isnan(f1):
            f1_values.append(f1)

        if not np.isnan(f2):
            f2_values.append(f2)

        if not np.isnan(f3):
            f3_values.append(f3)

    def safe_mean(values):
        if len(values) == 0:
            return 0.0

        return float(np.mean(values))

    return np.array([
        safe_mean(f1_values),
        safe_mean(f2_values),
        safe_mean(f3_values)
    ], dtype=np.float32)

