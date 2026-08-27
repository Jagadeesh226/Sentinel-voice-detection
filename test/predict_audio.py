import sys
from pathlib import Path

import torch

from src.personalization.baseline import SpeakerBaseline
from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio
from src.preprocessing.vad import VoiceActivityDetector
from src.speaker.verification import SpeakerVerifier

from src.segmentation.windows import (
    extract_speech_segments,
    create_overlapping_windows
)

from src.features.acoustic import (
    extract_acoustic_features
)

from src.features.wavlm import (
    WavLMFeatureExtractor
)

from src.features.temporal import (
    extract_temporal_features
)

from src.features.speech_rate import (
    SpeechRateAnalyzer,
    calculate_speech_rate
)

from src.models.fatigue_model import FatigueModel

BASELINE_DIR = "data/baselines"

baseline_manager = SpeakerBaseline(
    baseline_dir=BASELINE_DIR
)

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_PATH = Path(
    "models/best_fatigue_model_v3.pt"
)
CHECKPOINT = None
GLOBAL_MEAN = None
GLOBAL_STD = None

SAMPLE_RATE = 16000

WINDOW_SIZE = 1.0
HOP_SIZE = 0.5

SPEAKER_THRESHOLD = 0.25

CLASS_NAMES = [
    "Alert",
    "Mild Fatigue",
    "High Fatigue"
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global GLOBAL_MEAN
    global GLOBAL_STD

    print("\nLoading fatigue model...")

    model = FatigueModel(
        num_classes=3
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # LOAD NORMALIZATION STATISTICS
    # --------------------------------------------------------

    if "global_mean" in checkpoint:
        GLOBAL_MEAN = checkpoint["global_mean"].float()

    if "global_std" in checkpoint:
        GLOBAL_STD = checkpoint["global_std"].float()

    if GLOBAL_MEAN is None or GLOBAL_STD is None:

        print(
            "WARNING: Normalization statistics "
            "not found in checkpoint."
        )

    else:

        print(
            "✓ Normalization statistics loaded"
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    print("✓ Fatigue model loaded")

    return model
# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(
    audio_path
):

    print("\n")
    print("=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD AUDIO
    # --------------------------------------------------------

    print("\nLoading audio...")

    waveform, sample_rate = load_audio(
        str(audio_path)
    )

    print(
        f"✓ Audio loaded"
    )

    print(
        f"  Shape: {waveform.shape}"
    )

    print(
        f"  Sample rate: {sample_rate}"
    )

    duration = (
        waveform.shape[1] /
        sample_rate
    )

    print(
        f"  Duration: {duration:.2f}s"
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    waveform = normalize_audio(
        waveform
    )

    print(
        "✓ Audio normalized"
    )

    # --------------------------------------------------------
    # VAD
    # --------------------------------------------------------

    print(
        "\nRunning VAD..."
    )

    vad = VoiceActivityDetector()

    speech_timestamps = vad.detect(
        waveform
    )

    print(
        f"✓ Speech segments: "
        f"{len(speech_timestamps)}"
    )

    if len(speech_timestamps) == 0:

        raise ValueError(
            "No speech detected in audio."
        )

    # --------------------------------------------------------
    # SPEAKER VERIFICATION
    # --------------------------------------------------------

    print(
        "\nRunning speaker verification..."
    )

    verifier = SpeakerVerifier()

    enrollment_embedding = (
        verifier.get_embedding(
            waveform
        )
    )

    speech_segments = (
        extract_speech_segments(
            waveform,
            speech_timestamps,
            sample_rate
        )
    )

    target_segments = []

    for segment in speech_segments:

        segment_waveform = (
            segment["waveform"]
        )

        segment_embedding = (
            verifier.get_embedding(
                segment_waveform
            )
        )

        score = verifier.similarity(
            enrollment_embedding,
            segment_embedding
        )

        if score >= SPEAKER_THRESHOLD:

            target_segments.append(
                segment
            )

    print(
        f"✓ Target speaker segments: "
        f"{len(target_segments)}"
    )

    if len(target_segments) == 0:

        raise ValueError(
            "No valid target-speaker speech found."
        )

    # --------------------------------------------------------
    # WINDOWS
    # --------------------------------------------------------

    print(
        "\nCreating temporal windows..."
    )

    windows = []

    for segment in target_segments:

        segment_windows = (
            create_overlapping_windows(
                segment["waveform"],
                window_size=WINDOW_SIZE,
                hop_size=HOP_SIZE,
                sample_rate=sample_rate
            )
        )

        windows.extend(
            segment_windows
        )

    print(
        f"✓ Windows created: "
        f"{len(windows)}"
    )

    if len(windows) == 0:

        raise ValueError(
            "No temporal windows created."
        )

    # --------------------------------------------------------
    # LOAD WavLM
    # --------------------------------------------------------

    print(
        "\nLoading WavLM..."
    )

    wavlm_extractor = (
        WavLMFeatureExtractor()
    )

    # --------------------------------------------------------
    # ACOUSTIC + WavLM
    # --------------------------------------------------------

    acoustic_features = []
    wavlm_features = []

    for window_info in windows:

        window_waveform = (
            window_info["waveform"]
        )

        # Acoustic
        acoustic = (
            extract_acoustic_features(
                window_waveform,
                sample_rate
            )
        )

        acoustic = torch.tensor(
            acoustic,
            dtype=torch.float32
        )

        # WavLM
        wavlm = (
            wavlm_extractor.extract_embedding(
                window_waveform,
                sample_rate
            )
        )

        wavlm = wavlm.squeeze()
        wavlm = wavlm.detach().cpu()

        if acoustic.numel() != 40:

            raise ValueError(
                f"Expected 40 acoustic features, "
                f"got {acoustic.numel()}"
            )

        if wavlm.numel() != 768:

            raise ValueError(
                f"Expected 768 WavLM features, "
                f"got {wavlm.numel()}"
            )

        acoustic_features.append(
            acoustic
        )

        wavlm_features.append(
            wavlm
        )

    acoustic_features = torch.stack(
        acoustic_features
    )

    wavlm_features = torch.stack(
        wavlm_features
    )

    print(
        f"✓ Acoustic features: "
        f"{acoustic_features.shape}"
    )

    print(
        f"✓ WavLM features: "
        f"{wavlm_features.shape}"
    )

    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    print(
        "\nExtracting temporal features..."
    )

    total_audio_samples = (
        waveform.shape[1]
    )

    temporal_features = (
        extract_temporal_features(
            speech_timestamps,
            total_audio_samples,
            sample_rate
        )
    )

    temporal_features = torch.tensor(
        temporal_features,
        dtype=torch.float32
    )

    print(
        f"✓ Temporal features: "
        f"{temporal_features.shape}"
    )

    # --------------------------------------------------------
    # SPEECH RATE
    # --------------------------------------------------------

    print(
        "\nCalculating speech rate..."
    )

    speech_rate_analyzer = (
        SpeechRateAnalyzer()
    )

    transcript = (
        speech_rate_analyzer.transcribe(
            waveform,
            sample_rate
        )
    )

    speech_duration = 0.0

    for segment in speech_timestamps:

        segment_duration = (
            segment["end"]
            - segment["start"]
        ) / sample_rate

        speech_duration += (
            segment_duration
        )

    words_per_second, words_per_minute = (
        calculate_speech_rate(
            transcript,
            speech_duration
        )
    )

    speech_rate_features = torch.tensor(
        [
            words_per_second,
            words_per_minute
        ],
        dtype=torch.float32
    )

    print(
        f"✓ WPS: "
        f"{words_per_second:.2f}"
    )

    print(
        f"✓ WPM: "
        f"{words_per_minute:.2f}"
    )

    # --------------------------------------------------------
    # GLOBAL FEATURES
    # --------------------------------------------------------

    global_features = torch.cat(
        [
            temporal_features,
            speech_rate_features
        ],
        dim=0
    )

    if global_features.numel() != 11:

        raise ValueError(
            f"Expected 11 global features, "
            f"got {global_features.numel()}"
        )

    print(
        f"✓ Global features: "
        f"{global_features.shape}"
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {
        "acoustic": acoustic_features,
        "wavlm": wavlm_features,
        "global_features": global_features
    }


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    features,
    speaker_id
):

    acoustic = features[
        "acoustic"
    ]

    wavlm = features[
        "wavlm"
    ]

    global_features = features[
        "global_features"
    ].float()

    # ========================================================
    # RELATIVE SPEECH RATE
    # ========================================================
    #
    # Feature index:
    #
    # 0  Pause count
    # 1  Mean pause
    # 2  Std pause
    # 3  Max pause
    # 4  Total pause
    # 5  Speech duration
    # 6  Silence duration
    # 7  Speech activity
    # 8  Speech segments
    # 9  Words per second
    # 10 Words per minute
    #
    # New:
    #
    # 11 Relative speech rate
    #
    # For a real-time recording we need an ALERT
    # baseline speech rate.
    #
    # For the POC, use the alert WPM baseline
    # calculated from the paired real recordings.
    # ========================================================
    CURRENT_WPM=global_features[10].item()
    baseline_manager=SpeakerBaseline()
    baseline=baseline_manager.load(
        speaker_id
    )
    if baseline is not None:
        ALERT_BASELINE_WPM=baseline["wpm"]['mean']
        relative_speech_rate_value=(
            CURRENT_WPM/
            ALERT_BASELINE_WPM
        )
        print()
        print(
            f"Speaker:{speaker_id}"
        )
        print()
        print(
            f"Current WPM: "
            f"{CURRENT_WPM:.2f}"
        )
        print()
        print(
            f"Personal baseline WPM: "
            f"{ALERT_BASELINE_WPM:.2f}"
        )
        print()
        print(
            f"Relative speech rate: "
            f"{relative_speech_rate_value:4f}"
        )
    else:
        relative_speech_rate_value=1.0
        print()
        print(
            f"Speaker:{speaker_id}"
        )
        print()
        print("No personal baseline found")
        print(
            "Running initial prediction"
            "for candidate evaluation"
        )
        print()
        print(
            "Relative speech rate"
            "1.0000(temporary)"

        )
    relative_speech_rate=torch.tensor(
        [relative_speech_rate_value],
        dtype=torch.float32,
        device=global_features.device
    )

    global_features=torch.cat(
        [
            global_features,
            relative_speech_rate
        ],
        dim=0
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    if (
        GLOBAL_MEAN is not None
        and GLOBAL_STD is not None
    ):

        global_mean = (
            GLOBAL_MEAN
            .to(global_features.device)
        )

        global_std = (
            GLOBAL_STD
            .to(global_features.device)
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if global_mean.shape[0] != 12:

            raise ValueError(
                f"Expected GLOBAL_MEAN to contain "
                f"12 features, got "
                f"{global_mean.shape[0]}"
            )

        if global_std.shape[0] != 12:

            raise ValueError(
                f"Expected GLOBAL_STD to contain "
                f"12 features, got "
                f"{global_std.shape[0]}"
            )

        global_features = (
            (global_features - global_mean)
            / global_std
        )

        print(
            "\nNormalized global features:"
        )

        print(
            global_features
        )

    else:

        raise RuntimeError(
            "Global normalization statistics "
            "are missing"
        )

    # ========================================================
    # BATCH DIMENSION
    # ========================================================

    acoustic = acoustic.unsqueeze(
        0
    )

    wavlm = wavlm.unsqueeze(
        0
    )

    global_features = (
        global_features.unsqueeze(0)
    )

    sequence_length = (
        acoustic.shape[1]
    )

    attention_mask = torch.ones(
        1,
        sequence_length,
        dtype=torch.bool
    )

    # ========================================================
    # MOVE TO DEVICE
    # ========================================================

    acoustic = acoustic.to(
        DEVICE
    )

    wavlm = wavlm.to(
        DEVICE
    )

    global_features = global_features.to(
        DEVICE
    )

    attention_mask = attention_mask.to(
        DEVICE
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    with torch.no_grad():

        output = model(
            acoustic=acoustic,
            wavlm=wavlm,
            global_features=global_features,
            attention_mask=attention_mask
        )

        logits = output[
            "logits"
        ]

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]

        prediction = torch.argmax(
            probabilities
        ).item()

    return (
        prediction,
        probabilities
    )
# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Argument
    # --------------------------------------------------------

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python -m test.predict_audio "
            "path/to/audio.wav"
        )

        sys.exit(1)

    audio_path = Path(
        sys.argv[1]
    )

    if not audio_path.exists():

        print(
            f"\nERROR: File not found:"
            f"\n{audio_path}"
        )

        sys.exit(1)

    speaker_id=audio_path.stem
    print(
        f"\nSpeaker ID:{speaker_id}"
    )

    print("\n")
    print("=" * 60)
    print("SENTINEL FATIGUE DETECTION POC")
    print("=" * 60)

    print(
        f"\nAudio: {audio_path}"
    )

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = extract_features(
        audio_path
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FATIGUE PREDICTION")
    print("=" * 60)

    prediction, probabilities = predict(
        model,
        features,
        speaker_id=speaker_id
    )

    predicted_class = CLASS_NAMES[
        prediction
    ]

    confidence = (
        probabilities[prediction].item()
        * 100
    )

    print(
        f"\nFatigue Level: "
        f"{predicted_class}"
    )

    print(
        f"Confidence: "
        f"{confidence:.2f}%"
    )

    print("\nProbabilities:")

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = (
            probabilities[i].item()
            * 100
        )

        print(
            f"  {class_name:<15}: "
            f"{probability:.2f}%"
        )

    print("\n")
    print("=" * 60)
    print("POC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()