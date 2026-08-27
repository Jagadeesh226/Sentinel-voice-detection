import torch

from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalization import normalize_audio
from src.preprocessing.vad import VoiceActivityDetector

from src.segmentation.windows import (
    extract_speech_segments,
    create_overlapping_windows
)

from src.features.acoustic import extract_acoustic_features
from src.features.temporal import extract_temporal_features

from src.features.wavlm import WavLMFeatureExtractor

from src.models.projection import (
    AcousticProjection,
    WavLMProjection
)

from src.models.transformer import TemporalTransformer
from src.models.attention import AttentionPooling
from src.models.global_fusion import GlobalFeatureFusion
from src.models.classifier import FatigueClassifier
from src.features.speech_rate import (
    SpeechRateAnalyzer,
    calculate_speech_rate,
    count_words
)


TEST_FILE = "data/raw/test.wav"


print("\n")
print("========================================")
print(" SENTINEL VOICE FATIGUE")
print(" END-TO-END PIPELINE TEST")
print("========================================")


# ============================================================
# 1. LOAD AUDIO
# ============================================================

print("\n[1/9] Loading audio...")

audio, sample_rate = load_audio(TEST_FILE)

audio = normalize_audio(audio)

print("Audio shape:", audio.shape)
print("Sample rate:", sample_rate)


# ============================================================
# 2. VAD
# ============================================================

print("\n[2/9] Running VAD...")

vad = VoiceActivityDetector()

speech_timestamps = vad.detect(audio)

print(
    "Speech segments:",
    len(speech_timestamps)
)


# ============================================================
# 3. TARGET SPEAKER
# ============================================================

print("\n[3/9] Target speaker...")

print(
    "Use the target-speaker segments obtained from your "
    "existing ECAPA speaker-verification module."
)

print(
    "For this integration test, we will use the speech "
    "segments returned by VAD."
)

target_timestamps = speech_timestamps


# ============================================================
# 4. TEMPORAL WINDOWS
# ============================================================

print("\n[4/9] Creating temporal windows...")

segments = extract_speech_segments(
    audio,
    target_timestamps,
    sample_rate
)

all_windows = []

for segment in segments:

    duration = (
        segment["end"] -
        segment["start"]
    )

    if duration < 1.0:
        continue

    windows = create_overlapping_windows(
        segment["waveform"],
        window_size=1.0,
        hop_size=0.5,
        sample_rate=sample_rate
    )

    all_windows.extend(windows)


print(
    "Total windows:",
    len(all_windows)
)


if len(all_windows) == 0:

    raise RuntimeError(
        "No valid 1-second speech windows found."
    )


# ============================================================
# 5. LOAD FEATURE EXTRACTORS
# ============================================================

print("\n[5/9] Loading feature extractors...")

wavlm_extractor = WavLMFeatureExtractor()

acoustic_projection = AcousticProjection()

wavlm_projection = WavLMProjection()

print("Feature extractors loaded.")


# ============================================================
# 6. EXTRACT ACOUSTIC + WAVLM FEATURES
# ============================================================

print("\n[6/9] Extracting features...")

acoustic_features_list = []
wavlm_features_list = []


for i, window in enumerate(all_windows):

    waveform = window["waveform"]

    print(
        f"Processing window "
        f"{i + 1}/{len(all_windows)}"
    )

    # -------------------------
    # Acoustic
    # -------------------------

    acoustic = extract_acoustic_features(
        waveform,
        sample_rate
    )

    acoustic = torch.tensor(
        acoustic,
        dtype=torch.float32
    ).unsqueeze(0)

    acoustic_128 = acoustic_projection(
        acoustic
    )

    acoustic_features_list.append(
        acoustic_128.squeeze(0)
    )

    # -------------------------
    # WavLM
    # -------------------------

    wavlm = wavlm_extractor.extract_embedding(
        waveform,
        sample_rate
    )

    wavlm_128 = wavlm_projection(
        wavlm
    )

    wavlm_features_list.append(
        wavlm_128.squeeze(0)
    )


# Stack temporal features

acoustic_sequence = torch.stack(
    acoustic_features_list
)

wavlm_sequence = torch.stack(
    wavlm_features_list
)


print(
    "\nAcoustic sequence:",
    acoustic_sequence.shape
)

print(
    "WavLM sequence:",
    wavlm_sequence.shape
)


# ============================================================
# 7. FUSION + TRANSFORMER + ATTENTION
# ============================================================

print("\n[7/9] Running temporal model...")

# -----------------------------------------
# Feature fusion
# -----------------------------------------

temporal_sequence = torch.cat(
    [
        acoustic_sequence,
        wavlm_sequence
    ],
    dim=-1
)

print(
    "Fused sequence:",
    temporal_sequence.shape
)


# -----------------------------------------
# Add batch dimension
# -----------------------------------------

temporal_sequence = temporal_sequence.unsqueeze(0)

print(
    "Transformer input:",
    temporal_sequence.shape
)


# -----------------------------------------
# Transformer
# -----------------------------------------

transformer = TemporalTransformer(
    input_dim=256,
    num_heads=4,
    num_layers=2,
    feedforward_dim=512,
    dropout=0.1
)

transformer_output = transformer(
    temporal_sequence
)

print(
    "Transformer output:",
    transformer_output.shape
)


# -----------------------------------------
# Attention pooling
# -----------------------------------------

attention_pooling = AttentionPooling(
    input_dim=256
)

temporal_representation, attention_weights = (
    attention_pooling(
        transformer_output
    )
)

print(
    "Attention representation:",
    temporal_representation.shape
)


# ============================================================
# 8. GLOBAL FEATURES
# ============================================================

print("\n[8/9] Extracting global features...")

temporal_global = extract_temporal_features(
    target_timestamps,
    audio.shape[1],
    sample_rate
)

print(
    "Temporal global features:",
    temporal_global.shape
)

print(
    "Global feature dimension:",
    len(temporal_global)
)


# ------------------------------------------------------------
# Speech-rate features
# ------------------------------------------------------------

# ------------------------------------------------------------
# Whisper speech-rate analysis
# ------------------------------------------------------------

print("\nRunning Whisper...")

speech_rate_analyzer = SpeechRateAnalyzer()

transcript = speech_rate_analyzer.transcribe(
    audio,
    sample_rate
)

words_per_second, words_per_minute = (
    calculate_speech_rate(
        transcript,
        temporal_global[5]
    )
)

print("\nTranscript:")
print(transcript)

print(
    "\nWords per second:",
    round(words_per_second, 2)
)

print(
    "Words per minute:",
    round(words_per_minute, 2)
)


# ------------------------------------------------------------
# Combine global features
# ------------------------------------------------------------

speech_rate_features = torch.tensor(
    [
        words_per_second,
        words_per_minute
    ],
    dtype=torch.float32
)

temporal_global_tensor = torch.tensor(
    temporal_global,
    dtype=torch.float32
)

global_features = torch.cat(
    [
        temporal_global_tensor,
        speech_rate_features
    ],
    dim=0
)

print(
    "\nGlobal feature vector:",
    global_features.shape
)

print(
    "Global feature dimension:",
    global_features.shape[0]
)


# ============================================================
# FINAL FUSION
# ============================================================

print("\nRunning final fusion...")

global_features = global_features.unsqueeze(0)

global_fusion = GlobalFeatureFusion(
    temporal_dim=256,
    global_dim=11,
    output_dim=267
)

final_features = global_fusion(
    temporal_representation,
    global_features
)

print(
    "Temporal representation:",
    temporal_representation.shape
)

print(
    "Global features:",
    global_features.shape
)

print(
    "Final feature vector:",
    final_features.shape
)


# ============================================================
# FATIGUE CLASSIFIER
# ============================================================

print("\nRunning fatigue classifier...")

classifier = FatigueClassifier(
    input_dim=267,
    hidden_dim1=128,
    hidden_dim2=64,
    num_classes=3,
    dropout=0.3
)

logits = classifier(
    final_features
)

probabilities = torch.softmax(
    logits,
    dim=1
)

prediction = torch.argmax(
    probabilities,
    dim=1
)


# ============================================================
# RESULTS
# ============================================================

class_names = [
    "Alert",
    "Mild Fatigue",
    "High Fatigue"
]

predicted_class = class_names[
    prediction.item()
]

print("\n")
print("========================================")
print(" SENTINEL VOICE FATIGUE RESULT")
print("========================================")

print("\nTranscript:")
print(transcript)

print(
    "\nSpeech duration:",
    round(
        temporal_global[5],
        2
    ),
    "seconds"
)

print(
    "Word count:",
    count_words(transcript)
)

print(
    "Words per minute:",
    round(
        words_per_minute,
        2
    )
)

print(
    "\nFinal feature shape:",
    final_features.shape
)

print("\nFatigue probabilities:")

print(
    "Alert:",
    round(
        probabilities[0][0].item(),
        4
    )
)

print(
    "Mild Fatigue:",
    round(
        probabilities[0][1].item(),
        4
    )
)

print(
    "High Fatigue:",
    round(
        probabilities[0][2].item(),
        4
    )
)

print(
    "\nPredicted class:",
    predicted_class
)

print("========================================")

