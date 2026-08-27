import torch
import re 

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)


class SpeechRateAnalyzer:

    def __init__(
        self,
        model_name="openai/whisper-tiny"
    ):

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        print(
            f"Loading Whisper model on: "
            f"{self.device}"
        )

        # ----------------------------------------------------
        # PROCESSOR
        # ----------------------------------------------------

        self.processor = (
            WhisperProcessor.from_pretrained(
                model_name
            )
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = (
            WhisperForConditionalGeneration
            .from_pretrained(
                model_name,
                low_cpu_mem_usage=False
            )
        )

        # ----------------------------------------------------
        # MOVE MODEL TO DEVICE
        # ----------------------------------------------------

        self.model = (
            self.model.to(
                self.device
            )
        )

        self.model.eval()

    def transcribe(
        self,
        waveform,
        sample_rate=16000
    ):

        # Convert [1, samples] → [samples]
        audio = waveform.squeeze(0).cpu().numpy()

        # Whisper requires 16 kHz audio
        if sample_rate != 16000:
            raise ValueError(
                "Whisper requires audio sampled at 16 kHz"
            )

        # Convert audio into Whisper input features
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(
            self.device
        )

        # Generate transcription
        with torch.no_grad():

            predicted_ids = self.model.generate(
                input_features,
                language="english",
                task="transcribe"
            )

        # Convert tokens to text
        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return text


def count_words(text):

    words = re.findall(
        r"\b[\w']+\b",
        text
    )

    return len(words)


def calculate_speech_rate(
    text,
    speech_duration
):

    if speech_duration <= 0:
        return 0.0, 0.0

    word_count = count_words(text)

    words_per_second = (
        word_count / speech_duration
    )

    words_per_minute = (
        words_per_second * 60
    )

    return words_per_second, words_per_minute