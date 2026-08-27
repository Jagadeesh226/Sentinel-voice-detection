import torch

from transformers import (
    WavLMModel,
    Wav2Vec2FeatureExtractor
)


class WavLMFeatureExtractor:

    def __init__(self):

        print("Loading WavLM...")

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        model_name = "microsoft/wavlm-base-plus"

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name
        )

        self.model = WavLMModel.from_pretrained(
            model_name
        )

        self.model.to(self.device)

        self.model.eval()

    def extract_embedding(
        self,
        waveform,
        sample_rate=16000
    ):
        """
        Extract WavLM embedding from one audio window.

        Input:
            waveform: [1, samples]

        Output:
            embedding: [1, hidden_size]
        """

        if sample_rate != 16000:
            raise ValueError(
                "WavLM requires audio sampled at 16 kHz"
            )

        # [1, samples] → [samples]
        audio = waveform.squeeze(0).cpu().numpy()

        # Convert waveform into model input
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        input_values = inputs.input_values.to(
            self.device
        )

        # WavLM forward pass
        with torch.no_grad():

            outputs = self.model(
                input_values
            )

        # Hidden states
        hidden_states = outputs.last_hidden_state

        # Mean pooling over time
        embedding = hidden_states.mean(
            dim=1
        )

        return embedding.cpu()