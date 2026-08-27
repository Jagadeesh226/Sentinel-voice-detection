import torch
from silero_vad import load_silero_vad, get_speech_timestamps


SAMPLE_RATE = 16000


class VoiceActivityDetector:

    def __init__(self):
        self.model = load_silero_vad()

    def detect(self, waveform):
        """
        waveform:
            torch.Tensor with shape [1, samples]
        """

        # Silero VAD expects a 1-D tensor
        audio = waveform.squeeze(0)

        speech_timestamps = get_speech_timestamps(
            audio,
            self.model,
            sampling_rate=SAMPLE_RATE
        )

        return speech_timestamps