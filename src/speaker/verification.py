import torch
import torchaudio

from speechbrain.inference.speaker import SpeakerRecognition


SAMPLE_RATE = 16000


class SpeakerVerifier:

    def __init__(self):
        print("Loading ECAPA-TDNN...")

        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )

    def get_embedding(self, waveform):
        """
        Generate a speaker embedding from a waveform.

        waveform shape:
            [1, samples]
        """

        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)

        return embedding

    def similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two speaker embeddings.
        """

        embedding1 = embedding1.squeeze()
        embedding2 = embedding2.squeeze()

        similarity = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        )

        return similarity.item()

    def verify_segment(
        self,
        enrollment_embedding,
        segment_waveform,
        threshold=0.25
    ):
        """
        Determine whether a speech segment belongs
        to the target speaker.
        """

        segment_embedding = self.get_embedding(segment_waveform)

        score = self.similarity(
            enrollment_embedding,
            segment_embedding
        )

        is_target = score >= threshold

        return score, is_target