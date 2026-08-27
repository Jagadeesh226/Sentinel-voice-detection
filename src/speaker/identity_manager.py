import torch

from pathlib import Path


class SpeakerIdentityManager:

    def __init__(
        self,
        identity_dir="data/speaker_identities"
    ):

        self.identity_dir = Path(
            identity_dir
        )

        self.identity_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # GET SPEAKER PATH
    # ========================================================

    def get_identity_path(
        self,
        speaker_id
    ):

        return (
            self.identity_dir /
            f"{speaker_id}.pt"
        )

    # ========================================================
    # CHECK IF IDENTITY EXISTS
    # ========================================================

    def exists(
        self,
        speaker_id
    ):

        path = self.get_identity_path(
            speaker_id
        )

        return path.exists()

    # ========================================================
    # SAVE IDENTITY
    # ========================================================

    def save(
        self,
        speaker_id,
        embedding
    ):

        path = self.get_identity_path(
            speaker_id
        )

        embedding = (
            embedding
            .detach()
            .cpu()
        )

        torch.save(
            embedding,
            path
        )

        return path

    # ========================================================
    # LOAD IDENTITY
    # ========================================================

    def load(
        self,
        speaker_id
    ):

        path = self.get_identity_path(
            speaker_id
        )

        if not path.exists():

            return None

        return torch.load(
            path,
            map_location="cpu"
        )

    # ========================================================
    # GET ALL REGISTERED SPEAKERS
    # ========================================================

    def get_all_speakers(
        self
    ):

        identity_files = sorted(
            self.identity_dir.glob(
                "*.pt"
            )
        )

        speaker_ids = []

        for path in identity_files:

            speaker_ids.append(
                path.stem
            )

        return speaker_ids

    # ========================================================
    # CALCULATE EMBEDDING SIMILARITY
    # ========================================================

    def calculate_similarity(
        self,
        embedding1,
        embedding2
    ):

        embedding1 = (
            embedding1
            .squeeze()
            .float()
        )

        embedding2 = (
            embedding2
            .squeeze()
            .float()
        )

        similarity = (
            torch.nn.functional
            .cosine_similarity(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0)
            )
        )

        return similarity.item()

    # ========================================================
    # FIND EXISTING SPEAKER
    # ========================================================

    def find_existing_speaker(
        self,
        embedding,
        threshold=0.70,
        exclude_speaker_id=None
    ):

        """
        Compare an embedding against all registered
        speaker identities.

        Returns the best matching speaker if the
        similarity exceeds the threshold.
        """

        speaker_ids = (
            self.get_all_speakers()
        )

        best_match_id = None

        best_similarity = -1.0

        for existing_speaker_id in speaker_ids:

            # ----------------------------------------------
            # Skip a specific speaker if requested
            # ----------------------------------------------

            if (
                exclude_speaker_id is not None
                and
                str(existing_speaker_id)
                ==
                str(exclude_speaker_id)
            ):

                continue

            # ----------------------------------------------
            # Load existing embedding
            # ----------------------------------------------

            existing_embedding = (
                self.load(
                    existing_speaker_id
                )
            )

            if existing_embedding is None:

                continue

            # ----------------------------------------------
            # Calculate similarity
            # ----------------------------------------------

            similarity = (
                self.calculate_similarity(
                    embedding,
                    existing_embedding
                )
            )

            # ----------------------------------------------
            # Keep best match
            # ----------------------------------------------

            if similarity > best_similarity:

                best_similarity = similarity

                best_match_id = (
                    existing_speaker_id
                )

        # ----------------------------------------------------
        # CHECK THRESHOLD
        # ----------------------------------------------------

        if (
            best_match_id is not None
            and
            best_similarity >= threshold
        ):

            return {

                "match_found": True,

                "speaker_id": (
                    best_match_id
                ),

                "similarity": (
                    best_similarity
                ),

                "threshold": threshold
            }

        # ----------------------------------------------------
        # NO MATCH
        # ----------------------------------------------------

        return {

            "match_found": False,

            "speaker_id": None,

            "similarity": (
                best_similarity
            ),

            "threshold": threshold
        }

    # ========================================================
    # DELETE IDENTITY
    # ========================================================

    def delete(
        self,
        speaker_id
    ):

        path = self.get_identity_path(
            speaker_id
        )

        if path.exists():

            path.unlink()

            return True

        return False

