import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd


class SentinelVoiceDataset(Dataset):

    def __init__(
        self,
        split,
        metadata_file="data/dataset_split.csv",
        feature_dir="data/features",
        global_mean=None,
        global_std=None
    ):

        self.global_mean = global_mean
        self.global_std = global_std

        self.split = split

        self.metadata = pd.read_csv(
            metadata_file
        )

        self.metadata = self.metadata[
            self.metadata["split"] == split
        ].copy()

        self.feature_dir = Path(
            feature_dir
        )

        if len(self.metadata) == 0:

            raise ValueError(
                f"No samples found for split: {split}"
            )

        print(
            f"{split.capitalize()} samples: "
            f"{len(self.metadata)}"
        )

    def __len__(self):

        return len(self.metadata)

    # ========================================================
    # FIND ALERT BASELINE
    # ========================================================

    def get_alert_wpm(
        self,
        speaker_id,
        source
    ):

        """
        Find the Alert recording belonging to
        the same speaker and source.

        Preference:
        1. Paired Alert recording
        2. Any Alert recording for that speaker
        """

        # ----------------------------------------------------
        # First try paired Alert
        # ----------------------------------------------------

        alert_rows = self.metadata_all[
            (self.metadata_all["speaker_id"] == speaker_id) &
            (self.metadata_all["source"] == source) &
            (self.metadata_all["pairing"] == "paired") &
            (self.metadata_all["class_name"] == "alert")
        ]

        # ----------------------------------------------------
        # If unavailable, try any Alert recording
        # ----------------------------------------------------

        if len(alert_rows) == 0:

            alert_rows = self.metadata_all[
                (self.metadata_all["speaker_id"] == speaker_id) &
                (self.metadata_all["source"] == source) &
                (self.metadata_all["class_name"] == "alert")
            ]

        # ----------------------------------------------------
        # No baseline available
        # ----------------------------------------------------

        if len(alert_rows) == 0:

            return None

        alert_row = alert_rows.iloc[0]

        feature_file = (
            f"{alert_row['source']}_"
            f"{alert_row['pairing']}_"
            f"{alert_row['speaker_id']}_"
            f"{alert_row['class_name']}.pt"
        )

        feature_path = (
            self.feature_dir /
            feature_file
        )

        if not feature_path.exists():

            return None

        alert_data = torch.load(
            feature_path,
            map_location="cpu",
            weights_only=False
        )

        alert_global_features = (
            alert_data[
                "global_features"
            ].float()
        )

        # WPM is feature index 10
        alert_wpm = (
            alert_global_features[10].item()
        )

        return alert_wpm

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        speaker_id = row[
            "speaker_id"
        ]

        source = row[
            "source"
        ]

        pairing = row[
            "pairing"
        ]

        class_name = row[
            "class_name"
        ]

        # ----------------------------------------------------
        # Feature filename
        # ----------------------------------------------------

        feature_file = (
            f"{source}_"
            f"{pairing}_"
            f"{speaker_id}_"
            f"{class_name}.pt"
        )

        feature_path = (
            self.feature_dir /
            feature_file
        )

        if not feature_path.exists():

            raise FileNotFoundError(
                f"Feature file not found:\n"
                f"{feature_path}"
            )

        # ----------------------------------------------------
        # Load cached features
        # ----------------------------------------------------

        data = torch.load(
            feature_path,
            map_location="cpu",
            weights_only=False
        )

        acoustic = data[
            "acoustic_features"
        ].float()

        wavlm = data[
            "wavlm_features"
        ].float()

        global_features = data[
            "global_features"
        ].float()

        # ====================================================
        # RELATIVE SPEECH RATE
        # ====================================================

        current_wpm = (
            global_features[10].item()
        )

        alert_wpm = self.get_alert_wpm(
            speaker_id,
            source
        )

        if (
            alert_wpm is not None
            and alert_wpm > 0
        ):

            relative_wpm = (
                current_wpm /
                alert_wpm
            )

        else:

            # No speaker baseline available
            relative_wpm = 1.0

        relative_wpm_feature = torch.tensor(
            [relative_wpm],
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Add relative WPM
        #
        # Original:
        # 11 features
        #
        # New:
        # 12 features
        # ----------------------------------------------------

        global_features = torch.cat(
            [
                global_features,
                relative_wpm_feature
            ],
            dim=0
        )

        # ====================================================
        # GLOBAL FEATURE NORMALIZATION
        # ====================================================

        if self.global_mean is not None:

            global_features = (
                global_features
                - self.global_mean
            ) / self.global_std

        label = torch.tensor(
            int(data["label"]),
            dtype=torch.long
        )

        # ====================================================
        # DIMENSION CHECKS
        # ====================================================

        if acoustic.ndim != 2:

            raise ValueError(
                f"Acoustic features should "
                f"have shape [T, 40]. "
                f"Got {acoustic.shape}"
            )

        if wavlm.ndim != 2:

            raise ValueError(
                f"WavLM features should "
                f"have shape [T, 768]. "
                f"Got {wavlm.shape}"
            )

        if acoustic.shape[0] != wavlm.shape[0]:

            raise ValueError(
                "Acoustic and WavLM have "
                "different number of windows."
            )

        if acoustic.shape[1] != 40:

            raise ValueError(
                f"Expected acoustic dimension "
                f"40, got {acoustic.shape[1]}"
            )

        if wavlm.shape[1] != 768:

            raise ValueError(
                f"Expected WavLM dimension "
                f"768, got {wavlm.shape[1]}"
            )

        if global_features.shape[0] != 12:

            raise ValueError(
                f"Expected 12 global features, "
                f"got {global_features.shape[0]}"
            )

        return {
            "acoustic": acoustic,
            "wavlm": wavlm,
            "global_features": global_features,
            "label": label,
            "speaker_id": speaker_id,
            "source": source,
            "pairing": pairing
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    batch_size = len(batch)

    # --------------------------------------------------------
    # Find longest temporal sequence
    # --------------------------------------------------------

    max_length = max(
        item["acoustic"].shape[0]
        for item in batch
    )

    acoustic_dim = 40
    wavlm_dim = 768
    global_dim = 12

    # --------------------------------------------------------
    # Create padded tensors
    # --------------------------------------------------------

    acoustic_batch = torch.zeros(
        batch_size,
        max_length,
        acoustic_dim,
        dtype=torch.float32
    )

    wavlm_batch = torch.zeros(
        batch_size,
        max_length,
        wavlm_dim,
        dtype=torch.float32
    )

    attention_mask = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.bool
    )

    global_batch = torch.zeros(
        batch_size,
        global_dim,
        dtype=torch.float32
    )

    labels = torch.zeros(
        batch_size,
        dtype=torch.long
    )

    speaker_ids = []
    sources = []
    pairings = []

    # --------------------------------------------------------
    # Fill batch
    # --------------------------------------------------------

    for i, item in enumerate(batch):

        sequence_length = (
            item["acoustic"].shape[0]
        )

        acoustic_batch[
            i,
            :sequence_length
        ] = item["acoustic"]

        wavlm_batch[
            i,
            :sequence_length
        ] = item["wavlm"]

        attention_mask[
            i,
            :sequence_length
        ] = True

        global_batch[i] = (
            item["global_features"]
        )

        labels[i] = (
            item["label"]
        )

        speaker_ids.append(
            item["speaker_id"]
        )

        sources.append(
            item["source"]
        )

        pairings.append(
            item["pairing"]
        )

    return {
        "acoustic": acoustic_batch,
        "wavlm": wavlm_batch,
        "global_features": global_batch,
        "attention_mask": attention_mask,
        "label": labels,
        "speaker_id": speaker_ids,
        "source": sources,
        "pairing": pairings
    }


# ============================================================
# GLOBAL FEATURE NORMALIZATION STATISTICS
# ============================================================

def calculate_global_statistics(
    metadata_file="data/dataset_split.csv",
    feature_dir="data/features"
):

    dataset = SentinelVoiceDataset(
        split="train",
        metadata_file=metadata_file,
        feature_dir=feature_dir
    )

    # --------------------------------------------------------
    # Load ALL metadata separately.
    #
    # This is needed because the Alert baseline may belong
    # to a speaker whose Alert recording is in another split.
    # --------------------------------------------------------

    dataset.metadata_all = pd.read_csv(
        metadata_file
    )

    global_features = []

    for index in range(
        len(dataset)
    ):

        item = dataset[index]

        global_features.append(
            item["global_features"]
        )

    global_features = torch.stack(
        global_features
    ).float()

    mean = global_features.mean(
        dim=0
    )

    std = global_features.std(
        dim=0,
        unbiased=False
    )

    std = torch.where(
        std < 1e-8,
        torch.ones_like(std),
        std
    )

    return mean, std


# ============================================================
# DATALOADER
# ============================================================

def create_dataloader(
    split,
    batch_size=4,
    shuffle=False,
    global_mean=None,
    global_std=None
):

    dataset = SentinelVoiceDataset(
        split=split,
        global_mean=global_mean,
        global_std=global_std
    )

    # Needed for speaker-relative speech rate
    dataset.metadata_all = pd.read_csv(
        "data/dataset_split.csv"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return loader