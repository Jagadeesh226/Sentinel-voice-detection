import torch

from src.dataset import (
    SentinelVoiceDataset,
    create_dataloader
)


def main():

    print("\n")
    print("=" * 60)
    print("TESTING SENTINEL DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SentinelVoiceDataset(
        split="train"
    )

    print(
        "\nDataset size:",
        len(dataset)
    )

    # --------------------------------------------------------
    # First sample
    # --------------------------------------------------------

    sample = dataset[0]

    print("\nFirst sample:")

    print(
        "Acoustic:",
        sample["acoustic"].shape
    )

    print(
        "WavLM:",
        sample["wavlm"].shape
    )

    print(
        "Temporal:",
        sample["temporal"].shape
    )

    print(
        "Label:",
        sample["label"].item()
    )

    print(
        "Speaker:",
        sample["speaker_id"]
    )

    print(
        "Source:",
        sample["source"]
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = create_dataloader(
        split="train",
        batch_size=4,
        shuffle=True
    )

    batch = next(
        iter(loader)
    )

    print("\nBatch:")

    print(
        "Acoustic:",
        batch["acoustic"].shape
    )

    print(
        "WavLM:",
        batch["wavlm"].shape
    )

    print(
        "Temporal:",
        batch["temporal"].shape
    )

    print(
        "Attention mask:",
        batch["attention_mask"].shape
    )

    print(
        "Labels:",
        batch["label"].shape
    )

    print(
        "Labels:",
        batch["label"]
    )

    print("\nAttention mask:")

    print(
        batch["attention_mask"]
    )

    print("\n")
    print("=" * 60)
    print("DATASET TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()