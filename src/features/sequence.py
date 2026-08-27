import torch


def build_temporal_sequence(
    acoustic_features,
    wavlm_features
):
    """
    Combine acoustic and WavLM features
    for every temporal window.

    Inputs:
        acoustic_features:
            [T, 128]

        wavlm_features:
            [T, 128]

    Returns:
        fused_sequence:
            [T, 256]
    """

    if acoustic_features.shape[0] != wavlm_features.shape[0]:

        raise ValueError(
            "Acoustic and WavLM must have "
            "the same number of temporal windows."
        )

    fused_sequence = torch.cat(
        [
            acoustic_features,
            wavlm_features
        ],
        dim=-1
    )

    return fused_sequence