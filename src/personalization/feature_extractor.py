# ============================================================
# PERSONALIZED FEATURE EXTRACTION
# ============================================================

# Acoustic feature indices based on acoustic.py
#
# 0 - 12   → MFCC means
# 13 - 25  → MFCC standard deviations
# 26       → Mean F0
# 27       → Std F0
# 28       → Min F0
# 29       → Max F0
# 30       → Mean Energy
# 31       → Std Energy
# 32       → Min Energy
# 33       → Max Energy
# 34       → Jitter
# 35       → Shimmer
# 36       → HNR
# 37       → F1
# 38       → F2
# 39       → F3


PERSONALIZED_ACOUSTIC_INDICES = {

    "mean_f0": 26,

    "energy_mean": 30,

    "jitter": 34,

    "shimmer": 35,

    "hnr": 36
}


# ============================================================
# EXTRACT PERSONALIZED FEATURES
# ============================================================

def extract_personalized_features(
    features
):
    """
    Extract speaker-specific features from the complete
    feature dictionary.

    Input:
        features = {
            "acoustic": Tensor[num_windows, 40],
            "wavlm": Tensor[num_windows, 768],
            "global_features": Tensor[11]
        }

    Returns:
        Dictionary containing stable features used for
        personalized baseline creation.
    """

    # --------------------------------------------------------
    # GET ACOUSTIC FEATURES
    # --------------------------------------------------------

    acoustic_features = (
        features[
            "acoustic"
        ]
    )

    # --------------------------------------------------------
    # GET GLOBAL FEATURES
    # --------------------------------------------------------

    global_features = (
        features[
            "global_features"
        ]
    )

    # --------------------------------------------------------
    # SPEECH RATE
    # --------------------------------------------------------

    wps = (
        global_features[9]
        .item()
    )

    wpm = (
        global_features[10]
        .item()
    )

    # --------------------------------------------------------
    # ACOUSTIC FEATURES
    #
    # acoustic_features shape:
    #
    # [number_of_windows, 40]
    #
    # We calculate the mean value of each selected feature
    # across all temporal windows.
    # --------------------------------------------------------

    mean_f0 = (
        acoustic_features[
            :,
            PERSONALIZED_ACOUSTIC_INDICES[
                "mean_f0"
            ]
        ]
        .mean()
        .item()
    )

    energy_mean = (
        acoustic_features[
            :,
            PERSONALIZED_ACOUSTIC_INDICES[
                "energy_mean"
            ]
        ]
        .mean()
        .item()
    )

    jitter = (
        acoustic_features[
            :,
            PERSONALIZED_ACOUSTIC_INDICES[
                "jitter"
            ]
        ]
        .mean()
        .item()
    )

    shimmer = (
        acoustic_features[
            :,
            PERSONALIZED_ACOUSTIC_INDICES[
                "shimmer"
            ]
        ]
        .mean()
        .item()
    )

    hnr = (
        acoustic_features[
            :,
            PERSONALIZED_ACOUSTIC_INDICES[
                "hnr"
            ]
        ]
        .mean()
        .item()
    )

    # --------------------------------------------------------
    # RETURN PERSONALIZED FEATURES
    # --------------------------------------------------------

    return {

        "wpm": float(wpm),

        "wps": float(wps),

        "mean_f0": float(
            mean_f0
        ),

        "energy_mean": float(
            energy_mean
        ),

        "jitter": float(
            jitter
        ),

        "shimmer": float(
            shimmer
        ),

        "hnr": float(
            hnr
        )
    }