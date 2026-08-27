import json

from pathlib import Path

import numpy as np


class SpeakerBaseline:
    """
    Stores and manages a speaker's normal/alert voice baseline.

    The baseline is created from multiple reliable
    alert-state recordings.
    """

    def __init__(
        self,
        baseline_dir="data/baselines"
    ):

        self.baseline_dir = Path(
            baseline_dir
        )

        self.baseline_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # CREATE FEATURE STATISTICS
    # ========================================================

    def calculate_feature_statistics(
        self,
        values
    ):
        """
        Calculate mean, median and standard deviation
        for a single personalized feature.
        """

        values = np.array(
            values,
            dtype=np.float32
        )

        return {

            "mean": float(
                np.mean(values)
            ),

            "median": float(
                np.median(values)
            ),

            "std": float(
                np.std(values)
            )
        }

    # ========================================================
    # CREATE BASELINE
    # ========================================================

    def calculate_baseline(
        self,
        feature_records
    ):
        """
        Calculate a personalized speaker baseline
        from multiple reliable Alert recordings.

        Features:

        - WPM
        - WPS
        - Mean F0
        - Energy Mean
        - Jitter
        - Shimmer
        - HNR
        """

        if not feature_records:

            raise ValueError(
                "No feature records provided."
            )

        # ----------------------------------------------------
        # FEATURE NAMES
        # ----------------------------------------------------

        feature_names = [

            "wpm",

            "wps",

            "mean_f0",

            "energy_mean",

            "jitter",

            "shimmer",

            "hnr"
        ]

        # ----------------------------------------------------
        # CALCULATE STATISTICS
        # ----------------------------------------------------

        baseline = {}

        for feature_name in feature_names:

            values = [

                record[
                    feature_name
                ]

                for record in feature_records
            ]

            statistics = (
                self.calculate_feature_statistics(
                    values
                )
            )

            baseline[
                feature_name
            ] = statistics

        # ----------------------------------------------------
        # NUMBER OF RECORDINGS
        # ----------------------------------------------------

        baseline[
            "num_recordings"
        ] = len(
            feature_records
        )

        return baseline

    # ========================================================
    # SAVE BASELINE
    # ========================================================

    def save(
        self,
        speaker_id,
        baseline
    ):

        path = (
            self.baseline_dir /
            f"{speaker_id}.json"
        )

        with open(
            path,
            "w"
        ) as file:

            json.dump(
                baseline,
                file,
                indent=4
            )

        return path

    # ========================================================
    # LOAD BASELINE
    # ========================================================

    def load(
        self,
        speaker_id
    ):

        path = (
            self.baseline_dir /
            f"{speaker_id}.json"
        )

        if not path.exists():

            return None

        with open(
            path,
            "r"
        ) as file:

            return json.load(
                file
            )

    # ========================================================
    # CHECK BASELINE
    # ========================================================

    def exists(
        self,
        speaker_id
    ):

        path = (
            self.baseline_dir /
            f"{speaker_id}.json"
        )

        return path.exists()

    # ========================================================
    # RELATIVE SPEECH RATE
    # ========================================================

    def relative_speech_rate(
        self,
        current_wpm,
        baseline
    ):

        baseline_wpm = (
            baseline[
                "wpm"
            ][
                "mean"
            ]
        )

        if baseline_wpm <= 0:

            raise ValueError(
                "Baseline WPM must be greater than zero."
            )

        return (
            current_wpm /
            baseline_wpm
        )