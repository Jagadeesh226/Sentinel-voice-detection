import math


class PersonalizedDeviation:

    # ========================================================
    # SAFE Z-SCORE
    # ========================================================

    def calculate_z_score(
        self,
        current_value,
        baseline_mean,
        baseline_std,
        min_std=1e-6
    ):
        """
        Calculate how far the current value deviates
        from the speaker's normal baseline.

        Formula:

        Z = (current - baseline_mean) / baseline_std
        """

        if baseline_std < min_std:

            return 0.0

        z_score = (
            current_value
            -
            baseline_mean
        ) / baseline_std

        return float(
            z_score
        )

    # ========================================================
    # CALCULATE FEATURE DEVIATION
    # ========================================================

    def calculate_feature_deviation(
        self,
        current_value,
        baseline_feature
    ):
        """
        Calculate deviation for one personalized feature.
        """

        z_score = (
            self.calculate_z_score(
                current_value=current_value,
                baseline_mean=(
                    baseline_feature[
                        "mean"
                    ]
                ),
                baseline_std=(
                    baseline_feature[
                        "std"
                    ]
                )
            )
        )

        absolute_deviation = abs(
            z_score
        )

        return {

            "current_value": float(
                current_value
            ),

            "baseline_mean": float(
                baseline_feature[
                    "mean"
                ]
            ),

            "baseline_std": float(
                baseline_feature[
                    "std"
                ]
            ),

            "z_score": float(
                z_score
            ),

            "absolute_deviation": float(
                absolute_deviation
            ),

            "status": self.interpret(
                z_score
            )
        }

    # ========================================================
    # CALCULATE ALL PERSONALIZED DEVIATIONS
    # ========================================================

    def calculate(
        self,
        current_features,
        baseline
    ):
        """
        Compare all current personalized features
        against the speaker's baseline.

        Features:

        - WPM
        - WPS
        - Mean F0
        - Energy Mean
        - Jitter
        - Shimmer
        - HNR
        """

        feature_names = [

            "wpm",

            "wps",

            "mean_f0",

            "energy_mean",

            "jitter",

            "shimmer",

            "hnr"
        ]

        deviations = {}

        absolute_deviations = []

        # ----------------------------------------------------
        # CALCULATE DEVIATION FOR EACH FEATURE
        # ----------------------------------------------------

        for feature_name in feature_names:

            feature_deviation = (
                self.calculate_feature_deviation(
                    current_value=(
                        current_features[
                            feature_name
                        ]
                    ),
                    baseline_feature=(
                        baseline[
                            feature_name
                        ]
                    )
                )
            )

            deviations[
                feature_name
            ] = feature_deviation

            absolute_deviations.append(
                feature_deviation[
                    "absolute_deviation"
                ]
            )

        # ----------------------------------------------------
        # CALCULATE COMBINED DEVIATION SCORE
        # ----------------------------------------------------

        deviation_score = (
            self.calculate_deviation_score(
                absolute_deviations
            )
        )

        # ----------------------------------------------------
        # RETURN COMPLETE RESULT
        # ----------------------------------------------------

        return {

            "features": deviations,

            "deviation_score": (
                deviation_score
            ),

            "assessment": (
                self.interpret_score(
                    deviation_score
                )
            )
        }

    # ========================================================
    # CALCULATE COMBINED DEVIATION SCORE
    # ========================================================

    def calculate_deviation_score(
        self,
        absolute_deviations
    ):
        """
        Convert multiple absolute Z-scores into
        one normalized deviation score between 0 and 1.

        A Z-score of 3 or greater is treated
        as maximum deviation.
        """

        if not absolute_deviations:

            return 0.0

        normalized_scores = []

        for deviation in absolute_deviations:

            normalized = min(
                deviation / 3.0,
                1.0
            )

            normalized_scores.append(
                normalized
            )

        score = (

            sum(
                normalized_scores
            )

            /

            len(
                normalized_scores
            )

        )

        return float(
            score
        )

    # ========================================================
    # INTERPRET INDIVIDUAL DEVIATION
    # ========================================================

    def interpret(
        self,
        deviation
    ):
        """
        Interpret the magnitude of an individual
        feature deviation based on absolute Z-score.
        """

        absolute_deviation = abs(
            deviation
        )

        if absolute_deviation < 1:

            return "NORMAL"

        elif absolute_deviation < 2:

            return "SLIGHT_DEVIATION"

        elif absolute_deviation < 3:

            return "SIGNIFICANT_DEVIATION"

        else:

            return "EXTREME_DEVIATION"

    # ========================================================
    # INTERPRET COMBINED DEVIATION SCORE
    # ========================================================

    def interpret_score(
        self,
        score
    ):
        """
        Interpret the final personalized
        deviation score.
        """

        if score < 0.25:

            return "NORMAL"

        elif score < 0.50:

            return "MILD_DEVIATION"

        elif score < 0.75:

            return "SIGNIFICANT_DEVIATION"

        else:

            return "SEVERE_DEVIATION"