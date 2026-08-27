class FatigueFusion:
    """
    Combines the general ML fatigue prediction with
    the speaker's personalized deviation analysis.

    General classifier:
        0 -> Alert
        1 -> Mild Fatigue
        2 -> High Fatigue

    Personalized deviation score:
        0.0 -> No deviation
        1.0 -> Maximum deviation
    """

    CLASS_NAMES = [
        "Alert",
        "Mild Fatigue",
        "High Fatigue"
    ]

    # ========================================================
    # INITIALIZE
    # ========================================================

    def __init__(self):

        pass

    # ========================================================
    # GET CLASSIFIER INFORMATION
    # ========================================================

    def get_classifier_result(
        self,
        prediction,
        probabilities
    ):

        prediction = int(
            prediction
        )

        predicted_class = (
            self.CLASS_NAMES[
                prediction
            ]
        )

        confidence = float(
            probabilities[
                prediction
            ]
        )

        return {

            "prediction_index": prediction,

            "prediction": predicted_class,

            "confidence": confidence
        }

    # ========================================================
    # CALCULATE FUSION SCORE
    # ========================================================

    def calculate_fusion_score(
        self,
        prediction,
        probabilities,
        deviation_score
    ):
        """
        Combine the classifier fatigue probability
        with the personalized deviation score.

        The classifier contributes 70%.

        Personalized deviation contributes 30%.
        """

        prediction = int(
            prediction
        )

        probabilities = [
            float(probability)
            for probability in probabilities
        ]

        # ----------------------------------------------------
        # CLASSIFIER FATIGUE SCORE
        # ----------------------------------------------------
        #
        # Alert        -> 0.0
        # Mild Fatigue -> 0.5
        # High Fatigue -> 1.0
        #
        # Probability-weighted score
        # ----------------------------------------------------

        classifier_score = (

            probabilities[1] * 0.5

            +

            probabilities[2] * 1.0

        )

        # ----------------------------------------------------
        # PERSONALIZED DEVIATION SCORE
        # ----------------------------------------------------

        deviation_score = float(
            deviation_score
        )

        deviation_score = max(
            0.0,
            min(
                deviation_score,
                1.0
            )
        )

        # ----------------------------------------------------
        # FINAL FUSION SCORE
        # ----------------------------------------------------

        fusion_score = (

            0.70
            *
            classifier_score

            +

            0.30
            *
            deviation_score

        )

        return {

            "classifier_score": float(
                classifier_score
            ),

            "personalized_score": float(
                deviation_score
            ),

            "fusion_score": float(
                fusion_score
            )
        }

    # ========================================================
    # INTERPRET FINAL SCORE
    # ========================================================

    def interpret_fusion_score(
        self,
        fusion_score
    ):
        """
        Convert the combined score into a final
        fatigue assessment.
        """

        if fusion_score < 0.25:

            return {

                "final_assessment": "Alert",

                "risk_level": "LOW",

                "status": "NORMAL"
            }

        elif fusion_score < 0.50:

            return {

                "final_assessment": "Mild Fatigue",

                "risk_level": "MODERATE",

                "status": "MONITOR"
            }

        elif fusion_score < 0.75:

            return {

                "final_assessment": "Significant Fatigue",

                "risk_level": "HIGH",

                "status": "ATTENTION_REQUIRED"
            }

        else:

            return {

                "final_assessment": "High Fatigue",

                "risk_level": "SEVERE",

                "status": "IMMEDIATE_ATTENTION"
            }

    # ========================================================
    # COMPLETE FUSION
    # ========================================================

    def fuse(
        self,
        prediction,
        probabilities,
        deviation_score
    ):
        """
        Combine classifier prediction and
        personalized deviation into one result.
        """

        # ----------------------------------------------------
        # CLASSIFIER RESULT
        # ----------------------------------------------------

        classifier_result = (
            self.get_classifier_result(
                prediction,
                probabilities
            )
        )

        # ----------------------------------------------------
        # FUSION SCORE
        # ----------------------------------------------------

        fusion_result = (
            self.calculate_fusion_score(
                prediction,
                probabilities,
                deviation_score
            )
        )

        # ----------------------------------------------------
        # FINAL INTERPRETATION
        # ----------------------------------------------------

        interpretation = (
            self.interpret_fusion_score(
                fusion_result[
                    "fusion_score"
                ]
            )
        )

        # ----------------------------------------------------
        # RETURN COMPLETE RESULT
        # ----------------------------------------------------

        return {

            "classifier": classifier_result,

            "classifier_score": (
                fusion_result[
                    "classifier_score"
                ]
            ),

            "personalized_score": (
                fusion_result[
                    "personalized_score"
                ]
            ),

            "fusion_score": (
                fusion_result[
                    "fusion_score"
                ]
            ),

            "final_assessment": (
                interpretation[
                    "final_assessment"
                ]
            ),

            "risk_level": (
                interpretation[
                    "risk_level"
                ]
            ),

            "status": (
                interpretation[
                    "status"
                ]
            )
        }