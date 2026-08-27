from src.personalization.baseline import (
    SpeakerBaseline
)

from src.personalization.candidate_manager import (
    CandidateManager
)

from src.personalization.feature_extractor import (
    extract_personalized_features
)

from src.personalization.deviation import (
    PersonalizedDeviation
)

from src.personalization.fusion import (
    FatigueFusion
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Alert",
    "Mild Fatigue",
    "High Fatigue"
]


# ============================================================
# PERSONALIZED AUDIO WORKFLOW
# ============================================================

def process_audio_for_speaker(
    speaker_id,
    features,
    prediction,
    probabilities
):

    baseline_manager = (
        SpeakerBaseline()
    )

    candidate_manager = (
        CandidateManager()
    )

    deviation_calculator = (
        PersonalizedDeviation()
    )

    fatigue_fusion = (
        FatigueFusion()
    )

    # --------------------------------------------------------
    # EXTRACT PREDICTION INFORMATION
    # --------------------------------------------------------

    predicted_class = (
        CLASS_NAMES[
            prediction
        ]
    )

    confidence = (
        probabilities[
            prediction
        ]
        .item()
    )

    # --------------------------------------------------------
    # EXTRACT PERSONALIZED FEATURES
    # --------------------------------------------------------

    personalized_features = (
        extract_personalized_features(
            features
        )
    )

    # --------------------------------------------------------
    # EXTRACT SPEECH RATE
    # --------------------------------------------------------

    wpm = (
        personalized_features[
            "wpm"
        ]
    )

    wps = (
        personalized_features[
            "wps"
        ]
    )

    # --------------------------------------------------------
    # CHECK FOR EXISTING BASELINE
    # --------------------------------------------------------

    if baseline_manager.exists(
        speaker_id
    ):

        baseline = (
            baseline_manager.load(
                speaker_id
            )
        )

        # ----------------------------------------------------
        # CALCULATE PERSONALIZED DEVIATION
        # ----------------------------------------------------

        personalized_deviation = (
            deviation_calculator.calculate(
                current_features=(
                    personalized_features
                ),
                baseline=baseline
            )
        )

        deviation_score = (
            personalized_deviation[
                "deviation_score"
            ]
        )

        # ----------------------------------------------------
        # CALCULATE FUSION RESULT
        # ----------------------------------------------------

        fusion_result = (
            fatigue_fusion.fuse(
                prediction=prediction,
                probabilities=probabilities,
                deviation_score=deviation_score
            )
        )

        # ----------------------------------------------------
        # RELATIVE SPEECH RATE
        # ----------------------------------------------------

        relative_rate = (
            baseline_manager.relative_speech_rate(
                current_wpm=wpm,
                baseline=baseline
            )
        )

        # ----------------------------------------------------
        # RETURN BASELINE ANALYSIS
        # ----------------------------------------------------

        return {

            "status": "BASELINE_READY",

            "speaker_id": speaker_id,

            # --------------------------------------------
            # ORIGINAL CLASSIFIER RESULT
            # --------------------------------------------

            "prediction": predicted_class,

            "prediction_index": prediction,

            "confidence": confidence,

            # --------------------------------------------
            # PERSONALIZED FEATURES
            # --------------------------------------------

            "wpm": wpm,

            "wps": wps,

            "personalized_features": (
                personalized_features
            ),

            # --------------------------------------------
            # BASELINE
            # --------------------------------------------

            "baseline": baseline,

            "relative_rate": (
                relative_rate
            ),

            # --------------------------------------------
            # PERSONALIZED DEVIATION
            # --------------------------------------------

            "personalized_deviation": (
                personalized_deviation
            ),

            "deviation_score": (
                deviation_score
            ),

            "deviation_assessment": (
                personalized_deviation[
                    "assessment"
                ]
            ),

            # --------------------------------------------
            # FINAL FUSION RESULT
            # --------------------------------------------

            "fusion": (
                fusion_result
            ),

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
                fusion_result[
                    "final_assessment"
                ]
            ),

            "risk_level": (
                fusion_result[
                    "risk_level"
                ]
            ),

            "final_status": (
                fusion_result[
                    "status"
                ]
            )
        }

    # --------------------------------------------------------
    # NO BASELINE
    # TRY TO ADD CANDIDATE
    # --------------------------------------------------------

    candidate_result = (
        candidate_manager.add_candidate(
            speaker_id=speaker_id,
            personalized_features=(
                personalized_features
            ),
            prediction=prediction,
            confidence=confidence
        )
    )

    # --------------------------------------------------------
    # RECORDING REJECTED
    # --------------------------------------------------------

    if not candidate_result[
        "accepted"
    ]:

        return {

            "status": "CANDIDATE_REJECTED",

            "speaker_id": speaker_id,

            "prediction": predicted_class,

            "prediction_index": prediction,

            "confidence": confidence,

            "wpm": wpm,

            "wps": wps,

            "personalized_features": (
                personalized_features
            ),

            "reason": (
                candidate_result[
                    "reason"
                ]
            )
        }

    # --------------------------------------------------------
    # BASELINE NOT READY
    # --------------------------------------------------------

    candidate_count = (
        candidate_manager.count_candidates(
            speaker_id
        )
    )

    if not candidate_manager.is_ready_for_baseline(
        speaker_id
    ):

        return {

            "status": "BASELINE_IN_PROGRESS",

            "speaker_id": speaker_id,

            "prediction": predicted_class,

            "prediction_index": prediction,

            "confidence": confidence,

            "wpm": wpm,

            "wps": wps,

            "personalized_features": (
                personalized_features
            ),

            "candidate_count": (
                candidate_count
            ),

            "required_candidates": (
                candidate_manager.required_candidates
            )
        }

    # --------------------------------------------------------
    # CREATE BASELINE
    # --------------------------------------------------------

    baseline_result = (
        candidate_manager.create_baseline(
            speaker_id
        )
    )

    # --------------------------------------------------------
    # BASELINE CREATION FAILED
    # --------------------------------------------------------

    if not baseline_result[
        "success"
    ]:

        return {

            "status": "BASELINE_CREATION_FAILED",

            "speaker_id": speaker_id,

            "prediction": predicted_class,

            "prediction_index": prediction,

            "confidence": confidence,

            "wpm": wpm,

            "wps": wps,

            "personalized_features": (
                personalized_features
            ),

            "reason": (
                baseline_result[
                    "reason"
                ]
            )
        }

    # --------------------------------------------------------
    # BASELINE CREATED
    # --------------------------------------------------------

    baseline = (
        baseline_result[
            "baseline"
        ]
    )

    # --------------------------------------------------------
    # CALCULATE PERSONALIZED DEVIATION
    # --------------------------------------------------------

    personalized_deviation = (
        deviation_calculator.calculate(
            current_features=(
                personalized_features
            ),
            baseline=baseline
        )
    )

    deviation_score = (
        personalized_deviation[
            "deviation_score"
        ]
    )

    # --------------------------------------------------------
    # CALCULATE FUSION RESULT
    # --------------------------------------------------------

    fusion_result = (
        fatigue_fusion.fuse(
            prediction=prediction,
            probabilities=probabilities,
            deviation_score=deviation_score
        )
    )

    # --------------------------------------------------------
    # RELATIVE SPEECH RATE
    # --------------------------------------------------------

    relative_rate = (
        baseline_manager.relative_speech_rate(
            current_wpm=wpm,
            baseline=baseline
        )
    )

    # --------------------------------------------------------
    # RETURN BASELINE CREATED
    # --------------------------------------------------------

    return {

        "status": "BASELINE_CREATED",

        "speaker_id": speaker_id,

        # --------------------------------------------
        # ORIGINAL CLASSIFIER RESULT
        # --------------------------------------------

        "prediction": predicted_class,

        "prediction_index": prediction,

        "confidence": confidence,

        # --------------------------------------------
        # PERSONALIZED FEATURES
        # --------------------------------------------

        "wpm": wpm,

        "wps": wps,

        "personalized_features": (
            personalized_features
        ),

        # --------------------------------------------
        # BASELINE
        # --------------------------------------------

        "baseline": baseline,

        "relative_rate": (
            relative_rate
        ),

        # --------------------------------------------
        # PERSONALIZED DEVIATION
        # --------------------------------------------

        "personalized_deviation": (
            personalized_deviation
        ),

        "deviation_score": (
            deviation_score
        ),

        "deviation_assessment": (
            personalized_deviation[
                "assessment"
            ]
        ),

        # --------------------------------------------
        # FINAL FUSION RESULT
        # --------------------------------------------

        "fusion": (
            fusion_result
        ),

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
            fusion_result[
                "final_assessment"
            ]
        ),

        "risk_level": (
            fusion_result[
                "risk_level"
            ]
        ),

        "final_status": (
            fusion_result[
                "status"
            ]
        ),

        # --------------------------------------------
        # BASELINE CREATION INFORMATION
        # --------------------------------------------

        "valid_candidates": (
            baseline_result[
                "valid_candidates"
            ]
        ),

        "rejected_candidates": (
            baseline_result[
                "rejected_candidates"
            ]
        )
    }