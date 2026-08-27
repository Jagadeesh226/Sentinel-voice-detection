from test.predict_audio import (
    extract_features,
    predict
)

from src.personalization.baseline import (
    SpeakerBaseline
)

from src.personalization.workflow import (
    process_audio_for_speaker
)

from src.personalization.deviation import (
    PersonalizedDeviation
)


# ============================================================
# END-TO-END SPEAKER AUDIO ANALYSIS
# ============================================================

def analyze_speaker_audio(
    audio_path,
    speaker_id,
    model
):

    # --------------------------------------------------------
    # INITIALIZE MANAGERS
    # --------------------------------------------------------

    baseline_manager = (
        SpeakerBaseline()
    )

    deviation_calculator = (
        PersonalizedDeviation()
    )

    # --------------------------------------------------------
    # CHECK BASELINE STATUS BEFORE ANALYSIS
    # --------------------------------------------------------

    baseline_exists = (
        baseline_manager.exists(
            speaker_id
        )
    )

    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    print()

    print(
        "Extracting audio features..."
    )

    features = extract_features(
        audio_path
    )

    # --------------------------------------------------------
    # RUN MODEL PREDICTION
    # --------------------------------------------------------

    print()

    print(
        "Running fatigue prediction..."
    )

    prediction, probabilities = predict(
        model=model,
        features=features,
        speaker_id=speaker_id
    )

    # --------------------------------------------------------
    # PROCESS PERSONALIZED WORKFLOW
    # --------------------------------------------------------

    print()

    print(
        "Processing speaker personalization..."
    )

    workflow_result = (
        process_audio_for_speaker(
            speaker_id=speaker_id,
            features=features,
            prediction=prediction,
            probabilities=probabilities
        )
    )

    # --------------------------------------------------------
    # EXTRACT PREDICTION INFORMATION
    # --------------------------------------------------------

    predicted_class = (
        workflow_result[
            "prediction"
        ]
    )

    confidence = (
        workflow_result[
            "confidence"
        ]
    )

    # --------------------------------------------------------
    # EXTRACT SPEECH FEATURES
    # --------------------------------------------------------

    wpm = (
        workflow_result[
            "wpm"
        ]
    )

    wps = (
        workflow_result[
            "wps"
        ]
    )

    # ========================================================
    # PERSONALIZED DEVIATION ANALYSIS
    # ========================================================

    personalized_deviation = None

    # Get baseline from workflow result.
    # This can be:
    # 1. Existing baseline
    # 2. Newly created baseline
    baseline = (
        workflow_result.get(
            "baseline"
        )
    )

    if baseline is not None:

        personalized_deviation = (
            deviation_calculator.calculate(
                current_wpm=wpm,
                current_wps=wps,
                baseline=baseline
            )
        )

        # ----------------------------------------------------
        # INTERPRET WPM DEVIATION
        # ----------------------------------------------------

        personalized_deviation[
            "wpm_interpretation"
        ] = (
            deviation_calculator.interpret(
                personalized_deviation[
                    "wpm_z_score"
                ]
            )
        )

        # ----------------------------------------------------
        # INTERPRET WPS DEVIATION
        # ----------------------------------------------------

        personalized_deviation[
            "wps_interpretation"
        ] = (
            deviation_calculator.interpret(
                personalized_deviation[
                    "wps_z_score"
                ]
            )
        )

    # --------------------------------------------------------
    # RETURN COMPLETE RESULT
    # --------------------------------------------------------

    return {

        # ----------------------------------------------------
        # SPEAKER
        # ----------------------------------------------------

        "speaker_id": speaker_id,

        # ----------------------------------------------------
        # BASELINE STATUS
        # ----------------------------------------------------

        "baseline_existed_before_analysis": (
            baseline_exists
        ),

        "workflow_status": (
            workflow_result[
                "status"
            ]
        ),

        # ----------------------------------------------------
        # FATIGUE PREDICTION
        # ----------------------------------------------------

        "prediction": predicted_class,

        "prediction_index": (
            workflow_result[
                "prediction_index"
            ]
        ),

        "confidence": confidence,

        "probabilities": (
            probabilities
            .detach()
            .cpu()
            .tolist()
        ),

        # ----------------------------------------------------
        # SPEECH FEATURES
        # ----------------------------------------------------

        "wpm": wpm,

        "wps": wps,

        "relative_rate": (
            workflow_result.get(
                "relative_rate"
            )
        ),

        # ----------------------------------------------------
        # PERSONALIZED DEVIATION
        # ----------------------------------------------------

        "personalized_deviation": (
            personalized_deviation
        ),

        # ----------------------------------------------------
        # CANDIDATE INFORMATION
        # ----------------------------------------------------

        "candidate_count": (
            workflow_result.get(
                "candidate_count"
            )
        ),

        "required_candidates": (
            workflow_result.get(
                "required_candidates"
            )
        ),

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        "baseline": baseline,

        # ----------------------------------------------------
        # ERROR / STATUS REASON
        # ----------------------------------------------------

        "reason": (
            workflow_result.get(
                "reason"
            )
        )
    }