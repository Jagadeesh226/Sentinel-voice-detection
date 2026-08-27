import streamlit as st
import tempfile

from pathlib import Path


# ============================================================
# FATIGUE MODEL
# ============================================================

from test.predict_audio import (
    load_model,
    extract_features,
    predict
)


# ============================================================
# PERSONALIZATION WORKFLOW
# ============================================================

from src.personalization.workflow import (
    process_audio_for_speaker
)


# ============================================================
# SPEAKER IDENTITY
# ============================================================

from src.speaker.identity_manager import (
    SpeakerIdentityManager
)

from src.speaker.verification import (
    SpeakerVerifier
)

from src.preprocessing.audio_loader import (
    load_audio
)

from src.preprocessing.normalization import (
    normalize_audio
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sentinel Voice",
    page_icon="🎙️",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎙️ Sentinel Voice")

st.subheader(
    "Personalized Voice-Based Fatigue Detection"
)

st.write(
    "Enter the speaker ID and upload a voice recording "
    "to verify the speaker and analyze fatigue."
)


# ============================================================
# CONFIGURATION
# ============================================================

SIMILARITY_THRESHOLD = 0.55

DUPLICATE_SPEAKER_THRESHOLD = 0.70


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_fatigue_model():

    return load_model()


# ============================================================
# SPEAKER VERIFIER
# ============================================================

@st.cache_resource
def load_speaker_verifier():

    return SpeakerVerifier()


# ============================================================
# IDENTITY MANAGER
# ============================================================

@st.cache_resource
def load_identity_manager():

    return SpeakerIdentityManager()


# ============================================================
# SPEAKER ID
# ============================================================

st.divider()

speaker_id = st.text_input(
    "Speaker ID",
    placeholder="Example: worker_001"
)


# ============================================================
# AUDIO UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an audio recording",
    type=[
        "wav",
        "mp3",
        "m4a",
        "flac"
    ]
)


# ============================================================
# AUDIO PREVIEW
# ============================================================

if uploaded_file is not None:

    st.audio(
        uploaded_file
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_file is not None:

    if st.button(
        "🔍 Analyze Fatigue",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # VALIDATE SPEAKER ID
        # ----------------------------------------------------

        if not speaker_id:

            st.error(
                "Please enter a Speaker ID."
            )

            st.stop()

        # ----------------------------------------------------
        # SAVE AUDIO TEMPORARILY
        # ----------------------------------------------------

        suffix = Path(
            uploaded_file.name
        ).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_audio_path = (
                temp_file.name
            )

        try:

            # =================================================
            # LOAD SPEAKER VERIFICATION SYSTEM
            # =================================================

            with st.spinner(
                "Loading speaker verification system..."
            ):

                verifier = (
                    load_speaker_verifier()
                )

                identity_manager = (
                    load_identity_manager()
                )

            # =================================================
            # LOAD AND PREPROCESS AUDIO
            # =================================================

            with st.spinner(
                "Preparing audio for speaker verification..."
            ):

                audio, sample_rate = (
                    load_audio(
                        temp_audio_path
                    )
                )

                audio = (
                    normalize_audio(
                        audio
                    )
                )

            # =================================================
            # CREATE CURRENT SPEAKER EMBEDDING
            # =================================================

            with st.spinner(
                "Extracting speaker identity..."
            ):

                current_embedding = (
                    verifier.get_embedding(
                        audio
                    )
                )

            # =================================================
            # CHECK SPEAKER IDENTITY
            # =================================================

            with st.spinner(
                "Verifying speaker identity..."
            ):

                # =============================================
                # FIRST TIME SPEAKER
                # =============================================

                if not identity_manager.exists(
                    speaker_id
                ):

                    # -----------------------------------------
                    # CHECK AGAINST ALL EXISTING SPEAKERS
                    # -----------------------------------------

                    duplicate_result = (
                        identity_manager.find_existing_speaker(
                            embedding=current_embedding,
                            threshold=(
                                DUPLICATE_SPEAKER_THRESHOLD
                            )
                        )
                    )

                    # -----------------------------------------
                    # DUPLICATE SPEAKER FOUND
                    # -----------------------------------------

                    if duplicate_result[
                        "match_found"
                    ]:

                        duplicate_speaker_id = (
                            duplicate_result[
                                "speaker_id"
                            ]
                        )

                        similarity_score = (
                            duplicate_result[
                                "similarity"
                            ]
                        )

                        identity_verified = False

                        identity_status = (
                            "DUPLICATE_SPEAKER"
                        )

                    # -----------------------------------------
                    # GENUINELY NEW SPEAKER
                    # -----------------------------------------

                    else:

                        identity_manager.save(
                            speaker_id,
                            current_embedding
                        )

                        identity_verified = True

                        similarity_score = None

                        identity_status = (
                            "IDENTITY_CREATED"
                        )

                # =============================================
                # EXISTING SPEAKER
                # =============================================

                else:

                    stored_embedding = (
                        identity_manager.load(
                            speaker_id
                        )
                    )

                    similarity_score = (
                        verifier.similarity(
                            stored_embedding,
                            current_embedding
                        )
                    )

                    identity_verified = (
                        similarity_score
                        >= SIMILARITY_THRESHOLD
                    )

                    if identity_verified:

                        identity_status = (
                            "IDENTITY_VERIFIED"
                        )

                    else:

                        identity_status = (
                            "IDENTITY_REJECTED"
                        )

            # =================================================
            # IDENTITY RESULT
            # =================================================

            st.divider()

            st.subheader(
                "Speaker Identity"
            )

            # ------------------------------------------------
            # NEW SPEAKER
            # ------------------------------------------------

            if identity_status == "IDENTITY_CREATED":

                st.info(
                    "🆕 New speaker identity created."
                )

                st.caption(
                    "This voice will be used as the "
                    "identity reference for future recordings."
                )

            # ------------------------------------------------
            # DUPLICATE SPEAKER DETECTED
            # ------------------------------------------------

            elif identity_status == "DUPLICATE_SPEAKER":

                st.error(
                    "❌ This voice already appears to "
                    "belong to a registered speaker."
                )

                st.write(
                    f"Matching Speaker ID: "
                    f"**{duplicate_speaker_id}**"
                )

                st.metric(
                    "Voice Similarity",
                    f"{similarity_score:.3f}"
                )

                st.caption(
                    f"Duplicate detection threshold: "
                    f"{DUPLICATE_SPEAKER_THRESHOLD:.2f}"
                )

                st.stop()

            # ------------------------------------------------
            # VERIFIED SPEAKER
            # ------------------------------------------------

            elif identity_status == "IDENTITY_VERIFIED":

                st.success(
                    "✓ Speaker identity verified"
                )

                st.metric(
                    "Voice Similarity",
                    f"{similarity_score:.3f}"
                )

            # ------------------------------------------------
            # REJECTED SPEAKER
            # ------------------------------------------------

            elif identity_status == "IDENTITY_REJECTED":

                st.error(
                    "❌ Speaker identity verification failed."
                )

                st.write(
                    "The uploaded voice does not appear "
                    "to belong to the registered speaker."
                )

                st.metric(
                    "Voice Similarity",
                    f"{similarity_score:.3f}"
                )

                st.caption(
                    f"Required similarity: "
                    f"{SIMILARITY_THRESHOLD:.2f}"
                )

                st.stop()

            # =================================================
            # LOAD FATIGUE MODEL
            # =================================================

            with st.spinner(
                "Loading fatigue detection model..."
            ):

                model = (
                    load_fatigue_model()
                )

            # =================================================
            # FEATURE EXTRACTION
            # =================================================

            with st.spinner(
                "Analyzing voice recording..."
            ):

                features = (
                    extract_features(
                        temp_audio_path
                    )
                )

            # =================================================
            # FATIGUE PREDICTION
            # =================================================

            with st.spinner(
                "Predicting fatigue level..."
            ):

                prediction, probabilities = (
                    predict(
                        model=model,
                        features=features,
                        speaker_id=speaker_id
                    )
                )

            # =================================================
            # PERSONALIZED WORKFLOW
            # =================================================

            with st.spinner(
                "Processing personalized speaker profile..."
            ):

                workflow_result = (
                    process_audio_for_speaker(
                        speaker_id=speaker_id,
                        features=features,
                        prediction=prediction,
                        probabilities=probabilities
                    )
                )

            # =================================================
            # CLASS NAMES
            # =================================================

            class_names = [

                "Alert",

                "Mild Fatigue",

                "High Fatigue"
            ]

            predicted_class = (
                class_names[
                    prediction
                ]
            )

            confidence = (

                probabilities[
                    prediction
                ]
                .item()

                * 100
            )

            # =================================================
            # FATIGUE RESULT
            # =================================================

            st.divider()

            st.subheader(
                "Fatigue Detection Result"
            )

            if predicted_class == "Alert":

                st.success(
                    f"🟢 {predicted_class}"
                )

            elif predicted_class == "Mild Fatigue":

                st.warning(
                    f"🟡 {predicted_class}"
                )

            else:

                st.error(
                    f"🔴 {predicted_class}"
                )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            # =================================================
            # PREDICTION PROBABILITIES
            # =================================================

            st.subheader(
                "Prediction Probabilities"
            )

            for i, class_name in enumerate(
                class_names
            ):

                probability = (

                    probabilities[i]
                    .item()

                    * 100
                )

                st.write(
                    f"**{class_name}**"
                )

                st.progress(
                    min(
                        int(probability),
                        100
                    )
                )

                st.caption(
                    f"{probability:.2f}%"
                )

            # =================================================
            # PERSONALIZATION STATUS
            # =================================================

            st.divider()

            st.subheader(
                "Personalization Status"
            )

            workflow_status = (
                workflow_result[
                    "status"
                ]
            )

            # ------------------------------------------------
            # BASELINE IN PROGRESS
            # ------------------------------------------------

            if workflow_status == (
                "BASELINE_IN_PROGRESS"
            ):

                candidate_count = (
                    workflow_result[
                        "candidate_count"
                    ]
                )

                required_candidates = (
                    workflow_result[
                        "required_candidates"
                    ]
                )

                st.info(
                    "📊 Building personal voice baseline"
                )

                st.write(
                    f"Reliable recordings collected: "
                    f"**{candidate_count} / "
                    f"{required_candidates}**"
                )

                st.progress(
                    candidate_count
                    /
                    required_candidates
                )

            # ------------------------------------------------
            # BASELINE CREATED
            # ------------------------------------------------

            elif workflow_status == (
                "BASELINE_CREATED"
            ):

                st.success(
                    "✓ Personal voice baseline created."
                )

                baseline = (
                    workflow_result[
                        "baseline"
                    ]
                )

                st.write(
                    f"Baseline Speech Rate: "
                    f"**{baseline['wpm']['mean']:.2f} WPM**"
                )

            # ------------------------------------------------
            # BASELINE READY
            # ------------------------------------------------

            elif workflow_status == (
                "BASELINE_READY"
            ):

                st.success(
                    "✓ Personalized baseline active"
                )

            # ------------------------------------------------
            # CANDIDATE REJECTED
            # ------------------------------------------------

            elif workflow_status == (
                "CANDIDATE_REJECTED"
            ):

                st.warning(
                    "This recording was not added "
                    "to the personal baseline."
                )

                st.caption(
                    workflow_result[
                        "reason"
                    ]
                )

            # ------------------------------------------------
            # BASELINE CREATION FAILED
            # ------------------------------------------------

            elif workflow_status == (
                "BASELINE_CREATION_FAILED"
            ):

                st.error(
                    "Unable to create a reliable "
                    "speaker baseline."
                )

                st.caption(
                    workflow_result[
                        "reason"
                    ]
                )

            # =================================================
            # PERSONALIZED VOICE ANALYSIS
            # =================================================

            personalized_deviation = (
                workflow_result.get(
                    "personalized_deviation"
                )
            )

            deviation_score = (
                workflow_result.get(
                    "deviation_score"
                )
            )

            deviation_assessment = (
                workflow_result.get(
                    "deviation_assessment"
                )
            )

            if personalized_deviation is not None:

                st.divider()

                st.subheader(
                    "📊 Personalized Voice Analysis"
                )

                # ------------------------------------------------
                # OVERALL DEVIATION
                # ------------------------------------------------

                col1, col2 = (
                    st.columns(
                        2
                    )
                )

                with col1:

                    st.metric(
                        "Deviation Score",
                        f"{deviation_score:.2f}"
                    )

                with col2:

                    assessment_text = (
                        deviation_assessment
                        .replace(
                            "_",
                            " "
                        )
                        .title()
                    )

                    st.metric(
                        "Personalized Assessment",
                        assessment_text
                    )

                # ------------------------------------------------
                # FEATURE LEVEL DEVIATIONS
                # ------------------------------------------------

                st.markdown(
                    "#### Feature-Level Deviations"
                )

                feature_deviations = (
                    personalized_deviation.get(
                        "features",
                        {}
                    )
                )

                feature_labels = {

                    "wpm": (
                        "Speech Rate (WPM)"
                    ),

                    "wps": (
                        "Speech Rate (WPS)"
                    ),

                    "mean_f0": (
                        "Mean Pitch (F0)"
                    ),

                    "energy_mean": (
                        "Voice Energy"
                    ),

                    "jitter": (
                        "Jitter"
                    ),

                    "shimmer": (
                        "Shimmer"
                    ),

                    "hnr": (
                        "Harmonics-to-Noise Ratio"
                    )
                }

                for (
                    feature_name,
                    feature_data
                ) in feature_deviations.items():

                    label = (
                        feature_labels.get(
                            feature_name,
                            feature_name
                        )
                    )

                    current_value = (
                        feature_data.get(
                            "current_value"
                        )
                    )

                    baseline_mean = (
                        feature_data.get(
                            "baseline_mean"
                        )
                    )

                    z_score = (
                        feature_data.get(
                            "z_score"
                        )
                    )

                    status = (
                        feature_data.get(
                            "status",
                            "UNKNOWN"
                        )
                    )

                    status_text = (
                        status
                        .replace(
                            "_",
                            " "
                        )
                        .title()
                    )

                    col1, col2, col3 = (
                        st.columns(
                            3
                        )
                    )

                    with col1:

                        st.write(
                            f"**{label}**"
                        )

                        st.caption(
                            f"Status: {status_text}"
                        )

                    with col2:

                        if current_value is not None:

                            st.metric(
                                "Current",
                                f"{current_value:.3f}"
                            )

                    with col3:

                        if z_score is not None:

                            st.metric(
                                "Z-Score",
                                f"{z_score:.2f}"
                            )

                    if baseline_mean is not None:

                        st.caption(
                            f"Personal Baseline Mean: "
                            f"{baseline_mean:.3f}"
                        )

                    st.divider()

            # =================================================
            # FINAL FATIGUE FUSION RESULT
            # =================================================

            fusion_result = (
                workflow_result.get(
                    "fusion"
                )
            )

            if fusion_result is not None:

                st.divider()

                st.subheader(
                    "🧠 Final Fatigue Assessment"
                )

                final_assessment = (
                    workflow_result.get(
                        "final_assessment",
                        "UNKNOWN"
                    )
                )

                risk_level = (
                    workflow_result.get(
                        "risk_level",
                        "UNKNOWN"
                    )
                )

                final_status = (
                    workflow_result.get(
                        "final_status",
                        "UNKNOWN"
                    )
                )

                fusion_score = (
                    workflow_result.get(
                        "fusion_score"
                    )
                )

                classifier_score = (
                    workflow_result.get(
                        "classifier_score"
                    )
                )

                personalized_score = (
                    workflow_result.get(
                        "personalized_score"
                    )
                )

                # ------------------------------------------------
                # FINAL ASSESSMENT
                # ------------------------------------------------

                if final_assessment == "ALERT":

                    st.success(
                        "🟢 Final Assessment: ALERT"
                    )

                elif final_assessment == "MILD_FATIGUE":

                    st.warning(
                        "🟡 Final Assessment: MILD FATIGUE"
                    )

                elif final_assessment == "HIGH_FATIGUE":

                    st.error(
                        "🔴 Final Assessment: HIGH FATIGUE"
                    )

                else:

                    assessment_text = (
                        str(
                            final_assessment
                        )
                        .replace(
                            "_",
                            " "
                        )
                        .title()
                    )

                    st.info(
                        f"Final Assessment: "
                        f"{assessment_text}"
                    )

                # ------------------------------------------------
                # FUSION SCORES
                # ------------------------------------------------

                col1, col2, col3 = (
                    st.columns(
                        3
                    )
                )

                with col1:

                    if classifier_score is not None:

                        st.metric(
                            "Classifier Score",
                            f"{classifier_score:.3f}"
                        )

                with col2:

                    if personalized_score is not None:

                        st.metric(
                            "Personalized Score",
                            f"{personalized_score:.3f}"
                        )

                with col3:

                    if fusion_score is not None:

                        st.metric(
                            "Fusion Score",
                            f"{fusion_score:.3f}"
                        )

                # ------------------------------------------------
                # RISK INFORMATION
                # ------------------------------------------------

                st.markdown(
                    "#### Risk Assessment"
                )

                risk_text = (
                    str(
                        risk_level
                    )
                    .replace(
                        "_",
                        " "
                    )
                    .title()
                )

                status_text = (
                    str(
                        final_status
                    )
                    .replace(
                        "_",
                        " "
                    )
                    .title()
                )

                col1, col2 = (
                    st.columns(
                        2
                    )
                )

                with col1:

                    st.metric(
                        "Risk Level",
                        risk_text
                    )

                with col2:

                    st.metric(
                        "System Status",
                        status_text
                    )

                # ------------------------------------------------
                # FUSION EXPLANATION
                # ------------------------------------------------

                st.caption(
                    "The final assessment combines the "
                    "machine learning classifier result with "
                    "the speaker's personalized voice deviation."
                )

            # =================================================
            # SPEECH ANALYSIS
            # =================================================

            global_features = (
                features[
                    "global_features"
                ]
            )

            if global_features.shape[0] >= 11:

                wps = (
                    global_features[9]
                    .item()
                )

                wpm = (
                    global_features[10]
                    .item()
                )

                st.divider()

                st.subheader(
                    "Speech Analysis"
                )

                col1, col2 = st.columns(
                    2
                )

                with col1:

                    st.metric(
                        "Speech Rate",
                        f"{wpm:.1f} WPM"
                    )

                with col2:

                    st.metric(
                        "Speech Speed",
                        f"{wps:.2f} WPS"
                    )

            # =================================================
            # RELATIVE SPEECH RATE
            # =================================================

            if (
                "relative_rate"
                in workflow_result
            ):

                relative_rate = (
                    workflow_result[
                        "relative_rate"
                    ]
                )

                if relative_rate is not None:

                    st.divider()

                    st.subheader(
                        "Personalized Comparison"
                    )

                    st.metric(
                        "Relative Speech Rate",
                        f"{relative_rate:.3f}"
                    )

                    if relative_rate >= 0.90:

                        st.success(
                            "Speech rate is close to the "
                            "speaker's normal baseline."
                        )

                    elif relative_rate >= 0.70:

                        st.warning(
                            "Speech rate is noticeably slower "
                            "than the speaker's baseline."
                        )

                    else:

                        st.error(
                            "Speech rate is significantly slower "
                            "than the speaker's normal baseline."
                        )

        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

            st.exception(
                e
            )

        # =====================================================
        # REMOVE TEMPORARY FILE
        # =====================================================

        finally:

            try:

                Path(
                    temp_audio_path
                ).unlink()

            except Exception:

                pass