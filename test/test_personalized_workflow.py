from src.personalization.baseline import SpeakerBaseline
from src.personalization.candidate_manager import CandidateManager


# ============================================================
# PERSONALIZED SPEAKER WORKFLOW
# ============================================================

def process_speaker_sample(
    speaker_id,
    wpm,
    wps,
    prediction=0,
    confidence=0.90
):

    baseline_manager = SpeakerBaseline()

    candidate_manager = CandidateManager()

    # --------------------------------------------------------
    # CHECK FOR EXISTING BASELINE
    # --------------------------------------------------------

    if baseline_manager.exists(
        speaker_id
    ):

        baseline = baseline_manager.load(
            speaker_id
        )

        relative_rate = (
            baseline_manager.relative_speech_rate(
                current_wpm=wpm,
                baseline=baseline
            )
        )

        return {
            "status": "BASELINE_READY",
            "speaker_id": speaker_id,
            "baseline": baseline,
            "relative_rate": relative_rate
        }

    # --------------------------------------------------------
    # NO BASELINE
    # ADD AS CANDIDATE
    # --------------------------------------------------------

    candidate_result = (
        candidate_manager.add_candidate(
            speaker_id=speaker_id,
            wpm=wpm,
            wps=wps,
            prediction=prediction,
            confidence=confidence
        )
    )

    # --------------------------------------------------------
    # RECORDING NOT ELIGIBLE
    # --------------------------------------------------------

    if not candidate_result["accepted"]:

        return {
            "status": "CANDIDATE_REJECTED",
            "speaker_id": speaker_id,
            "reason": candidate_result["reason"]
        }

    # --------------------------------------------------------
    # CHECK IF ENOUGH CANDIDATES
    # --------------------------------------------------------

    candidate_count = (
        candidate_manager.count_candidates(
            speaker_id
        )
    )

    required_candidates = (
        candidate_manager.required_candidates
    )

    if not candidate_manager.is_ready_for_baseline(
        speaker_id
    ):

        return {
            "status": "BASELINE_IN_PROGRESS",
            "speaker_id": speaker_id,
            "candidate_count": candidate_count,
            "required_candidates": required_candidates
        }

    # --------------------------------------------------------
    # CREATE BASELINE
    # --------------------------------------------------------

    baseline_result = (
        candidate_manager.create_baseline(
            speaker_id
        )
    )

    if not baseline_result["success"]:

        return {
            "status": "BASELINE_CREATION_FAILED",
            "speaker_id": speaker_id,
            "reason": baseline_result["reason"],
            "valid_candidates": (
                baseline_result.get(
                    "valid_candidates",
                    0
                )
            ),
            "rejected_candidates": (
                baseline_result.get(
                    "rejected_candidates",
                    0
                )
            )
        }

    baseline = (
        baseline_result["baseline"]
    )

    relative_rate = (
        baseline_manager.relative_speech_rate(
            current_wpm=wpm,
            baseline=baseline
        )
    )

    return {
        "status": "BASELINE_CREATED",
        "speaker_id": speaker_id,
        "baseline": baseline,
        "relative_rate": relative_rate,
        "valid_candidates": (
            baseline_result["valid_candidates"]
        ),
        "rejected_candidates": (
            baseline_result["rejected_candidates"]
        )
    }


# ============================================================
# TEST
# ============================================================

def main():

    # Use a new speaker ID every time you want
    # to test from scratch.

    speaker_id = "workflow_test_speaker_v2"

    samples = [

        {
            "wpm": 180.0,
            "wps": 3.00,
            "prediction": 0,
            "confidence": 0.95
        },

        {
            "wpm": 185.0,
            "wps": 3.08,
            "prediction": 0,
            "confidence": 0.92
        },

        {
            "wpm": 175.0,
            "wps": 2.92,
            "prediction": 0,
            "confidence": 0.90
        }

    ]

    print()

    print("=" * 60)

    print(
        "PERSONALIZED SPEAKER WORKFLOW TEST"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # PROCESS CANDIDATES
    # --------------------------------------------------------

    for index, sample in enumerate(
        samples,
        start=1
    ):

        result = process_speaker_sample(

            speaker_id=speaker_id,

            wpm=sample["wpm"],

            wps=sample["wps"],

            prediction=sample["prediction"],

            confidence=sample["confidence"]
        )

        print()

        print(
            f"Sample {index}"
        )

        print("-" * 60)

        for key, value in result.items():

            print(
                f"{key}: {value}"
            )

    # ========================================================
    # TEST EXISTING BASELINE
    # ========================================================

    print()

    print("=" * 60)

    print(
        "TESTING EXISTING BASELINE"
    )

    print("=" * 60)

    result = process_speaker_sample(

        speaker_id=speaker_id,

        wpm=150.0,

        wps=2.50,

        prediction=1,

        confidence=0.90
    )

    print()

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":

    main()