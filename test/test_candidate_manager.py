from src.personalization.candidate_manager import (
    CandidateManager
)


def main():

    print("\n" + "=" * 60)
    print("CANDIDATE BASELINE TEST")
    print("=" * 60)

    manager = CandidateManager(
        min_confidence=0.80,
        required_candidates=3
    )

    speaker_id = "test_speaker"

    # --------------------------------------------------------
    # Candidate 1
    # --------------------------------------------------------

    result = manager.add_candidate(
        speaker_id=speaker_id,
        wpm=185.0,
        wps=3.08,
        prediction=0,
        confidence=0.91
    )

    print("\nCandidate 1:")
    print(result)

    # --------------------------------------------------------
    # Candidate 2
    # --------------------------------------------------------

    result = manager.add_candidate(
        speaker_id=speaker_id,
        wpm=182.0,
        wps=3.03,
        prediction=0,
        confidence=0.87
    )

    print("\nCandidate 2:")
    print(result)

    # --------------------------------------------------------
    # Candidate 3
    # --------------------------------------------------------

    result = manager.add_candidate(
        speaker_id=speaker_id,
        wpm=188.0,
        wps=3.13,
        prediction=0,
        confidence=0.94
    )

    print("\nCandidate 3:")
    print(result)

    # --------------------------------------------------------
    # Invalid candidate
    # --------------------------------------------------------

    result = manager.add_candidate(
        speaker_id=speaker_id,
        wpm=130.0,
        wps=2.16,
        prediction=1,
        confidence=0.95
    )

    print("\nFatigued recording:")
    print(result)

    # --------------------------------------------------------
    # Show candidates
    # --------------------------------------------------------

    print("\nStored candidates:")

    candidates = manager.get_candidates(
        speaker_id
    )

    for candidate in candidates:

        print(
            f"\nWPM: {candidate['wpm']}"
        )

        print(
            f"WPS: {candidate['wps']}"
        )

        print(
            f"Confidence: "
            f"{candidate['confidence']:.2%}"
        )

    print("\nReady for baseline:")

    print(
        manager.is_ready_for_baseline(
            speaker_id
        )
    )
        # --------------------------------------------------------
    # CREATE FINAL BASELINE
    # --------------------------------------------------------

    print("\nCreating final baseline...")

    result = manager.create_baseline(
        speaker_id
    )

    print("\nBaseline result:")

    print(result)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()
    