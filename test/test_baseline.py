from src.personalization.baseline import (
    SpeakerBaseline
)


def main():

    print("\n")
    print("=" * 60)
    print("SPEAKER BASELINE TEST")
    print("=" * 60)

    baseline_manager = SpeakerBaseline()

    # --------------------------------------------------------
    # Simulated alert recordings
    # --------------------------------------------------------

    alert_recordings = [

        {
            "wpm": 200.0,
            "wps": 3.33
        },

        {
            "wpm": 205.0,
            "wps": 3.42
        },

        {
            "wpm": 210.0,
            "wps": 3.50
        }

    ]

    # --------------------------------------------------------
    # Calculate baseline
    # --------------------------------------------------------

    baseline = (
        baseline_manager.calculate_baseline(
            alert_recordings
        )
    )

    print("\nCalculated baseline:")

    print(
        f"  WPM: "
        f"{baseline['wpm']:.2f}"
    )

    print(
        f"  WPS: "
        f"{baseline['wps']:.2f}"
    )

    print(
        f"  Recordings: "
        f"{baseline['num_recordings']}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    path = baseline_manager.save(
        "test_speaker",
        baseline
    )

    print(
        f"\n✓ Baseline saved:"
        f"\n  {path}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    loaded = baseline_manager.load(
        "test_speaker"
    )

    print("\nLoaded baseline:")

    print(
        loaded
    )

    # --------------------------------------------------------
    # Relative speech rate
    # --------------------------------------------------------

    current_wpm = 170.0

    relative_rate = (
        baseline_manager.relative_speech_rate(
            current_wpm,
            loaded
        )
    )

    print(
        f"\nCurrent WPM: "
        f"{current_wpm:.2f}"
    )

    print(
        f"Baseline WPM: "
        f"{loaded['wpm']:.2f}"
    )

    print(
        f"Relative speech rate: "
        f"{relative_rate:.4f}"
    )

    print("\n")
    print("=" * 60)
    print("BASELINE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()