from src.utils.relative_speech_rate import (
    calculate_relative_speech_rate
)


def main():

    baseline_wpm = 180.0
    current_wpm = 135.0

    result = calculate_relative_speech_rate(
        baseline_wpm=baseline_wpm,
        current_wpm=current_wpm
    )

    print("\n")
    print("=" * 60)
    print("RELATIVE SPEECH RATE TEST")
    print("=" * 60)

    print(
        f"\nBaseline WPM: "
        f"{result['baseline_wpm']:.2f}"
    )

    print(
        f"Current WPM: "
        f"{result['current_wpm']:.2f}"
    )

    print(
        f"Relative WPM: "
        f"{result['relative_wpm']:.4f}"
    )

    print(
        f"WPM Drop: "
        f"{result['wpm_drop']:.2f}"
    )

    print("\n")
    print("=" * 60)


if __name__ == "__main__":
    main()