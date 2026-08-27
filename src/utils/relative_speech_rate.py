def calculate_relative_speech_rate(
    baseline_wpm,
    current_wpm
):
    """
    Calculate current speech rate relative
    to the person's alert baseline.

    relative_wpm = current_wpm / baseline_wpm
    """

    if baseline_wpm <= 0:
        raise ValueError(
            "Baseline WPM must be greater than zero."
        )

    relative_wpm = (
        current_wpm / baseline_wpm
    )

    wpm_drop = (
        baseline_wpm - current_wpm
    )

    return {
        "baseline_wpm": baseline_wpm,
        "current_wpm": current_wpm,
        "relative_wpm": relative_wpm,
        "wpm_drop": wpm_drop
    }