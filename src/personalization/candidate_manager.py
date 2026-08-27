import json
import numpy as np

from pathlib import Path
from datetime import datetime

from src.personalization.baseline import SpeakerBaseline


class CandidateManager:

    def __init__(
        self,
        candidate_dir="data/baseline_candidates",
        min_confidence=0.80,
        required_candidates=3,

        # Maximum relative difference for
        # individual feature duplicate comparison
        duplicate_thresholds=None,

        # Minimum number of features that must
        # be extremely similar to call it duplicate
        duplicate_match_count=5
    ):

        self.candidate_dir = Path(
            candidate_dir
        )

        self.candidate_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.min_confidence = (
            min_confidence
        )

        self.required_candidates = (
            required_candidates
        )

        # ----------------------------------------------------
        # DUPLICATE THRESHOLDS
        # ----------------------------------------------------

        if duplicate_thresholds is None:

            duplicate_thresholds = {

                "wpm": 0.02,

                "mean_f0": 0.03,

                "energy_mean": 0.03,

                "jitter": 0.05,

                "shimmer": 0.05,

                "hnr": 0.05
            }

        self.duplicate_thresholds = (
            duplicate_thresholds
        )

        self.duplicate_match_count = (
            duplicate_match_count
        )

    # ========================================================
    # SPEAKER DIRECTORY
    # ========================================================

    def get_speaker_dir(
        self,
        speaker_id
    ):

        speaker_dir = (
            self.candidate_dir /
            str(speaker_id)
        )

        speaker_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        return speaker_dir

    # ========================================================
    # CHECK IF RECORDING IS ELIGIBLE
    # ========================================================

    def is_eligible(
        self,
        prediction,
        confidence
    ):

        # Alert class = 0

        if prediction != 0:

            return False

        if confidence < self.min_confidence:

            return False

        return True

    # ========================================================
    # COUNT CANDIDATES
    # ========================================================

    def count_candidates(
        self,
        speaker_id
    ):

        speaker_dir = (
            self.get_speaker_dir(
                speaker_id
            )
        )

        candidates = list(
            speaker_dir.glob(
                "candidate_*.json"
            )
        )

        return len(
            candidates
        )

    # ========================================================
    # GET ALL CANDIDATES
    # ========================================================

    def get_candidates(
        self,
        speaker_id
    ):

        speaker_dir = (
            self.get_speaker_dir(
                speaker_id
            )
        )

        candidate_files = sorted(
            speaker_dir.glob(
                "candidate_*.json"
            )
        )

        candidates = []

        for path in candidate_files:

            with open(
                path,
                "r"
            ) as file:

                candidate = json.load(
                    file
                )

            candidates.append(
                candidate
            )

        return candidates

    # ========================================================
    # CALCULATE RELATIVE DIFFERENCE
    # ========================================================

    def calculate_relative_difference(
        self,
        value1,
        value2
    ):

        denominator = max(
            abs(value2),
            1e-6
        )

        return abs(
            value1 - value2
        ) / denominator

    # ========================================================
    # CHECK DUPLICATE CANDIDATE
    # ========================================================

    def is_duplicate_candidate(
        self,
        speaker_id,
        personalized_features
    ):

        candidates = (
            self.get_candidates(
                speaker_id
            )
        )

        if not candidates:

            return False

        # ----------------------------------------------------
        # FEATURES USED FOR DUPLICATE DETECTION
        # ----------------------------------------------------

        feature_names = [

            "wpm",

            "mean_f0",

            "energy_mean",

            "jitter",

            "shimmer",

            "hnr"
        ]

        # ----------------------------------------------------
        # COMPARE AGAINST EVERY EXISTING CANDIDATE
        # ----------------------------------------------------

        for candidate in candidates:

            matching_features = 0

            # ----------------------------------------------
            # COMPARE EACH FEATURE
            # ----------------------------------------------

            for feature_name in feature_names:

                current_value = float(
                    personalized_features[
                        feature_name
                    ]
                )

                existing_value = float(
                    candidate[
                        feature_name
                    ]
                )

                difference = (
                    self.calculate_relative_difference(
                        current_value,
                        existing_value
                    )
                )

                threshold = (
                    self.duplicate_thresholds[
                        feature_name
                    ]
                )

                if difference <= threshold:

                    matching_features += 1

            # ----------------------------------------------
            # DUPLICATE DECISION
            # ----------------------------------------------

            if (
                matching_features
                >=
                self.duplicate_match_count
            ):

                return True

        return False

    # ========================================================
    # ADD CANDIDATE
    # ========================================================

    def add_candidate(
        self,
        speaker_id,
        personalized_features,
        prediction,
        confidence
    ):

        # ----------------------------------------------------
        # CHECK IF RECORDING IS ELIGIBLE
        # ----------------------------------------------------

        if not self.is_eligible(
            prediction,
            confidence
        ):

            return {

                "accepted": False,

                "reason": (
                    "Recording is not a sufficiently "
                    "confident Alert prediction."
                )
            }

        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        wpm = float(
            personalized_features[
                "wpm"
            ]
        )

        wps = float(
            personalized_features[
                "wps"
            ]
        )

        # ----------------------------------------------------
        # CHECK FOR DUPLICATE CANDIDATE
        # ----------------------------------------------------

        is_duplicate = (
            self.is_duplicate_candidate(
                speaker_id=speaker_id,
                personalized_features=(
                    personalized_features
                )
            )
        )

        if is_duplicate:

            return {

                "accepted": False,

                "reason": (
                    "This recording is too similar "
                    "to an existing baseline candidate."
                )
            }

        # ----------------------------------------------------
        # GET SPEAKER DIRECTORY
        # ----------------------------------------------------

        speaker_dir = (
            self.get_speaker_dir(
                speaker_id
            )
        )

        # ----------------------------------------------------
        # DETERMINE CANDIDATE NUMBER
        # ----------------------------------------------------

        candidate_number = (
            self.count_candidates(
                speaker_id
            )
            + 1
        )

        candidate_path = (
            speaker_dir /
            f"candidate_{candidate_number:03d}.json"
        )

        # ----------------------------------------------------
        # CREATE CANDIDATE DATA
        # ----------------------------------------------------

        candidate_data = {

            "speaker_id": speaker_id,

            "wpm": wpm,

            "wps": wps,

            "mean_f0": float(
                personalized_features[
                    "mean_f0"
                ]
            ),

            "energy_mean": float(
                personalized_features[
                    "energy_mean"
                ]
            ),

            "jitter": float(
                personalized_features[
                    "jitter"
                ]
            ),

            "shimmer": float(
                personalized_features[
                    "shimmer"
                ]
            ),

            "hnr": float(
                personalized_features[
                    "hnr"
                ]
            ),

            "prediction": "Alert",

            "confidence": float(
                confidence
            ),

            "created_at": (
                datetime.now().isoformat()
            )
        }

        # ----------------------------------------------------
        # SAVE CANDIDATE
        # ----------------------------------------------------

        with open(
            candidate_path,
            "w"
        ) as file:

            json.dump(
                candidate_data,
                file,
                indent=4
            )

        # ----------------------------------------------------
        # COUNT TOTAL CANDIDATES
        # ----------------------------------------------------

        total_candidates = (
            self.count_candidates(
                speaker_id
            )
        )

        return {

            "accepted": True,

            "candidate_path": str(
                candidate_path
            ),

            "total_candidates": (
                total_candidates
            ),

            "required_candidates": (
                self.required_candidates
            ),

            "ready_for_baseline": (
                total_candidates
                >=
                self.required_candidates
            )
        }

    # ========================================================
    # CHECK BASELINE READINESS
    # ========================================================

    def is_ready_for_baseline(
        self,
        speaker_id
    ):

        count = (
            self.count_candidates(
                speaker_id
            )
        )

        return (
            count
            >=
            self.required_candidates
        )

    # ========================================================
    # CREATE FINAL BASELINE
    # ========================================================

    def create_baseline(
        self,
        speaker_id,
        max_wpm_deviation=0.20
    ):

        # ----------------------------------------------------
        # CHECK CANDIDATE COUNT
        # ----------------------------------------------------

        if not self.is_ready_for_baseline(
            speaker_id
        ):

            return {

                "success": False,

                "reason": (
                    "Not enough reliable candidates "
                    "to create a baseline."
                )
            }

        # ----------------------------------------------------
        # LOAD CANDIDATES
        # ----------------------------------------------------

        candidates = (
            self.get_candidates(
                speaker_id
            )
        )

        wpm_values = [

            candidate["wpm"]

            for candidate in candidates
        ]

        median_wpm = float(
            np.median(
                wpm_values
            )
        )

        # ----------------------------------------------------
        # REMOVE ABNORMAL CANDIDATES
        # ----------------------------------------------------

        valid_candidates = []

        rejected_candidates = []

        for candidate in candidates:

            wpm = candidate[
                "wpm"
            ]

            deviation = abs(
                wpm - median_wpm
            ) / max(
                median_wpm,
                1e-6
            )

            if deviation <= max_wpm_deviation:

                valid_candidates.append(
                    candidate
                )

            else:

                rejected_candidates.append(
                    candidate
                )

        # ----------------------------------------------------
        # NEED ENOUGH VALID CANDIDATES
        # ----------------------------------------------------

        if len(
            valid_candidates
        ) < self.required_candidates:

            return {

                "success": False,

                "reason": (
                    "Too many inconsistent "
                    "candidate recordings."
                ),

                "valid_candidates": len(
                    valid_candidates
                ),

                "rejected_candidates": len(
                    rejected_candidates
                )
            }

        # ----------------------------------------------------
        # CONVERT INTO BASELINE RECORDS
        # ----------------------------------------------------

        feature_records = []

        for candidate in valid_candidates:

            feature_records.append(

                {

                    "wpm": candidate[
                        "wpm"
                    ],

                    "wps": candidate[
                        "wps"
                    ],

                    "mean_f0": candidate[
                        "mean_f0"
                    ],

                    "energy_mean": candidate[
                        "energy_mean"
                    ],

                    "jitter": candidate[
                        "jitter"
                    ],

                    "shimmer": candidate[
                        "shimmer"
                    ],

                    "hnr": candidate[
                        "hnr"
                    ]
                }
            )

        # ----------------------------------------------------
        # CREATE BASELINE
        # ----------------------------------------------------

        baseline_manager = (
            SpeakerBaseline()
        )

        baseline = (
            baseline_manager.calculate_baseline(
                feature_records
            )
        )

        baseline_path = (
            baseline_manager.save(
                speaker_id,
                baseline
            )
        )

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return {

            "success": True,

            "baseline": baseline,

            "baseline_path": str(
                baseline_path
            ),

            "valid_candidates": len(
                valid_candidates
            ),

            "rejected_candidates": len(
                rejected_candidates
            )
        }