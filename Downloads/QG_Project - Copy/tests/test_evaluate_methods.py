import csv
import tempfile
import unittest
from pathlib import Path

from src.evaluate_methods import evaluate_csv, evaluate_rows


class EvaluateMethodsSyntheticTests(unittest.TestCase):
    """Synthetic fixtures validate evaluator mechanics, not model performance."""

    def test_metrics_and_correlation_use_matched_windows(self):
        rows = [
            {
                "session_id": "s1",
                "subject_id": "p1",
                "window_number": number,
                "start_seconds": float(number - 1) * 10,
                "end_seconds": float(number) * 10,
                "reference_hr_bpm": reference,
                "predictions": {
                    "CNN": prediction,
                    "POS": prediction,
                    "CHROM": prediction,
                    "GREEN": prediction,
                },
                "accepted": True,
            }
            for number, reference, prediction in (
                (1, 60.0, 61.0),
                (2, 70.0, 69.0),
                (3, 80.0, 82.0),
            )
        ]
        result = evaluate_rows(rows)["CNN"]
        self.assertEqual(result["matched_windows"], 3)
        self.assertAlmostEqual(result["mae_bpm"], 4 / 3)
        self.assertAlmostEqual(result["signed_bias_bpm"], 2 / 3)
        self.assertEqual(result["within_5_bpm_percent"], 100.0)
        self.assertIsNotNone(result["correlation"])

    def test_legacy_csv_is_unavailable(self):
        fields = [
            "test_id", "video", "subject", "actual_hr", "predicted_hr",
            "error_bpm", "error_percent", "notes",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict.fromkeys(fields, "1"))
            result = evaluate_csv(path)
        self.assertFalse(result["available"])
        self.assertIn("missing columns", result["unavailable_reason"])

    def test_duplicate_windows_are_unavailable(self):
        fields = [
            "session_id", "subject_id", "window_number", "start_seconds",
            "end_seconds", "reference_hr_bpm", "cnn_hr", "pos_hr",
            "chrom_hr", "green_hr", "accepted",
        ]
        row = {
            "session_id": "s1", "subject_id": "p1", "window_number": "1",
            "start_seconds": "0", "end_seconds": "10",
            "reference_hr_bpm": "70", "cnn_hr": "71", "pos_hr": "70",
            "chrom_hr": "69", "green_hr": "70", "accepted": "true",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
                writer.writerow(row)
            result = evaluate_csv(path)
        self.assertFalse(result["available"])
        self.assertIn("duplicate", result["unavailable_reason"])

    def test_missing_reference_and_invalid_prediction_are_counted(self):
        rows = [
            {
                "session_id": "s1",
                "subject_id": "p1",
                "window_number": 1,
                "start_seconds": 0.0,
                "end_seconds": 10.0,
                "reference_hr_bpm": None,
                "predictions": {"CNN": 70.0, "POS": None, "CHROM": None, "GREEN": None},
                "accepted": True,
            },
            {
                "session_id": "s1",
                "subject_id": "p1",
                "window_number": 2,
                "start_seconds": 10.0,
                "end_seconds": 20.0,
                "reference_hr_bpm": 70.0,
                "predictions": {"CNN": None, "POS": 70.0, "CHROM": None, "GREEN": None},
                "accepted": True,
            },
            {
                "session_id": "s1",
                "subject_id": "p1",
                "window_number": 3,
                "start_seconds": 20.0,
                "end_seconds": 30.0,
                "reference_hr_bpm": 70.0,
                "predictions": {"CNN": 70.0, "POS": None, "CHROM": None, "GREEN": None},
                "accepted": False,
            },
        ]
        result = evaluate_rows(rows)["CNN"]
        self.assertFalse(result["available"])
        self.assertEqual(result["rejection_counts"]["missing_reference"], 1)
        self.assertEqual(result["rejection_counts"]["missing_or_invalid_prediction"], 1)
        self.assertEqual(result["rejection_counts"]["rejected_quality"], 1)


if __name__ == "__main__":
    unittest.main()
