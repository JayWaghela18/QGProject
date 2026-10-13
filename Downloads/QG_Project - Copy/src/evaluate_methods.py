"""Evaluate matched, synchronized per-window heart-rate predictions.

The evaluator intentionally does not contain fallback or demonstration data.
See ``EVALUATION_COLUMNS`` for the required wide CSV format.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHOD_COLUMNS = {
    "CNN": "cnn_hr",
    "POS": "pos_hr",
    "CHROM": "chrom_hr",
    "GREEN": "green_hr",
}
EVALUATION_COLUMNS = (
    "session_id",
    "subject_id",
    "window_number",
    "start_seconds",
    "end_seconds",
    "reference_hr_bpm",
    "cnn_hr",
    "pos_hr",
    "chrom_hr",
    "green_hr",
    "accepted",
)
MIN_VALID_BPM = 20.0
MAX_VALID_BPM = 240.0


def _finite_bpm(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or not MIN_VALID_BPM <= number <= MAX_VALID_BPM:
        return None
    return number


def _parse_accepted(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError("accepted must be true/false (or 1/0, yes/no)")


def _validate_row(row, row_number):
    missing = [column for column in EVALUATION_COLUMNS if column not in row]
    if missing:
        raise ValueError(
            f"missing required columns: {', '.join(missing)}"
        )

    session_id = row["session_id"].strip()
    subject_id = row["subject_id"].strip()
    if not session_id or not subject_id:
        raise ValueError(f"row {row_number}: session_id and subject_id are required")

    try:
        window_number = int(row["window_number"])
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"row {row_number}: invalid window number or boundaries"
        ) from error
    if window_number < 1 or not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError(f"row {row_number}: invalid window number or boundaries")

    reference = _finite_bpm(row["reference_hr_bpm"])
    predictions = {
        method: _finite_bpm(row[column])
        for method, column in METHOD_COLUMNS.items()
    }
    return {
        "session_id": session_id,
        "subject_id": subject_id,
        "window_number": window_number,
        "start_seconds": start,
        "end_seconds": end,
        "reference_hr_bpm": reference,
        "predictions": predictions,
        "accepted": _parse_accepted(row["accepted"]),
    }


def load_evaluation_csv(path):
    """Load and structurally validate the documented wide evaluation CSV."""
    rows = []
    seen_windows = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        missing = [column for column in EVALUATION_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(
                "CSV is not the Stage G1 format; missing columns: "
                + ", ".join(missing)
            )
        for row_number, row in enumerate(reader, start=2):
            parsed = _validate_row(row, row_number)
            key = (parsed["session_id"], parsed["window_number"])
            if key in seen_windows:
                raise ValueError(
                    f"duplicate or mismatched window: "
                    f"{parsed['session_id']} / {parsed['window_number']}"
                )
            seen_windows.add(key)
            rows.append(parsed)
    if not rows:
        raise ValueError("CSV contains no evaluation rows")
    return rows


def calculate_mae(predicted, ground_truth):
    predicted = np.asarray(predicted, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    return float(np.mean(np.abs(predicted - ground_truth)))


def calculate_rmse(predicted, ground_truth):
    predicted = np.asarray(predicted, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    return float(np.sqrt(np.mean((predicted - ground_truth) ** 2)))


def calculate_error_percentage(predicted, ground_truth):
    predicted = np.asarray(predicted, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    return float(np.mean(np.abs(predicted - ground_truth) / ground_truth) * 100)


def _method_result(method, rows):
    reference_rows = [row for row in rows if row["reference_hr_bpm"] is not None]
    matched = [
        row for row in reference_rows
        if row["accepted"] and row["predictions"][method] is not None
    ]
    rejected_quality = sum(
        not row["accepted"] for row in reference_rows
    )
    missing_predictions = sum(
        row["accepted"] and row["predictions"][method] is None
        for row in reference_rows
    )
    references = np.asarray(
        [row["reference_hr_bpm"] for row in matched],
        dtype=float,
    )
    predictions = np.asarray(
        [row["predictions"][method] for row in matched],
        dtype=float,
    )
    result = {
        "method": method,
        "available": bool(matched),
        "mae_bpm": None,
        "rmse_bpm": None,
        "signed_bias_bpm": None,
        "correlation": None,
        "within_5_bpm_percent": None,
        "matched_windows": len(matched),
        "reference_windows": len(reference_rows),
        "valid_coverage_percent": (
            100.0 * len(matched) / len(reference_rows)
            if reference_rows else 0.0
        ),
        "rejection_counts": {
            "missing_reference": len(rows) - len(reference_rows),
            "rejected_quality": rejected_quality,
            "missing_or_invalid_prediction": missing_predictions,
        },
        "unavailable_reason": None,
    }
    if not matched:
        result["unavailable_reason"] = (
            "no accepted windows have both a valid reference and prediction"
        )
        return result

    errors = predictions - references
    result["mae_bpm"] = calculate_mae(predictions, references)
    result["rmse_bpm"] = calculate_rmse(predictions, references)
    result["signed_bias_bpm"] = float(np.mean(errors))
    result["within_5_bpm_percent"] = float(
        np.mean(np.abs(errors) <= 5.0) * 100
    )
    if len(matched) >= 3 and np.std(predictions) > 0 and np.std(references) > 0:
        result["correlation"] = float(np.corrcoef(predictions, references)[0, 1])
    else:
        result["unavailable_reason"] = (
            "correlation unavailable: at least three matched windows with "
            "non-constant predictions and references are required"
        )
    return result


def evaluate_rows(rows):
    """Evaluate all four methods on the same structurally matched windows."""
    return {
        method: _method_result(method, rows)
        for method in METHOD_COLUMNS
    }


def evaluate_csv(path):
    try:
        rows = load_evaluation_csv(path)
    except (OSError, ValueError) as error:
        return {
            "available": False,
            "unavailable_reason": str(error),
            "methods": {},
        }
    return {
        "available": any(
            result["available"] for result in evaluate_rows(rows).values()
        ),
        "matched_window_count": len(rows),
        "methods": evaluate_rows(rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate synchronized Stage G1 per-window HR predictions."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()
    result = evaluate_csv(args.csv_path)
    output = json.dumps(result, indent=2)
    print(output)
    if args.json_path:
        args.json_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
