
"""Compare GREEN, CHROM and POS on timestamped multi-ROI RGB data."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, filtfilt, welch

REGIONS = ("forehead", "left_cheek", "right_cheek", "chin")
LOW_HZ = 0.7
HIGH_HZ = 3.0
MAX_GAP = 0.25
MIN_DURATION = 8.0


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def resample_segment(times, rgb):
    """Interpolate one uninterrupted RGB segment onto a uniform time grid."""
    gaps = np.diff(times)
    if len(times) < 2 or np.any(gaps <= 0) or np.max(gaps) > MAX_GAP:
        raise ValueError("Invalid timestamps or unsafe interpolation gap")

    dt = float(np.median(gaps))
    uniform_times = np.arange(times[0], times[-1], dt)
    uniform_rgb = np.column_stack([
        np.interp(uniform_times, times, rgb[:, channel])
        for channel in range(3)
    ])
    return uniform_times, uniform_rgb, 1.0 / dt


def normalize_rgb(rgb):
    """Remove channel DC levels before colour-based pulse extraction."""
    means = np.mean(rgb, axis=0)
    if np.any(means <= 1e-8):
        raise ValueError("Invalid RGB channel mean")
    normalized = rgb / means - 1.0
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Non-finite normalized RGB values")
    return normalized


def extract_green(rgb):
    return normalize_rgb(rgb)[:, 1]


def extract_chrom(rgb):
    """CHROM-style colour projection."""
    r, g, b = normalize_rgb(rgb).T
    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b

    std_y = np.std(y)
    if std_y < 1e-10:
        raise ValueError("CHROM channel has insufficient variation")

    alpha = np.std(x) / std_y
    return x - alpha * y


def extract_pos(rgb):
    """POS-style projection, applied to one complete segment."""
    r, g, b = normalize_rgb(rgb).T
    s1 = g - b
    s2 = g + b - 2.0 * r

    std_s2 = np.std(s2)
    if std_s2 < 1e-10:
        raise ValueError("POS channel has insufficient variation")

    alpha = np.std(s1) / std_s2
    return s1 + alpha * s2


def estimate_hr(signal, fps):
    """Bandpass-filter and estimate the dominant frequency with Welch PSD."""
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 30 or fps <= 2 * HIGH_HZ:
        raise ValueError("Insufficient samples or sampling rate")

    signal = detrend(signal, type="linear")
    b, a = butter(
        3,
        [LOW_HZ / (fps / 2), HIGH_HZ / (fps / 2)],
        btype="band",
    )

    if len(signal) <= 3 * max(len(a), len(b)):
        raise ValueError("Signal too short for filter")

    filtered = filtfilt(b, a, signal)
    frequencies, power = welch(
        filtered,
        fs=fps,
        nperseg=min(len(filtered), int(round(fps * 8))),
        nfft=8192,
    )

    valid = (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ)
    if not np.any(valid):
        raise ValueError("No frequencies inside HR band")

    frequencies = frequencies[valid]
    power = power[valid]
    peak_index = int(np.argmax(power))

    return {
        "hr_bpm": round(float(frequencies[peak_index] * 60), 2),
        "peak_to_median_power": round(
            float(power[peak_index] / max(np.median(power), 1e-15)), 3
        ),
    }


def analyse_region(rows, region):
    times = np.array(
        [get_float(row, "time_seconds") for row in rows], dtype=float
    )
    rgb = np.array([
        [get_float(row, f"{region}_{channel}") for channel in ("R", "G", "B")]
        for row in rows
    ], dtype=float)

    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError(f"{region}: timestamps invalid")

    # Missing RGB values split the recording into separate safe segments.
    valid = np.all(np.isfinite(rgb), axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return {"status": "insufficient_valid_rgb_samples"}

    splits = np.flatnonzero(
        (np.diff(indices) != 1)
        | (np.diff(times[indices]) > MAX_GAP)
    ) + 1
    groups = np.split(indices, splits)
    segment_results = []

    methods = {
        "GREEN": extract_green,
        "CHROM": extract_chrom,
        "POS": extract_pos,
    }

    for group in groups:
        if len(group) < 2:
            continue

        segment_times = times[group]
        duration = float(segment_times[-1] - segment_times[0])
        if duration < MIN_DURATION:
            continue

        try:
            _, uniform_rgb, fps = resample_segment(
                segment_times, rgb[group]
            )
        except ValueError:
            continue

        result = {
            "duration_seconds": round(duration, 2),
            "sampling_rate_hz": round(fps, 2),
            "methods": {},
        }

        for name, extractor in methods.items():
            try:
                pulse = extractor(uniform_rgb)
                result["methods"][name] = estimate_hr(pulse, fps)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
                result["methods"][name] = {"status": "failed", "reason": str(e)}

        segment_results.append(result)

    if not segment_results:
        return {
            "status": "no_safe_segment_longer_than_minimum_duration",
            "valid_rgb_samples": int(len(indices)),
        }

    # Keep every segment in the report; do not silently combine them.
    return {
        "status": "ok",
        "valid_rgb_samples": int(len(indices)),
        "segments": segment_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path, default=Path("outputs/multi_roi_rgb.csv")
    )
    parser.add_argument(
        "--json", type=Path,
        default=Path("outputs/classical_method_comparison.json")
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if len(rows) < 2:
        raise SystemExit("CSV has fewer than two rows")

    report = {
        "csv": str(args.csv),
        "rows": len(rows),
        "warning": (
            "These are estimates, not accuracy scores. Peak-to-median power "
            "is not calibrated confidence. Validate against synchronized "
            "reference HR before judging accuracy."
        ),
        "regions": {
            region: analyse_region(rows, region)
            for region in REGIONS
        },
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {args.json}")


if __name__ == "__main__":
    main()
