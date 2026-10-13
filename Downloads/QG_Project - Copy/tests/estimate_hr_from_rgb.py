"""Validate timestamped multi-ROI RGB data and estimate classical GREEN HR."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, filtfilt, welch


REGIONS = ("forehead", "left_cheek", "right_cheek", "chin")
CHANNELS = ("R", "G", "B")
LOW_HZ = 0.7
HIGH_HZ = 3.0
MAX_INTERPOLATION_GAP_SECONDS = 0.25
MIN_SEGMENT_SECONDS = 8.0
FILTER_ORDER = 3


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def validate_timestamps(rows):
    timestamps = np.asarray(
        [_float_or_nan(row.get("time_seconds")) for row in rows],
        dtype=np.float64,
    )
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("at least two timestamps are required")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps contain non-finite values")
    differences = np.diff(timestamps)
    if not np.all(differences > 0):
        raise ValueError("timestamps must be strictly increasing")
    return timestamps


def contiguous_segments(timestamps, max_gap=MAX_INTERPOLATION_GAP_SECONDS):
    gaps = np.diff(timestamps)
    split_indices = np.flatnonzero(gaps > max_gap) + 1
    boundaries = np.concatenate(([0], split_indices, [len(timestamps)]))
    return [
        (int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end - start >= 2
    ]


def uniform_resample(timestamps, values, max_gap=MAX_INTERPOLATION_GAP_SECONDS):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(timestamps) != len(values) or len(timestamps) < 2:
        raise ValueError("timestamp and signal lengths are invalid")
    if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(values)):
        raise ValueError("timestamps and signal must be finite")
    gaps = np.diff(timestamps)
    if np.any(gaps <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if np.max(gaps) > max_gap:
        raise ValueError("segment contains an unsafe interpolation gap")

    interval = float(np.median(gaps))
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("invalid median sampling interval")
    target = np.arange(timestamps[0], timestamps[-1] + interval / 2, interval)
    return target, np.interp(target, timestamps, values), interval


def bandpass_and_estimate(signal, fps):
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 30:
        raise ValueError("signal is too short")
    if not np.isfinite(fps) or fps <= 2 * HIGH_HZ:
        raise ValueError("sampling rate is too low for the requested band")

    filtered = detrend(signal, type="linear")
    nyquist = fps / 2.0
    b, a = butter(
        FILTER_ORDER,
        [LOW_HZ / nyquist, HIGH_HZ / nyquist],
        btype="band",
    )
    pad_length = 3 * max(len(a), len(b))
    if len(filtered) <= pad_length:
        raise ValueError("signal is too short for filter edge handling")
    filtered = filtfilt(b, a, filtered)

    frequencies, power = welch(
        filtered,
        fs=fps,
        nperseg=min(len(filtered), max(64, int(round(fps * 8)))),
        nfft=4096,
    )
    valid = (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ)
    if not np.any(valid):
        raise ValueError("no spectrum inside heart-rate band")
    band_power = power[valid]
    band_frequencies = frequencies[valid]
    peak_index = int(np.argmax(band_power))
    peak_hz = float(band_frequencies[peak_index])
    median_power = float(np.median(band_power))
    peak_to_median_power = (
        float(band_power[peak_index] / median_power)
        if median_power > 0 else None
    )
    return {
        "hr_bpm": peak_hz * 60.0,
        "peak_hz": peak_hz,
        "peak_to_median_power": peak_to_median_power,
        "filtered": filtered,
        "frequencies": band_frequencies,
        "power": band_power,
    }


def estimate_region(rows, region):
    timestamps = validate_timestamps(rows)
    values = np.asarray(
        [_float_or_nan(row.get(f"{region}_G")) for row in rows],
        dtype=np.float64,
    )
    finite = np.isfinite(values)
    valid_timestamps = timestamps[finite]
    valid_values = values[finite]
    if len(valid_values) < 2:
        return {
            "region": region,
            "status": "insufficient_samples",
            "input_samples": len(rows),
            "valid_samples": int(len(valid_values)),
        }

    results = []
    for start, end in contiguous_segments(valid_timestamps):
        segment_timestamps = valid_timestamps[start:end]
        segment_values = valid_values[start:end]
        duration = float(segment_timestamps[-1] - segment_timestamps[0])
        if duration < MIN_SEGMENT_SECONDS:
            continue
        try:
            uniform_timestamps, uniform_values, interval = uniform_resample(
                segment_timestamps,
                segment_values,
            )
            spectrum = bandpass_and_estimate(
                uniform_values,
                1.0 / interval,
            )
            results.append({
                "start_seconds": float(uniform_timestamps[0]),
                "end_seconds": float(uniform_timestamps[-1]),
                "duration_seconds": duration,
                "resampled_samples": len(uniform_values),
                "median_sampling_rate_hz": 1.0 / interval,
                "max_gap_seconds": float(np.max(np.diff(segment_timestamps))),
                "hr_bpm": spectrum["hr_bpm"],
                "peak_to_median_power": spectrum["peak_to_median_power"],
            })
        except ValueError as error:
            results.append({
                "start_seconds": float(segment_timestamps[0]),
                "end_seconds": float(segment_timestamps[-1]),
                "duration_seconds": duration,
                "status": "rejected",
                "reason": str(error),
            })

    if not results:
        return {
            "region": region,
            "status": "no_safe_segment",
            "input_samples": len(rows),
            "valid_samples": int(len(valid_values)),
            "large_gap_count": int(
                np.sum(np.diff(valid_timestamps) > MAX_INTERPOLATION_GAP_SECONDS)
            ),
        }
    return {
        "region": region,
        "status": "ok",
        "input_samples": len(rows),
        "valid_samples": int(len(valid_values)),
        "large_gap_count": int(
            np.sum(np.diff(valid_timestamps) > MAX_INTERPOLATION_GAP_SECONDS)
        ),
        "segments": results,
        "best_segment": max(
            results,
            key=lambda result: result.get("duration_seconds", 0.0),
        ),
    }


def analyze_csv(path):
    rows = load_csv(path)
    timestamps = validate_timestamps(rows)
    report = {
        "csv": str(path),
        "rows": len(rows),
        "recorded_duration_seconds": float(timestamps[-1] - timestamps[0]),
        "median_interval_seconds": float(np.median(np.diff(timestamps))),
        "mean_interval_seconds": float(np.mean(np.diff(timestamps))),
        "max_interval_seconds": float(np.max(np.diff(timestamps))),
        "intervals_over_0_1_seconds": int(np.sum(np.diff(timestamps) > 0.1)),
        "peak_to_median_power_interpretation": (
            "relative spectral peak power divided by median in-band power; "
            "not a calibrated confidence score"
        ),
        "regions": {
            region: estimate_region(rows, region)
            for region in REGIONS
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/multi_roi_rgb.csv"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("outputs/classical_rppg_results.json"),
    )
    args = parser.parse_args()
    report = analyze_csv(args.csv)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
