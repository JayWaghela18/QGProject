"""Plot raw and resampled multi-ROI RGB channels without requiring a GUI."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REGIONS = ("forehead", "left_cheek", "right_cheek", "chin")
CHANNELS = ("R", "G", "B")


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_csv(csv_path, output_path):
    rows = load_rows(csv_path)
    times = np.asarray([float(row["time_seconds"]) for row in rows])
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    for axis, region in zip(axes, REGIONS):
        for channel in CHANNELS:
            values = []
            valid_times = []
            for time_value, row in zip(times, rows):
                try:
                    value = float(row[f"{region}_{channel}"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    valid_times.append(time_value)
                    values.append(value)
            axis.plot(valid_times, values, label=channel, linewidth=0.8)
        axis.set_title(region.replace("_", " ").title())
        axis.set_ylabel("RGB mean")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Elapsed monotonic time (seconds)")
    fig.suptitle("Raw RGB Signals from Four Facial Regions")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/multi_roi_rgb.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/multi_roi_rgb_plot.png"),
    )
    args = parser.parse_args()
    plot_csv(args.csv, args.output)
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()
