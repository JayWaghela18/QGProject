"""Collect timestamped MediaPipe multi-ROI RGB samples from a webcam.

The camera timestamp is measured independently from the nominal camera FPS.
MediaPipe receives a strictly increasing VIDEO timestamp, while the CSV keeps
the original elapsed monotonic timestamp for later validation and resampling.
"""

import csv
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path("models/face_landmarker.task")
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "multi_roi_rgb.csv"
DIAGNOSTICS_PATH = OUTPUT_DIR / "multi_roi_rgb_diagnostics.json"
DURATION_SECONDS = 30.0
CAMERA_INDEX = 0
MIN_ROI_WIDTH = 10
MIN_ROI_HEIGHT = 10

ROI_RANGES = {
    "forehead": (0.20, 0.08, 0.80, 0.28),
    "left_cheek": (0.12, 0.38, 0.43, 0.62),
    "right_cheek": (0.57, 0.38, 0.88, 0.62),
    "chin": (0.30, 0.68, 0.70, 0.90),
}
CHANNELS = ("R", "G", "B")


def get_face_box(landmarks, frame_shape):
    height, width = frame_shape[:2]
    points = []
    for point in landmarks:
        try:
            x = float(point.x)
            y = float(point.y)
        except (AttributeError, TypeError, ValueError):
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            points.append((x, y))

    if len(points) < 3:
        return None

    x1 = max(0, int(round(min(x for x, _ in points) * width)))
    y1 = max(0, int(round(min(y for _, y in points) * height)))
    x2 = min(width, int(round(max(x for x, _ in points) * width)))
    y2 = min(height, int(round(max(y for _, y in points) * height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def empty_roi_sample():
    return {
        f"{region}_{channel}": ""
        for region in ROI_RANGES
        for channel in CHANNELS
    }


def extract_rois(rgb_frame, face_box):
    sample = empty_roi_sample()
    boxes = {}
    if face_box is None:
        return sample, boxes

    x, y, width, height = face_box
    frame_height, frame_width = rgb_frame.shape[:2]
    for region, (left, top, right, bottom) in ROI_RANGES.items():
        x1 = max(0, min(frame_width, int(round(x + left * width))))
        y1 = max(0, min(frame_height, int(round(y + top * height))))
        x2 = max(0, min(frame_width, int(round(x + right * width))))
        y2 = max(0, min(frame_height, int(round(y + bottom * height))))
        roi_width = x2 - x1
        roi_height = y2 - y1
        if roi_width < MIN_ROI_WIDTH or roi_height < MIN_ROI_HEIGHT:
            continue

        crop = rgb_frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        mean_rgb = crop.mean(axis=(0, 1))
        if mean_rgb.shape != (3,) or not np.isfinite(mean_rgb).all():
            continue
        for channel, value in zip(CHANNELS, mean_rgb):
            sample[f"{region}_{channel}"] = round(float(value), 3)
        boxes[region] = (x1, y1, x2, y2)
    return sample, boxes


def _backup_existing(path):
    if not path.exists():
        return None
    backup = path.with_name(
        f"{path.stem}.pre_capture_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"
    )
    shutil.copy2(path, backup)
    return str(backup)


def collect():
    model_path = Path(MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(f"Face Landmarker model not found: {model_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        detector.close()
        raise RuntimeError("Could not open webcam.")

    rows = []
    frame_number = 0
    face_frames = 0
    read_failures = 0
    timestamp_discontinuities = 0
    elapsed_timestamps = []
    previous_elapsed = None
    previous_video_timestamp = -1
    start_time = time.monotonic()
    stop_reason = "duration_reached"

    try:
        print(f"Collecting RGB samples for {DURATION_SECONDS:.1f} seconds.")
        print("Press Q to stop early.")
        while True:
            before_read = time.monotonic()
            if before_read - start_time >= DURATION_SECONDS:
                break
            success, frame = camera.read()
            if not success:
                read_failures += 1
                if time.monotonic() - start_time >= DURATION_SECONDS:
                    break
                continue

            frame_number += 1
            elapsed = time.monotonic() - start_time
            video_timestamp = max(
                previous_video_timestamp + 1,
                int(round(elapsed * 1000)),
            )
            if previous_elapsed is not None and elapsed <= previous_elapsed:
                timestamp_discontinuities += 1
            previous_elapsed = elapsed
            previous_video_timestamp = video_timestamp
            elapsed_timestamps.append(elapsed)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            result = detector.detect_for_video(image, video_timestamp)
            display = frame.copy()
            sample = {
                "frame": frame_number,
                "time_seconds": round(elapsed, 6),
                "mediapipe_timestamp_ms": video_timestamp,
                "face_detected": 0,
                **empty_roi_sample(),
            }
            face_box = (
                get_face_box(result.face_landmarks[0], frame.shape)
                if result.face_landmarks
                else None
            )
            if face_box is not None:
                face_frames += 1
                sample["face_detected"] = 1
                roi_values, boxes = extract_rois(rgb_frame, face_box)
                sample.update(roi_values)
                x, y, width, height = face_box
                cv2.rectangle(
                    display, (x, y), (x + width, y + height),
                    (255, 255, 255), 1,
                )
                for region, (x1, y1, x2, y2) in boxes.items():
                    cv2.rectangle(
                        display, (x1, y1), (x2, y2), (0, 255, 0), 1
                    )
                    cv2.putText(
                        display, region, (x1, max(20, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
                    )
            else:
                cv2.putText(
                    display, "Face not detected", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            rows.append(sample)
            cv2.putText(
                display,
                f"RGB collection: {elapsed:.1f}/{DURATION_SECONDS:.1f}s",
                (15, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.imshow("Multi-ROI RGB Collection", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_reason = "user_stopped"
                break
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    actual_duration = (
        elapsed_timestamps[-1] if elapsed_timestamps else 0.0
    )
    intervals = [
        later - earlier
        for earlier, later in zip(elapsed_timestamps, elapsed_timestamps[1:])
    ]
    diagnostics = {
        "requested_duration_seconds": DURATION_SECONDS,
        "actual_duration_seconds": actual_duration,
        "duration_achieved": actual_duration >= DURATION_SECONDS * 0.99,
        "stop_reason": stop_reason,
        "frames_captured": len(rows),
        "face_detected_frames": face_frames,
        "read_failures": read_failures,
        "timestamp_discontinuities": timestamp_discontinuities,
        "median_interval_seconds": (
            float(sorted(intervals)[len(intervals) // 2])
            if intervals else None
        ),
        "max_interval_seconds": max(intervals) if intervals else None,
        "intervals_over_0_1_seconds": sum(
            interval > 0.1 for interval in intervals
        ),
        "camera_reported_fps": camera.get(cv2.CAP_PROP_FPS),
        "camera_reported_width": camera.get(cv2.CAP_PROP_FRAME_WIDTH),
        "camera_reported_height": camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "roi_valid_samples": {
            region: sum(
                all(row.get(f"{region}_{channel}", "") != "" for channel in CHANNELS)
                for row in rows
            )
            for region in ROI_RANGES
        },
    }
    if rows:
        backup = _backup_existing(OUTPUT_PATH)
        with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        diagnostics["csv_backup"] = backup
    DIAGNOSTICS_PATH.write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    collect()
