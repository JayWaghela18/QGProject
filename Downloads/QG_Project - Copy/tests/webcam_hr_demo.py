"""Beginner-friendly rolling-window webcam rPPG demo.

This is an experimental signal-processing demo, not a medical measurement.
It uses the forehead GREEN channel because it is the simplest classical
baseline already used by this project.
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from scipy.signal import butter, detrend, filtfilt, welch

sys.path.insert(0, str(Path(__file__).parent))
from test_multi_roi_rgb import extract_rois, get_face_box


MODEL_PATH = Path("models/face_landmarker.task")
CAMERA_INDEX = 0
WINDOW_SECONDS = 10.0
LOW_HZ = 0.7
HIGH_HZ = 3.0
MAX_GAP_SECONDS = 0.25
MIN_QUALITY_RATIO = 3.0
MIN_HR = 40.0
MAX_HR = 180.0


def estimate_rolling_hr(timestamps, green_values):
    """Return (bpm, quality_ratio, status) for one rolling signal window."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    green_values = np.asarray(green_values, dtype=np.float64)
    if len(timestamps) < 30:
        return None, None, "Measuring..."
    if (
        len(timestamps) != len(green_values)
        or not np.all(np.isfinite(timestamps))
        or not np.all(np.isfinite(green_values))
        or np.any(np.diff(timestamps) <= 0)
    ):
        return None, None, "Low signal quality"

    gaps = np.diff(timestamps)
    if len(gaps) == 0 or np.max(gaps) > MAX_GAP_SECONDS:
        return None, None, "Low signal quality"

    fps = 1.0 / float(np.median(gaps))
    if not np.isfinite(fps) or fps <= 2 * HIGH_HZ:
        return None, None, "Low signal quality"

    uniform_times = np.arange(
        timestamps[0],
        timestamps[-1] + 0.5 / fps,
        1.0 / fps,
    )
    signal = np.interp(uniform_times, timestamps, green_values)
    if len(signal) < 30:
        return None, None, "Measuring..."

    try:
        signal = detrend(signal, type="linear")
        b, a = butter(
            3,
            [LOW_HZ / (fps / 2.0), HIGH_HZ / (fps / 2.0)],
            btype="band",
        )
        pad_length = 3 * max(len(a), len(b))
        if len(signal) <= pad_length:
            return None, None, "Measuring..."
        filtered = filtfilt(b, a, signal)
        frequencies, power = welch(
            filtered,
            fs=fps,
            nperseg=min(len(filtered), max(64, int(fps * 8))),
            nfft=4096,
        )
    except (ValueError, FloatingPointError):
        return None, None, "Low signal quality"

    valid = (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ)
    if not np.any(valid):
        return None, None, "Low signal quality"

    band_power = power[valid]
    band_frequencies = frequencies[valid]
    peak_index = int(np.argmax(band_power))
    median_power = float(np.median(band_power))
    if median_power <= 0 or not np.isfinite(median_power):
        return None, None, "Low signal quality"

    quality_ratio = float(band_power[peak_index] / median_power)
    bpm = float(band_frequencies[peak_index] * 60.0)
    if (
        not np.isfinite(bpm)
        or bpm < MIN_HR
        or bpm > MAX_HR
        or quality_ratio < MIN_QUALITY_RATIO
    ):
        return None, quality_ratio, "Low signal quality"
    return bpm, quality_ratio, "Experimental rPPG estimate"


def run_demo():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"MediaPipe model not found: {MODEL_PATH}")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        detector.close()
        raise RuntimeError("Could not open webcam.")

    timestamps = deque()
    green_values = deque()
    start_time = time.monotonic()
    previous_video_timestamp = -1

    try:
        print("Rolling webcam rPPG demo started. Press Q to quit.")
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read webcam frame.")
                break

            elapsed = time.monotonic() - start_time
            video_timestamp = max(
                previous_video_timestamp + 1,
                int(round(elapsed * 1000)),
            )
            previous_video_timestamp = video_timestamp

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            result = detector.detect_for_video(image, video_timestamp)
            face_box = (
                get_face_box(result.face_landmarks[0], frame.shape)
                if result.face_landmarks
                else None
            )
            roi_values, roi_boxes = extract_rois(rgb_frame, face_box)
            green_text = "Measuring..."
            if face_box is not None:
                for x1, y1, x2, y2 in roi_boxes.values():
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                forehead_green = roi_values.get("forehead_G", "")
                if forehead_green != "":
                    timestamps.append(elapsed)
                    green_values.append(float(forehead_green))
                    green_text = "Face/ROI valid"
            else:
                green_text = "Face not detected"

            while timestamps and elapsed - timestamps[0] > WINDOW_SECONDS:
                timestamps.popleft()
                green_values.popleft()

            bpm, quality_ratio, quality_status = estimate_rolling_hr(
                list(timestamps),
                list(green_values),
            )
            if bpm is None:
                bpm_text = "-- BPM"
            else:
                bpm_text = f"{bpm:.1f} BPM"

            cv2.putText(
                frame,
                f"Heart rate: {bpm_text}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if bpm is not None else (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                quality_status,
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if bpm is not None else (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"{green_text} | samples: {len(green_values)}",
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            if quality_ratio is not None:
                cv2.putText(
                    frame,
                    f"Relative peak/median power: {quality_ratio:.1f}",
                    (15, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            cv2.imshow("Experimental rPPG Heart Rate", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
