import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_DIR / "data" / "processed" / "dataset_manifest.json"
OUTPUT_DIR = PROJECT_DIR / "data" / "raw"
LANDMARKER_PATH = Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"

ROI_NAMES = ("forehead", "left_cheek", "right_cheek", "chin")

# MediaPipe Face Mesh landmark groups.
# Each group is averaged to obtain one RGB signal per frame.
ROI_LANDMARKS = {
    "forehead": [10, 67, 69, 104, 108, 109, 151, 337, 338, 297, 299],
    "left_cheek": [50, 101, 118, 119, 120, 123, 147, 187, 205, 206, 207],
    "right_cheek": [280, 330, 347, 348, 349, 352, 376, 411, 425, 426, 427],
    "chin": [152, 175, 176, 199, 200, 396, 419],
}


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"MediaPipe task model not found: {LANDMARKER_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = json.loads(MANIFEST.read_text(encoding="utf-8"))

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        for item in subjects:
            subject = item["subject"]
            dataset = item["dataset"]
            output_file = OUTPUT_DIR / f"{dataset}_{subject}_rgb.csv"

            if output_file.exists():
                print(f"Skipping existing: {output_file.name}")
                continue

            cap = cv2.VideoCapture(item["video_path"])
            if not cap.isOpened():
                print(f"WARNING: Cannot open video for {dataset}/{subject}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            rows = []
            frame_index = 0
            detected_count = 0

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    timestamp_seconds = frame_index / fps if fps > 0 else frame_index / 30.0
                    timestamp_ms = max(frame_index, round(timestamp_seconds * 1000))

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    row = {
                        "frame": frame_index,
                        "time_seconds": timestamp_seconds,
                        "face_detected": 0,
                    }

                    for roi in ROI_NAMES:
                        for channel in "RGB":
                            row[f"{roi}_{channel}"] = ""

                    if result.face_landmarks:
                        landmarks = result.face_landmarks[0]
                        height, width = frame.shape[:2]

                        for roi, indices in ROI_LANDMARKS.items():
                            points = np.array(
                                [
                                    [
                                        landmarks[i].x * width,
                                        landmarks[i].y * height,
                                    ]
                                    for i in indices
                                ],
                                dtype=np.int32,
                            )

                            # Small patch around each landmark, clipped to the frame.
                            values = []
                            for x, y in points:
                                x0, x1 = max(0, x - 3), min(width, x + 4)
                                y0, y1 = max(0, y - 3), min(height, y + 4)
                                if x1 > x0 and y1 > y0:
                                    patch = frame[y0:y1, x0:x1]
                                    values.append(patch.reshape(-1, 3).mean(axis=0))

                            if values:
                                b, g, r = np.mean(values, axis=0)
                                row[f"{roi}_R"] = float(r)
                                row[f"{roi}_G"] = float(g)
                                row[f"{roi}_B"] = float(b)

                        row["face_detected"] = 1
                        detected_count += 1

                    rows.append(row)
                    frame_index += 1

            finally:
                cap.release()

            if not rows:
                print(f"WARNING: No frames read for {dataset}/{subject}")
                continue

            fieldnames = list(rows[0].keys())
            with output_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(
                f"{dataset}/{subject}: {len(rows)} frames, "
                f"{detected_count} face detections -> {output_file.name}"
            )


if __name__ == "__main__":
    main()