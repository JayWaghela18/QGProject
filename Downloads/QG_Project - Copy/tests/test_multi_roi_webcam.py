
import cv2
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Project file locations
MODEL_PATH = os.path.abspath("models/face_landmarker.task")

# Facial regions relative to the detected face box:
# (left, top, right, bottom)
ROI_RANGES = {
    "Forehead": (0.20, 0.08, 0.80, 0.28),
    "Left cheek": (0.12, 0.38, 0.43, 0.62),
    "Right cheek": (0.57, 0.38, 0.88, 0.62),
    "Chin": (0.30, 0.68, 0.70, 0.90),
}

# OpenCV uses BGR colors
ROI_COLORS = {
    "Forehead": (0, 255, 0),       # Green
    "Left cheek": (255, 160, 0),   # Orange/blue shade
    "Right cheek": (255, 0, 255),  # Pink
    "Chin": (0, 255, 255),         # Yellow
}


def get_face_box(landmarks, frame_shape):
    """Calculate a bounding box around the detected facial landmarks."""

    height, width = frame_shape[:2]

    points = [
        (int(point.x * width), int(point.y * height))
        for point in landmarks
        if 0 <= point.x <= 1 and 0 <= point.y <= 1
    ]

    if len(points) < 3:
        return None

    x1 = max(0, min(point[0] for point in points))
    y1 = max(0, min(point[1] for point in points))
    x2 = min(width - 1, max(point[0] for point in points))
    y2 = min(height - 1, max(point[1] for point in points))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2 - x1, y2 - y1


def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Face Landmarker model not found: {MODEL_PATH}")
        return

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    detector = vision.FaceLandmarker.create_from_options(options)
    camera = None

    try:
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("ERROR: Could not open webcam.")
            return

        start_time = time.monotonic()

        print("Four-ROI webcam preview started.")
        print("Look straight at the camera and keep your face visible.")
        print("Press Q in the webcam window to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read a webcam frame.")
                break

            # MediaPipe VIDEO mode requires increasing timestamps.
            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            result = detector.detect_for_video(
                image, timestamp_ms
            )

            if result.face_landmarks:
                face_box = get_face_box(
                    result.face_landmarks[0],
                    frame.shape,
                )

                if face_box is not None:
                    x, y, w, h = face_box

                    # White outline around the detected face
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (255, 255, 255),
                        1,
                    )

                    # Draw each proposed facial ROI
                    for name, (left, top, right, bottom) in ROI_RANGES.items():
                        x1 = max(
                            0, min(frame.shape[1] - 1,
                                   int(x + left * w))
                        )
                        y1 = max(
                            0, min(frame.shape[0] - 1,
                                   int(y + top * h))
                        )
                        x2 = max(
                            0, min(frame.shape[1] - 1,
                                   int(x + right * w))
                        )
                        y2 = max(
                            0, min(frame.shape[0] - 1,
                                   int(y + bottom * h))
                        )

                        if x2 - x1 < 20 or y2 - y1 < 10:
                            continue

                        color = ROI_COLORS[name]

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2,
                        )

                        cv2.putText(
                            frame,
                            name,
                            (x1, max(20, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )

                    cv2.putText(
                        frame,
                        "ROI PREVIEW - NOT MEASURING HR",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

            else:
                cv2.putText(
                    frame,
                    "Face not detected - face the camera",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Four Facial ROIs", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if camera is not None:
            camera.release()

        detector.close()
        cv2.destroyAllWindows()
        print("Four-ROI preview closed.")


if __name__ == "__main__":
    main()
