import os
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def main():
    model_path = os.path.abspath("models/face_landmarker.task")

    if not os.path.isfile(model_path):
        print("ERROR: Face Landmarker model asset not found.")
        return

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )

    detector = vision.FaceLandmarker.create_from_options(options)
    camera = None

    try:
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("ERROR: Could not open webcam.")
            return

        print("Look at the camera and keep your face visible.")
        print("Capturing a test frame...")

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not capture a frame.")
            return

        # MediaPipe expects an RGB image.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = detector.detect(mp_image)

        print("Frame shape:", frame.shape)
        print("Faces detected:", len(result.face_landmarks))

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            print("Landmarks detected:", len(landmarks))
            print("MEDIAPIPE FACE TEST PASSED")
        else:
            print("No face detected. Try better lighting and face the camera.")

    finally:
        if camera is not None:
            camera.release()

        detector.close()


if __name__ == "__main__":
    main()