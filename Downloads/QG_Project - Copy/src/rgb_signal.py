import cv2
import numpy as np
import time

DURATION = 30
CAMERA_INDEX = 0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Starting 30-second rPPG capture...")
print("Keep your face still and look at the camera.")

rgb_signal = []
timestamps = []

start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    elapsed = time.time() - start_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) > 0:

        # Select the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Upper-face ROI
        roi_x = x + int(0.25 * w)
        roi_y = y + int(0.15 * h)
        roi_w = int(0.50 * w)
        roi_h = int(0.25 * h)

        roi = frame[
            roi_y:roi_y + roi_h,
            roi_x:roi_x + roi_w
        ]

        if roi.size > 0:

            # OpenCV uses BGR
            mean_bgr = np.mean(roi, axis=(0, 1))

            blue = mean_bgr[0]
            green = mean_bgr[1]
            red = mean_bgr[2]

            rgb_signal.append([red, green, blue])
            timestamps.append(elapsed)

            cv2.rectangle(
                frame,
                (roi_x, roi_y),
                (roi_x + roi_w, roi_y + roi_h),
                (0, 255, 0),
                2
            )

    remaining = max(0, DURATION - elapsed)

    cv2.putText(
        frame,
        f"Recording: {remaining:.1f}s",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("rPPG RGB Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if elapsed >= DURATION:
        break

cap.release()
cv2.destroyAllWindows()

rgb_signal = np.array(rgb_signal)
timestamps = np.array(timestamps)

print("\n========== CAPTURE COMPLETE ==========")
print("Samples:", len(rgb_signal))

if len(timestamps) > 1:
    duration = timestamps[-1] - timestamps[0]
    fps = len(rgb_signal) / duration

    print(f"Duration: {duration:.2f} seconds")
    print(f"Effective FPS: {fps:.2f}")

print("RGB signal shape:", rgb_signal.shape)

np.save("rgb_signal.npy", rgb_signal)
np.save("timestamps.npy", timestamps)

print("Saved:")
print("  rgb_signal.npy")
print("  timestamps.npy")