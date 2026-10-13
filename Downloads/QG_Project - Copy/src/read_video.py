import cv2
import time

CAMERA_INDEX = 0
DURATION = 30
OUTPUT_FILE = "camera_test.avi"

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

fps = 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(
    OUTPUT_FILE,
    fourcc,
    fps,
    (width, height)
)

print("Camera started.")
print("Press Q to stop early.")

start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    elapsed = time.time() - start_time
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

    cv2.imshow("rPPG Camera Test", frame)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if elapsed >= DURATION:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("\nRecording complete.")
print(f"Saved as: {OUTPUT_FILE}")