import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        # Define an upper-face skin ROI
        roi_x = x + int(0.25 * w)
        roi_y = y + int(0.15 * h)
        roi_w = int(0.50 * w)
        roi_h = int(0.25 * h)

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Draw rPPG ROI
        cv2.rectangle(
            frame,
            (roi_x, roi_y),
            (roi_x + roi_w, roi_y + roi_h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "rPPG ROI",
            (roi_x, roi_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("rPPG ROI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()