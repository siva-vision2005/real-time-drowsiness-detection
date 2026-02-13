import cv2
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_trained_model, predict_state, init_alarm, play_alarm, stop_alarm

# Load trained model
model = load_trained_model("models/mobilenetv2_sleep_model.h5")

# Initialize alarm
init_alarm()

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


cap = cv2.VideoCapture(0)

sleep_start_time = None
ALERT_THRESHOLD = 2  # seconds

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(100, 100)
)


    for (x, y, w, h) in faces:

        # Crop upper-middle region (both eyes area)
        eye_region = frame[
            y + int(h * 0.25) : y + int(h * 0.55),
            x + int(w * 0.15) : x + int(w * 0.85)
        ]

        state, confidence = predict_state(model, eye_region)

        color = (0, 255, 0) if state == "awake" else (0, 0, 255)

        # Draw rectangle around cropped region
        cv2.rectangle(
            frame,
            (x + int(w * 0.15), y + int(h * 0.25)),
            (x + int(w * 0.85), y + int(h * 0.55)),
            color,
            2
        )

        cv2.putText(frame,
                    f"{state} ({confidence:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

        # Sleep logic
        if state == "sleepy":
            if sleep_start_time is None:
                sleep_start_time = time.time()
            elif time.time() - sleep_start_time >= ALERT_THRESHOLD:
                play_alarm("alarm.wav")
        else:
            sleep_start_time = None
            stop_alarm()

    cv2.imshow("Sleep Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stop_alarm()
