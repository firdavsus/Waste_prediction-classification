# detect_webcam.py
import cv2
from ultralytics import YOLO

MODEL_PATH = r"weights/best.pt"  # change to your .pt
CONF = 0.5

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF)
        # results[0].plot() returns a numpy BGR image with boxes drawn
        out = results[0].plot()
        cv2.imshow("Webcam YOLOv8", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
