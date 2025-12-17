import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

# Load model
model = YOLO("yolov8s.pt")

# Load video
cap = cv2.VideoCapture("veh2.mp4")

# Load class names
with open("coco.txt", "r") as f:
    class_list = f.read().splitlines()

tracker = Tracker()

# Garis counting
cy1 = 323
cy2 = 367
offset = 6

vh_down = {}
vh_up = {}
counter_down = []
counter_up = []

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # YOLO inference
    results = model(frame, stream=False)
    boxes = results[0].boxes

    detections = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        class_name = class_list[cls_id]

        # Filter only CAR
        if class_name == "car":
            detections.append([x1, y1, x2, y2])

    # Tracking
    tracked = tracker.update(detections)

    for x1, y1, x2, y2, oid in tracked:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ---------- ARAH TURUN ----------
        if cy1 - offset < cy < cy1 + offset:
            vh_down[oid] = cy

        if oid in vh_down:
            if cy2 - offset < cy < cy2 + offset:
                if oid not in counter_down:
                    counter_down.append(oid)

                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(oid), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ---------- ARAH NAIK ----------
        if cy2 - offset < cy < cy2 + offset:
            vh_up[oid] = cy

        if oid in vh_up:
            if cy1 - offset < cy < cy1 + offset:
                if oid not in counter_up:
                    counter_up.append(oid)

                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(oid), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Garis hitung
    cv2.line(frame, (267, cy1), (829, cy1), (255, 255, 255), 1)
    cv2.putText(frame, "L1", (267, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.line(frame, (167, cy2), (932, cy2), (255, 255, 255), 1)
    cv2.putText(frame, "L2", (167, cy2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Counter text
    cv2.putText(frame, f"Down: {len(counter_down)}", (60, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Up   : {len(counter_up)}", (60, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Vehicle Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
