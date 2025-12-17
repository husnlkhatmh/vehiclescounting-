import cv2
import pandas as pd
import time
from ultralytics import YOLO
from tracker import Tracker

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture("veh2.mp4")

with open("coco.txt", "r") as f:
    class_list = f.read().splitlines()

tracker = Tracker()

cy1 = 322
cy2 = 368
offset = 6

vh_down = {}
vh_up   = {}
counter_down = []
counter_up   = []

DISTANCE = 10  # meter

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype(float)

    detections = []

    for _, row in px.iterrows():
        x1=int(row[0]); y1=int(row[1])
        x2=int(row[2]); y2=int(row[3])
        cid=int(row[5])
        c=class_list[cid]
        if "car" in c:
            detections.append([x1,y1,x2,y2])

    tracked = tracker.update(detections)

    for x1,y1,x2,y2,oid in tracked:
        cx = (x1+x2)//2
        cy = (y1+y2)//2

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

        # DOWN
        if cy1 < cy + offset and cy1 > cy - offset:
            vh_down[oid] = time.time()

        if oid in vh_down:
            if cy2 < cy + offset and cy2 > cy - offset:
                if oid not in counter_down:
                    counter_down.append(oid)

                    elapsed = time.time() - vh_down[oid]
                    speed_ms = DISTANCE / elapsed
                    speed_kmh = speed_ms * 3.6

                    cv2.putText(frame, f"{int(speed_kmh)} km/h", (x2, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()