from ultralytics import YOLO
import cv2
import cvzone
from dataclasses import dataclass
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.detection.core import Detections
import numpy as np
from Detector_License import *

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

path_yolo_full = r"E:\DATN\DoneVehicle\Train\best.pt"
path_yolo_lisence = r"E:\DATN\DoneLisence\Train\best.pt"
path_yolo_helmet = r"C:\Users\doant\Downloads\train3\train3\weights\best.pt"
path_img = r"C:\Users\doant\Downloads\Untitled design (5).jpg"
path_origin = r"C:\Users\doant\Downloads\IMG_2987.mp4"
path_tracker = r"bytetrack.yaml"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Demo.mp4', fourcc, 20.0, (800, 500))
fourccc = cv2.VideoWriter_fourcc(*'mp4v')
out_2 = cv2.VideoWriter("DemoCounterByteTrack.mp4", fourccc, 20.0, (800, 500))

class_names = ["Motorcycle", "Car", "Bus", "Truck", "Bicycle"]

cap = cv2.VideoCapture(path_origin)
mask = cv2.imread(path_img)
mask = cv2.resize(mask, (800, 500))

mode_all = YOLO(path_yolo_full)
model_all_name = mode_all.names
model_lisence = YOLO(path_yolo_lisence)
model_lisence_name = model_lisence.names

print(model_lisence.names)
byte_tracker = ByteTrack()

# print("Name all: ", model_all_name)
#

count_all = []
count_motor = []
count_car = []
count_bus = []
count_truck = []

count_num_all = 0
count_motor_num = 0
count_car_num = 0
count_bus_num = 0
count_truck_num = 0

count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800,500))
    frame = cv2.bitwise_and(mask, frame)
    result_full = mode_all.track(frame, tracker=path_tracker, persist=True)
    result_lisence = model_lisence.track(frame, tracker=path_tracker)

    cv2.putText(frame, f"Objects: {count_num_all}", (300, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"Motors: {count_motor_num}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"Cars: {count_car_num}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    # cv2.putText(frame, f"Bus: {count_bus_num}", (700, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    # cv2.putText(frame, f"Trucks: {count_truck_num}", (700, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    for r_full in result_full:
        boxes = r_full.boxes
        conf = boxes.conf
        xywh = boxes.xywh
        xyxy = boxes.xyxy
        for box_full in r_full.boxes:
            x1_full, y1_full, x2_full, y2_full = box_full.xyxy[0]
            x1_full, y1_full, x2_full, y2_full = int(x1_full), int(y1_full), int(x2_full), int(y2_full)
            w_full, h_full = x2_full - x1_full, y2_full - y1_full
            cvzone.cornerRect(frame, (x1_full, y1_full, w_full, h_full), l=10, rt=3)
            confidence = box_full.conf[0]
            name_objects = model_all_name[int(box_full.cls)]
            if name_objects == "Car":
                img = frame[y1_full:y1_full+w_full, x1_full:x1_full+h_full]
                cv2.imwrite(f"Car_Violations/{count}.jpg", img)
                count += 1

    detections = Detections.from_ultralytics(r_full)
    tracks = byte_tracker.update_with_detections(detections=detections)
    for tracking in tracks.class_id:
        for track_id in tracks.tracker_id:
            print(tracking)
            if name_objects == "Motorcycle":
                if tracking == 0:
                    if track_id not in count_all:
                        if confidence > 0.5:
                            count_all.append(track_id)

            if name_objects == "Car":
                if tracking == 1:
                    if track_id not in count_all:
                        if confidence > 0.8:
                            count_all.append(track_id)

            if name_objects == "Bus":
                if tracking == 2:
                    if track_id not in count_all:
                        if confidence > 0.75:
                            count_all.append(track_id)

            if name_objects == "Truck":
                if tracking == 3:
                    if track_id not in count_all:
                        if confidence > 0.7:
                            count_all.append(track_id)

            if name_objects == "Motorcycle":
                if tracking == 0:
                    if track_id not in count_car and track_id not in count_motor and track_id not in count_bus and track_id not in count_truck:
                        if confidence > 0.5:
                            count_motor.append(track_id)
            if name_objects == "Car" or name_objects == "Truck" or name_objects == "Bus":
                if tracking == 1:
                    if track_id not in count_car and track_id not in count_motor and track_id not in count_bus and track_id not in count_truck:
                        if confidence > 0.6:
                            count_car.append(track_id)
                if tracking == 2:
                    if track_id not in count_car and track_id not in count_motor and track_id not in count_bus and track_id not in count_truck:
                        if confidence > 0.8:
                            count_car.append(track_id)
                if tracking == 3:
                    if track_id not in count_car and track_id not in count_motor and track_id not in count_bus and track_id not in count_truck:
                        if confidence > 0.7:
                            count_car.append(track_id)

            cv2.putText(frame, f"Objects: {count_num_all}", (300, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, f"Motors: {count_motor_num}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, f"Cars: {count_car_num}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            # cv2.putText(frame, f"Bus: {count_bus_num}", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            # cv2.putText(frame, f"Trucks: {count_truck_num}", (350, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    for r_lisence in result_lisence:
        for box_lisence in r_lisence.boxes:
            xywh_lisence = box_lisence.xywh[0]
            confg_lisence = box_lisence.conf[0]
            x1_lisence, y1_lisence, x2_lisence, y2_lisence = box_lisence.xyxy[0]
            x1_lisence, y1_lisence, x2_lisence, y2_lisence = int(x1_lisence), int(y1_lisence), int(x2_lisence), int(y2_lisence)
            w_lisence, h_lisence = x2_lisence - x1_lisence, y2_lisence - y1_lisence
            # cvzone.cornerRect(frame, (x1_lisence, y1_lisence, w_lisence, h_lisence), l=2, rt=1, colorR=(255,255,0), colorC=(0,0,255))
            name_lisence = model_lisence_name[int(box_lisence.cls)]

            if name_lisence == "Lisence" and confg_lisence > 0:
                roi = frame[y1_lisence:y1_lisence+h_lisence, x1_lisence:x1_lisence+w_lisence]
                blur = cv2.GaussianBlur(roi, (7,7), 5)
                frame[y1_lisence:y1_lisence + h_lisence, x1_lisence:x1_lisence + w_lisence] = blur

    count_num_all = len(count_all)
    count_motor_num = len(count_motor)
    count_car_num = len(count_car)
    count_bus_num = len(count_bus)
    count_truck_num = len(count_truck)

    print("Count All: ", count_all)
    print("Count Motor: ", count_motor)
    print("Count Car: ", count_car)
    print("Count Bus: ", count_bus)
    print("Count Truck: ", count_truck)

    anotated_frame = result_full[0].plot()
    cv2.imshow("anotated_frame", anotated_frame)

    out.write(frame)
    out_2.write(anotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()