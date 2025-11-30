from ultralytics import YOLO
import cv2
import time, datetime
import os

#1
model = YOLO("yolov8n.pt")

video_path = 'videos'
os.makedirs(video_path, exist_ok=True)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", 640, 360)

#2
if not cap.isOpened():
    print('Camera not found')
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = float(cap.get(cv2.CAP_PROP_FPS))

if frame_fps == 0:
    frame_fps = 30.0

date = datetime.datetime.now().strftime('%d%m%Y_%H%M%S')


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_name = f'{video_path}/video_{date}.mp4'
out = cv2.VideoWriter(video_name, fourcc, frame_fps, (frame_width, frame_height))


classes = ['person']

while True:
    fps_start = time.time()
    fps_start = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Frame not found")
        break

    result = model(frame, stream=True, conf=0.5)

    for i in result:
        for n in i.boxes:
            cls = int(n.cls[0])
            label = model.names[cls]
            conf = float(n.conf[0])

            if label not  in classes:
                continue



restricted_area = (100, 100, 500, 400)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not found")
        break

    result = model(frame, stream=True, conf=0.5)
    count_people_in_area = 0  # Счётчик людей в зоне

    for i in result:
        for n in i.boxes:
            cls = int(n.cls[0])
            label = model.names[cls]
            conf = float(n.conf[0])

            if label != 'person':
                continue


            x1, y1, x2, y2 = map(int, n.xyxy[0])


            rx1, ry1, rx2, ry2 = restricted_area
            if x2 > rx1 and x1 < rx2 and y2 > ry1 and y1 < ry2:
                count_people_in_area += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
    cv2.putText(frame, f'People in area: {count_people_in_area}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
