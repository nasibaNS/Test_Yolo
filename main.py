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
