from ultralytics import YOLO
import cv2
import time, datetime
import os

model = YOLO("yolov8n.pt")

video_path = 'videos'
os.makedirs(video_path, exist_ok=True)

cap = cv2.VideoCapture('test.mp4')

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", 640, 360)

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


restricted_area = (800, 320, 1400, 650)


total_visitors = 0
visited_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not found")
        break


    results = model.track(frame, stream=True, conf=0.5, persist=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label != 'person':
                continue

            x, y, h, w = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else None

            rx1, ry1, rx2, ry2 = restricted_area


            if h > rx1 and x < rx2 and w > ry1 and y < ry2:
                cv2.rectangle(frame, (x, y), (h, w), (0, 0, 255), 2)


                if track_id is not None and track_id not in visited_ids:
                    visited_ids.add(track_id)
                    total_visitors += 1

    rx1, ry1, rx2, ry2 = restricted_area
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)


    cv2.putText(frame, f'Total visitors: {total_visitors}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()