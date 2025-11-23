from ultralytics import YOLO
import cv2
import time, datetime
import os

model = YOLO("yolov8n.pt")

video_path = 'videos'
os.makedirs(video_path, exist_ok=True)

img_path = 'media'
os.makedirs(img_path, exist_ok=True)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", 640, 360)
