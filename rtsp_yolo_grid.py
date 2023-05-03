import cv2
from yolov8 import YOLOv8
import numpy as np
import math

model_path = "C:/Users/vijus/Documents/nivedha/Happymonk/yolov8n.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

rtsp_links = ['rtsp://happymonk:admin123@streams.ckdr.co.in:1554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif',
            'rtsp://test:test123456789@streams.ckdr.co.in:2554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif',
            'rtsp://happymonk:admin123@streams.ckdr.co.in:3554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif',
            'rtsp://happymonk:admin123@streams.ckdr.co.in:4554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif',
            'rtsp://happymonk:admin123@streams.ckdr.co.in:5554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif',
            'rtsp://happymonk:admin123@streams.ckdr.co.in:6554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif']

captures = []

frames = []
        
if len(rtsp_links)%2==0:
    num_rows = int(math.sqrt(len(rtsp_links)))
    num_cols = math.ceil(len(rtsp_links) / num_rows)

    for path in rtsp_links:
        captures.append(cv2.VideoCapture(path))

    while True:
        frames = []
        for capture in captures:
            ret, frame = capture.read()
            
            frame = cv2.resize(frame, (480,480))
            boxes, scores, class_ids = yolov8_detector(frame)
            img = yolov8_detector.draw_detections(frame)
            frames.append(img)

        grid_img = None

        for i in range(num_rows): # 2
            row = None
            for j in range(num_cols): # 3
                index = i * num_cols + j
                if index < len(frames):
                    if row is None:
                        row = frames[index]
                    else:
                        row = cv2.hconcat([row, frames[index]])
            if grid_img is None:
                grid_img = row
            else:
                grid_img = cv2.vconcat([grid_img, row])

        cv2.imshow('Grid', grid_img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
else:
    print("\n**The number of links provided should be even!!**\n")


