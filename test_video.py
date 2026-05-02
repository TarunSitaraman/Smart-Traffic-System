import os

import cv2

video_path = "intersection_demo.mp4"
cap = cv2.VideoCapture(video_path)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"SUCCESS: Opened {video_path}")
        print(
            f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        )
        print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Total Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    else:
        print(f"FAILURE: Opened {video_path} but could not read frame")
else:
    print(f"FAILURE: Could not open {video_path}")
cap.release()
