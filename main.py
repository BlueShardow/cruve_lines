import numpy as np
import cv2 as cv
import math
import os
import time

def resize_frame(frame):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_height = 480
    new_width = int(new_height * aspect_ratio)

    return cv.resize(frame, (new_width, new_height))

def enhance_contrast(frame):
    frame = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX)

    return np.uint8(frame)

def preprocess_frame(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.bilateralFilter(frame, 15, 75, 75)

    return frame

def frame_rate(frame, fps):
    delay = 1 / fps
    time.sleep(delay)

    return frame

def main():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    fps = 15

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        #frame = frame_rate(frame, fps)
        frame = preprocess_frame(frame)
        frame = resize_frame(frame)
        frame = enhance_contrast(frame)

        cv.imshow('Camera Feed', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
