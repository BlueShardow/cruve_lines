import numpy as np
import cv2 as cv
import math
import os
import time
import sys

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

"""(frame, max_line_gap=85, toward_tolerance=75, away_tolerance=75, merge_angle_tolerance=65,
                              distance_threshold=999999999, min_distance=250, min_line_length=150,
                              min_overlap_ratio=0.8,
                              proximity_threshold=20):"""

def detect_lines(frame):
    edges = cv.Canny(frame, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, rho = 1, theta = np.pi / 180, threshold = 100, minLineLength = 50, maxLineGap = 85)

    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, lines if lines is not None else []

def recursive_merge_lines(lines, merge_angle_tolerance=35, distance_threshold=50, min_overlap_ratio=0.5):
    merged_lines = []
    unmerged_lines = []
    lines = [line.tolist() for line in lines]

    while len(lines) > 0:
        line = lines.pop(0)
        x1, y1, x2, y2 = line[0]

        merged_line = [line]

        for i in range(len(lines)):
            next_line = lines[i]
            x3, y3, x4, y4 = next_line[0]

            angle = math.atan2(y2 - y1, x2 - x1) - math.atan2(y4 - y3, x4 - x3)
            angle = math.degrees(angle)
            angle = abs(angle)

            if angle < merge_angle_tolerance:
                distance = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

                if distance < distance_threshold:
                    distance = math.sqrt((x4 - x2) ** 2 + (y4 - y2) ** 2)

                    if distance < distance_threshold:
                        overlap = 0

                        if x1 == x2:
                            overlap = (min(y1, y2) - max(y3, y4)) / (min(y1, y2) - max(y1, y2))
                        else:
                            overlap = (min(x1, x2) - max(x3, x4)) / (min(x1, x2) - max(x1, x2))

                        if overlap > min_overlap_ratio:
                            merged_line.append(next_line)

        if len(merged_line) > 1:
            merged_lines.append(merged_line)

        else:
            unmerged_lines.append(line)

    return merged_lines, unmerged_lines

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

        frame = cv.imread(r"c:\Users\Owner\Pictures\Screenshots\Screenshot 2025-02-02 202610.png")

        #frame = frame_rate(frame, fps)
        pre_frame = preprocess_frame(frame)
        resized_frame = resize_frame(pre_frame)
        contrast_frame = enhance_contrast(resized_frame)

        cv.imshow('Camera Feed', contrast_frame)

        line_frame, lines = detect_lines(contrast_frame)
        
        cv.imshow('Line Detection', line_frame)

        frame = resize_frame(frame)

        if len(lines) > 0:
            merged_lines, unmerged_lines = recursive_merge_lines(lines)

            if len(unmerged_lines) > 0:
                for unmerged_lines in unmerged_lines:
                    x1, y1, x2, y2 = unmerged_lines[0]
                    cv.line(frame, (x1, y1), (x2, y2), (0 , 255, 0), 2)

            if len(merged_lines) > 0:
                for merged_line in merged_lines:
                    x1, y1, x2, y2 = merged_line[0][0]
                    x3, y3, x4, y4 = merged_line[-1][0]
                    cv.line(frame, (x1, y1), (x4, y4), (0, 255, 0), 2)

        cv.imshow('Merged Lines', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
