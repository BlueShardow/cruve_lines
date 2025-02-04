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

def recursive_merge_lines(lines, same_merge_angle_tolerance = 35, away_merge_angle_tolerance = 25, distance_threshold = 50, max_line_gap = 25):
    merged_lines = []
    unmerged_lines = []
    lines = [line.tolist() for line in lines]

    while len(lines) > 0:
        line = lines.pop(0)
        x1, y1, x2, y2 = line[0]
        merged_line = [line]

        for i in range(len(lines)):
            same_direction = False
            tolerance = away_merge_angle_tolerance
            next_line = lines[i]
            x3, y3, x4, y4 = next_line[0]

            dir1 = (x2 - x1, y2 - y1)
            dir2 = (x4 - x3, y4 - y3)

            angle1 = math.atan2(dir1[1], dir1[0])
            angle2 = math.atan2(dir2[1], dir2[0])

            #print(math.degrees(angle1), math.degrees(angle2))

            if (x1 or x2 < x3 or x4) and angle1 < 0 and angle2 < 0: # if x1 is on left and same direction
                same_direction = True
            
            elif (x1 or x2 > x3 or x4) and angle1 > 0 and angle2 > 0: # if x1 is on right and same direction
                same_direction = True

            if same_direction:
                tolerance = same_merge_angle_tolerance
            
            else:
                tolerance = away_merge_angle_tolerance
                
            angle_diff = abs(math.degrees(angle1 - angle2))

            if angle_diff > 90:
                angle_diff = 180 - angle_diff

                if angle_diff < tolerance:
                    distance_from_start_to_start = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
                    distance_from_end_to_end = math.sqrt((x4 - x2) ** 2 + (y4 - y2) ** 2)
                    gap = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2) # distance between end of first line and start of next line

                    if (distance_from_start_to_start < distance_threshold) or (distance_from_end_to_end < distance_threshold) or (gap < max_line_gap):
                        merged_line.append(next_line)

        if len(merged_line) > 1:
            merged_lines.append(merged_line)
            
        else:
            unmerged_lines.append(line)

    return merged_lines, unmerged_lines

def detect_curved_lines(frame, distance_threshold = 50, contour_approx_tolerance = 75):
    if len(frame.shape) == 3:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(frame, 50, 150, apertureSize = 3)
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    approx_contours = []

    for contour in contours:
        epsilon = contour_approx_tolerance * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        approx_contours.append(approx)

    return approx_contours

def is_parallel(line1, line2, tolerance=5):
    pass

    """if (x1 or x2 < x3 or x4) and angle1 < 0 and angle2 > 0: # if x1 is on left and towards
                towards = True
            
            elif (x1 or x2 > x3 or x4) and angle1 > 0 and angle2 < 0: # if x1 is on right and towards
                towards = True

            if towards:"""

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

        frame = cv.imread("/Users/pl1001515/Downloads/cruved.jpeg") 

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

        #print("Merged Lines:", len(merged_lines), "\nUnmerged Lines:", len(unmerged_lines)) # not working i think

        cv.imshow('Merged Lines', frame)

        contours = detect_curved_lines(contrast_frame)

        for contour in contours:
            for i in range(len(contour)):
                start_point = tuple(contour[i][0])
                end_point = tuple(contour[(i + 1) % len(contour)][0])
                cv.line(frame, start_point, end_point, (0, 255, 0), 2)

        cv.drawContours(frame, contours, -1, (0, 0, 255), 2)
        cv.imshow('Curved Line Detection', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
