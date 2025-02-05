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

def line_length(line):
    x1, y1, x2, y2 = line
    return np.hypot((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(line):
    """
    Calculates the angle of a line in degrees.
    """
    x1, y1, x2, y2 = line
    angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
    while angle < 0:
        angle += 180
    while angle > 180:
        angle -= 180
    return angle

def merge_lines(lines, min_distance = 250, merge_angle_tolerance = 65, vertical_leeway=1.5, horizontal_leeway=1.5):
    def weighted_average(p1, w1, p2, w2):
        """Computes a weighted average of two points."""
        return (p1 * w1 + p2 * w2) / (w1 + w2)

    def merge_once(lines):
        merged_lines = []
        used = [False] * len(lines)

        for i, line1 in enumerate(lines):
            if used[i]:
                continue

            x1, y1, x2, y2 = line1
            angle1 = calculate_angle(line1)
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
            line_weight = line_length(line1)

            for j, line2 in enumerate(lines):
                if i != j and not used[j]:
                    x3, y3, x4, y4 = line2
                    angle2 = calculate_angle(line2)

                    # Check parallelism
                    if is_parallel(line1, line2, merge_angle_tolerance, merge_angle_tolerance, min_distance):
                        is_horizontal1 = abs(y2 - y1) < abs(x2 - x1)
                        is_horizontal2 = abs(y4 - y3) < abs(x4 - x3)

                        # Apply separate logic for merging based on orientation
                        if is_horizontal1 and is_horizontal2:
                            # Horizontal lines: More leeway in horizontal, less in vertical
                            vertical_distance = abs((y1 + y2) / 2 - (y3 + y4) / 2)
                            horizontal_distance = abs((x1 + x2) / 2 - (x3 + x4) / 2)
                            if vertical_distance > min_distance * horizontal_leeway or horizontal_distance > min_distance:
                                continue
                        elif not is_horizontal1 and not is_horizontal2:
                            # Vertical lines: More leeway in vertical, less in horizontal
                            vertical_distance = abs((y1 + y2) / 2 - (y3 + y4) / 2)
                            horizontal_distance = abs((x1 + x2) / 2 - (x3 + x4) / 2)
                            if vertical_distance > min_distance or horizontal_distance > min_distance * vertical_leeway:
                                continue

                        # Merge lines using weighted averages
                        l2_len = line_length(line2)
                        new_x1 = weighted_average(new_x1, line_weight, x3, l2_len)
                        new_y1 = weighted_average(new_y1, line_weight, y3, l2_len)
                        new_x2 = weighted_average(new_x2, line_weight, x4, l2_len)
                        new_y2 = weighted_average(new_y2, line_weight, y4, l2_len)
                        line_weight += l2_len
                        used[j] = True

            merged_lines.append((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
            used[i] = True

        return merged_lines

    # Perform iterative merging until lines stabilize
    prev_lines = []
    while prev_lines != lines:
        prev_lines = lines
        lines = merge_once(lines)

    return lines

def detect_curved_lines(frame, contour_approx_tolerance = .005):
    edges = cv.Canny(frame, 50, 150, apertureSize=3)
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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

def display_fps(frame, start_time):
    fps = int(1 / (time.time() - start_time))
    cv.putText(frame, f'FPS: {fps}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    return frame

def main():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        start_time = time.time()
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
            merged_lines = merge_lines(lines)

            """
            if len(unmerged_lines) > 0:
                for unmerged_line in unmerged_lines:
                    x1, y1, x2, y2 = unmerged_line[0]
                    cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            """

            if len(merged_lines) > 0:
                for merged_line in merged_lines:
                    x1, y1, x2, y2 = merged_line[0]
                    x3, y3, x4, y4 = merged_line[-1]
                    cv.line(frame, (x1, y1), (x4, y4), (0, 255, 0), 2)

        print("Merged Lines:", len(merged_lines))
        
        #"\nUnmerged Lines:", len(unmerged_lines))

        cv.imshow('Merged Lines', frame)

        contour_frame = frame.copy()
        contours = detect_curved_lines(contrast_frame)

        for contour in contours:
            cv.drawContours(contour_frame, [contour], -1, (0, 0, 255), 2)
            
        contour_frame = display_fps(contour_frame, start_time)
        cv.imshow('Curved Line Detection', contour_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
