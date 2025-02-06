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
    frame = cv.bilateralFilter(frame, 9, 75, 75)

    return frame

def detect_lines(frame):
    edges = cv.Canny(frame, 50, 150, apertureSize = 3)
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
    return np.hypot((x2 - x1), (y2 - y1))

def calculate_angle(line):
    x1, y1, x2, y2 = line
    angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi

    while angle < 0:
        angle += 180

    while angle > 180:
        angle -= 180

    return angle

def is_parallel(line1, line2, toward_tolerance, away_tolerance, distance_threshold):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    angle1 = calculate_angle(line1)
    angle2 = calculate_angle(line2)

    angle_diff = abs(angle1 - angle2)
    if angle_diff > toward_tolerance and (180 - angle_diff) > away_tolerance:
        return False

    # Horizontal or vertical check
    is_horizontal1 = abs(y2 - y1) < abs(x2 - x1)
    is_horizontal2 = abs(y4 - y3) < abs(x4 - x3)

    # Check alignment and proximity
    if is_horizontal1 and is_horizontal2:
        vertical_distance = abs((y1 + y2) / 2 - (y3 + y4) / 2)

        return vertical_distance < distance_threshold

    elif not is_horizontal1 and not is_horizontal2:
        horizontal_distance = abs((x1 + x2) / 2 - (x3 + x4) / 2)

        return horizontal_distance < distance_threshold

    return False

def merge_lines(lines, min_distance = 50, merge_angle_tolerance = 20, vertical_leeway = 1.5, horizontal_leeway = 1.5):
    def weighted_average(p1, w1, p2, w2):
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

                    if is_parallel(line1, line2, merge_angle_tolerance, merge_angle_tolerance, min_distance):
                        is_horizontal1 = abs(y2 - y1) < abs(x2 - x1)
                        is_horizontal2 = abs(y4 - y3) < abs(x4 - x3)

                        if is_horizontal1 and is_horizontal2:
                            vertical_distance = abs((y1 + y2) / 2 - (y3 + y4) / 2)
                            horizontal_distance = abs((x1 + x2) / 2 - (x3 + x4) / 2)

                            if vertical_distance > min_distance * horizontal_leeway or horizontal_distance > min_distance:
                                continue

                        elif not is_horizontal1 and not is_horizontal2:
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

    # Convert to a flattened list of (x1, y1, x2, y2)
    lines = [line[0] for line in lines]

    # Perform iterative merging until lines stabilize
    prev_lines = []

    while prev_lines != lines:
        prev_lines = lines.copy()
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

def rate_of_change(contour):
    contour_points = contour[:, 0, :]
    differences = np.diff(contour_points, axis = 0)
    rate_of_change = np.linalg.norm(differences, axis = 1)

    return rate_of_change

def calculate_distances(contour1, contour2): # WORK ON ADDING THIS IN _______________________________________________________________________________________
    distances = []
    for point1 in contour1:
        for point2 in contour2:
            x1, y1 = point1[0]
            x2, y2 = point2[0]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append(distance)
    return distances

"""
# Example usage
contour1 = np.array([[[0, 0]], [[1, 1]], [[2, 2]]])
contour2 = np.array([[[3, 3]], [[4, 4]], [[5, 5]]])

distances = calculate_distances(contour1, contour2)
print("Distances:", distances)
"""

def merge_curved_lines(contours, min_distance = 75, merge_angle_tolerance = 50, difference_tolerance = 50):
    contours = [np.array(contour) if isinstance(contour, tuple) else contour for contour in contours]

    def weighted_average(p1, w1, p2, w2):
        return (p1 * w1 + p2 * w2) / (w1 + w2)
    
    def fix_contour_shape(contour):
        # Case 1: If contour is already in a valid shape, return it directly
        if len(contour.shape) == 3 and contour.shape[1] == 1:
            return contour
        if len(contour.shape) == 2 and contour.shape[1] == 2:
            return contour
        
        # Case 2: If contour is flat like (4,), reshape it to [[x1, y1], [x2, y2]]
        if contour.shape == (4,):
            return np.array([[[contour[0], contour[1]]], [[contour[2], contour[3]]]])

        # Case 3: If it’s a single point or invalid, skip it
        print(f"Skipping contour due to invalid shape: {contour.shape}")
        return None

    def merge_once(contours):
        merged_contours = []
        used = [False] * len(contours)

        for i, contour1 in enumerate(contours):
            if used[i]:
                continue
            
            contour1 = np.array(contour1)
            result = fix_contour_shape(contour1)

            if result is None: # HERE ___________________________________________________________________________________________ (prints before but no after)
                continue

            (new_x1, new_y1), (new_x2, new_y2) = result
            contour_weight1 = cv.arcLength(contour1, True)
            print("a")

            for j, contour2 in enumerate(contours):
                if i != j and not used[j]:
                    contour2 = np.array(contour2)
                    distance_sim = cv.matchShapes(contour1, contour2, cv.CONTOURS_MATCH_I1, 0)
                    difference_roc = difference_roc = abs(np.mean(rate_of_change(contour1)) - np.mean(rate_of_change(contour2)))
                    contour_weight1 = cv.arcLength(contour1, True)
                    contour_weight2 = cv.arcLength(contour2, True)

                    #print("Distance Similarity:", distance_sim, "Difference ROC:", difference_roc)

                    if distance_sim < difference_tolerance and difference_roc < merge_angle_tolerance:
                        (x1, y1), (x2, y2) = fix_contour_shape(contour2)

                        if x1 is not None and y1 is not None:  
                            new_x1 = weighted_average(new_x1, contour_weight1, x1, contour_weight2)
                            new_y1 = weighted_average(new_y1, contour_weight1, y1, contour_weight2)
                            new_x2 = weighted_average(new_x2, contour_weight1, x2, contour_weight2)
                            new_y2 = weighted_average(new_y2, contour_weight1, y2, contour_weight2)
                    
                    used[j] = True

            merged_contours.append((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
            used[i] = True

        return merged_contours

    # Perform iterative merging until contours stabilize
    prev_contours = []

    while prev_contours != contours:
        prev_contours = contours.copy()
        contours = merge_once(contours)

    return contours

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

        frame = resize_frame(frame)
        pre_frame = preprocess_frame(frame)
        contrast_frame = enhance_contrast(pre_frame)

        cv.imshow('Camera Feed', contrast_frame)

        line_frame, lines = detect_lines(contrast_frame)

        cv.imshow('Line Detection', line_frame)

        if len(lines) > 0:
            merged_lines = merge_lines(lines)

            for merged_line in merged_lines:
                x1, y1, x2, y2 = merged_line
                cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print("Merged Lines:", len(merged_lines))

        cv.imshow('Merged Lines', frame)

        contour_frame = frame.copy()
        contours = detect_curved_lines(contrast_frame)

        for contour in contours:
            cv.drawContours(contour_frame, [contour], -1, (0, 0, 255), 2)
            
        contour_frame = display_fps(contour_frame, start_time)
        cv.imshow('Curved Line Detection', contour_frame)

        if len(contours) > 0:
            merged_contours = merge_curved_lines(contours)

            for merged_contour in merged_contours: # NO merged_contours ___________________________________________________________
                x1, y1, x2, y2 = merged_contour
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print("Merged Contours:", len(merged_contours))

        cv.imshow('Merged Curved Lines', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
