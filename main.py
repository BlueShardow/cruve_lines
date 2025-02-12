import numpy as np
import cv2 as cv
import math
import os
import time
import sys
from scipy.spatial.distance import cdist, directed_hausdorff
from skimage.morphology import skeletonize

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

def skeletonize_frame(frame):
    frame = cv.threshold(frame, 127, 255, cv.THRESH_BINARY)[1]
    frame = skeletonize(frame)
    frame = (frame * 255).astype(np.uint8)  # Convert boolean to uint8
    
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

def is_parallel_lines(line1, line2, toward_tolerance, away_tolerance, distance_threshold):
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

def is_parallel_curves(contour1, contour2, roc_tolerance, distance_threshold):

    distance_sim = cv.matchShapes(contour1, contour2, cv.CONTOURS_MATCH_I1, 0)
    difference_roc = abs(np.mean(rate_of_change(contour1)) - np.mean(rate_of_change(contour2)))

    if distance_sim < distance_threshold and difference_roc < roc_tolerance:
        return True
    
    else:
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
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
            line_weight = line_length(line1)

            for j, line2 in enumerate(lines):
                if i != j and not used[j]:
                    x3, y3, x4, y4 = line2

                    if is_parallel_lines(line1, line2, merge_angle_tolerance, merge_angle_tolerance, min_distance):
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

    while not np.array_equal(prev_lines, lines):
        prev_lines = lines.copy()
        lines = merge_once(lines)

    return lines

def detect_curved_lines(frame, contour_approx_tolerance = .002):
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

def calculate_distances(contour1, contour2):
    points1 = contour1[:, 0, :]
    points2 = contour2[:, 0, :]

    distances = np.mean(cdist(points1, points2))

    return distances.flatten()  # Return a 1D array of all distances

"""
# Example usage
contour1 = np.array([[[0, 0]], [[1, 1]], [[2, 2]]])
contour2 = np.array([[[3, 3]], [[4, 4]], [[5, 5]]])

distances = calculate_distances(contour1, contour2)
print("Distances:", distances)
"""

def contours_are_stable(prev_contours, new_contours, length_tolerance = 2, shape_tolerance = .05):
    if len(prev_contours) != len(new_contours):
        return False
    
    for c1, c2 in zip(prev_contours, new_contours):
        # Simplify shapes to reduce noise in comparisons
        epsilon1 = 0.01 * cv.arcLength(c1, True)
        epsilon2 = 0.01 * cv.arcLength(c2, True)
        simplified_c1 = cv.approxPolyDP(c1, epsilon1, True)
        simplified_c2 = cv.approxPolyDP(c2, epsilon2, True)
        
        # Check arc length similarity
        length_diff = abs(cv.arcLength(simplified_c1, True) - cv.arcLength(simplified_c2, True))
        
        # Check shape similarity using matchShapes
        shape_diff = cv.matchShapes(simplified_c1, simplified_c2, cv.CONTOURS_MATCH_I1, 0)
        
        if length_diff > length_tolerance or shape_diff > shape_tolerance:
            return False
    
    return True
"""
def merge_curved_lines(contours, min_distance = 75, merge_angle_tolerance = 50, difference_tolerance = 50, contour_approx_tolerance = .005):
    contours = [np.array(contour) if isinstance(contour, tuple) else contour for contour in contours]
    iteration_count = 0

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
        else:
            print(f"Skipping contour due to invalid shape: {contour.shape}")
            return None

    def merge_once(contours):
        merged_contours = []
        used = [False] * len(contours)

        for i, contour1 in enumerate(contours):
            if used[i]:
                continue

            # Start with all points from contour1
            merged_points = list(contour1[:, 0, :])
            #contour_weight1 = cv.arcLength(contour1, True)

            for j, contour2 in enumerate(contours):
                if i != j and not used[j]:
                    distance_sim = cv.matchShapes(contour1, contour2, cv.CONTOURS_MATCH_I1, 0)
                    difference_roc = abs(np.mean(rate_of_change(contour1)) - np.mean(rate_of_change(contour2)))
                    distance_points = calculate_distances(contour1, contour2)

                    if distance_sim < difference_tolerance and difference_roc < merge_angle_tolerance and distance_points < min_distance:
                        # Merge points from contour2 into contour1
                        merged_points.extend(contour2[:, 0, :])
                        used[j] = True

            merged_contour = np.array(merged_points).reshape((-1, 1, 2)).astype(np.int32)
            epsilon = contour_approx_tolerance * cv.arcLength(merged_contour, True)
            simplified_contour = cv.approxPolyDP(merged_contour, epsilon, True)
            merged_contours.append(simplified_contour)

            used[i] = True

        return merged_contours

    contours = [fix_contour_shape(c) for c in contours if fix_contour_shape(c) is not None]

    # Perform iterative merging until contours stabilize
    prev_contours = []

    while iteration_count < 25:
        if iteration_count >= 24:
            print("\nContours did not stabilize within 25 iterations.\n")
            break

        new_contours = merge_once(prev_contours)

        if contours_are_stable(prev_contours, new_contours):
            break

        prev_contours = new_contours
        iteration_count += 1

    return contours
"""

def draw_midline_lines(frame, lines):
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            x1, y1, x2, y2 = line1

            line2 = lines[j]
            x3, y3, x4, y4 = line2

            if is_parallel_lines(line1, line2, 25, 15, 9999):
                x_mid1 = (x1 + x3) // 2
                y_mid1 = (y1 + y3) // 2
                x_mid2 = (x2 + x4) // 2
                y_mid2 = (y2 + y4) // 2

                cv.line(frame, (x_mid1, y_mid1), (x_mid2, y_mid2), (255, 0, 0), 2)

    return frame

def smooth_midline_curves ():
    pass

def draw_midline_curves(frame, contours):
    center_lines = []

    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            contour1 = contours[i]
            contour2 = contours[j]

            if is_parallel_curves(contour1, contour2, 50, 50):
                min_length = min(len(contour1), len(contour2))

                for k in range(min_length):
                    x1, y1 = contour1[k][0]
                    x2, y2 = contour2[k][0]

                    x_mid = (x1 + x2) // 2
                    y_mid = (y1 + y2) // 2

                    if k < min_length - 1:
                        x1_next, y1_next = contour1[k + 1][0]
                        x2_next, y2_next = contour2[k + 1][0]

                        x_mid_next = (x1_next + x2_next) // 2
                        y_mid_next = (y1_next + y2_next) // 2

                        center_lines.append((x_mid, y_mid, x_mid_next, y_mid_next))

                        cv.line(frame, (x_mid, y_mid), (x_mid_next, y_mid_next), (255, 0, 0), 2)

                    else:
                        cv.line(frame, (x_mid, y_mid), (x_mid, y_mid), (255, 0, 0), 2)

                        center_lines.append((x_mid, y_mid, x_mid, y_mid))

                    

    return frame

#__________________________________________________________________________________________________________________________________________________________________
"""
def merge_curved_lines(contours, hausdorff_threshold = 50, match_shapes_threshold = .3, centroid_threshold = 75, contour_approx_tolerance = .002, roc_tolerance = 25, distance_threshold = 200):
"""
def merge_curved_lines(contours, hausdorff_threshold = 50, match_shapes_threshold = .75, centroid_threshold = 75, contour_approx_tolerance = .002, roc_tolerance = 25, distance_threshold = 275):
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
        else:
            print(f"Skipping contour due to invalid shape: {contour.shape}")
            return None

    def hausdorff_distance(c1, c2):
        points1 = c1[:, 0, :]
        points2 = c2[:, 0, :]

        return max(directed_hausdorff(points1, points2)[0], directed_hausdorff(points2, points1)[0])

    def get_centroid(contour):
        moments = cv.moments(contour)

        if moments["m00"] == 0:
            return np.array([0, 0])

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        return np.array([cx, cy])

    def contours_are_similar(c1, c2):
        # Calculate Hausdorff distance
        hausdorff_dist = hausdorff_distance(c1, c2)
        
        # Calculate shape similarity using cv.matchShapes
        shape_similarity = cv.matchShapes(c1, c2, cv.CONTOURS_MATCH_I1, 0)
        
        # Calculate centroid distance
        centroid_dist = np.linalg.norm(get_centroid(c1) - get_centroid(c2))

        # Calculate rate of change difference
        roc_difference = abs(np.mean(rate_of_change(c1)) - np.mean(rate_of_change(c2))) 

        # Calculate distance between points
        distance_points = calculate_distances(c1, c2)
        
        # Check all criteria
        print("\nHausdorff Distance:", hausdorff_dist < hausdorff_threshold, "\nShape Similarity:", shape_similarity < match_shapes_threshold, "\nCentroid Distance:", centroid_dist < centroid_threshold, "\nROC Difference:", roc_difference < roc_tolerance, "\nDistance Points:", distance_points < distance_threshold)
        
        return (hausdorff_dist < hausdorff_threshold and shape_similarity < match_shapes_threshold and centroid_dist < centroid_threshold and roc_difference < roc_tolerance and distance_points < distance_threshold)

    def merge_contours(c1, c2):
        merged_points = np.vstack((c1, c2)).reshape((-1, 1, 2))
        epsilon = contour_approx_tolerance * cv.arcLength(merged_points, True)
        return cv.approxPolyDP(merged_points, epsilon, True)

    contours = [fix_contour_shape(c) for c in contours if fix_contour_shape(c) is not None]
    merged_contours = []

    while True:
        used = [False] * len(contours)
        temp_merged = []

        for i, c1 in enumerate(contours):
            if used[i]:
                continue

            merged = c1

            for j, c2 in enumerate(contours):
                if i != j and not used[j] and contours_are_similar(merged, c2):
                    merged = merge_contours(merged, c2)
                    used[j] = True

            temp_merged.append(merged)
            used[i] = True

        if contours_are_stable(merged_contours, temp_merged):
            break

        merged_contours = temp_merged

    return merged_contours
# _______________________________________________________________________________

def display_fps(frame, start_time):
    fps = int(1 / (time.time() - start_time))
    cv.putText(frame, f'FPS: {fps}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    return frame

def main():
    cap = cv.VideoCapture(0)
    merged_lines = []
    merged_contours = []

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv.imread("/Users/pl1001515/Downloads/cruved2.jpeg") 

        frame = resize_frame(frame)
        pre_frame = preprocess_frame(frame)
        contrast_frame = enhance_contrast(pre_frame)
        #skele_frame = skeletonize_frame(contrast_frame)

        cv.imshow('Camera Feed', contrast_frame)

        line_frame, lines = detect_lines(contrast_frame)

        cv.imshow('Line Detection', line_frame)

        if len(lines) > 0:
            merged_lines = merge_lines(lines)

            for merged_line in merged_lines:
                x1, y1, x2, y2 = merged_line
                cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        try:
            print("Merged Lines:", len(merged_lines))

        except:
            print("Merged Lines: 0")

        cv.imshow('Merged Lines', frame)

        contour_frame = frame.copy()
        contours = detect_curved_lines(contrast_frame)

        for contour in contours:
            cv.drawContours(contour_frame, [contour], -1, (0, 0, 255), 2)
            
        contour_frame = display_fps(contour_frame, start_time)
        cv.imshow('Curved Line Detection', contour_frame)

        if len(contours) > 0:
            merged_contours = merge_curved_lines(contours)

            for merged_contour in merged_contours:
                cv.drawContours(frame, [merged_contour], -1, (0, 255, 255), 2)

        try:
            print("Merged Contours:", len(merged_contours))

        except:
            print("Merged Contours: 0")

        cv.imshow('Merged Curved Lines', frame)

        if len(merged_lines) > 0:
            draw_midline_lines(frame, merged_lines)
            cv.imshow('Middle Lines', frame)

        if len(merged_contours) > 0:
            draw_midline_curves(frame, merged_contours)
            cv.imshow('Middle Curved Lines', frame)

        #cv.imshow("Skeleton", skele_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
