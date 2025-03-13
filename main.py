import numpy as np
import cv2 as cv
import math
import time
from scipy.spatial.distance import directed_hausdorff, cdist
from collections import deque

def resize_frame(frame):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_height = 360
    new_width = int(new_height * aspect_ratio)

    return cv.resize(frame, (new_width, new_height)), new_height, new_width, aspect_ratio

def enhance_contrast(frame):
    # Convert the image to LAB color space
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    
    # Split the LAB image into different channels
    l_channel, a, b = cv.split(lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv.createCLAHE(clipLimit = 3, tileGridSize = (2, 2))
    cl = clahe.apply(l_channel)
    
    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv.merge((cl, a, b))
    
    # Convert the LAB image back to BGR format
    enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    
    return enhanced_img
    
def enhance_color_contrast(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define HSV range for white color
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask_white = cv.inRange(hsv, lower_white, upper_white)

    # Define HSV range for yellow color
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([75, 255, 255])
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)

    # Combine the masks
    mask = cv.bitwise_or(mask_white, mask_yellow)

    # Apply the mask to the original frame
    result = cv.bitwise_and(frame, frame, mask = mask)

    # Enhance the contrast of the masked regions
    result = enhance_contrast(result)

    # Combine the enhanced regions with the original frame
    enhanced_frame = cv.addWeighted(frame, .3, result, .7, 0)

    return enhanced_frame

def process_frame(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.medianBlur(frame, 5)
    frame = cv.bilateralFilter(frame, 9, 75, 75)

    return frame

def draw_arrow(frame, lines):
    best_x = 0
    best_line = None

    for line in lines:
        x1, y1, x2, y2 = line
        avg_x = (x1 + x2) // 2

        if avg_x > best_x:
            best_x = avg_x
            best_line = line
            #print("best line", best_line)

    if best_line is not None:
        #print("yippee")
        x1, y1, x2, y2 = best_line
        angle = calculate_angle(best_line)

        arrowx1, arrowy1 = 25, 75
        arrowx2 = int(arrowx1 - 50 * math.cos(math.radians(angle + 7)))
        arrowy2 = int(arrowy1 - 50 * math.sin(math.radians(angle + 7)))

        cv.arrowedLine(frame, (arrowx1, arrowy1), (arrowx2, arrowy2), (0, 255, 0), 2)

    else:
        cv.arrowedLine(frame, (25, 75), (25, 25), (0, 255, 0), 2)
        #print("no line")

    return frame

def get_perspective_transform(frame, roi_points, width, height):
    src_pts = np.float32(roi_points)
    dst_pts = np.float32([
        [0, height],
        [width, height],
        [width, 0],
        [0, 0]
    ])

    H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
    warped = cv.warpPerspective(frame, H, (width, height))

    return warped

def canny_edges(frame):
    return cv.Canny(frame, 50, 150, apertureSize = 3)

def sobel_edges(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)

    edges = cv.magnitude(sobel_x, sobel_y)
    edges = cv.normalize(edges, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    _, binary_edges = cv.threshold(edges, 50, 255, cv.THRESH_BINARY)
    binary_edges = cv.dilate(binary_edges, None, iterations = 1) # do this multiple times, and next function
    binary_edges = cv.erode(binary_edges, None, iterations = 1)
    binary_edges = cv.medianBlur(binary_edges, 5)
    binary_edges = cv.bilateralFilter(binary_edges, 9, 75, 75)

    contours, _ = cv.findContours(binary_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    binary_edges_bgr = cv.cvtColor(binary_edges, cv.COLOR_GRAY2BGR)

    lines = cv.HoughLinesP(binary_edges, rho = 1, theta = np.pi / 180, threshold = 100, minLineLength = 100, maxLineGap = 150)
    line_frame = frame.copy()

    filtered_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if calculate_angle(line[0]) > 55 and calculate_angle(line[0]) < 125:
                filtered_lines.append(line)

        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    contour_frame = frame.copy()
    cv.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
    """
    # Draw bounding boxes on the original frame
    output_frame = frame.copy()
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    """

    return contour_frame, binary_edges_bgr, line_frame, filtered_lines

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

def calculate_distance(x1, y1, x2, y2, x3, y3, x4, y4):
    num = abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1)
    den = np.hypot((y2 - y1), (x2 - x1))

    return num / den

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

def merge_lines(lines, width, height = 360, min_distance = 75, merge_angle_tolerance = 20, vertical_leeway = 1.5, horizontal_leeway = 1.1):
    def weighted_average(p1, w1, p2, w2):
        return (p1 * w1 + p2 * w2) / (w1 + w2)
    
    def extend_line(x1, y1, x2, y2, height):
        if x1 <= x2 + 25 or x1 >= x2 - 25:
            return x1, 0, x2, height
        
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            new_x1 = (0 - intercept) / slope
            new_x2 = (height - intercept) / slope

            return int(new_x1), 0, int(new_x2), height

    def sort_line_endpoints(line):
        x1, y1, x2, y2 = line

        if x1 > x2:
            return x2, y2, x1, y1
        
        else:
            return x1, y1, x2, y2
        
    def adjust_towards_center(x1, y1, x2, y2, width):
        center_x = width // 2
        adjustment_factor = 0.1  
        slope = (y2 - y1) / (x2 - x1)

        if slope == 0:
            return x1, y1, x2, y2
        
        elif slope > 0 and (x1 < center_x or x2 < center_x):
            return x1, y1, x2, y2
        
        elif slope > 0 and (x1 > center_x or x2 > center_x):
            adjustment_factor = slope / 100

            x1 = int(x1 - adjustment_factor * (x1 - center_x))
            x2 = int(x2 - adjustment_factor * (x2 - center_x))

            return x1, y1, x2, y2


        elif slope < 0 and (x1 > center_x or x2 > center_x):
            return x1, y1, x2, y2
        
        elif slope < 0 and (x1 < center_x or x2 < center_x):
            adjustment_factor = slope / 100

            x1 = int(x1 - adjustment_factor * (x1 - center_x))
            x2 = int(x2 - adjustment_factor * (x2 - center_x))

            return x1, y1, x2, y2

        else:
            return x1, y1, x2, y2
        
    def fix_slope(line, width):
        x1, y1, x2, y2 = line

        if x1 < width // 2 and x2 < width // 2:
            return x2, y1, x1, y2
        
        else:
            return x1, y1, x2, y2

    def merge_once(lines):
        merged_lines = []
        used = [False] * len(lines)

        for i, line1 in enumerate(lines):
            if used[i]:
                continue

            x1, y1, x2, y2 = sort_line_endpoints(line1)
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
            line_weight = line_length(line1)

            for j, line2 in enumerate(lines):
                if i != j and not used[j]:
                    x3, y3, x4, y4 = sort_line_endpoints(line2)

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

            new_x1, new_y1, new_x2, new_y2 = extend_line(new_x1, new_y1, new_x2, new_y2, height)
            new_x1, new_y1, new_x2, new_y2 = fix_slope((new_x1, new_y1, new_x2, new_y2), width)
            #new_x1, new_y1, new_x2, new_y2 = adjust_towards_center(new_x1, new_y1, new_x2, new_y2, width)
            
            merged_lines.append((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
            #print("merge lines in function", merged_lines)
            used[i] = True

        #print("merged lines in 2", merged_lines)
        return merged_lines

    # Convert to a flattened list of (x1, y1, x2, y2)
    lines = [line[0] for line in lines]

    # Perform iterative merging until lines stabilize
    prev_lines = []

    while not np.array_equal(prev_lines, lines):
        prev_lines = lines.copy()
        lines = merge_once(lines)

    return lines

def draw_midline_lines(warped_frame, blended_lines, width):
    if not isinstance(blended_lines, list):
        print(f"Error: blended_lines should be a list, but got {type(blended_lines)}")
        return
    
    for i in range(len(blended_lines)):
        for j in range(i + 1, len(blended_lines)):
            line1 = blended_lines[i]
            line2 = blended_lines[j]

            if isinstance(line1, tuple) and isinstance(line2, tuple):
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2

                if is_parallel_lines(line1, line2, 25, 15, 9999):
                    x_mid1 = (x1 + x3) // 2
                    y_mid1 = (y1 + y3) // 2
                    x_mid2 = (x2 + x4) // 2
                    y_mid2 = (y2 + y4) // 2

                    cv.line(warped_frame, (x_mid1, y_mid1), (x_mid2, y_mid2), (255, 0, 0), 2)

            elif isinstance(line1, int) and isinstance(line2, int):
                if line1 + 3 < len(blended_lines) and line2 + 3 < len(blended_lines):
                    x1, y1, x2, y2 = blended_lines[line1:line1 + 4]
                    x3, y3, x4, y4 = blended_lines[line2:line2 + 4]

                    if is_parallel_lines((x1, y1, x2, y2), (x3, y3, x4, y4), 25, 15, 9999):
                        x_mid1 = (x1 + x3) // 2
                        y_mid1 = (y1 + y3) // 2
                        x_mid2 = (x2 + x4) // 2
                        y_mid2 = (y2 + y4) // 2

                        cv.line(warped_frame, (x_mid1, y_mid1), (x_mid2, y_mid2), (255, 0, 0), 2)
                    
                else:
                    print(f"Skipping invalid line indices: {line1}, {line2}")
            else:
                print(f"Skipping invalid line format: {line1} (Type: {type(line1)}), {line2} (Type: {type(line2)})")

    return warped_frame

def blend_lines(old_lines, new_lines, alpha = .35):
    if not old_lines or not new_lines:
        return new_lines  # If no history, use new lines directly

    blended = []
    threshold_distance = 25  # Maximum distance to consider a line match
    angle_tolerance = 15  # Maximum angle difference to consider a match

    for new_line in new_lines:
        x1_new, y1_new, x2_new, y2_new = new_line
        best_match = None
        min_distance = float('inf')

        for old_line in old_lines:
            x1_old, y1_old, x2_old, y2_old = old_line

            # Compute distance between midpoints of lines
            mid_x_new = (x1_new + x2_new) / 2
            mid_y_new = (y1_new + y2_new) / 2
            mid_x_old = (x1_old + x2_old) / 2
            mid_y_old = (y1_old + y2_old) / 2
            distance = np.hypot(mid_x_new - mid_x_old, mid_y_new - mid_y_old)

            # Compute angle difference
            angle_new = calculate_angle(new_line)
            angle_old = calculate_angle(old_line)
            angle_diff = abs(angle_new - angle_old)

            if distance < threshold_distance and angle_diff < angle_tolerance:
                if distance < min_distance:
                    min_distance = distance
                    best_match = old_line

        if best_match:
            # Blend old and new lines using weighted average
            x1_old, y1_old, x2_old, y2_old = best_match
            blended_x1 = int(alpha * x1_new + (1 - alpha) * x1_old)
            blended_y1 = int(alpha * y1_new + (1 - alpha) * y1_old)
            blended_x2 = int(alpha * x2_new + (1 - alpha) * x2_old)
            blended_y2 = int(alpha * y2_new + (1 - alpha) * y2_old)

            blended.append((blended_x1, blended_y1, blended_x2, blended_y2))

        else:
            blended.append(new_line)

    return blended

def detect_shadows(frame, old_frame, lines):
    factors = 0
    height, width = frame.shape[:2]
    frame = frame[:height // 4 + 50, width // 4 - 50:]
    mean_brightness = np.mean(frame)
    shadow_bool = False

    if old_frame is not None:
        old_brightness = np.mean(old_frame)

    else:
        old_brightness = mean_brightness

    #print("mean brightness", mean_brightness)

    if len(lines) > 5:
        factors += 1

    if mean_brightness < 120:
        factors += 1

    if old_brightness > mean_brightness + 10:
        factors += 1

    if factors >= 2:
        print("Shadow Detected\nmean brightness", mean_brightness, "\n")
        shadow_bool = True

    return frame, shadow_bool

def calculate_optimal_lane_lines(frame, lines, old_lines, width):
    optimal_lanes = []
    optimal_left_lanes = []
    optimal_right_lanes = []
    lanes_final = []
    optimal_lane_frame = frame.copy()

    cv.line(optimal_lane_frame, (width // 2 - width // 4, 0), (width // 2 - width // 4, frame.shape[0]), (0, 0, 255), 2)
    cv.line(optimal_lane_frame, (width // 2 + width // 4, 0), (width // 2 + width // 4, frame.shape[0]), (0, 0, 255), 2)

    for line in lines:
        x1, y1, x2, y2 = line

        if x1 > width // 2 + width // 4 and x2 > width // 2 + width // 4:
            optimal_right_lanes.append(line)

        if width // 2 - width // 4 < x1 < width // 2 + width // 4 and width // 2 - width // 4 < x2 < width // 2 + width // 4:
            optimal_left_lanes.append(line)

    if optimal_right_lanes:
        for optimal_right_lane in optimal_right_lanes:
            best_angle = 0
            x1, y1, x2, y2 = optimal_right_lane
            angle = calculate_angle(optimal_right_lane)

            if angle > best_angle:
                best_angle = angle
                optimal_lanes.append(optimal_right_lane)

    if optimal_left_lanes:
        for optimal_left_lane in optimal_left_lanes:
            best_angle = 0
            x1, y1, x2, y2 = optimal_left_lane
            angle = calculate_angle(optimal_left_lane)

            if angle > best_angle:
                best_angle = angle
                optimal_lanes.append(optimal_left_lane)

    if len(old_lines) > 0:
        if len(optimal_lanes) > 0:
            for old_line in old_lines:
                #x1, y1, x2, y2 = old_line
                old_angle = calculate_angle(old_line)

                for optimal_lane in optimal_lanes:
                    #x3, y3, x4, y4 = optimal_lane
                    new_angle = calculate_angle(optimal_lane)

                    old_angle = abs(old_angle - 90)
                    new_angle = abs(new_angle - 90)

                    if old_angle < new_angle:
                        lanes_final.append(old_line)

                    else:
                        lanes_final.append(optimal_lane)

    if len(lanes_final) > 0:
        for lane_final in lanes_final:
            x1, y1, x2, y2 = lane_final
            cv.line(optimal_lane_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return optimal_lane_frame, lanes_final
        
    else:
        for optimal_lane in optimal_lanes:
            x1, y1, x2, y2 = optimal_lane
            cv.line(optimal_lane_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return optimal_lane_frame, optimal_lanes


"""cant we 
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
        if len(contour) < 5:
            continue

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

def is_parallel_curves(contour1, contour2, roc_tolerance, distance_threshold):

    distance_sim = cv.matchShapes(contour1, contour2, cv.CONTOURS_MATCH_I1, 0)
    difference_roc = abs(np.mean(rate_of_change(contour1)) - np.mean(rate_of_change(contour2)))

    if distance_sim < distance_threshold and difference_roc < roc_tolerance:
        return True
    
    else:
        return False

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

                        center_lines.append(np.array([[[x_mid, y_mid]], [[x_mid_next, y_mid_next]]]))

                        cv.line(frame, (x_mid, y_mid), (x_mid_next, y_mid_next), (255, 0, 0), 2)

                    else:
                        cv.line(frame, (x_mid, y_mid), (x_mid, y_mid), (255, 0, 0), 2)

                        center_lines.append(np.array([[[x_mid, y_mid]], [[x_mid, y_mid]]]))

    mid_lines = merge_curved_lines(center_lines)

    for mid_line in mid_lines:
        for i in range(len(mid_line) - 1):
            x1, y1 = mid_line[i][0]
            x2, y2 = mid_line[i + 1][0]

            cv.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return frame

def merge_curved_lines(contours, hausdorff_threshold = 40, match_shapes_threshold = .6, centroid_threshold = 65, contour_approx_tolerance = .005, roc_tolerance = 30, distance_threshold = 285):
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
        hausdorff_dist = hausdorff_distance(c1, c2)
        shape_similarity = cv.matchShapes(c1, c2, cv.CONTOURS_MATCH_I1, 0)
        centroid_dist = np.linalg.norm(get_centroid(c1) - get_centroid(c2))
        roc_difference = abs(np.mean(rate_of_change(c1)) - np.mean(rate_of_change(c2))) 
        distance_points = calculate_distances(c1, c2)
        
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
"""

def display_fps(frame, start_time):
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

    return frame

def main():
    vid_path = R"c:\Users\Owner\Downloads\pwp_data\videoplayback.webm" # vid path ________________________________________________________________________
    cap = cv.VideoCapture(vid_path)

    merged_lines = []
    merged_contours = []
    last_merged_lines = []
    blended_lines = []
    line_history = deque (maxlen = 100)
    frame_skip = 10
    frame_by_frame_mode = False
    old_frame = None
    optimal_lane_lines = []

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1

        if frame_count < 200:
            continue

        if frame_count % frame_skip != 0:
            continue

        #frame = cv.imread(r"c:\Users\Owner\Downloads\pwp_data\testroad.jpg")
        frame, height, width, ratio = resize_frame(frame)
        #contrast_frame = enhance_contrast(frame)
        #enhance_color_frame = enhance_color_contrast(contrast_frame)
        preprocessed_frame = process_frame(frame)

        display_fps(frame, start_time)

        #"""
        roi_points = [
            (0, height - 15),  # bottom left
            (width - 175, height - 20),  # bottom right
            (width - 235, 250),  # top right
            (150, 250)  # top left
        ]
        #"""

        """
        roi_points = [
            (10, height),  # bottom left
            (width - 35, height),  # bottom right
            (width - 125, 250),  # top right
            (175, 250)  # top left
        ]
        """

        mask = np.zeros_like(preprocessed_frame)
        roi_corners = np.array(roi_points, dtype=np.int32)
        cv.fillPoly(mask, [roi_corners], 255)
        roi_frame = cv.bitwise_and(preprocessed_frame, mask)

        roi_points_np = np.array(roi_points, np.int32).reshape((-1, 1, 2))
        cv.polylines(frame, [roi_points_np], True, (0, 255, 0), 2)

        warped_frame = get_perspective_transform(preprocessed_frame, roi_points, width, height)
        contour_frame, binary_frame, line_frame, lines = sobel_edges(warped_frame)

        warped_width = warped_frame.shape[1]
        merge_line_frame = warped_frame.copy()
        blend_frame = warped_frame.copy()
        shadow_frame = warped_frame.copy()
        optimal_frame = warped_frame.copy()

        if lines is not None:
            new_merged_lines = merge_lines(lines, warped_width)
            draw_midline_lines(merge_line_frame, new_merged_lines, warped_width)
            shadow_frame, shadow_bool = detect_shadows(shadow_frame, old_frame, new_merged_lines)
            #_, optimal_frame = calculate_optimal_lane_lines(optimal_frame, new_merged_lines, optimal_lane_lines, warped_width)
            #optimal_lane_lines.append(calculate_optimal_lane_lines(optimal_frame, new_merged_lines, optimal_lane_lines, warped_width))

            if shadow_bool:
                if last_merged_lines:
                    shadow_lines = line_history[-1]

                    for shadow_line in shadow_lines:
                        x1, y1, x2, y2 = shadow_line
                        cv.line(blend_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            else:
                for new_merged_line in new_merged_lines:
                    x1, y1, x2, y2 = new_merged_line
                    cv.line(merge_line_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if last_merged_lines:
                    blended_lines = blend_lines(last_merged_lines, new_merged_lines)

                else:
                    blended_lines = new_merged_lines

                last_merged_lines = blended_lines
                line_history.append(blended_lines)
                old_frame = shadow_frame

                if blended_lines is not None:
                    draw_midline_lines(blend_frame, blended_lines, warped_width)
                    
                    for blended_lines in blended_lines:
                        x1, y1, x2, y2 = blended_lines
                        cv.line(blend_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            #print("Blended Lines:", len(blended_lines))

        else:
            new_merged_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #print("Lines:", len(lines))

        if new_merged_lines is not None:
            frame = draw_arrow(frame, new_merged_lines)


        """
        if blended_lines is not None:
            #print("Blended Lines aaaa:", len(blended_lines))
            draw_midline_lines(merge_line_frame, blended_lines)
            #print("after midlines-----------------------------------------------------" [blended_lines])

        print("Blended Lines:", blended_lines)
        """

        """
        line_frame, lines = detect_lines(warped_frame)

        if len(lines) > 0:
            merged_lines = merge_lines(lines)

            for merged_line in merged_lines:
                x1, y1, x2, y2 = merged_line
                cv.line(warped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        try:
            print("Merged Lines:", len(merged_lines))

        except:
            print("Merged Lines: 0")

        contour_frame = warped_frame.copy()
        contours = detect_curved_lines(contour_frame)

        for contour in contours:
            cv.drawContours(contour_frame, [contour], -1, (0, 0, 255), 2)
            
        contour_frame = display_fps(contour_frame, start_time)

        if len(contours) > 0:
            merged_contours = merge_curved_lines(contours)

            for merged_contour in merged_contours:
                cv.drawContours(warped_frame, [merged_contour], -1, (0, 255, 255), 2)

        try:
            print("Merged Contours:", len(merged_contours))

        except:
            print("Merged Contours: 0")

        if len(merged_lines) > 0:
            draw_midline_lines(warped_frame, merged_lines)
            cv.imshow('Middle Lines', warped_frame)

        if len(merged_contours) > 0:
            draw_midline_curves(warped_frame, merged_contours)
            cv.imshow('Middle Curved Lines', warped_frame)
        

        cv.imshow('Merged Curved Lines', warped_frame)
        cv.imshow('Curved Line Detection', contour_frame)
        cv.imshow('Merged Lines', warped_frame)
        cv.imshow('Line Detection', line_frame)
        """

        cv.imshow("Frame", frame)
        cv.imshow("Preprocessed Frame", preprocessed_frame)
        cv.imshow("ROI Frame", roi_frame)
        cv.imshow("Warped Frame", warped_frame)
        cv.imshow("Binary Frame", binary_frame)
        cv.imshow("Contour Frame", contour_frame)
        cv.imshow("Line Frame", line_frame)
        cv.imshow("Merged Lines", merge_line_frame)
        #cv.imshow("Contrast Frame", contrast_frame)
        #cv.imshow("Enhanced Color Frame", enhance_color_frame)
        cv.imshow("Blended Lines", blend_frame)
        cv.imshow("Shadow Frame", shadow_frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('f'):
            frame_by_frame_mode = not frame_by_frame_mode

        if frame_by_frame_mode:
            while True:
                key = cv.waitKey(0) & 0xFF

                if key == ord('n'):
                    break

                elif key == ord('q'):
                    cap.release()
                    cv.destroyAllWindows()
                    return

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
