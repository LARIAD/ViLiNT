import numpy as np
import cv2

def waypoints_px_to_img(image, trajectories, chosen_trajectory, chosen_waypoint):
    # Convert image to OpenCV format if it's a PIL image
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Get the width and height of the image
    img_height, img_width, _ = img.shape

    # Define the bottom-center position for drawing the waypoints
    bottom_y = img_height - 10  # Slightly above the bottom edge for some padding
    center_x = img_width // 2   # Horizontal center of the image
    center_y = img_height // 2
    center = (center_x, center_y)

    # Define the scaling factor
    scaling_factor = 20  # Adjust this value to fit the waypoints within the image size

    # Iterate through the list of trajectories
    for t_idx, trajectory in enumerate(trajectories):
        if np.linalg.norm(np.sum(trajectory)) < 1e-6:
            continue
        # Scale the waypoints to image coordinates and position them at the bottom center
        scaled_waypoints = []
        for wp in trajectory:
            # Correctly scale the waypoints
            scaled_wp = np.array([center_x + wp[0] * scaling_factor, bottom_y - wp[1] * scaling_factor])

            # Ensure the scaled waypoints are within the image bounds
            scaled_wp = np.clip(scaled_wp, [0, 0], [img_width-1, img_height-1])  # Ensure within image bounds

            scaled_waypoints.append(scaled_wp)

        scaled_waypoints = rotate_trajectory(scaled_waypoints, center)
        
        # Draw lines connecting consecutive waypoints (representing the trajectory)
        for i in range(len(scaled_waypoints) - 1):
            wp1 = scaled_waypoints[i]
            wp2 = scaled_waypoints[i + 1]
            
            # Choose color based on whether it's the chosen trajectory
            if t_idx == chosen_trajectory:
                color = (255, 0, 0)  # Blue color for the chosen trajectory
            else:
                color = (0, 255, 255)  # Yellow color for other trajectories
            
            # Draw a line between the two consecutive waypoints (representing the trajectory)
            cv2.line(img, (int(wp1[0]), int(wp1[1])),
                    (int(wp2[0]), int(wp2[1])), color, 2)

        # Draw circles at each waypoint for visualization
        for i, wp in enumerate(scaled_waypoints):
            # Highlight the waypoint for the chosen trajectory
            if t_idx == chosen_trajectory and i == chosen_waypoint:
                color = (0, 0, 255)  # Red color for the selected waypoint
            elif t_idx == chosen_trajectory:
                color = (255, 0, 0)  # Blue color for the other waypoints in the chosen trajectory
            else:
                color = (0, 255, 255)  # Yellow color for all other trajectories' waypoints
            
            # Draw a circle at the waypoint
            cv2.circle(img, (int(wp[0]), int(wp[1])), 2, color, -1)  # Circle with the appropriate color

    # Return the image with the drawn waypoints and trajectories
    return img

def rotate_trajectory(waypoints, image_center):
    # Extract the first waypoint
    P1 = np.array(waypoints[0])

    # Calculate the tangent vector (difference between the first and second waypoint)
    tangent = np.array(waypoints[1]) - P1

    # Calculate the vector from the first waypoint to the image center
    center_vector = np.array(image_center) - P1

    # Normalize the tangent and center vectors
    tangent_norm = np.linalg.norm(tangent)
    center_norm = np.linalg.norm(center_vector)

    # Compute the cosine of the angle between the tangent and the center vector
    cos_angle = np.dot(tangent, center_vector) / (tangent_norm * center_norm)

    # Ensure the cosine value is within [-1, 1] range due to floating point errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Calculate the angle in radians
    angle = np.arccos(cos_angle)

    # To determine the sign of the angle (clockwise or counterclockwise), 
    # we check the direction of the perpendicular vector (2D cross product equivalent)
    perp = np.array([-tangent[1], tangent[0]])  # 90 degree rotation of tangent
    cross_product = np.dot(perp, center_vector)

    # If cross product is negative, the angle is clockwise
    if cross_product < 0:
        angle = -angle

    # Create the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # Rotate each waypoint around the first waypoint
    rotated_waypoints = []
    for wp in waypoints:
        wp = np.array(wp)
        # Translate to origin, rotate, and then translate back
        rotated_wp = np.dot(rotation_matrix, wp - P1) + P1
        rotated_waypoints.append(rotated_wp.tolist())

    return rotated_waypoints

def pub_waypoints(curr_img, raw_naction, waypoint):
    img_waypoints = waypoints_px_to_img(curr_img, raw_naction, 0, waypoint)
    return img_waypoints