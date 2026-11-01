import numpy as np
from tqdm import tqdm

from src.dataset import (
    load_calibration,
    get_intrinsic_matrix,
    get_image_paths,
    load_image,
)
from src.features.orb import detect_orb_features
from src.features.matching import match_features
from src.geometry.essential import (
    extract_matched_points,
    estimate_essential_matrix,
    recover_camera_pose,
)
from src.geometry.trajectory import update_pose

# Estimate the camera trajectory from an entire sequence of frames by repeatedly estimating relative camera motion and accumulating the resulting poses
# Lets us store each estimated camera pose as a pose matrix (each represented by a 4x4 transformation matrix)
def run_visual_odometry(calibration_path, image_dir, num_frames=100):

    calibration = load_calibration(calibration_path)
    K = get_intrinsic_matrix(calibration)
    image_paths = get_image_paths(image_dir)[:num_frames]

    # At least two images are needed to estimate relative motion
    if len(image_paths) < 2:
        raise ValueError("At least two frames are required.")

    current_pose = np.eye(4)  # initialize accumulated camera pose as pose at frame 0, so identity matrix

    trajectory = [current_pose.copy()]  # store initial pose as the first trajectory entry

    # Load frame 0 and detect its ORB features
    image1 = load_image(image_paths[0])
    keypoints1, descriptors1 = detect_orb_features(image1)

    # Process consecutive frame pairs i and i+1
    for i in tqdm(range(len(image_paths) - 1), desc="Processing frames", unit="pair"):

        # Load frame i+1 and detect its ORB features
        image2 = load_image(image_paths[i + 1])
        keypoints2, descriptors2 = detect_orb_features(image2)

        # Match features between frames i and i+1
        matches = match_features(descriptors1, descriptors2)

        if len(matches) < 5:
            raise ValueError(f"Insufficient matches between frames {i} and {i+1}.")

        # Extract the pixel coordinates of matched keypoints
        points1, points2 = extract_matched_points(keypoints1, keypoints2, matches)

        # Estimate the essential matrix using RANSAC
        E, mask = estimate_essential_matrix(points1, points2, K)

        # Recover relative rotation and translation direction
        try:
            R, t, pose_mask = recover_camera_pose(E, points1, points2, K, mask)

            # Incorporate the motion from frame i to frame i+1 into the accumulated pose relative to frame 0
            current_pose = update_pose(current_pose, R, t)  # returns pose info for frame i+1

        except ValueError:
            tqdm.write(
                f"Warning: Pose recovery failed between frames {i} and {i + 1}. "
                f"Keeping previous pose."
            )

        # Store the accumulated camera pose at frame i+1
        trajectory.append(current_pose.copy())

        # Frame i+1 becomes frame i in the next iteration
        keypoints1, descriptors1 = keypoints2, descriptors2

    return np.array(trajectory)  # returns Nx4x4 array, where N is number of frames
