import cv2
import numpy as np


# Function match_features() returns keypoint objects but want to get pixel coordinates of those keypoints (to be used for essential matrix estimation)
def extract_matched_points(
    keypoints1: list[cv2.KeyPoint],  # recall: keypoints1, descriptors1 = detect_orb_features(image1)
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    
    # For every match, retrieve the coordinates of its keypoint in the first image
    points1 = np.float32([
        keypoints1[match.queryIdx].pt  # queryIdx is index of keypoint in image 1 and .pt gets the coordinate from the keypoints1 object 
        for match in matches
    ])

    # For every match, retrieve the coordinates of its corresponding keypoint in the second image
    points2 = np.float32([
        keypoints2[match.trainIdx].pt  # trainIdx is index of keypoint in image 2
        for match in matches
    ])

    return points1, points2


# ORB matches are not necessarily all correct: two different objects might have similar-looking descriptors, causing the matcher to pair them incorrectly
# Estimate essential matrix (describes the geometric relationship between two camera views), letting us identify matches consistent with the geometry of camera motion
def estimate_essential_matrix(
    points1: np.ndarray,  # (N, 2) array of matched pixel coordinates from image 1
    points2: np.ndarray,
    K: np.ndarray,  # (3, 3) camera intrinsic matrix
) -> tuple[np.ndarray, np.ndarray]:

    E, mask = cv2.findEssentialMat(  # E is essential matrix; mask indicates which matches are geometrically consistent (1 = inlier, 0 = outlier)
        points1,  # pixel coordinates of matched features in image 1
        points2,
        cameraMatrix=K,  # camera intrinsic matrix used to interpret pixel coordinates
        method=cv2.RANSAC,  # use RANSAC algorithm to estimate E (uses random subsets of matches to identify and reject outliers)
        prob=0.999,  # desired 99.9% confidence of sampling an outlier-free subs
        threshold=1.0,  # maximum geometric error (approximately 1 pixel) for classifying an inlier
    )

    if E is None or mask is None:
        raise ValueError("Essential matrix estimation failed.")

    return E, mask


# Recover relative camera pose (rotation and translation direction) by decomposing E
def recover_camera_pose(
    E: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # this decomposition produces 4 possible combinations of rotations and translations, so OpenCV selects the pose that produces the most geometrically valid points
    # some of the inliers pre-selection might not produce valid 3D points during pose recovery, so number of inliers and mask change

    num_inliers, R, t, pose_mask = cv2.recoverPose(  # R: 3x3 rotation matrix, t: 3x1 translation direction, pose_mask: inilier mask after pose recovery
        E,
        points1,
        points2,
        cameraMatrix=K,
        mask=mask.copy(),
    )

    # Camera motion cannot be estimated if there are no matches
    if num_inliers == 0:
        raise ValueError("Camera pose recovery failed: no valid inliers.")

    return R, t, pose_mask

