import numpy as np

from src.dataset import load_calibration, get_intrinsic_matrix, get_image_paths, load_image, load_poses
from src.features.orb import detect_orb_features
from src.features.matching import match_features
from src.geometry.essential import (
    extract_matched_points,
    estimate_essential_matrix,
    recover_camera_pose,
)


def main():
    calibration = load_calibration("data/calib/00/calib.txt")
    K = get_intrinsic_matrix(calibration)

    image_paths = get_image_paths("data/sequences/00/image_0")
    ground_truth = load_poses("data/poses/00.txt")

    # Normal frame pairs and regions containing large rotation errors
    frame_indices = (
        [100, 200, 700]
        + list(range(360, 371))
        + list(range(455, 466))
        + list(range(620, 651))
    )

    results = []

    for i in frame_indices:
        image1 = load_image(image_paths[i])
        image2 = load_image(image_paths[i + 1])

        keypoints1, descriptors1 = detect_orb_features(image1)
        keypoints2, descriptors2 = detect_orb_features(image2)

        matches = match_features(descriptors1, descriptors2)
        points1, points2 = extract_matched_points(
            keypoints1, keypoints2, matches
        )

        E, mask = estimate_essential_matrix(points1, points2, K)
        ransac_inliers = np.count_nonzero(mask)

        try:
            R, t, pose_mask = recover_camera_pose(
                E, points1, points2, K, mask
            )
            pose_inliers = np.count_nonzero(pose_mask)
        except ValueError:
            # Match the existing pipeline's fallback behaviour
            R = np.eye(3)
            pose_inliers = 0

        # Ground-truth rotation from camera i to camera i+1
        R_gt = (
            ground_truth[i + 1, :3, :3].T
            @ ground_truth[i, :3, :3]
        )

        # Difference between estimated and actual relative rotations
        R_error = R_gt.T @ R
        cos_angle = (np.trace(R_error) - 1) / 2
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        error_degrees = np.degrees(np.arccos(cos_angle))

        results.append((
            i,
            len(matches),
            ransac_inliers,
            pose_inliers,
            error_degrees,
        ))

    print(
        f"{'Frame pair':<14}"
        f"{'Matches':>10}"
        f"{'RANSAC':>10}"
        f"{'Pose':>10}"
        f"{'Error (°)':>14}"
    )

    for i, matches, ransac, pose, error in results:
        print(
            f"{f'{i} → {i + 1}':<14}"
            f"{matches:>10}"
            f"{ransac:>10}"
            f"{pose:>10}"
            f"{error:>14.4f}"
        )


if __name__ == "__main__":
    main()

