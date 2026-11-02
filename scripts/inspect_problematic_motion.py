import numpy as np

from src.dataset import load_calibration, get_intrinsic_matrix, get_image_paths, load_image, load_poses
from src.features.orb import detect_orb_features
from src.features.matching import match_features
from src.geometry.essential import (
    extract_matched_points,
    estimate_essential_matrix,
    recover_camera_pose,
)


def rotation_angle(R):
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def main():
    calibration = load_calibration("data/calib/00/calib.txt")
    K = get_intrinsic_matrix(calibration)

    image_paths = get_image_paths("data/sequences/00/image_0")
    ground_truth = load_poses("data/poses/00.txt")

    # Problematic pairs and nearby comparisons
    frame_indices = [363, 364, 365, 461, 462, 463, 465,
                     621, 622, 623, 643, 644, 645, 646, 700]

    print(
        f"{'Pair':<12}"
        f"{'Estimated R':>14}"
        f"{'Actual R':>14}"
        f"{'GT distance':>15}"
        f"{'Median flow':>15}"
    )

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

        try:
            R, t, pose_mask = recover_camera_pose(
                E, points1, points2, K, mask
            )
            estimated_angle = rotation_angle(R)
        except ValueError:
            estimated_angle = 0.0

        # Actual relative camera rotation
        R_gt = (
            ground_truth[i + 1, :3, :3].T
            @ ground_truth[i, :3, :3]
        )
        actual_angle = rotation_angle(R_gt)

        # Actual distance travelled between frames
        position1 = ground_truth[i, :3, 3]
        position2 = ground_truth[i + 1, :3, 3]
        actual_distance = np.linalg.norm(position2 - position1)

        # Median pixel displacement of matched features
        pixel_displacements = np.linalg.norm(
            points2 - points1, axis=1
        )
        median_flow = np.median(pixel_displacements)

        print(
            f"{f'{i} → {i + 1}':<12}"
            f"{estimated_angle:>13.4f}°"
            f"{actual_angle:>13.4f}°"
            f"{actual_distance:>14.4f} m"
            f"{median_flow:>14.2f} px"
        )


if __name__ == "__main__":
    main()

