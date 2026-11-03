import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import (
    load_calibration,
    get_intrinsic_matrix,
    get_image_paths,
    load_image,
    load_poses,
)
from src.features.orb import detect_orb_features
from src.features.matching import match_features
from src.geometry.essential import (
    extract_matched_points,
    estimate_essential_matrix,
    recover_camera_pose,
)
from src.geometry.trajectory import update_pose
from src.geometry.temporal import (
    rotation_angle,
    detect_rotation_outliers,
    correct_rotation_outliers,
)
from src.evaluation import align_scale, calculate_ate


def main():
    num_frames = 1000
    window = 5
    minimum_angle = 2.0
    median_multiplier = 3.0

    calibration = load_calibration("data/calib/00/calib.txt")
    K = get_intrinsic_matrix(calibration)
    image_paths = get_image_paths("data/sequences/00/image_0")[:num_frames]

    num_pairs = num_frames - 1

    rotations = [None] * num_pairs
    translations = [None] * num_pairs

    image1 = load_image(image_paths[0])
    keypoints1, descriptors1 = detect_orb_features(image1)

    # Estimate relative motion for every frame pair
    for i in tqdm(range(num_pairs), desc="Processing frames", unit="pair"):
        image2 = load_image(image_paths[i + 1])
        keypoints2, descriptors2 = detect_orb_features(image2)

        matches = match_features(descriptors1, descriptors2)

        if len(matches) < 5:
            raise ValueError(
                f"Insufficient matches between frames {i} and {i + 1}."
            )

        points1, points2 = extract_matched_points(
            keypoints1, keypoints2, matches
        )

        E, mask = estimate_essential_matrix(points1, points2, K)

        try:
            R, t, pose_mask = recover_camera_pose(
                E, points1, points2, K, mask
            )

            rotations[i] = R
            translations[i] = t

        except ValueError:
            # Keep the previous pose when recovery fails
            pass

        keypoints1, descriptors1 = keypoints2, descriptors2

    # Identify rotations inconsistent with neighbouring estimates
    flagged = detect_rotation_outliers(
        rotations,
        window=window,
        minimum_angle=minimum_angle,
        median_multiplier=median_multiplier,
    )

    # Replace suspicious rotations using neighbouring rotation vectors
    corrected_rotations = correct_rotation_outliers(
        rotations,
        flagged,
        window=window,
    )

    # Reconstruct both trajectories from the same motion estimates
    baseline_pose = np.eye(4)
    corrected_pose = np.eye(4)

    baseline_trajectory = [baseline_pose.copy()]
    corrected_trajectory = [corrected_pose.copy()]

    for i in range(num_pairs):
        if rotations[i] is not None:
            baseline_pose = update_pose(
                baseline_pose, rotations[i], translations[i]
            )
            corrected_pose = update_pose(
                corrected_pose, corrected_rotations[i], translations[i]
            )

        baseline_trajectory.append(baseline_pose.copy())
        corrected_trajectory.append(corrected_pose.copy())

    baseline_trajectory = np.array(baseline_trajectory)
    corrected_trajectory = np.array(corrected_trajectory)

    # Evaluate both trajectories against KITTI ground truth
    ground_truth = load_poses("data/poses/00.txt")[:num_frames]
    ground_truth_positions = ground_truth[:, :3, 3]

    baseline_positions = baseline_trajectory[:, :3, 3]
    corrected_positions = corrected_trajectory[:, :3, 3]

    aligned_baseline, baseline_scale = align_scale(
        baseline_positions, ground_truth_positions
    )
    aligned_corrected, corrected_scale = align_scale(
        corrected_positions, ground_truth_positions
    )

    baseline_ate = calculate_ate(
        aligned_baseline, ground_truth_positions
    )
    corrected_ate = calculate_ate(
        aligned_corrected, ground_truth_positions
    )

    # Compare final accumulated orientation errors
    R_gt = ground_truth[-1, :3, :3]

    baseline_rotation_error = rotation_angle(
        R_gt.T @ baseline_trajectory[-1, :3, :3]
    )
    corrected_rotation_error = rotation_angle(
        R_gt.T @ corrected_trajectory[-1, :3, :3]
    )

    print(f"\nFlagged rotations: {np.count_nonzero(flagged)}")
    print(f"Flagged frame pairs: {np.flatnonzero(flagged).tolist()}")

    print(f"\nBaseline scale factor: {baseline_scale:.4f}")
    print(f"Corrected scale factor: {corrected_scale:.4f}")

    print(f"\nBaseline ATE RMSE: {baseline_ate:.4f} metres")
    print(f"Corrected ATE RMSE: {corrected_ate:.4f} metres")

    print(
        f"\nBaseline final rotation error: "
        f"{baseline_rotation_error:.4f} degrees"
    )
    print(
        f"Corrected final rotation error: "
        f"{corrected_rotation_error:.4f} degrees"
    )

    # Visualize the original and corrected trajectories
    ground_truth_relative = (
        ground_truth_positions - ground_truth_positions[0]
    )

    plt.figure(figsize=(9, 9))

    plt.plot(
        ground_truth_relative[:, 0],
        ground_truth_relative[:, 2],
        label="KITTI ground truth",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        aligned_baseline[:, 0],
        aligned_baseline[:, 2],
        label="Original baseline",
    )

    plt.plot(
        aligned_corrected[:, 0],
        aligned_corrected[:, 2],
        label="Temporal rotation correction",
    )

    plt.xlabel("X position (metres)")
    plt.ylabel("Z position (metres)")
    plt.title("Temporal Rotation Correction: 1,000 Frames")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        "results/temporal_rotation_correction_1000.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
