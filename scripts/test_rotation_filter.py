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
from src.evaluation import align_scale, calculate_ate


def rotation_angle(R):
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def main():
    num_frames = 1000
    rotation_threshold = 2.0  # degrees

    calibration = load_calibration("data/calib/00/calib.txt")
    K = get_intrinsic_matrix(calibration)
    image_paths = get_image_paths("data/sequences/00/image_0")[:num_frames]

    baseline_pose = np.eye(4)
    filtered_pose = np.eye(4)

    baseline_trajectory = [baseline_pose.copy()]
    filtered_trajectory = [filtered_pose.copy()]
    rejected_frames = []

    image1 = load_image(image_paths[0])
    keypoints1, descriptors1 = detect_orb_features(image1)

    for i in tqdm(range(num_frames - 1), desc="Processing frames", unit="pair"):
        image2 = load_image(image_paths[i + 1])
        keypoints2, descriptors2 = detect_orb_features(image2)

        matches = match_features(descriptors1, descriptors2)

        if len(matches) < 5:
            raise ValueError(f"Insufficient matches between frames {i} and {i + 1}.")

        points1, points2 = extract_matched_points(
            keypoints1, keypoints2, matches
        )

        E, mask = estimate_essential_matrix(points1, points2, K)

        try:
            R, t, pose_mask = recover_camera_pose(
                E, points1, points2, K, mask
            )

            # Original pose accumulation
            baseline_pose = update_pose(baseline_pose, R, t)

            # Experimental pose accumulation
            if rotation_angle(R) > rotation_threshold:
                filtered_R = np.eye(3)
                rejected_frames.append(i)
            else:
                filtered_R = R

            filtered_pose = update_pose(filtered_pose, filtered_R, t)

        except ValueError:
            tqdm.write(
                f"Pose recovery failed between frames {i} and {i + 1}."
            )

        baseline_trajectory.append(baseline_pose.copy())
        filtered_trajectory.append(filtered_pose.copy())

        keypoints1, descriptors1 = keypoints2, descriptors2

    baseline_trajectory = np.array(baseline_trajectory)
    filtered_trajectory = np.array(filtered_trajectory)

    ground_truth = load_poses("data/poses/00.txt")[:num_frames]
    ground_truth_positions = ground_truth[:, :3, 3]

    baseline_positions = baseline_trajectory[:, :3, 3]
    filtered_positions = filtered_trajectory[:, :3, 3]

    # Evaluate both trajectories using global scale alignment
    aligned_baseline, baseline_scale = align_scale(
        baseline_positions, ground_truth_positions
    )
    aligned_filtered, filtered_scale = align_scale(
        filtered_positions, ground_truth_positions
    )

    baseline_ate = calculate_ate(
        aligned_baseline, ground_truth_positions
    )
    filtered_ate = calculate_ate(
        aligned_filtered, ground_truth_positions
    )

    # Compare final accumulated camera orientations
    R_gt = ground_truth[-1, :3, :3]

    baseline_rotation_error = rotation_angle(
        R_gt.T @ baseline_trajectory[-1, :3, :3]
    )
    filtered_rotation_error = rotation_angle(
        R_gt.T @ filtered_trajectory[-1, :3, :3]
    )

    print(f"\nRejected rotations: {len(rejected_frames)}")
    print(f"Rejected frame pairs: {rejected_frames}")

    print(f"\nBaseline scale factor: {baseline_scale:.4f}")
    print(f"Filtered scale factor: {filtered_scale:.4f}")

    print(f"\nBaseline ATE RMSE: {baseline_ate:.4f} metres")
    print(f"Filtered ATE RMSE: {filtered_ate:.4f} metres")

    print(f"\nBaseline final rotation error: {baseline_rotation_error:.4f} degrees")
    print(f"Filtered final rotation error: {filtered_rotation_error:.4f} degrees")

    # Compare trajectories
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
        aligned_filtered[:, 0],
        aligned_filtered[:, 2],
        label="Rotation-filtered trajectory",
    )

    plt.xlabel("X position (metres)")
    plt.ylabel("Z position (metres)")
    plt.title("Rotation Filtering Experiment: 1,000 Frames")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        "results/rotation_filter_comparison_1000.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()

