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


def rotation_angle(R):
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def main():
    num_frames = 1000
    window = 5
    minimum_angle = 2.0
    median_multiplier = 3.0
    error_threshold = 2.0

    calibration = load_calibration("data/calib/00/calib.txt")
    K = get_intrinsic_matrix(calibration)
    image_paths = get_image_paths("data/sequences/00/image_0")[:num_frames]
    ground_truth = load_poses("data/poses/00.txt")[:num_frames]

    num_pairs = num_frames - 1

    estimated_angles = np.full(num_pairs, np.nan)
    actual_angles = np.zeros(num_pairs)
    rotation_errors = np.full(num_pairs, np.nan)

    image1 = load_image(image_paths[0])
    keypoints1, descriptors1 = detect_orb_features(image1)

    # Estimate relative rotations for every frame pair
    for i in tqdm(range(num_pairs), desc="Processing frames", unit="pair"):
        image2 = load_image(image_paths[i + 1])
        keypoints2, descriptors2 = detect_orb_features(image2)

        matches = match_features(descriptors1, descriptors2)

        if len(matches) < 5:
            raise ValueError(f"Insufficient matches between frames {i} and {i + 1}.")

        points1, points2 = extract_matched_points(
            keypoints1, keypoints2, matches
        )

        E, mask = estimate_essential_matrix(points1, points2, K)

        # KITTI rotation from camera i to camera i+1
        R_gt = (
            ground_truth[i + 1, :3, :3].T
            @ ground_truth[i, :3, :3]
        )
        actual_angles[i] = rotation_angle(R_gt)

        try:
            R, t, pose_mask = recover_camera_pose(
                E, points1, points2, K, mask
            )

            estimated_angles[i] = rotation_angle(R)
            rotation_errors[i] = rotation_angle(R_gt.T @ R)

        except ValueError:
            # Failed pose recoveries are excluded from this diagnostic
            pass

        keypoints1, descriptors1 = keypoints2, descriptors2

    # Compare each estimated rotation with neighbouring estimates
    nearby_medians = np.full(num_pairs, np.nan)
    flagged = np.zeros(num_pairs, dtype=bool)

    for i in range(num_pairs):
        if not np.isfinite(estimated_angles[i]):
            continue

        before = estimated_angles[max(0, i - window):i]
        after = estimated_angles[i + 1:min(num_pairs, i + window + 1)]

        neighbours = np.concatenate([before, after])
        neighbours = neighbours[np.isfinite(neighbours)]

        if len(neighbours) == 0:
            continue

        nearby_medians[i] = np.median(neighbours)

        threshold = max(
            minimum_angle,
            median_multiplier * nearby_medians[i],
        )

        flagged[i] = estimated_angles[i] > threshold

    # Use KITTI only to evaluate the detector
    valid = np.isfinite(rotation_errors)
    actual_errors = valid & (rotation_errors > error_threshold)

    true_positives = np.count_nonzero(flagged & actual_errors)
    false_positives = np.count_nonzero(flagged & ~actual_errors)
    false_negatives = np.count_nonzero(~flagged & actual_errors)
    true_negatives = np.count_nonzero(valid & ~flagged & ~actual_errors)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0 else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0 else 0.0
    )

    print(f"\nValid rotation estimates: {np.count_nonzero(valid)}")
    print(f"Failed pose recoveries: {np.count_nonzero(~valid)}")
    print(f"Flagged rotations: {np.count_nonzero(flagged)}")
    print(f"Actual rotation errors above {error_threshold:.1f}°: "
          f"{np.count_nonzero(actual_errors)}")

    print(f"\nTrue positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"True negatives: {true_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print("\nFlagged frame pairs:")
    for i in np.flatnonzero(flagged):
        print(
            f"{i} → {i + 1}: "
            f"estimated={estimated_angles[i]:.4f}°, "
            f"nearby median={nearby_medians[i]:.4f}°, "
            f"GT error={rotation_errors[i]:.4f}°"
        )

    # Plot the estimated rotations and suspicious frame pairs
    plt.figure(figsize=(12, 5))

    plt.plot(
        estimated_angles,
        label="Estimated rotation",
        linewidth=1,
    )
    plt.plot(
        nearby_medians,
        label="Nearby median",
        linewidth=1,
    )
    plt.scatter(
        np.flatnonzero(flagged),
        estimated_angles[flagged],
        color="red",
        s=22,
        label="Flagged rotation",
        zorder=3,
    )

    plt.xlabel("Frame pair (i → i+1)")
    plt.ylabel("Rotation angle (degrees)")
    plt.title("Temporal Rotation Consistency: First 1,000 Frames")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        "results/rotation_consistency_1000.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()

