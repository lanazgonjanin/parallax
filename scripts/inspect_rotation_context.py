import numpy as np
import matplotlib.pyplot as plt

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
    calibration = load_calibration("data/calib/00/calib.txt")
    K = get_intrinsic_matrix(calibration)

    image_paths = get_image_paths("data/sequences/00/image_0")
    ground_truth = load_poses("data/poses/00.txt")

    # Known inaccurate rotations
    problematic_frames = [364, 462, 465, 622, 644, 645]

    # Candidate genuine turns from sustained high-rotation intervals
    turning_frames = [109, 423, 741]

    frame_indices = problematic_frames + turning_frames

    # Store results to avoid processing overlapping frame pairs twice
    rotation_cache = {}

    def get_rotation_angles(i):
        if i in rotation_cache:
            return rotation_cache[i]

        image1 = load_image(image_paths[i])
        image2 = load_image(image_paths[i + 1])

        keypoints1, descriptors1 = detect_orb_features(image1)
        keypoints2, descriptors2 = detect_orb_features(image2)

        matches = match_features(descriptors1, descriptors2)

        try:
            points1, points2 = extract_matched_points(
                keypoints1, keypoints2, matches
            )

            E, mask = estimate_essential_matrix(points1, points2, K)
            R, t, pose_mask = recover_camera_pose(
                E, points1, points2, K, mask
            )

            estimated_angle = rotation_angle(R)

        except ValueError:
            estimated_angle = np.nan

        # Ground-truth relative rotation
        R_gt = (
            ground_truth[i + 1, :3, :3].T
            @ ground_truth[i, :3, :3]
        )
        actual_angle = rotation_angle(R_gt)

        rotation_cache[i] = (estimated_angle, actual_angle)
        return rotation_cache[i]

    print(
        f"{'Pair':<14}"
        f"{'Estimated':>12}"
        f"{'GT':>12}"
        f"{'Nearby est.':>15}"
        f"{'Nearby GT':>15}"
    )

    for i in frame_indices:
        estimated_angle, actual_angle = get_rotation_angles(i)

        # Examine five frame pairs before and five after
        nearby_indices = list(range(i - 5, i)) + list(range(i + 1, i + 6))
        nearby_results = [get_rotation_angles(j) for j in nearby_indices]

        nearby_estimated = np.array([r[0] for r in nearby_results])
        nearby_actual = np.array([r[1] for r in nearby_results])

        print(
            f"{f'{i} → {i + 1}':<14}"
            f"{estimated_angle:>11.4f}°"
            f"{actual_angle:>11.4f}°"
            f"{np.nanmedian(nearby_estimated):>14.4f}°"
            f"{np.median(nearby_actual):>14.4f}°"
        )

    # Visualise the images surrounding each frame pair
    fig, axes = plt.subplots(
        len(frame_indices),
        4,
        figsize=(16, 18),
    )

    for row, i in enumerate(frame_indices):
        frames = [i - 20, i, i + 1, i + 20]
        labels = ["20 before", "Frame i", "Frame i+1", "20 after"]

        for col, (frame, label) in enumerate(zip(frames, labels)):
            image = load_image(image_paths[frame])

            axes[row, col].imshow(image, cmap="gray")
            axes[row, col].set_title(f"{label}: {frame}", fontsize=10)
            axes[row, col].axis("off")

        category = "Spike" if i in problematic_frames else "Candidate turn"
        axes[row, 0].set_ylabel(
            f"{category}\n{i} → {i + 1}",
            fontsize=10,
        )

    fig.suptitle("Image Context Around Selected Rotations", fontsize=16)
    plt.tight_layout()

    plt.savefig(
        "results/rotation_context_contact_sheet.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()

