import numpy as np
import matplotlib.pyplot as plt

from src.dataset import load_poses
from src.odometry import run_visual_odometry
from src.evaluation import align_scale, calculate_ate


def main():
    
    num_frames = 1000

    trajectory = run_visual_odometry(calibration_path="data/calib/00/calib.txt", image_dir="data/sequences/00/image_0", num_frames=num_frames)

    estimated_positions = trajectory[:, :3, 3]  # get accumulated positions matrix
    estimated_displacements = np.diff(estimated_positions, axis=0)  # get estimated frame-to-frame translation vectors

    ground_truth = load_poses("data/poses/00.txt")[:num_frames]
    ground_truth_positions = ground_truth[:, :3, 3]  # get ground-truth positions

    aligned_positions, scale = align_scale(estimated_positions, ground_truth_positions)  # apply global scaling
    baseline_ate = calculate_ate(aligned_positions, ground_truth_positions)

    # Calculate actual frame-to-frame distances from ground truth
    ground_truth_displacements = np.diff(ground_truth_positions, axis=0)
    actual_distances = np.linalg.norm(ground_truth_displacements, axis=1)

    # Replace unit-length estimated displacements with actual distances
    # To keep estimated direction, scale estimated translation vector with actual distance, thus only changing the magnitude 
    corrected_displacements = estimated_displacements * actual_distances[:, None]

    corrected_positions = np.vstack([np.zeros(3), np.cumsum(corrected_displacements, axis=0)])  # resconstruct accumulated positions matrix
    corrected_ate = calculate_ate(corrected_positions, ground_truth_positions)

    print(f"Baseline global scale factor: {scale:.4f}")
    print(f"Baseline ATE RMSE: {baseline_ate:.4f} metres")
    print(f"Diagnostic ATE RMSE: {corrected_ate:.4f} metres")
    print(f"Zero-displacement pairs: {np.count_nonzero(np.linalg.norm(estimated_displacements, axis=1) < 1e-10)}")

    ground_truth_relative = ground_truth_positions - ground_truth_positions[0]  # express ground-truth positions relative to frame 0

    plt.figure(figsize=(9, 9))

    plt.plot(
        ground_truth_relative[:, 0],
        ground_truth_relative[:, 2],
        label="KITTI ground truth",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        aligned_positions[:, 0],
        aligned_positions[:, 2],
        label="Baseline (global scale)",
        linewidth=2,
    )

    plt.plot(
        corrected_positions[:, 0],
        corrected_positions[:, 2],
        label="Diagnostic (actual frame-to-frame distances)",
        linewidth=2,
    )

    plt.xlabel("X position (metres)")
    plt.ylabel("Z position (metres)")
    plt.title(f"Translation Scale Diagnostic: First {num_frames} Frames")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        f"results/scale_diagnostic_{num_frames}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()

