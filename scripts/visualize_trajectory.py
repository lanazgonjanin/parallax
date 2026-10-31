import matplotlib.pyplot as plt
import numpy as np

from src.dataset import load_poses
from src.odometry import run_visual_odometry
from src.evaluation import align_scale, calculate_ate


def main():

    num_frames = 100

    # Estimate the accumulated camera poses for the first 100 KITTI frames
    trajectory = run_visual_odometry(
        calibration_path="data/calib/00/calib.txt",
        image_dir="data/sequences/00/image_0",
        num_frames=num_frames,
    )

    # Extract the camera positions from the last column of each 4x4 pose
    positions = trajectory[:, :3, 3]

    # Extract the horizontal (x) and forward (z) coordinates
    x = positions[:, 0]
    z = positions[:, 2]

    # Create a top-down plot of the estimated camera trajectory
    plt.figure(figsize=(8, 8))
    plt.plot(x, z, label="Estimated trajectory", linewidth=2)

    # Mark the starting and ending camera positions
    plt.scatter(x[0], z[0], color="green", label="Start", zorder=3)
    plt.scatter(x[-1], z[-1], color="red", label="End", zorder=3)

    plt.xlabel("X position (arbitrary units)")
    plt.ylabel("Z position (arbitrary units)")
    plt.title("Estimated Camera Trajectory: KITTI Sequence 00, First 100 Frames")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/estimated_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close()


    # Load the corresponding KITTI ground-truth poses
    ground_truth = load_poses("data/poses/00.txt")[:num_frames]

    # Extract the ground-truth camera positions from the 4x4 pose matrices
    ground_truth_positions = ground_truth[:, :3, 3]

    # Align the estimated trajectory to KITTI ground truth using one global scale factor
    aligned_positions, scale = align_scale(positions, ground_truth_positions)

    # Express ground truth relative to frame 0 for comparison
    ground_truth_positions = ground_truth_positions - ground_truth_positions[0]

    # Extract the horizontal (x) and forward (z) coordinates
    estimated_x = aligned_positions[:, 0]
    estimated_z = aligned_positions[:, 2]

    ground_truth_x = ground_truth_positions[:, 0]
    ground_truth_z = ground_truth_positions[:, 2]

    # Create a top-down comparison of the estimated and ground-truth trajectories
    plt.figure(figsize=(8, 8))

    plt.plot(
        estimated_x,
        estimated_z,
        label="Estimated trajectory (scale-aligned)",
        linewidth=2,
    )

    plt.plot(
        ground_truth_x,
        ground_truth_z,
        label="KITTI ground truth",
        linewidth=2,
        linestyle="--",
    )

    # Mark the starting and ending positions of both trajectories
    plt.scatter(
        ground_truth_x[0],
        ground_truth_z[0],
        color="green",
        label="Start",
        zorder=3,
    )

    plt.scatter(
        estimated_x[-1],
        estimated_z[-1],
        color="red",
        label="Estimated end",
        zorder=3,
    )

    plt.scatter(
        ground_truth_x[-1],
        ground_truth_z[-1],
        color="purple",
        label="Ground-truth end",
        zorder=3,
    )

    plt.xlabel("X position (metres)")
    plt.ylabel("Z position (metres)")
    plt.title("Estimated vs. Ground-Truth Camera Trajectory: KITTI Sequence 00, First 100 Frames")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/trajectory_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Calculate the 3D ATE after global scale alignment
    ate_rmse = calculate_ate(aligned_positions, ground_truth_positions)
    print(f"Global scale factor: {scale:.4f}")
    print(f"Scale-aligned ATE RMSE: {ate_rmse:.4f} metres")


if __name__ == "__main__":
    main()