import numpy as np
import matplotlib.pyplot as plt

from src.dataset import load_poses
from src.odometry import run_visual_odometry


def main():

    num_frames = 1000

    trajectory = run_visual_odometry(
        calibration_path="data/calib/00/calib.txt",
        image_dir="data/sequences/00/image_0",
        num_frames=num_frames,
    )

    ground_truth = load_poses("data/poses/00.txt")[:num_frames]

    # Extract accumulated camera orientations
    estimated_rotations = trajectory[:, :3, :3]
    ground_truth_rotations = ground_truth[:, :3, :3]

    rotation_errors = []

    for i in range(num_frames - 1):

        # Calculate frame-to-frame rotations
        estimated_relative = estimated_rotations[i].T @ estimated_rotations[i + 1]
        ground_truth_relative = ground_truth_rotations[i].T @ ground_truth_rotations[i + 1]

        # Calculate the difference between estimated and ground-truth rotations
        error_rotation = ground_truth_relative.T @ estimated_relative

        # Convert the rotation difference into an angle in degrees
        cos_angle = (np.trace(error_rotation) - 1) / 2
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        error_angle = np.degrees(np.arccos(cos_angle))

        rotation_errors.append(error_angle)

    rotation_errors = np.array(rotation_errors)

    print(f"Mean rotation error: {np.mean(rotation_errors):.4f} degrees")
    print(f"Median rotation error: {np.median(rotation_errors):.4f} degrees")
    print(f"Maximum rotation error: {np.max(rotation_errors):.4f} degrees")
    print(f"Rotation RMSE: {np.sqrt(np.mean(rotation_errors ** 2)):.4f} degrees")

    plt.figure(figsize=(12, 5))
    plt.plot(rotation_errors, linewidth=1)
    plt.xlabel("Frame pair (i → i+1)")
    plt.ylabel("Rotation error (degrees)")
    plt.title(f"Frame-to-Frame Rotation Error: First {num_frames} Frames")
    plt.grid(True)

    plt.savefig(
        f"results/rotation_error_{num_frames}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()

