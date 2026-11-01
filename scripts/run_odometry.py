from src.dataset import load_poses
from src.evaluation import align_scale, calculate_ate
from src.odometry import run_visual_odometry


# Testing Parallax on longer sequences of images
def main():

    calibration_path = "data/calib/00/calib.txt"  # sequence camera calibration
    image_dir = "data/sequences/00/image_0"  # sequence images
    poses_path = "data/poses/00.txt"  # sequence ground-truth poses

    frame_counts = [100, 250, 500, 1000]  # sequence lengths to be evaluated

    ground_truth_poses = load_poses(poses_path)

    # Run visual odometry for the longest trajectory (to optimize, shorter trajectories are evaluated using subsets of this)
    trajectory = run_visual_odometry(calibration_path, image_dir, num_frames=max(frame_counts))  # output is max(frame_counts) pose matrices

    # Extract camera positions at each frame from the pose matrices
    estimated_positions = trajectory[:, :3, 3]
    ground_truth_positions = ground_truth_poses[:, :3, 3]

    for num_frames in frame_counts:
        estimated = estimated_positions[:num_frames]
        ground_truth = ground_truth_positions[:num_frames]

        aligned_positions, scale = align_scale(estimated, ground_truth)  # find global scale factor

        ate_rmse = calculate_ate(aligned_positions, ground_truth)

        print(
            f"Frames: {num_frames:4d} | "
            f"Scale: {scale:.4f} | "
            f"ATE RMSE: {ate_rmse:.4f} m"
        )


if __name__ == "__main__":
    main()
