import numpy as np

# Since our estimated positions are not in distance units but ground-truth is, we need a scale factor to bring them both to the same scale
# Compute a single global scale factor that minimizes the squared position differences between our estimated positions and KITTI's ground-truth positions

# Align estimated camera positions to ground truth using one global scale factor
def align_scale(estimated_positions, ground_truth_positions):

    # Express both trajectories relative to their starting positions at frame 0
    estimated = estimated_positions - estimated_positions[0]
    ground_truth = ground_truth_positions - ground_truth_positions[0]

    # Calculate the global scale factor that minimizes the squared position differences between the estimated trajectory and ground truth across all frames
    numerator = np.sum(estimated * ground_truth)
    denominator = np.sum(estimated ** 2)
    if denominator == 0:  # zero denominator means the estimated camera never moved from its starting position
        raise ValueError("Cannot align a trajectory with zero displacement.")
    scale = numerator / denominator

    # Apply the same scale factor to every estimated camera position to align the scale without changing the estimated trajectory's shape
    aligned_positions = scale * estimated  # aligned_positions is still relative to frame 0

    return aligned_positions, scale


# Absolute Trajectory Error (ATE) measures how far our estimated camera positions are from the actual camera positions provided by KITTI ground truth
# Calculate the root mean squared ATE
def calculate_ate(aligned_positions, ground_truth_positions):

    # Express ground-truth positions relative to frame 0, matching the aligned trajectory
    ground_truth = ground_truth_positions - ground_truth_positions[0]

    # Calculate the 3D Euclidean distance between the estimated and actual camera positions at each frame i
    errors = np.linalg.norm(aligned_positions - ground_truth, axis=1)

    # Calculate the root mean squared absolute trajectory error in metres
    ate_rmse = np.sqrt(np.mean(errors ** 2))

    return ate_rmse
