## Project Structure

### Core Implementation

| File | Description |
|---|---|
| `src/dataset.py` | Loads KITTI camera calibration, ground-truth poses, timestamps, and images. |
| `src/features/orb.py` | Detects ORB keypoints and computes feature descriptors. |
| `src/features/matching.py` | Matches ORB descriptors between consecutive frames using brute-force Hamming distance. |
| `src/geometry/essential.py` | Estimates the essential matrix using RANSAC and recovers relative camera rotation and translation direction. |
| `src/geometry/trajectory.py` | Accumulates relative camera transformations to update the estimated camera pose. |
| `src/odometry.py` | Runs the complete monocular visual odometry pipeline over a sequence of images. |
| `src/evaluation.py` | Performs global scale alignment and calculates Absolute Trajectory Error (ATE). |

### Evaluation and Diagnostic Scripts

| File | Description |
|---|---|
| `scripts/run_odometry.py` | Runs the visual odometry pipeline and evaluates trajectory accuracy over different frame ranges. |
| `scripts/visualize_trajectory.py` | Generates estimated trajectory plots and comparisons against KITTI ground truth. |
| `scripts/inspect_frames.py` | Visualizes selected KITTI frames to investigate pose recovery failures. |
| `scripts/diagnose_scale.py` | Reconstructs a trajectory using estimated frame-to-frame motion scaled by ground-truth distances to investigate translation scale errors. |
| `scripts/diagnose_rotation.py` | Measures frame-to-frame estimated rotation error against KITTI ground truth. |
| `scripts/diagnose_accumulated_rotation.py` | Measures accumulated estimated camera orientation error over the sequence. |
| `scripts/inspect_rotation_errors.py` | Examines feature-match counts, RANSAC inliers, pose inliers, and rotation errors at selected frame pairs. |
| `scripts/inspect_problematic_motion.py` | Compares estimated and actual rotations, ground-truth displacement, and feature movement at problematic frames. |
| `scripts/test_rotation_filter.py` | Tests an experimental fixed rotation threshold and evaluates its effect on trajectory accuracy. |
| `scripts/inspect_rotation_context.py` | Examines surrounding images and neighbouring rotation estimates to distinguish rotation errors from actual turns. |
| `scripts/evaluate_rotation_consistency.py` | Implement and test a temporal consistency rule across the sequence to identify suspicious rotation estimates. |
| `scripts/test_temporal_rotation_correction.py` | Identifies inconsistent rotation estimates using neighbouring frames, replaces suspicious rotations with estimates derived from surrounding motion, and evaluates the corrected trajectory against the original baseline. |
| `scripts/validate_temporal_correction.py` | Evaluates temporal rotation correction on an unseen segment of KITTI sequence 00, comparing trajectory accuracy and final accumulated rotation error against the original pipeline. |

The diagnostic scripts are separate from the core visual odometry pipeline. KITTI ground truth is used for evaluation and analysis, not for estimating camera motion in the baseline implementation.

### Data and Outputs

| Directory | Contents |
|---|---|
| `data/calib/00/` | Camera calibration and timestamps for KITTI sequence 00. |
| `data/poses/` | KITTI ground-truth camera poses. |
| `data/sequences/00/image_0/` | Left-camera images for KITTI sequence 00, accessed through a local link to external storage. |
| `results/` | Generated trajectory plots, diagnostic visualisations, and evaluation data. |

### Automated Tests

| File | Description |
|---|---|
| `test/test_geometry.py` | Tests camera pose updates, including identity transformations, translation, rotation, and accumulation of consecutive poses. |
| `test/test_evaluation.py` | Tests global scale alignment and Absolute Trajectory Error (ATE) calculations, including edge cases. |
| `test/test_dataset.py` | Tests KITTI pose loading, timestamp loading, and image path ordering using synthetic data and temporary files. |

Tests are implemented using `pytest` and verify core functionality without requiring the KITTI dataset or external storage.