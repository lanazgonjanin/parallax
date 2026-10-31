# Parallax: Monocular Visual Odometry for Autonomous Driving

Parallax is a Python-based monocular visual odometry system that estimates camera motion and reconstructs vehicle trajectories from sequences of images. Using classical computer vision techniques, including feature detection and matching, epipolar geometry, and robust motion estimation, the system tracks changes in camera position and orientation without relying on GPS.

Developed using OpenCV and evaluated on the KITTI Visual Odometry dataset, the project explores the accuracy, computational efficiency, and limitations of vision-based localisation for autonomous driving applications.

## Methodology

### ORB Keypoint Detection

ORB (Oriented FAST and Rotated BRIEF) is used to detect distinctive image features and compute binary descriptors for matching across consecutive frames. The FAST algorithm detects keypoints by comparing pixel intensities in their surrounding neighbourhoods, while BRIEF generates binary descriptors through pairwise intensity comparisons around each keypoint. The following visualization shows the ORB keypoints detected in the first frame of a driving sequence.

![ORB keypoint detection](results/orb_keypoints.png)

### Feature Matching

ORB descriptors are matched between consecutive frames using a brute-force matcher. Each ORB descriptor is a 256-bit binary string representing the local visual appearance around a keypoint. The matcher compares every descriptor in one frame against those in the next, measuring their similarity using Hamming distance, which counts the number of differing bits between two binary descriptors:

$$
d_H(\mathbf{a},\mathbf{b}) = \sum_{i=1}^{256} \mathbb{1}(a_i \neq b_i)
$$

where $\mathbb{1}$ is the indicator function, equal to 1 when the corresponding bits differ and 0 otherwise. A smaller Hamming distance indicates greater descriptor similarity.

For each descriptor, the matcher identifies its nearest neighbour (the descriptor with the smallest Hamming distance). Cross-checking is then applied to retain only pairs where both descriptors are each other's nearest neighbours, reducing ambiguous matches.

The following visualization shows the 50 best descriptor matches between the first two frames of a driving sequence, with coloured lines connecting corresponding keypoints.

![ORB feature matching](results/orb_matches.png)

### Essential Matrix Estimation

ORB feature matching is not necessarily correct: two different image features might have similar-looking descriptors, causing the matcher to pair them incorrectly. To mitigate this, we estimate an essential matrix to describe the geometric relationship between two camera views, letting us identify matches consistent with the geometry of camera motion. For a feature detected in the first image, the essential matrix predicts an epipolar line in the second image. If the feature match is geometrically correct, its corresponding point in the second image should lie on or very close to that line. 

The essential matrix is estimated using RANSAC (Random Sample Consensus). RANSAC estimates the essential matrix by repeatedly sampling subsets of feature correspondences and evaluating how well the resulting models explain the remaining matches. Matches consistent with the estimated geometry are classified as *inliers*, while inconsistent matches are classified as *outliers*. A geometric error threshold of 1 pixel is used to distinguish inliers from outliers, allowing for small inaccuracies in feature localisation.

Using the first two frames of the same driving sequence, the implementation produced the following results:

| Metric | Result |
|---|---|
| Total ORB matches | 1,221 |
| RANSAC inliers | 868 |
| RANSAC outliers | 353 |
| Inlier ratio | 71.09% |

The following visualization shows a subset of feature correspondences classified as inliers and outliers between the first two frames.

![RANSAC inlier and outlier matches](results/ransac_matches.png)

Some visually correct feature correspondences may still be classified as outliers because RANSAC evaluates their consistency with the estimated epipolar geometry rather than their visual similarity. Small inaccuracies in feature localization can cause a correct match to exceed the 1-pixel geometric error threshold and be rejected.

### Camera Pose Recovery

The essential matrix encodes the relative rotation and translation direction between two camera views:

$$
E = [\mathbf{t}]_\times R
$$

where $R$ is the $3 \times 3$ rotation matrix, $\mathbf{t}$ is the $3 \times 1$ translation vector, and $[\mathbf{t}]_\times$ is the $3 \times 3$ skew-symmetric matrix constructed from $\mathbf{t}$.

The estimated essential matrix is decomposed to recover the camera's relative rotation and translation direction. Using the matched pixel coordinates and a candidate pose, the 3D positions of corresponding features can be estimated through triangulation, allowing us to check whether they have positive depth in both camera views. Since multiple camera poses can satisfy the same essential matrix, the pose that places the most reconstructed 3D points in front of the camera in both frames is selected. As a result, some matches previously classified as RANSAC inliers may fail this additional geometric check and are excluded from the final set of inliers.

For the first two frames of the driving sequence, 699 of the 868 RANSAC inliers passed pose recovery validation. The recovered rotation was close to the identity matrix, indicating minimal change in orientation, while the estimated camera movement was predominantly forward. Because monocular visual odometry cannot determine absolute scale from two images alone, the translation is recovered as a direction rather than a distance.

### Pose Accumulation

To reconstruct the camera's trajectory across multiple frames, the relative camera poses are accumulated into a common coordinate system, with the first camera frame serving as the origin.

For each pair of consecutive frames $i$ and $i+1$, the recovered rotation $R$ and translation direction $\mathbf{t}$ are combined into a homogeneous transformation matrix:

$$
T_{i \to i+1} =
\begin{bmatrix}
R & \mathbf{t} \\
\mathbf{0}^T & 1
\end{bmatrix}
$$

Since this transformation maps 3D point coordinates from frame $i$ to frame $i+1$, it is inverted before being combined with the existing accumulated camera pose:

$$
T_{i+1} = T_i T_{i \to i+1}^{-1}
$$

where $T_i$ represents the existing accumulated camera pose at frame $i$, relative to frame 0, and $T_0=I$. Note that the inversion is necessary because the recovered transformation maps coordinates into the new camera frame, whereas pose accumulation requires the reverse transformation to express the new camera's position and orientation relative to the previous frame.

Because monocular visual odometry cannot recover absolute translation scale, each estimated translation is treated as a unit vector. The resulting trajectory therefore represents camera motion in arbitrary units rather than metres.

## Trajectory Evaluation

### Global Scale Alignment

Since estimated trajectories are represented by arbitrary units, they must aligned to ground truth using a single global scale factor. Both trajectories are first expressed relative to their initial positions.

The scale factor $s$ is calculated by minimising the sum of squared 3D position differences:

$$
s = \frac{\sum_i \hat{\mathbf{p}}_i \cdot \mathbf{p}_i^{GT}}{\sum_i \|\hat{\mathbf{p}}_i\|^2}
$$

where $\hat{\mathbf{p}}_i$ is the estimated camera position at frame $i$ and $\mathbf{p}_i^{GT}$ is the corresponding ground-truth camera position, both relative to their initial positions at frame 0.

The scale factor $s$ represents the single multiplicative factor that best matches the estimated trajectory's overall scale to ground truth, converting its arbitrary displacement units into metres.

The aligned estimated positions are then calculated as:

$$
\hat{\mathbf{p}}_i^{aligned} = s\hat{\mathbf{p}}_i
$$

Ground truth is used only for scale alignment and evaluation, not for estimating camera motion.

### Absolute Trajectory Error

Trajectory accuracy is measured using scale-aligned Absolute Trajectory Error (ATE), calculated as the root mean squared Euclidean distance between corresponding estimated and ground-truth camera positions:

$$
\mathrm{ATE}_{RMSE} =
\sqrt{\frac{1}{N}\sum_{i=0}^{N-1}
\left\|\hat{\mathbf{p}}_i^{aligned}-\mathbf{p}_i^{GT}\right\|_2^2}
$$

ATE measures the overall positional accuracy of the reconstructed trajectory after global scale alignment. Lower values indicate closer agreement with ground truth.

## Results

The visual odometry pipeline was evaluated on the first 100 frames of KITTI Odometry sequence 00.

| Metric | Result |
|---|---|
| Frames evaluated | 100 |
| Global scale factor | 0.9220 |
| Scale-aligned ATE RMSE | 2.3665 m |

### Estimated Camera Trajectory

![Estimated camera trajectory](results/estimated_trajectory.png)

The estimated trajectory shows predominantly forward motion along the Z-axis, with smaller lateral displacement along the X-axis. Positions are expressed in arbitrary units because absolute translation scale is not observable from monocular images alone.

### Comparison with Ground Truth

![Estimated trajectory compared with KITTI ground truth](results/trajectory_comparison.png)

After global scale alignment, the estimated trajectory follows the overall direction of the KITTI ground-truth trajectory. However, the estimated positions gradually diverge from ground truth, particularly in the lateral direction, indicating accumulated trajectory error.

The scale-aligned ATE RMSE over the first 100 frames is 2.3665 metres. This means that after adjusting for scale, Parallax's estimated camera positions were off by about 2.37 metres overall compared to the actual positions across the first 100 frames.

These results establish an initial baseline for the monocular visual odometry pipeline. Evaluation over longer sequences is needed to better characterize accumulated drift and performance during more complex camera motion.

## Dataset

This project uses the [KITTI Visual Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) to develop and evaluate camera-based vehicle localization algorithms.

The dataset is used for educational and research purposes and is not redistributed through this repository. Users must download the data separately from the official KITTI website and comply with its licensing and usage terms.

**Reference:**

Geiger, A., Lenz, P., & Urtasun, R. (2012). *Are we ready for autonomous driving? The KITTI Vision Benchmark Suite.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

## Licence

The original source code in this repository is licensed under the MIT Licence. See the [LICENSE](LICENSE) file for details.

The KITTI dataset and any third-party materials are subject to their respective licences and are not covered by this project's MIT Licence.


