# Parallax: Monocular Visual Odometry for Autonomous Driving

Parallax is a Python-based monocular visual odometry system that estimates camera motion and reconstructs vehicle trajectories from sequences of images. Using classical computer vision techniques, including feature detection and matching, epipolar geometry, and robust motion estimation, the system tracks changes in camera position and orientation without relying on GPS.

Developed using OpenCV and evaluated on the KITTI Visual Odometry dataset, the project explores the accuracy, computational efficiency, and limitations of vision-based localisation for autonomous driving applications.

## Feature Detection and Matching

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

where \(R\) is the \(3 \times 3\) rotation matrix, \(\mathbf{t}\) is the translation vector, and \([\mathbf{t}]_\times\) is the skew-symmetric matrix representing the cross product with \(\mathbf{t}\).

The estimated essential matrix is decomposed to recover the camera's relative rotation and translation direction. Using the matched pixel coordinates and a candidate pose, the 3D positions of corresponding features can be estimated through triangulation, allowing us to check whether they have positive depth in both camera views. Since multiple camera poses can satisfy the same essential matrix, the pose that places the most reconstructed 3D points in front of the camera in both frames is selected. As a result, some matches previously classified as RANSAC inliers may fail this additional geometric check and are excluded from the final set of inliers.

For the first two frames of the driving sequence, 699 of the 868 RANSAC inliers passed pose recovery validation. The recovered rotation was close to the identity matrix, indicating minimal change in orientation, while the estimated camera movement was predominantly forward. Because monocular visual odometry cannot determine absolute scale from two images alone, the translation is recovered as a direction rather than a distance.

## Dataset

This project uses the [KITTI Visual Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) to develop and evaluate camera-based vehicle localization algorithms.

The dataset is used for educational and research purposes and is not redistributed through this repository. Users must download the data separately from the official KITTI website and comply with its licensing and usage terms.

**Reference:**

Geiger, A., Lenz, P., & Urtasun, R. (2012). *Are we ready for autonomous driving? The KITTI Vision Benchmark Suite.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

## Licence

The original source code in this repository is licensed under the MIT Licence. See the [LICENSE](LICENSE) file for details.

The KITTI dataset and any third-party materials are subject to their respective licences and are not covered by this project's MIT Licence.


