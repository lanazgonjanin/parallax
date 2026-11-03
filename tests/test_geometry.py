import numpy as np

from src.geometry.trajectory import update_pose


def test_identity_pose():
    current_pose = np.eye(4)
    R = np.eye(3)
    t = np.zeros((3, 1))

    new_pose = update_pose(current_pose, R, t)

    np.testing.assert_allclose(new_pose, np.eye(4))


def test_translation():
    current_pose = np.eye(4)
    R = np.eye(3)
    t = np.array([[0], [0], [1]])

    new_pose = update_pose(current_pose, R, t)

    np.testing.assert_allclose(
        new_pose[:3, 3],
        [0, 0, -1],
    )


def test_rotation():
    current_pose = np.eye(4)

    # 90-degree rotation about the Z-axis
    R = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ])

    t = np.zeros((3, 1))

    new_pose = update_pose(current_pose, R, t)

    np.testing.assert_allclose(new_pose[:3, :3], R.T)


def test_pose_accumulation():
    current_pose = np.eye(4)
    R = np.eye(3)
    t = np.array([[0], [0], [1]])

    pose1 = update_pose(current_pose, R, t)
    pose2 = update_pose(pose1, R, t)

    np.testing.assert_allclose(
        pose2[:3, 3],
        [0, 0, -2],
    )
