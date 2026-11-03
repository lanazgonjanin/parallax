import numpy as np

from src.dataset import load_poses, load_timestamps, get_image_paths


def test_load_poses(tmp_path):
    pose_file = tmp_path / "poses.txt"

    # KITTI stores each 3x4 pose matrix as 12 values
    pose_file.write_text("1 0 0 4 0 1 0 5 0 0 1 6\n")  # create temporary text file containing one example KITTI camera pose

    poses = load_poses(str(pose_file))

    assert poses.shape == (1, 4, 4)
    np.testing.assert_allclose(poses[0, :3, :3], np.eye(3))
    np.testing.assert_allclose(poses[0, :3, 3], [4, 5, 6])
    np.testing.assert_allclose(poses[0, 3], [0, 0, 0, 1])


def test_load_timestamps(tmp_path):
    timestamp_file = tmp_path / "times.txt"
    timestamp_file.write_text("0.0\n0.1\n0.2\n")

    timestamps = load_timestamps(str(timestamp_file))

    np.testing.assert_allclose(timestamps, [0.0, 0.1, 0.2])


def test_get_image_paths(tmp_path):
    (tmp_path / "000002.png").touch()
    (tmp_path / "000000.png").touch()
    (tmp_path / "000001.png").touch()
    (tmp_path / "notes.txt").touch()

    paths = get_image_paths(str(tmp_path))

    filenames = [
        str(path).split("/")[-1]
        for path in paths
    ]

    assert filenames == [
        "000000.png",
        "000001.png",
        "000002.png",
    ]

