import cv2
import numpy as np

from src.geometry.temporal import (
    rotation_angle,
    detect_rotation_outliers,
    correct_rotation_outliers,
)


# Create a rotation about the Y-axis
def make_rotation(angle):
    rotation_vector = np.array([0, np.radians(angle), 0])
    return cv2.Rodrigues(rotation_vector)[0]


# Verifies that a rotation matrix is converted into the correct angle
def test_rotation_angle():
    R = make_rotation(30)

    assert np.isclose(rotation_angle(R), 30)


# Checks that an isolated 8° rotation is flagged
def test_detect_rotation_spike():
    # One incorrect rotation surrounded by small rotations
    rotations = [
        make_rotation(angle)
        for angle in [0.3, 0.4, 0.3, 8.0, 0.4, 0.3, 0.4]
    ]

    flagged = detect_rotation_outliers(rotations, window=2)

    assert np.flatnonzero(flagged).tolist() == [3]


# Checks that sustained 3° rotations are not flagged
def test_preserve_genuine_turn():
    # Sustained large rotations representing a genuine turn
    rotations = [
        make_rotation(angle)
        for angle in [3.0, 3.2, 3.1, 3.3, 3.0, 3.2, 3.1]
    ]

    flagged = detect_rotation_outliers(rotations, window=2)

    assert not np.any(flagged)


# Verifies that a suspicious rotation is replaced using neighbouring rotations
def test_correct_rotation_spike():
    rotations = [
        make_rotation(angle)
        for angle in [0.4, 0.4, 0.4, 8.0, 0.4, 0.4, 0.4]
    ]

    flagged = detect_rotation_outliers(rotations, window=3)
    corrected = correct_rotation_outliers(rotations, flagged, window=3)

    # Incorrect 8-degree rotation should be replaced with 0.4 degrees
    assert np.isclose(rotation_angle(corrected[3]), 0.4)

    # Other rotations should remain unchanged
    for i in [0, 1, 2, 4, 5, 6]:
        np.testing.assert_allclose(corrected[i], rotations[i])


# Checks that failed pose recoveries are excluded from detection and correction
def test_exclude_failed_recoveries():
    rotations = [
        make_rotation(0.4),
        None,
        make_rotation(0.4),
        make_rotation(8.0),
        make_rotation(0.4),
        make_rotation(0.4),
        make_rotation(0.4),
    ]

    flagged = detect_rotation_outliers(rotations, window=3)
    corrected = correct_rotation_outliers(rotations, flagged, window=3)

    assert not flagged[1]
    assert corrected[1] is None
    assert flagged[3]
    assert np.isclose(rotation_angle(corrected[3]), 0.4)

