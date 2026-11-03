import numpy as np
import pytest

from src.evaluation import align_scale, calculate_ate


def test_scale_alignment():
    estimated = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
    ])

    ground_truth = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [4, 0, 0],
    ])

    aligned, scale = align_scale(estimated, ground_truth)

    assert scale == pytest.approx(2.0)
    np.testing.assert_allclose(aligned, ground_truth)


def test_scale_alignment_with_offset():
    estimated = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
    ])

    ground_truth = np.array([
        [10, 5, 0],
        [12, 5, 0],
        [14, 5, 0],
    ])

    aligned, scale = align_scale(estimated, ground_truth)

    assert scale == pytest.approx(2.0)
    np.testing.assert_allclose(
        aligned,
        [[0, 0, 0], [2, 0, 0], [4, 0, 0]],
    )


def test_ate_perfect_trajectory():
    positions = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [2, 0, 2],
    ])

    ate = calculate_ate(positions, positions)

    assert ate == pytest.approx(0.0)


def test_ate_with_error():
    estimated = np.array([
        [0, 0, 0],
        [2, 0, 0],
    ])

    ground_truth = np.array([
        [0, 0, 0],
        [3, 0, 0],
    ])

    ate = calculate_ate(estimated, ground_truth)

    assert ate == pytest.approx(np.sqrt(0.5))


def test_scale_alignment_zero_movement():
    estimated = np.zeros((3, 3))
    ground_truth = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
    ])

    with pytest.raises(ValueError):
        align_scale(estimated, ground_truth)

