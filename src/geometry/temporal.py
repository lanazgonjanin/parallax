import cv2
import numpy as np


def rotation_angle(R):
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


# Identifying suspicious rotation estimates
def detect_rotation_outliers(rotations, window=5, minimum_angle=2.0, median_multiplier=3.0):
    num_pairs = len(rotations)

    estimated_angles = np.array([
        rotation_angle(R) if R is not None else np.nan
        for R in rotations
    ])

    flagged = np.zeros(num_pairs, dtype=bool)

    for i in range(num_pairs):
        if not np.isfinite(estimated_angles[i]):
            continue

        before = estimated_angles[max(0, i - window):i]
        after = estimated_angles[i + 1:min(num_pairs, i + window + 1)]

        neighbours = np.concatenate([before, after])
        neighbours = neighbours[np.isfinite(neighbours)]

        if len(neighbours) == 0:
            continue

        nearby_median = np.median(neighbours)
        threshold = max(minimum_angle, median_multiplier * nearby_median)

        flagged[i] = estimated_angles[i] > threshold

    return flagged


# Replacing suspicious rotations using neighbouring estimates
def correct_rotation_outliers(rotations, flagged, window=5):
    num_pairs = len(rotations)
    corrected_rotations = rotations.copy()

    for i in np.flatnonzero(flagged):
        nearby_indices = list(range(max(0, i - window), i))
        nearby_indices += list(range(i + 1, min(num_pairs, i + window + 1)))

        # Exclude failed recoveries and other flagged rotations
        nearby_indices = [
            j for j in nearby_indices
            if rotations[j] is not None and not flagged[j]
        ]

        if len(nearby_indices) < 3:
            continue

        rotation_vectors = np.array([
            cv2.Rodrigues(rotations[j])[0].ravel()
            for j in nearby_indices
        ])

        median_rotation_vector = np.median(rotation_vectors, axis=0)
        corrected_rotations[i] = cv2.Rodrigues(median_rotation_vector)[0]

    return corrected_rotations

