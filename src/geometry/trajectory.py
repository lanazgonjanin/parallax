import numpy as np

# Update the accumulated camera pose using the newest relative motion
# Lets us store pose info at frame i+1 using motion info from frames 0 to i
def update_pose(
    current_pose: np.ndarray,  # contains all the accumulated movement from frame 0 to i
    R: np.ndarray,  # relative rotation between frame i and i+1
    t: np.ndarray,  # relative translation direction between frame i and i+1
) -> np.ndarray:

    # Build the transformation describing how 3D points change from frame i to i+1
    relative_transform = np.eye(4)  # create 4x4 identity matrix
    relative_transform[:3, :3] = R
    relative_transform[:3, 3] = t.ravel()  # .ravel() converts t into a 1D array

    # Invert it to obtain the current camera's pose relative to the previous camera frame

    # Invert the transformation to express the pose of camera i+1 relative to camera i
    inverse_transform = np.linalg.inv(relative_transform)

    # Combine the accumulated pose at frame i (relative to frame 0) with the motion from frame i to i+1
    new_pose = current_pose @ inverse_transform  # contains all accumulated movement from frame 0 to i+1

    return new_pose

