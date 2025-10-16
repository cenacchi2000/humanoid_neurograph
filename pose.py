# humanoid_neurograph/app/pose.py
import math
import numpy as np
import cv2

# Canonical 3D face points
FACE_3D_CANON_7 = np.array([
    [  0.0,   0.0,   0.0],   # nose tip
    [ -30.0,  35.0, -30.0],  # left eye center
    [  30.0,  35.0, -30.0],  # right eye center
    [ -25.0, -20.0, -30.0],  # left mouth corner
    [  25.0, -20.0, -30.0],  # right mouth corner
    [   0.0, -55.0, -20.0],  # chin
    [   0.0,  65.0, -20.0],  # forehead
], dtype=np.float64)
FACE_3D_CANON_5 = FACE_3D_CANON_7[:5].copy()

def solve_headpose(pts2d: np.ndarray, W: int, H: int):
    """Return yaw, pitch, roll (deg). Uses ITERATIVE if >=6 pts, else EPNP."""
    cam = np.array([[W*1.0, 0.0,   W/2.0],
                    [0.0,   W*1.0, H/2.0],
                    [0.0,   0.0,   1.0 ]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    if pts2d.shape[0] >= 6:
        flags = cv2.SOLVEPNP_ITERATIVE
        obj = FACE_3D_CANON_7[:pts2d.shape[0]]
    else:
        flags = cv2.SOLVEPNP_EPNP
        obj = FACE_3D_CANON_5[:pts2d.shape[0]]
    ok, rvec, _ = cv2.solvePnP(obj, pts2d, cam, dist, flags=flags)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    pitch = math.degrees(math.atan2(-R[2,0], sy))
    yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))
    roll  = math.degrees(math.atan2(R[2,1], R[2,2]))
    return yaw, pitch, roll
