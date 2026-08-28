"""
calibration_utils.py
====================
Helpers for loading a calibration.yaml produced by calibrate.py and
adapting it to any target image resolution.

Typical usage in downstream code:

    from calibration_utils import load_calibration
    K, D, R, t = load_calibration('calibration.yaml',
                                   image_width=640, image_height=480)
    # K is already scaled to 640×480; D, R, t are unchanged.
"""

import numpy as np
import yaml


def load_calibration(path: str,
                     image_width: int  = None,
                     image_height: int = None):
    """
    Load a calibration.yaml and return intrinsics/extrinsics ready to use.

    Parameters
    ----------
    path         : path to calibration.yaml
    image_width  : width  of the images you will be processing (pixels)
    image_height : height of the images you will be processing (pixels)
                   Pass None (or omit) to get K at the original calibration
                   resolution with no scaling applied.

    Returns
    -------
    K  : (3, 3) float64  — camera intrinsic matrix scaled to the target resolution
    D  : (N,)   float64  — distortion coefficients (resolution-independent)
    R  : (3, 3) float64  — rotation    LiDAR → camera  (resolution-independent)
    t  : (3,)   float64  — translation LiDAR → camera  (resolution-independent)

    Notes
    -----
    To project a LiDAR point p_L into the image:

        import cv2, numpy as np
        rvec, _ = cv2.Rodrigues(R)
        pts2d, _ = cv2.projectPoints(p_L.reshape(-1, 1, 3), rvec,
                                      t.reshape(3, 1), K, D)
    """
    with open(path) as f:
        doc = yaml.safe_load(f)

    K = np.array(doc['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
    D = np.array(doc['dist_coeffs']['data'],   dtype=np.float64).flatten()
    R = np.array(doc['R_cam_from_lidar']['data'], dtype=np.float64).reshape(3, 3)
    t = np.array(doc['t_cam_from_lidar']['data'], dtype=np.float64).flatten()

    calib_w = doc.get('image_width')
    calib_h = doc.get('image_height')

    if image_width is not None and image_height is not None:
        if calib_w is None or calib_h is None:
            raise ValueError(
                f"{path} does not contain image_width / image_height. "
                "Re-save with the current version of calibrate.py.")
        if (image_width, image_height) != (calib_w, calib_h):
            sx = image_width  / calib_w
            sy = image_height / calib_h
            if abs(sx - sy) > 1e-4:
                raise ValueError(
                    f"Non-uniform scale: calibration was {calib_w}×{calib_h}, "
                    f"target is {image_width}×{image_height}  "
                    f"(sx={sx:.4f}, sy={sy:.4f}). Aspect ratios must match.")
            K = K.copy()
            K[0] *= sx   # fx, cx
            K[1] *= sy   # fy, cy

    return K, D, R, t


def scale_K(K: np.ndarray, src_w: int, src_h: int,
            dst_w: int, dst_h: int) -> np.ndarray:
    """
    Scale an intrinsic matrix from one resolution to another.
    Useful when you already have K in hand and just need to rescale it.
    """
    sx = dst_w / src_w
    sy = dst_h / src_h
    K_new = K.copy()
    K_new[0] *= sx
    K_new[1] *= sy
    return K_new
