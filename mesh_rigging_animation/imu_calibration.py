#!/usr/bin/env python3
"""
Quaternion math primitives for IMU-to-BVH conversion.

Provides the three functions imported by c3d_to_bvh:
    - quaternion_multiply(): Hamilton product of two [w, x, y, z] quaternions
    - quaternion_inverse():  Conjugate inverse for unit quaternions
    - quaternion_to_euler(): Quaternion to Euler angles via scipy (any order)
"""

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions: q_result = q1 * q2.

    Quaternion format: [w, x, y, z]

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]

    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion inverse (conjugate for unit quaternions).

    For unit quaternion q = [w, x, y, z], inverse is q^-1 = [w, -x, -y, -z]

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Inverse quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_to_euler(q: np.ndarray, order: str = 'ZXY') -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (in degrees).

    BVH files typically use ZXY rotation order (intrinsic rotations).

    Args:
        q: Quaternion [w, x, y, z]
        order: Euler angle order (default 'ZXY' for BVH)

    Returns:
        Tuple of (angle1, angle2, angle3) in degrees
    """
    # scipy.Rotation uses [x, y, z, w] quaternion order
    q_scipy = np.array([q[1], q[2], q[3], q[0]])
    rot = Rotation.from_quat(q_scipy)
    euler = rot.as_euler(order.lower(), degrees=True)
    return tuple(euler)
