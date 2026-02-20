#!/usr/bin/env python3
"""
IMU calibration for aligning Cometa sensor coordinate frame to BVH skeleton frame.

This module computes T-pose calibration quaternions that transform sensor
orientations from the Cometa hardware coordinate frame into the BVH skeleton
coordinate frame used by CMU mocap data.

Functions:
    - compute_tpose_calibration(): Compute calibration quaternions from T-pose frames
    - apply_calibration(): Apply calibration to transform quaternions
    - quaternion_multiply(): Multiply two quaternions
    - quaternion_inverse(): Compute quaternion inverse
    - quaternion_to_euler(): Convert quaternion to Euler angles (ZXY order for BVH)
"""

from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from cometa_parser import IMUFrame, QuaternionFrame


@dataclass
class CalibrationData:
    """Calibration quaternions for transforming Cometa frame to BVH frame."""
    chest_offset: np.ndarray  # [w, x, y, z]
    upper_arm_offset: np.ndarray
    forearm_offset: np.ndarray
    hand_offset: np.ndarray

    def __str__(self):
        return (
            f"Chest:     [{self.chest_offset[0]:+.4f}, {self.chest_offset[1]:+.4f}, "
            f"{self.chest_offset[2]:+.4f}, {self.chest_offset[3]:+.4f}]\n"
            f"Upper Arm: [{self.upper_arm_offset[0]:+.4f}, {self.upper_arm_offset[1]:+.4f}, "
            f"{self.upper_arm_offset[2]:+.4f}, {self.upper_arm_offset[3]:+.4f}]\n"
            f"Forearm:   [{self.forearm_offset[0]:+.4f}, {self.forearm_offset[1]:+.4f}, "
            f"{self.forearm_offset[2]:+.4f}, {self.forearm_offset[3]:+.4f}]\n"
            f"Hand:      [{self.hand_offset[0]:+.4f}, {self.hand_offset[1]:+.4f}, "
            f"{self.hand_offset[2]:+.4f}, {self.hand_offset[3]:+.4f}]"
        )


def compute_tpose_calibration(
    frames: List[IMUFrame],
    tpose_start: int,
    tpose_end: int,
    verbose: bool = False
) -> CalibrationData:
    """
    Compute calibration quaternions from T-pose frames.

    The T-pose in BVH (CMU mocap convention):
    - Character facing +Z (forward), Y-up, X-right (right-handed)
    - Chest/spine: Upright, facing forward → identity rotation (1, 0, 0, 0)
    - Right upper arm: Extended horizontally to the right (+X direction)
      → 90° rotation around Y-axis
    - Right forearm: Continuation of upper arm (straight elbow)
      → same as upper arm (90° around Y)
    - Right hand: Palm down, fingers forward
      → same as forearm with possible wrist offset

    Args:
        frames: List of all IMU frames
        tpose_start: Start frame index of T-pose region
        tpose_end: End frame index of T-pose region
        verbose: Print calibration details

    Returns:
        CalibrationData containing offset quaternions for each sensor

    Raises:
        ValueError: If T-pose frames are invalid
    """
    if tpose_start < 0 or tpose_end > len(frames):
        raise ValueError(f"Invalid T-pose range: {tpose_start}-{tpose_end}, frames: {len(frames)}")

    if tpose_end <= tpose_start:
        raise ValueError(f"T-pose end ({tpose_end}) must be after start ({tpose_start})")

    tpose_frames = frames[tpose_start:tpose_end]

    if verbose:
        print(f"Computing T-pose calibration from {len(tpose_frames)} frames...")

    # Average quaternions over T-pose duration
    chest_avg = _average_quaternions([f.chest for f in tpose_frames])
    upper_arm_avg = _average_quaternions([f.upper_arm for f in tpose_frames])
    forearm_avg = _average_quaternions([f.forearm for f in tpose_frames])
    hand_avg = _average_quaternions([f.hand for f in tpose_frames])

    if verbose:
        print(f"\n  T-pose average quaternions (Cometa frame):")
        print(f"    Chest:     [{chest_avg[0]:+.4f}, {chest_avg[1]:+.4f}, {chest_avg[2]:+.4f}, {chest_avg[3]:+.4f}]")
        print(f"    Upper Arm: [{upper_arm_avg[0]:+.4f}, {upper_arm_avg[1]:+.4f}, {upper_arm_avg[2]:+.4f}, {upper_arm_avg[3]:+.4f}]")
        print(f"    Forearm:   [{forearm_avg[0]:+.4f}, {forearm_avg[1]:+.4f}, {forearm_avg[2]:+.4f}, {forearm_avg[3]:+.4f}]")
        print(f"    Hand:      [{hand_avg[0]:+.4f}, {hand_avg[1]:+.4f}, {hand_avg[2]:+.4f}, {hand_avg[3]:+.4f}]")

    # Define BVH T-pose target quaternions
    # BVH coordinate system: Y-up, Z-forward, X-right
    #
    # Strategy: The sensors are mounted on the body and have their own coordinate frame.
    # We need to find the rotation that transforms from sensor frame to BVH bone frame.
    #
    # Based on analysis of T-pose sensor data, the sensors appear to use a coordinate
    # system where their axes don't align with anatomical/skeletal axes. We empirically
    # determine the transform by analyzing what rotation makes the T-pose sensor reading
    # map to the desired BVH T-pose (identity rotation).
    #
    # For now, we'll use identity targets and let the calibration figure out the offset.
    # If the arms still twist incorrectly, we may need to add an additional axis remapping.

    chest_target = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
    upper_arm_target = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
    forearm_target = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
    hand_target = np.array([1.0, 0.0, 0.0, 0.0])  # Identity

    if verbose:
        print(f"\n  BVH T-pose target quaternions:")
        print(f"    Chest:     [{chest_target[0]:+.4f}, {chest_target[1]:+.4f}, {chest_target[2]:+.4f}, {chest_target[3]:+.4f}]")
        print(f"    Upper Arm: [{upper_arm_target[0]:+.4f}, {upper_arm_target[1]:+.4f}, {upper_arm_target[2]:+.4f}, {upper_arm_target[3]:+.4f}]")
        print(f"    Forearm:   [{forearm_target[0]:+.4f}, {forearm_target[1]:+.4f}, {forearm_target[2]:+.4f}, {forearm_target[3]:+.4f}]")
        print(f"    Hand:      [{hand_target[0]:+.4f}, {hand_target[1]:+.4f}, {hand_target[2]:+.4f}, {hand_target[3]:+.4f}]")

    # Compute calibration offsets: q_offset = q_target * q_measured_inverse
    # When applied: q_calibrated = q_offset * q_measured
    chest_offset = quaternion_multiply(chest_target, quaternion_inverse(chest_avg))
    upper_arm_offset = quaternion_multiply(upper_arm_target, quaternion_inverse(upper_arm_avg))
    forearm_offset = quaternion_multiply(forearm_target, quaternion_inverse(forearm_avg))
    hand_offset = quaternion_multiply(hand_target, quaternion_inverse(hand_avg))

    calib = CalibrationData(chest_offset, upper_arm_offset, forearm_offset, hand_offset)

    if verbose:
        print(f"\n  Computed calibration offsets (q_target * q_measured^-1):")
        print(f"    {calib}")

    return calib


def apply_calibration(
    frame: IMUFrame,
    calib: CalibrationData
) -> IMUFrame:
    """
    Apply calibration to transform a single IMU frame from Cometa to BVH coordinates.

    Args:
        frame: Raw IMU frame (Cometa coordinate frame)
        calib: Calibration data

    Returns:
        Calibrated IMU frame (BVH coordinate frame)
    """
    chest_cal = quaternion_multiply(calib.chest_offset, frame.chest.as_array())
    upper_arm_cal = quaternion_multiply(calib.upper_arm_offset, frame.upper_arm.as_array())
    forearm_cal = quaternion_multiply(calib.forearm_offset, frame.forearm.as_array())
    hand_cal = quaternion_multiply(calib.hand_offset, frame.hand.as_array())

    return IMUFrame(
        timestamp=frame.timestamp,
        chest=QuaternionFrame(frame.timestamp, chest_cal[0], chest_cal[1], chest_cal[2], chest_cal[3]),
        upper_arm=QuaternionFrame(frame.timestamp, upper_arm_cal[0], upper_arm_cal[1], upper_arm_cal[2], upper_arm_cal[3]),
        forearm=QuaternionFrame(frame.timestamp, forearm_cal[0], forearm_cal[1], forearm_cal[2], forearm_cal[3]),
        hand=QuaternionFrame(frame.timestamp, hand_cal[0], hand_cal[1], hand_cal[2], hand_cal[3])
    )


def _average_quaternions(quaternions: List[QuaternionFrame]) -> np.ndarray:
    """
    Average a list of quaternions using normalized sum method.

    For small angular differences (like T-pose stability), simple averaging
    followed by normalization is sufficient. For large differences, would need
    more sophisticated methods (e.g., quaternion mean on SO(3) manifold).

    Args:
        quaternions: List of quaternion frames

    Returns:
        Average quaternion as numpy array [w, x, y, z]
    """
    if not quaternions:
        raise ValueError("Cannot average empty quaternion list")

    q_arrays = np.array([q.as_array() for q in quaternions])

    # Ensure all quaternions are in the same hemisphere (avoid sign ambiguity)
    # Flip quaternions that point away from the first quaternion
    q_ref = q_arrays[0]
    for i in range(1, len(q_arrays)):
        if np.dot(q_arrays[i], q_ref) < 0:
            q_arrays[i] = -q_arrays[i]

    # Average and normalize
    q_avg = np.mean(q_arrays, axis=0)
    q_avg = q_avg / np.linalg.norm(q_avg)

    return q_avg


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


if __name__ == "__main__":
    print("IMU Calibration Library")
    print("=" * 70)
    print("\nThis is a library module. Import it in your scripts:")
    print("  from imu_calibration import compute_tpose_calibration, apply_calibration")
    print("  calib = compute_tpose_calibration(frames, tpose_start, tpose_end)")
    print("  calibrated_frame = apply_calibration(raw_frame, calib)")
