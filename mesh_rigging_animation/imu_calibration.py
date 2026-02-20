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
    """
    Calibration quaternions for transforming Cometa frame to BVH frame.

    Uses anatomical T-pose calibration with correct skeletal targets:
    - Chest: calibrated to identity (upright, facing forward)
    - Upper arm: calibrated to -90° Y rotation (horizontal extension to the right)
    - Forearm: calibrated to match upper arm in T-pose (straight elbow)
    - Hand: calibrated to match forearm in T-pose (straight wrist)
    - This ensures correct T-pose while preserving motion range
    """
    chest_offset: np.ndarray  # [w, x, y, z] - calibration to identity (upright)
    upper_arm_offset: np.ndarray  # [w, x, y, z] - calibration to horizontal (-90° Y)
    forearm_offset: np.ndarray  # [w, x, y, z] - calibration to match upper arm in T-pose
    hand_offset: np.ndarray  # [w, x, y, z] - calibration to match forearm in T-pose

    def __str__(self):
        return (
            f"Calibration offsets:\n"
            f"  Chest:     [{self.chest_offset[0]:+.4f}, {self.chest_offset[1]:+.4f}, "
            f"{self.chest_offset[2]:+.4f}, {self.chest_offset[3]:+.4f}]\n"
            f"  Upper Arm: [{self.upper_arm_offset[0]:+.4f}, {self.upper_arm_offset[1]:+.4f}, "
            f"{self.upper_arm_offset[2]:+.4f}, {self.upper_arm_offset[3]:+.4f}]\n"
            f"  Forearm:   [{self.forearm_offset[0]:+.4f}, {self.forearm_offset[1]:+.4f}, "
            f"{self.forearm_offset[2]:+.4f}, {self.forearm_offset[3]:+.4f}]\n"
            f"  Hand:      [{self.hand_offset[0]:+.4f}, {self.hand_offset[1]:+.4f}, "
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
    - Right upper arm: Extended horizontally to the right (-X direction from shoulder)
      → -90° rotation around Y-axis
    - Right forearm: Continuation of upper arm (straight elbow)
      → same as upper arm (-90° around Y)
    - Right hand: Palm down, fingers forward
      → same as forearm with possible wrist offset

    This function computes sensor-to-bone calibration offsets that transform
    raw sensor orientations to these anatomically correct T-pose targets.

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
        print(f"\n  T-pose average quaternions (raw sensor frame):")
        print(f"    Chest:     [{chest_avg[0]:+.4f}, {chest_avg[1]:+.4f}, {chest_avg[2]:+.4f}, {chest_avg[3]:+.4f}]")
        print(f"    Upper Arm: [{upper_arm_avg[0]:+.4f}, {upper_arm_avg[1]:+.4f}, {upper_arm_avg[2]:+.4f}, {upper_arm_avg[3]:+.4f}]")
        print(f"    Forearm:   [{forearm_avg[0]:+.4f}, {forearm_avg[1]:+.4f}, {forearm_avg[2]:+.4f}, {forearm_avg[3]:+.4f}]")
        print(f"    Hand:      [{hand_avg[0]:+.4f}, {hand_avg[1]:+.4f}, {hand_avg[2]:+.4f}, {hand_avg[3]:+.4f}]")

    # ANATOMICAL T-POSE CALIBRATION STRATEGY:
    #
    # Problem: Each sensor has its own mounting orientation. We need to calibrate
    # sensors to match the anatomical T-pose in BVH coordinate frame.
    #
    # BVH T-pose (CMU mocap convention):
    # - Character faces +Z (forward), +Y is up, +X is right
    # - Chest: Upright, facing forward → identity rotation
    # - Right arm: Extended horizontally to the right → -90° rotation around Y-axis
    # - Right forearm: Continuation of upper arm (straight elbow) → match upper arm
    # - Right hand: Palm down, fingers forward → match forearm
    #
    # Solution: Use anatomically correct T-pose targets:
    # 1. Calibrate chest to identity (upright)
    # 2. Calibrate upper arm to horizontal extension (-90° Y rotation)
    # 3. Calibrate forearm to match upper arm (straight elbow)
    # 4. Calibrate hand to match forearm (straight wrist)

    identity = np.array([1.0, 0.0, 0.0, 0.0])

    # Right arm horizontal extension: -90° rotation around Y-axis
    # Using scipy for reliable conversion
    arm_tpose_rot = Rotation.from_euler('Y', -90, degrees=True)
    arm_tpose_quat_xyzw = arm_tpose_rot.as_quat()  # [x, y, z, w] format
    arm_tpose_target = np.array([
        arm_tpose_quat_xyzw[3],  # w
        arm_tpose_quat_xyzw[0],  # x
        arm_tpose_quat_xyzw[1],  # y
        arm_tpose_quat_xyzw[2]   # z
    ])

    # Step 1: Compute absolute calibration offsets for chest and upper arm
    chest_offset = quaternion_multiply(identity, quaternion_inverse(chest_avg))
    upper_arm_offset = quaternion_multiply(arm_tpose_target, quaternion_inverse(upper_arm_avg))

    # Step 2: For forearm and hand, calibrate them to MATCH their parent in T-pose
    # This ensures zero relative rotation when joints are straight
    # Target: forearm should match arm orientation in T-pose (straight elbow)
    # Target: hand should match forearm orientation in T-pose (straight wrist)

    # Apply arm calibration to get calibrated arm orientation in T-pose
    upper_arm_cal_tpose = quaternion_multiply(upper_arm_offset, upper_arm_avg)  # Should be identity

    # Forearm should match calibrated arm orientation
    # forearm_offset * forearm_avg = upper_arm_cal_tpose
    # forearm_offset = upper_arm_cal_tpose * forearm_avg^-1
    forearm_offset = quaternion_multiply(upper_arm_cal_tpose, quaternion_inverse(forearm_avg))

    # Hand should match calibrated forearm orientation
    # First get calibrated forearm in T-pose
    forearm_cal_tpose = quaternion_multiply(forearm_offset, forearm_avg)
    # hand_offset * hand_avg = forearm_cal_tpose
    # hand_offset = forearm_cal_tpose * hand_avg^-1
    hand_offset = quaternion_multiply(forearm_cal_tpose, quaternion_inverse(hand_avg))

    if verbose:
        print(f"\n  Calibration offsets:")
        print(f"    Chest:     [{chest_offset[0]:+.4f}, {chest_offset[1]:+.4f}, "
              f"{chest_offset[2]:+.4f}, {chest_offset[3]:+.4f}] -> identity (upright)")
        print(f"    Upper Arm: [{upper_arm_offset[0]:+.4f}, {upper_arm_offset[1]:+.4f}, "
              f"{upper_arm_offset[2]:+.4f}, {upper_arm_offset[3]:+.4f}] -> -90° Y (horizontal)")
        print(f"    Forearm:   [{forearm_offset[0]:+.4f}, {forearm_offset[1]:+.4f}, "
              f"{forearm_offset[2]:+.4f}, {forearm_offset[3]:+.4f}] -> match arm")
        print(f"    Hand:      [{hand_offset[0]:+.4f}, {hand_offset[1]:+.4f}, "
              f"{hand_offset[2]:+.4f}, {hand_offset[3]:+.4f}] -> match forearm")

        # Verify T-pose calibration
        print(f"\n  T-pose verification (calibrated orientations):")
        print(f"    Upper Arm: [{upper_arm_cal_tpose[0]:+.4f}, {upper_arm_cal_tpose[1]:+.4f}, "
              f"{upper_arm_cal_tpose[2]:+.4f}, {upper_arm_cal_tpose[3]:+.4f}]")
        print(f"    Forearm:   [{forearm_cal_tpose[0]:+.4f}, {forearm_cal_tpose[1]:+.4f}, "
              f"{forearm_cal_tpose[2]:+.4f}, {forearm_cal_tpose[3]:+.4f}]")
        hand_cal_tpose = quaternion_multiply(hand_offset, hand_avg)
        print(f"    Hand:      [{hand_cal_tpose[0]:+.4f}, {hand_cal_tpose[1]:+.4f}, "
              f"{hand_cal_tpose[2]:+.4f}, {hand_cal_tpose[3]:+.4f}]")

    calib = CalibrationData(chest_offset, upper_arm_offset, forearm_offset, hand_offset)

    if verbose:
        print(f"\n  Calibration computed:")
        print(f"    {calib}")

    return calib


def apply_calibration(
    frame: IMUFrame,
    calib: CalibrationData
) -> IMUFrame:
    """
    Apply calibration to transform a single IMU frame from Cometa to BVH coordinates.

    Uses anatomical T-pose calibration offsets:
    - Chest: calibrated to identity (upright)
    - Upper arm: calibrated to -90° Y rotation (horizontal)
    - Forearm: calibrated to match upper arm in T-pose
    - Hand: calibrated to match forearm in T-pose

    This ensures correct anatomical T-pose while preserving motion range.

    Args:
        frame: Raw IMU frame (Cometa coordinate frame)
        calib: Calibration data with anatomical offsets

    Returns:
        Calibrated IMU frame (BVH coordinate frame)
    """
    # Apply calibration offsets directly (simple quaternion multiplication)
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
