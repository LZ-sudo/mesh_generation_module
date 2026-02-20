#!/usr/bin/env python3
"""
BVH file writer for IMU-based motion capture data.

This module creates BVH (Biovision Hierarchy) files from calibrated IMU
quaternion data. The skeleton is a minimal right-arm chain with 4 bones:
Root (Hips) → Chest → RightShoulder → RightElbow → RightWrist.

BVH Format:
- HIERARCHY section: Skeleton structure with bone offsets
- MOTION section: Frame count, frame time, and per-frame channel data

Functions:
    - write_bvh_file(): Write complete BVH file from IMU frames
    - downsample_frames(): Reduce frame rate (e.g., 2000 Hz → 120 Hz)
"""

from pathlib import Path
from typing import List

from cometa_parser import IMUFrame
from imu_calibration import quaternion_to_euler, quaternion_multiply, quaternion_inverse


def write_bvh_file(
    frames: List[IMUFrame],
    output_path: Path,
    target_fps: float = 120.0,
    verbose: bool = False
) -> bool:
    """
    Write calibrated IMU frames to BVH file.

    The skeleton hierarchy:
        ROOT Hips (static root, position channels + rotation)
          JOINT Chest (rotation from chest sensor)
            JOINT RightShoulder (rotation from upper arm sensor)
              JOINT RightElbow (rotation from forearm sensor)
                JOINT RightWrist (rotation from hand sensor)
                  End Site (hand endpoint)

    Args:
        frames: List of calibrated IMU frames (must be in BVH coordinate frame)
        output_path: Path to output .bvh file
        target_fps: Target frame rate (will downsample if needed)
        verbose: Print progress

    Returns:
        True if successful, False otherwise
    """
    if not frames:
        print("Error: No frames to write")
        return False

    # Downsample if needed
    source_fps = 1.0 / (frames[1].timestamp - frames[0].timestamp) if len(frames) > 1 else target_fps
    if source_fps > target_fps * 1.1:  # Downsample if source is >10% faster
        if verbose:
            print(f"Downsampling from {source_fps:.0f} Hz to {target_fps:.0f} Hz...")
        frames = downsample_frames(frames, source_fps, target_fps)
        if verbose:
            print(f"  Downsampled to {len(frames)} frames")

    frame_time = 1.0 / target_fps

    if verbose:
        print(f"Writing BVH file: {output_path.name}")
        print(f"  Frames: {len(frames)}")
        print(f"  Duration: {frames[-1].timestamp:.2f}s")
        print(f"  FPS: {target_fps:.0f}")
        print(f"  Frame time: {frame_time:.6f}s")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write HIERARCHY section
            _write_hierarchy(f)

            # Write MOTION section
            _write_motion(f, frames, frame_time, verbose)

        if verbose:
            print(f"  BVH file written successfully")

        return True

    except Exception as e:
        print(f"Error writing BVH file: {e}")
        return False


def _write_hierarchy(f):
    """
    Write BVH HIERARCHY section matching CMU mocap format.

    Creates a CMU-style full-body skeleton compatible with existing animation assets:
    - Uses ZYX rotation order (Zrotation Yrotation Xrotation) to match reference BVHs
    - CMU bone naming: LowerBack, Spine, Spine1, RightArm, RightForeArm, RightHand
    - Spine chain: Hips -> LowerBack -> Spine -> Spine1 (Spine1 has IMU chest data)
    - Head: Neck -> Neck1 -> Head (dummy, static)
    - Right arm: RightShoulder -> RightArm -> RightForeArm -> RightHand (IMU data)
    - Left arm: LeftShoulder -> LeftArm (dummy, static)
    - Legs: LHipJoint/RHipJoint with full leg chains (dummy, static)

    This structure matches animation_assets/*.bvh format for consistency.
    Only the right arm and chest (Spine1) are animated from IMU data.

    Bone offsets are in centimeters (BVH standard units).
    """
    hierarchy = """HIERARCHY
ROOT Hips
{
\tOFFSET 0.00 0.00 0.00
\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
\tJOINT LHipJoint
\t{
\t\tOFFSET 0 0 0
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tJOINT LeftUpLeg
\t\t{
\t\t\tOFFSET 1.59 -1.84 0.72
\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\tJOINT LeftLeg
\t\t\t{
\t\t\t\tOFFSET 2.51 -6.88 0.00
\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\tJOINT LeftFoot
\t\t\t\t{
\t\t\t\t\tOFFSET 2.63 -7.23 0.00
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT LeftToeBase
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET 0.24 -0.65 1.73
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET 0.00 -0.00 0.93
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}
\t}
\tJOINT RHipJoint
\t{
\t\tOFFSET 0 0 0
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tJOINT RightUpLeg
\t\t{
\t\t\tOFFSET -1.51 -1.84 0.72
\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\tJOINT RightLeg
\t\t\t{
\t\t\t\tOFFSET -2.55 -6.99 0.00
\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\tJOINT RightFoot
\t\t\t\t{
\t\t\t\t\tOFFSET -2.66 -7.31 0.00
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT RightToeBase
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET -0.23 -0.63 2.04
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET -0.00 -0.00 1.07
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}
\t}
\tJOINT LowerBack
\t{
\t\tOFFSET 0 0 0
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tJOINT Spine
\t\t{
\t\t\tOFFSET -0.03 1.86 -0.11
\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\tJOINT Spine1
\t\t\t{
\t\t\t\tOFFSET 0.01 1.86 0.04
\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\tJOINT Neck
\t\t\t\t{
\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT Neck1
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET -0.02 1.81 0.09
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tJOINT Head
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET 0.06 1.76 -0.38
\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\tOFFSET 0.02 1.83 -0.14
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tJOINT LeftShoulder
\t\t\t\t{
\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT LeftArm
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET 3.47 1.51 0.14
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tJOINT LeftForeArm
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET 4.78 -0.00 0.00
\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\tJOINT LeftHand
\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\tOFFSET 3.59 -0.00 -0.00
\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\tJOINT LeftFingerBase
\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\t\tJOINT LeftHandIndex1
\t\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\t\tOFFSET 0.66 -0.00 0.00
\t\t\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\t\t\tOFFSET 0.53 -0.00 0.00
\t\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\tJOINT LThumb
\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\t\tOFFSET 0.54 -0.00 0.54
\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tJOINT RightShoulder
\t\t\t\t{
\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT RightArm
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET -3.32 1.61 0.35
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tJOINT RightForeArm
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET -4.49 -0.00 0.00
\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\tJOINT RightHand
\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\tOFFSET -3.71 -0.00 0.00
\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\tJOINT RightFingerBase
\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\t\tJOINT RightHandIndex1
\t\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\t\tOFFSET -0.45 -0.00 0.00
\t\t\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\t\t\tOFFSET -0.36 -0.00 0.00
\t\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\tJOINT RThumb
\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\t\tOFFSET -0.37 -0.00 0.37
\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}
\t}
}
"""
    f.write(hierarchy)


def _write_motion(f, frames: List[IMUFrame], frame_time: float, verbose: bool):
    """
    Write BVH MOTION section matching CMU mocap format.

    Uses ZYX rotation order (Zrotation Yrotation Xrotation) to match reference BVHs.

    Channel order for each joint (from HIERARCHY):
    - Hips: Xposition Yposition Zposition Zrotation Yrotation Xrotation (6 channels)
    - LHipJoint: Zrotation Yrotation Xrotation (3 channels) - static
    - LeftUpLeg: Zrotation Yrotation Xrotation (3 channels) - static
    - LeftLeg: Zrotation Yrotation Xrotation (3 channels) - static
    - LeftFoot: Zrotation Yrotation Xrotation (3 channels) - static
    - LeftToeBase: Zrotation Yrotation Xrotation (3 channels) - static
    - RHipJoint: Zrotation Yrotation Xrotation (3 channels) - static
    - RightUpLeg: Zrotation Yrotation Xrotation (3 channels) - static
    - RightLeg: Zrotation Yrotation Xrotation (3 channels) - static
    - RightFoot: Zrotation Yrotation Xrotation (3 channels) - static
    - RightToeBase: Zrotation Yrotation Xrotation (3 channels) - static
    - LowerBack: Zrotation Yrotation Xrotation (3 channels) - static
    - Spine: Zrotation Yrotation Xrotation (3 channels) - static
    - Spine1: Zrotation Yrotation Xrotation (3 channels) - from chest IMU
    - Neck: Zrotation Yrotation Xrotation (3 channels) - static
    - Neck1: Zrotation Yrotation Xrotation (3 channels) - static
    - Head: Zrotation Yrotation Xrotation (3 channels) - static
    - LeftShoulder: Zrotation Yrotation Xrotation (3 channels) - static
    - LeftArm: Zrotation Yrotation Xrotation (3 channels) - static
    - LeftForeArm: Zrotation Yrotation Xrotation (3 channels) - static
    - LeftHand: Zrotation Yrotation Xrotation (3 channels) - static
    - LeftFingerBase: Zrotation Yrotation Xrotation (3 channels) - static
    - LeftHandIndex1: Zrotation Yrotation Xrotation (3 channels) - static
    - LThumb: Zrotation Yrotation Xrotation (3 channels) - static
    - RightShoulder: Zrotation Yrotation Xrotation (3 channels) - static
    - RightArm: Zrotation Yrotation Xrotation (3 channels) - from upper_arm IMU
    - RightForeArm: Zrotation Yrotation Xrotation (3 channels) - from forearm IMU
    - RightHand: Zrotation Yrotation Xrotation (3 channels) - from hand IMU
    - RightFingerBase: Zrotation Yrotation Xrotation (3 channels) - static
    - RightHandIndex1: Zrotation Yrotation Xrotation (3 channels) - static
    - RThumb: Zrotation Yrotation Xrotation (3 channels) - static

    Total: 96 channels per frame (6 + 30*3)

    Args:
        f: File handle
        frames: List of calibrated IMU frames
        frame_time: Time per frame in seconds
        verbose: Print progress
    """
    f.write("MOTION\n")
    f.write(f"Frames: {len(frames)}\n")
    f.write(f"Frame Time: {frame_time:.6f}\n")

    # Static rotation for all dummy bones (identity rotation in ZYX order)
    static = (0.0, 0.0, 0.0)

    for i, frame in enumerate(frames):
        # Root (Hips) position: static at origin
        hips_pos = (0.0, 0.0, 0.0)
        hips_rot = static

        # IMU-driven rotations - compute hierarchical local rotations
        # Chest (Spine1): relative to Spine (which is static/identity)
        q_chest = frame.chest.as_array()
        spine1_rot = quaternion_to_euler(q_chest, order='ZYX')

        # Right Arm: relative to Chest/Spine1 (via static RightShoulder)
        # RightShoulder is a child of Spine1 and parent of RightArm
        # Since RightShoulder is static (identity), RightArm inherits Spine1's orientation
        # So we need: arm rotation relative to chest orientation
        q_arm_world = frame.upper_arm.as_array()
        q_arm_local = quaternion_multiply(quaternion_inverse(q_chest), q_arm_world)
        right_arm_rot = quaternion_to_euler(q_arm_local, order='ZYX')

        # Right Forearm: relative to Right Arm (HIERARCHICAL - compute local rotation)
        q_forearm_world = frame.forearm.as_array()
        q_forearm_local = quaternion_multiply(quaternion_inverse(q_arm_world), q_forearm_world)
        right_forearm_rot = quaternion_to_euler(q_forearm_local, order='ZYX')

        # Right Hand: relative to Right Forearm (HIERARCHICAL - compute local rotation)
        q_hand_world = frame.hand.as_array()
        q_hand_local = quaternion_multiply(quaternion_inverse(q_forearm_world), q_hand_world)
        right_hand_rot = quaternion_to_euler(q_hand_local, order='ZYX')

        # Write all 96 channels in hierarchy order
        f.write(
            # Hips (6 channels)
            f"{hips_pos[0]:.6f} {hips_pos[1]:.6f} {hips_pos[2]:.6f} "
            f"{hips_rot[0]:.6f} {hips_rot[1]:.6f} {hips_rot[2]:.6f} "
            # LHipJoint (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # LeftUpLeg (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # LeftLeg (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # LeftFoot (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # LeftToeBase (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # RHipJoint (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # RightUpLeg (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # RightLeg (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # RightFoot (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # RightToeBase (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # LowerBack (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # Spine (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # Spine1 (3) - IMU chest data
            f"{spine1_rot[0]:.6f} {spine1_rot[1]:.6f} {spine1_rot[2]:.6f} "
            # Neck (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # Neck1 (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # Head (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # LeftShoulder (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # LeftArm (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # LeftForeArm (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # LeftHand (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # LeftFingerBase (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # LeftHandIndex1 (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # LThumb (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # RightShoulder (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # RightArm (3) - IMU upper_arm data
            f"{right_arm_rot[0]:.6f} {right_arm_rot[1]:.6f} {right_arm_rot[2]:.6f} "
            # RightForeArm (3) - IMU forearm data
            f"{right_forearm_rot[0]:.6f} {right_forearm_rot[1]:.6f} {right_forearm_rot[2]:.6f} "
            # RightHand (3) - IMU hand data
            f"{right_hand_rot[0]:.6f} {right_hand_rot[1]:.6f} {right_hand_rot[2]:.6f} "
            # RightFingerBase (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # RightHandIndex1 (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f} "
            # RThumb (3)
            f"{static[0]:.6f} {static[1]:.6f} {static[2]:.6f}\n"
        )

        if verbose and (i + 1) % 500 == 0:
            print(f"    Written {i+1}/{len(frames)} frames...")


def downsample_frames(
    frames: List[IMUFrame],
    source_fps: float,
    target_fps: float
) -> List[IMUFrame]:
    """
    Downsample frames by decimation (select every Nth frame).

    For example, 2000 Hz → 120 Hz requires keeping every ~17th frame.

    Args:
        frames: Original frames at source_fps
        source_fps: Source frame rate
        target_fps: Target frame rate

    Returns:
        Downsampled list of frames
    """
    if source_fps <= target_fps:
        return frames  # No downsampling needed

    decimation_factor = int(round(source_fps / target_fps))

    if decimation_factor <= 1:
        return frames

    # Select every Nth frame
    downsampled = frames[::decimation_factor]

    return downsampled


if __name__ == "__main__":
    print("BVH Writer Library")
    print("=" * 70)
    print("\nThis is a library module. Import it in your scripts:")
    print("  from bvh_writer import write_bvh_file")
    print("  success = write_bvh_file(calibrated_frames, Path('output.bvh'), target_fps=120)")
