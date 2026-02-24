#!/usr/bin/env python3
"""
C3D to BVH Converter - Convert Cometa C3D joint angles to BVH animation format.

Reads pre-computed anatomical joint angles from a Cometa Systems C3D file
(channels exported by Cometa EMG and Motion Tools software) and converts them
to BVH (Biovision Hierarchy) animation format.

Unlike the quaternion-based imu_to_bvh pipeline, this approach uses Cometa's
pre-calibrated joint angles directly, bypassing the AHRS world-correction step.
The Cometa software performs its own T-pose calibration and sensor-to-segment
alignment internally, exposing the result as named degree-valued channels in
the C3D ANALOG section.

Angle-to-BVH channel mapping (ZYX rotation order, arm rest direction = -X):
  Shoulder angles use a spherical-to-ZYX conversion (see _shoulder_to_zyx).
  Cometa reports (abd, horiz) as spherical coordinates: abd = elevation above
  horizontal, horiz = azimuth forward from the coronal plane.  Direct use as
  ZYX Euler channels over-rotates the arm forward at high abduction (Euler
  coupling).  The corrected BVH ZYX angles are derived as:
    theta_z = atan2(-sin(abd), cos(abd)*cos(horiz))
    theta_y = asin(cos(abd)*sin(horiz))
    theta_x = shoulder_vert  (axial/internal-external rotation)
  Elbow Flexion (+bend)            -> RightForeArm Z = -fe
  Elbow Pronation/Supination       -> RightForeArm X = +ps
  Elbow Deviation                  -> RightForeArm Y = +dev
  Wrist Flexion (+bend)            -> RightHand  Z = -fe
  Wrist Ulnar/Radial Deviation     -> RightHand  Y = +rad
  Wrist CW/CCW Rotation            -> RightHand  X = +rot

Usage:
    python c3d_to_bvh.py -i capture.c3d -o animation.bvh
    python c3d_to_bvh.py -i capture.c3d -o animation.bvh --fps 120 --verbose
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import ezc3d

from bvh_writer import write_bvh_hierarchy


# ---------------------------------------------------------------------------
# Cometa channel label constants (Cometa EMG and Motion Tools naming)
# ---------------------------------------------------------------------------

_CHANNEL_LABELS = {
    'shoulder_horiz': 'Right Shoulder :Horizontal Flexion/Extension',
    'shoulder_vert':  'Right Shoulder :Vertical Flexion/Extension',
    'shoulder_abd':   'Right Shoulder :Abduction/Adduction',
    'elbow_fe':       'Right Elbow :Flexion/Extension',
    'elbow_ps':       'Right Elbow :Pronation/Supination',
    'elbow_dev':      'Right Elbow :Deviation',
    'wrist_fe':       'Right Wrist :Flexion/Extension',
    'wrist_rad':      'Right Wrist :Ulnar/Radial Deviation',
    'wrist_rot':      'Right Wrist :CW/CCW Rotation',
}

_STATIC_3 = "0.000000 0.000000 0.000000 "
_STATIC_6 = "0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 "


@dataclass
class JointAngleFrame:
    """One frame of pre-computed anatomical joint angles from Cometa C3D."""
    timestamp: float
    # Shoulder (3 DOF)
    shoulder_horiz: float   # Horizontal Flex/Ext [deg], +forward
    shoulder_vert: float    # Vertical Flex/Ext [deg]
    shoulder_abd: float     # Abduction/Adduction [deg], +upward
    # Elbow (3 DOF)
    elbow_fe: float         # Flexion/Extension [deg], +flexed
    elbow_ps: float         # Pronation/Supination [deg]
    elbow_dev: float        # Deviation [deg]
    # Wrist (3 DOF)
    wrist_fe: float         # Flexion/Extension [deg]
    wrist_rad: float        # Ulnar/Radial Deviation [deg]
    wrist_rot: float        # CW/CCW Rotation [deg]


def parse_c3d_joint_angles(
    c3d_path: Path,
    verbose: bool = False,
) -> List[JointAngleFrame]:
    """
    Parse pre-computed joint angles from a Cometa C3D file.

    Cometa stores all analog data at 2000 Hz but the actual IMU update rate
    is ~142.857 Hz (every 14th sample is unique). This function extracts the
    unique frames at the effective IMU rate by stepping through the 2000 Hz
    data at the IMU stride interval.

    Args:
        c3d_path: Path to Cometa C3D file
        verbose: Print parsing details

    Returns:
        List of JointAngleFrame at ~142.857 Hz

    Raises:
        ValueError: If required joint angle channels are not found in the C3D file
    """
    c = ezc3d.c3d(str(c3d_path))

    analog_rate = float(c['header']['analogs']['frame_rate'])
    imu_rate = float(c['parameters']['ANALOG']['IMU_RATE']['value'][0])
    step = int(round(analog_rate / imu_rate))

    analogs = c['data']['analogs']  # shape: (1, n_channels, n_total_frames)
    an_labels = c['parameters']['ANALOG']['LABELS']['value']
    label_to_idx = {lbl.strip(): i for i, lbl in enumerate(an_labels)}

    def _get_channel(key: str):
        label = _CHANNEL_LABELS[key]
        idx = label_to_idx.get(label)
        if idx is None:
            raise ValueError(f"Channel not found in C3D file: '{label}'")
        return analogs[0, idx, :]

    shoulder_horiz = _get_channel('shoulder_horiz')
    shoulder_vert  = _get_channel('shoulder_vert')
    shoulder_abd   = _get_channel('shoulder_abd')
    elbow_fe       = _get_channel('elbow_fe')
    elbow_ps       = _get_channel('elbow_ps')
    elbow_dev      = _get_channel('elbow_dev')
    wrist_fe       = _get_channel('wrist_fe')
    wrist_rad      = _get_channel('wrist_rad')
    wrist_rot      = _get_channel('wrist_rot')

    n_total = analogs.shape[2]
    frames = []
    for i in range(0, n_total, step):
        frames.append(JointAngleFrame(
            timestamp=i / analog_rate,
            shoulder_horiz=float(shoulder_horiz[i]),
            shoulder_vert=float(shoulder_vert[i]),
            shoulder_abd=float(shoulder_abd[i]),
            elbow_fe=float(elbow_fe[i]),
            elbow_ps=float(elbow_ps[i]),
            elbow_dev=float(elbow_dev[i]),
            wrist_fe=float(wrist_fe[i]),
            wrist_rad=float(wrist_rad[i]),
            wrist_rot=float(wrist_rot[i]),
        ))

    if verbose:
        print(f"  Analog rate: {analog_rate:.0f} Hz, IMU rate: {imu_rate:.3f} Hz (stride={step})")
        print(f"  Extracted {len(frames)} unique frames")
        print(f"  Duration: {frames[-1].timestamp:.2f}s")

    return frames


def _downsample_joint_angles(
    frames: List[JointAngleFrame],
    source_fps: float,
    target_fps: float,
) -> List[JointAngleFrame]:
    """Decimate frames from source_fps to target_fps by selecting every Nth frame."""
    if source_fps <= target_fps * 1.1:
        return frames
    step = int(round(source_fps / target_fps))
    return frames[::step]


def _shoulder_to_zyx(
    shoulder_abd_deg: float,
    shoulder_horiz_deg: float,
    shoulder_vert_deg: float,
) -> Tuple[float, float, float]:
    """
    Convert Cometa spherical shoulder angles to BVH ZYX Euler angles.

    Cometa reports (abd, horiz) as spherical coordinates of the upper arm direction:
      - abd:   elevation above horizontal (0 = T-pose, +90 = arm straight up)
      - horiz: azimuth forward from the coronal plane (0 = pure abduction, +90 = pure forward flex)
      - vert:  axial rotation of the humerus (internal/external rotation)

    Arm unit vector from Cometa spherical coordinates:
      arm = (-cos(abd)*cos(horiz),  sin(abd),  cos(abd)*sin(horiz))

    BVH ZYX arm direction (arm rest at -X):
      arm = (-cos(ty)*cos(tz),  -cos(ty)*sin(tz),  sin(ty))

    Plugging Cometa angles directly into ZYX channels amplifies the forward lean
    (horiz term) by Euler coupling at high abduction.  This function solves for the
    exact ZYX angles that reproduce the same arm direction.

    Returns:
        (theta_z_deg, theta_y_deg, theta_x_deg) for BVH ZYX channels
    """
    abd = math.radians(shoulder_abd_deg)
    horiz = math.radians(shoulder_horiz_deg)

    theta_y = math.asin(math.cos(abd) * math.sin(horiz))
    theta_z = math.atan2(-math.sin(abd), math.cos(abd) * math.cos(horiz))
    theta_x = math.radians(shoulder_vert_deg)

    return math.degrees(theta_z), math.degrees(theta_y), math.degrees(theta_x)


def _map_angles_to_bvh(
    frame: JointAngleFrame,
) -> Tuple[
    Tuple[float, float, float],   # Spine1 (chest): Z, Y, X
    Tuple[float, float, float],   # RightArm: Z, Y, X
    Tuple[float, float, float],   # RightForeArm: Z, Y, X
    Tuple[float, float, float],   # RightHand: Z, Y, X
]:
    """
    Map Cometa anatomical joint angles to BVH ZYX Euler channel values.

    The BVH skeleton uses CMU convention: +Y up, +Z forward, right arm at -X.
    All rotations are ZYX intrinsic Euler angles relative to the parent bone.

    Shoulder angles use a spherical-to-ZYX conversion because Cometa reports
    abd (elevation) and horiz (azimuth) as independent spherical coordinates,
    not as sequential Euler angles.  Direct substitution would over-rotate the
    arm forward at high abduction due to Euler coupling.
    """
    # Chest (Spine1): no dedicated chest-relative angle available; kept static.
    chest = (0.0, 0.0, 0.0)

    # Shoulder -> RightArm ZYX (spherical coordinate conversion)
    arm = _shoulder_to_zyx(frame.shoulder_abd, frame.shoulder_horiz, frame.shoulder_vert)

    # Elbow -> RightForeArm ZYX
    # Z = -fe:  +flexion = forearm rotates toward +Y in arm frame = R_z(-angle)
    # Y = +dev: deviation (near zero, sagittal plane)
    # X = +ps:  pronation/supination = axial rotation of forearm
    forearm = (
        -frame.elbow_fe,
        +frame.elbow_dev,
        +frame.elbow_ps,
    )

    # Wrist -> RightHand ZYX
    # Z = -fe:  +flexion = hand rotates toward +Y in forearm frame = R_z(-angle)
    # Y = +rad: ulnar/radial deviation
    # X = +rot: CW/CCW rotation = axial rotation
    hand = (
        -frame.wrist_fe,
        +frame.wrist_rad,
        +frame.wrist_rot,
    )

    return chest, arm, forearm, hand


def write_bvh_from_joint_angles(
    frames: List[JointAngleFrame],
    output_path: Path,
    target_fps: float = 120.0,
    verbose: bool = False,
) -> bool:
    """
    Write joint angle frames to a BVH file using the CMU mocap skeleton.

    Args:
        frames: Joint angle frames at any sample rate
        output_path: Output BVH file path
        target_fps: Target frame rate for BVH output (default 120 Hz)
        verbose: Print progress

    Returns:
        True on success, False on failure
    """
    if not frames:
        print("Error: No frames to write")
        return False

    source_fps = (
        1.0 / (frames[1].timestamp - frames[0].timestamp)
        if len(frames) > 1
        else target_fps
    )
    frames = _downsample_joint_angles(frames, source_fps, target_fps)
    # Use actual inter-frame interval so Blender plays at the correct speed.
    # If decimation step rounded to 1 (e.g. 142 Hz -> 120 Hz), the source
    # rate is preserved rather than lying to BVH readers about the frame time.
    frame_time = (
        frames[1].timestamp - frames[0].timestamp
        if len(frames) > 1
        else 1.0 / target_fps
    )

    if verbose:
        print(f"  Frames: {len(frames)}, FPS: {target_fps:.0f}, frame_time: {frame_time:.6f}s")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            write_bvh_hierarchy(f)
            f.write("MOTION\n")
            f.write(f"Frames: {len(frames)}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")

            for i, frame in enumerate(frames):
                chest, arm, forearm, hand = _map_angles_to_bvh(frame)
                # 96 channels total: 6 (Hips) + 30 joints x 3
                # Animated: Spine1, RightArm, RightForeArm, RightHand
                line = (
                    _STATIC_6                                                       # Hips (pos + rot)
                    + _STATIC_3 * 5                                                 # LHipJoint..LeftToeBase
                    + _STATIC_3 * 5                                                 # RHipJoint..RightToeBase
                    + _STATIC_3 * 2                                                 # LowerBack, Spine
                    + f"{chest[0]:.6f} {chest[1]:.6f} {chest[2]:.6f} "            # Spine1
                    + _STATIC_3 * 3                                                 # Neck, Neck1, Head
                    + _STATIC_3 * 7                                                 # LeftShoulder..LThumb
                    + _STATIC_3                                                     # RightShoulder
                    + f"{arm[0]:.6f} {arm[1]:.6f} {arm[2]:.6f} "                  # RightArm
                    + f"{forearm[0]:.6f} {forearm[1]:.6f} {forearm[2]:.6f} "      # RightForeArm
                    + f"{hand[0]:.6f} {hand[1]:.6f} {hand[2]:.6f} "               # RightHand
                    + _STATIC_3 * 2                                                 # RightFingerBase, RightHandIndex1
                    + "0.000000 0.000000 0.000000\n"                               # RThumb
                )
                f.write(line)

                if verbose and (i + 1) % 500 == 0:
                    print(f"    Written {i+1}/{len(frames)} frames...")

        return True

    except Exception as e:
        print(f"Error writing BVH file: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Convert Cometa C3D joint angles to BVH animation format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python c3d_to_bvh.py -i imu_data/capture.c3d -o output/anim.bvh
  python c3d_to_bvh.py -i capture.c3d -o anim.bvh --fps 120 --verbose

Notes:
  - Input must be a Cometa Systems C3D file with pre-computed joint angle channels.
  - Required channels: Right Shoulder/Elbow/Wrist (3 DOF each).
  - Chest is kept static; only the right arm chain is animated.
  - Output BVH uses CMU mocap full-body skeleton (same as imu_to_bvh pipeline).
        """,
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to Cometa C3D file (.c3d)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to output BVH file (.bvh)')
    parser.add_argument('--fps', type=float, default=120.0,
                        help='Target frame rate for BVH output (default: 120 Hz)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed progress information')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("C3D TO BVH CONVERTER (Cometa Pre-Computed Joint Angles)")
    print("=" * 70)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Target FPS: {args.fps}")
    print()

    print("Step 1: Parsing C3D joint angles...")
    try:
        frames = parse_c3d_joint_angles(input_path, verbose=args.verbose)
        print(f"  Parsed {len(frames)} frames")
        print(f"  Duration: {frames[-1].timestamp:.2f}s")
    except Exception as e:
        print(f"  ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    print("\nStep 2: Writing BVH file...")
    try:
        success = write_bvh_from_joint_angles(
            frames,
            output_path,
            target_fps=args.fps,
            verbose=args.verbose,
        )
        if not success:
            print("  ERROR: BVH writing failed")
            return 1
        file_size = output_path.stat().st_size
        print(f"  BVH file written: {file_size / 1024:.1f} KB")
    except Exception as e:
        print(f"  ERROR: Failed to write BVH: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print(f"Duration: {frames[-1].timestamp:.2f}s at {args.fps:.0f} FPS")
    print(f"\nImport in Blender: File -> Import -> Motion Capture (.bvh)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
