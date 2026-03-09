#!/usr/bin/env python3
"""
BVH Axis Probe - Empirically determine the axis convention of the CMU skeleton.

Generates several BVH test files that isolate individual rotation channels of
the RightArm joint.  Load each file in Blender and observe the direction of arm
movement to determine what each channel physically controls.

Also computes the _euler_shoulder_to_zyx() output for the key calibration frames
(movements 1-4) and writes them as a static-pose BVH for direct visual inspection.

Output files (written to the same directory as this script):
    probe_arm_z.bvh      -- RightArm Zrotation sweeps 0 -> +60 -> 0 -> -60 -> 0 deg
    probe_arm_y.bvh      -- RightArm Yrotation sweeps 0 -> +60 -> 0 -> -60 -> 0 deg
    probe_arm_x.bvh      -- RightArm Xrotation sweeps 0 -> +60 -> 0 -> -60 -> 0 deg
    probe_key_frames.bvh -- Static poses for movement 1/2 channel values

Usage:
    python bvh_axis_probe.py
    python bvh_axis_probe.py --out-dir /path/to/output

Expected Blender workflow:
    1. Import probe_arm_z.bvh  (File -> Import -> Motion Capture .bvh)
    2. Play the animation.  The arm sweeps positive then negative along one axis.
    3. Note which physical direction corresponds to positive Zrotation.
    4. Repeat for probe_arm_y.bvh and probe_arm_x.bvh.
    5. Import probe_key_frames.bvh and step through frames to inspect each pose.
"""

import argparse
import math
from pathlib import Path

from bvh_writer import write_bvh_hierarchy
from c3d_to_bvh import _euler_shoulder_to_zyx, _RIGHT_SHOULDER_SIGNS


_STATIC_3 = "0.000000 0.000000 0.000000 "
_STATIC_6 = "0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 "

_FPS = 30
_FRAME_TIME = 1.0 / _FPS


def _arm_line(tx: float, tz: float, ty: float) -> str:
    """
    Build a single BVH motion line with the given RightArm XZY angles (degrees)
    and all other joints at zero (static T-pose).

    RightArm declares CHANNELS 3 Xrotation Zrotation Yrotation, so values are
    written in (X, Z, Y) order.

    Channel layout (96 total):
        Hips (6) | LHipJoint..LeftToeBase (5x3) | RHipJoint..RightToeBase (5x3)
        LowerBack,Spine (2x3) | Spine1 (3) | Neck,Neck1,Head (3x3)
        LeftShoulder..LThumb (7x3) | RightShoulder (3)
        RightArm (3) | RightForeArm (3) | RightHand (3)
        RightFingerBase,RightHandIndex1 (2x3) | RThumb (3)
    """
    arm_str = f"{tx:.6f} {tz:.6f} {ty:.6f} "
    return (
        _STATIC_6           # Hips pos + rot
        + _STATIC_3 * 5     # LHipJoint..LeftToeBase
        + _STATIC_3 * 5     # RHipJoint..RightToeBase
        + _STATIC_3 * 2     # LowerBack, Spine
        + _STATIC_3         # Spine1
        + _STATIC_3 * 3     # Neck, Neck1, Head
        + _STATIC_3 * 7     # LeftShoulder..LThumb
        + _STATIC_3         # RightShoulder
        + arm_str           # RightArm  <-- the animated joint
        + _STATIC_3         # RightForeArm
        + _STATIC_3         # RightHand
        + _STATIC_3 * 2     # RightFingerBase, RightHandIndex1
        + "0.000000 0.000000 0.000000\n"  # RThumb
    )


def _write_sweep_bvh(path: Path, channel: str, peak_deg: float = 60.0) -> None:
    """
    Write a BVH where one RightArm ZYX channel sweeps 0 -> +peak -> 0 -> -peak -> 0.

    Segment layout (5 segments x 18 frames each = 90 frames total at 30 fps):
        seg 0 (frames  0-17): hold at 0 deg              -- T-pose reference
        seg 1 (frames 18-35): ramp 0 -> +peak_deg        -- positive direction
        seg 2 (frames 36-53): hold at +peak_deg          -- positive extreme
        seg 3 (frames 54-71): ramp +peak_deg -> 0        -- return
        seg 4 (frames 72-89): hold at 0 deg              -- second reference hold
        seg 5 (frames 90-107): ramp 0 -> -peak_deg       -- negative direction
        seg 6 (frames 108-125): hold at -peak_deg        -- negative extreme
        seg 7 (frames 126-143): ramp -peak_deg -> 0      -- return

    Args:
        path:      Output BVH file path.
        channel:   One of 'z', 'y', 'x' — which RightArm channel to sweep.
        peak_deg:  Peak angle magnitude in degrees (default 60).
    """
    n_hold = 18
    n_ramp = 18
    frames = []

    def _make_frame(angle: float) -> str:
        tx = angle if channel == 'x' else 0.0
        tz = angle if channel == 'z' else 0.0
        ty = angle if channel == 'y' else 0.0
        return _arm_line(tx, tz, ty)

    # Segment 0: hold at 0
    frames.extend([_make_frame(0.0)] * n_hold)
    # Segment 1: ramp 0 -> +peak
    for i in range(n_ramp):
        frames.append(_make_frame(peak_deg * (i / (n_ramp - 1))))
    # Segment 2: hold at +peak
    frames.extend([_make_frame(peak_deg)] * n_hold)
    # Segment 3: ramp +peak -> 0
    for i in range(n_ramp):
        frames.append(_make_frame(peak_deg * (1.0 - i / (n_ramp - 1))))
    # Segment 4: hold at 0
    frames.extend([_make_frame(0.0)] * n_hold)
    # Segment 5: ramp 0 -> -peak
    for i in range(n_ramp):
        frames.append(_make_frame(-peak_deg * (i / (n_ramp - 1))))
    # Segment 6: hold at -peak
    frames.extend([_make_frame(-peak_deg)] * n_hold)
    # Segment 7: ramp -peak -> 0
    for i in range(n_ramp):
        frames.append(_make_frame(-peak_deg * (1.0 - i / (n_ramp - 1))))

    with open(path, 'w', encoding='utf-8') as f:
        write_bvh_hierarchy(f)
        f.write("MOTION\n")
        f.write(f"Frames: {len(frames)}\n")
        f.write(f"Frame Time: {_FRAME_TIME:.6f}\n")
        for line in frames:
            f.write(line)


def _write_key_frames_bvh(path: Path) -> None:
    """
    Write a BVH with static poses derived from _euler_shoulder_to_zyx() for
    the key calibration trial frames.  All frames are held for 1 second each
    so they are easy to step through in Blender.

    Poses included (frame ranges at 30 fps):
        frames  0-29:  T-pose reference (all zeros)
        frames 30-59:  Pure ABD = +30 deg, vert=0, horiz=0
                       Expected: arm elevated in coronal plane
        frames 60-89:  Movement 1 approx: abd=+8.5, vert=0, horiz=+89.5
                       Expected: arm forward + slightly elevated
        frames 90-119: Movement 2 approx: abd=-3.86, vert=0, horiz=+90.0
                       Expected: arm forward + slightly depressed
        frames 120-149: Pure ABD = -30 deg, vert=0, horiz=0
                        Expected: arm depressed in coronal plane
    """
    signs = _RIGHT_SHOULDER_SIGNS

    poses = [
        ("T-pose reference",               0.0,    0.0, 0.0),
        ("Pure ABD +30 (coronal raise)",  30.0,    0.0, 0.0),
        ("Movement 1: abd=+8.5 horiz=+89.5", 8.5, 0.0, 89.5),
        ("Movement 2: abd=-3.86 horiz=+90.0", -3.86, 0.0, 90.0),
        ("Pure ABD -30 (coronal lower)", -30.0,    0.0, 0.0),
    ]

    print("\n--- Key frame poses (_euler_shoulder_to_zyx output, XZY channel order) ---")
    print(f"{'Pose':<42}  {'abd':>7}  {'vert':>7}  {'horiz':>7}  "
          f"{'tx_out':>8}  {'tz_out':>8}  {'ty_out':>8}")
    print("-" * 105)

    frames = []
    for label, abd, vert, horiz in poses:
        tx, tz, ty = _euler_shoulder_to_zyx(abd, vert, horiz, **signs)
        print(f"  {label:<40}  {abd:>7.2f}  {vert:>7.2f}  {horiz:>7.2f}  "
              f"{tx:>8.3f}  {tz:>8.3f}  {ty:>8.3f}")
        line = _arm_line(tx, tz, ty)
        frames.extend([line] * _FPS)  # hold for 1 second

    print()

    with open(path, 'w', encoding='utf-8') as f:
        write_bvh_hierarchy(f)
        f.write("MOTION\n")
        f.write(f"Frames: {len(frames)}\n")
        f.write(f"Frame Time: {_FRAME_TIME:.6f}\n")
        for line in frames:
            f.write(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Generate BVH axis probe files for CMU skeleton axis validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Load each output BVH in Blender (File -> Import -> Motion Capture .bvh) and
observe the arm direction when the angle is positive vs negative.

probe_arm_z.bvh:
    Positive Zrotation -- note whether arm goes UP, DOWN, FORWARD, or BACKWARD.
    Cross-reference with _euler_shoulder_to_zyx() output in probe_key_frames.bvh.

probe_arm_y.bvh:
    Positive Yrotation -- note direction.

probe_arm_x.bvh:
    Positive Xrotation -- note direction (axial/twist).

probe_key_frames.bvh:
    Step through poses with arrow keys (or use timeline).  Each pose holds for
    30 frames (1 second at 30 fps).  Verify the arm position matches physical
    expectation for each labelled pose.
        """,
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='Output directory for BVH files (default: same directory as this script)',
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    probes = [
        ('z', out_dir / 'probe_arm_z.bvh'),
        ('y', out_dir / 'probe_arm_y.bvh'),
        ('x', out_dir / 'probe_arm_x.bvh'),
    ]

    print("Generating axis sweep BVH files...")
    for channel, path in probes:
        _write_sweep_bvh(path, channel)
        print(f"  Written: {path.name}  (RightArm {channel.upper()}rotation sweeps +/-60 deg)")

    key_path = out_dir / 'probe_key_frames.bvh'
    print(f"\nGenerating key frame poses BVH...")
    _write_key_frames_bvh(key_path)
    print(f"  Written: {key_path.name}")

    print("\nBlender import: File -> Import -> Motion Capture (.bvh)")
    print("Timeline: each sweep is 144 frames (4.8 s) at 30 fps")
    print("Key frames: 5 poses x 30 frames each = 150 frames total\n")

    print("=== What to record from Blender ===")
    print("For each probe_arm_?.bvh:")
    print("  Play frames 18-35 (positive ramp).  Note arm direction at frame 35.")
    print("  Play frames 90-107 (negative ramp). Note arm direction at frame 107.")
    print("  Fill in the table:")
    print()
    print("  Channel     +60 deg direction     -60 deg direction")
    print("  Zrotation   ???                   ???")
    print("  Yrotation   ???                   ???")
    print("  Xrotation   ???                   ???")
    print()
    print("For probe_key_frames.bvh:")
    print("  Frame  0: T-pose (baseline, arm horizontal right)")
    print("  Frame 30: Pure ABD +30 -- arm should raise in coronal plane")
    print("  Frame 60: Movement 1 (abd=+8.5, horiz=+89.5) -- arm forward, slightly UP?")
    print("  Frame 90: Movement 2 (abd=-3.86, horiz=+90.0) -- arm forward, slightly DOWN?")
    print("  Frame 120: Pure ABD -30 -- arm should lower from T-pose")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
