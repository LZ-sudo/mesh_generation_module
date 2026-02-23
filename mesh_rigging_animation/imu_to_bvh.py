#!/usr/bin/env python3
"""
IMU to BVH Converter - Convert Cometa IMU sensor data to BVH animation format.

This script converts Cometa Systems IMU sensor data (tab-separated text format)
to BVH (Biovision Hierarchy) animation files compatible with motion capture
pipelines and 3D animation software like Blender.

Pipeline:
1. Parse Cometa TXT file -> extract pre-fused quaternions (Cometa firmware)
2. Detect T-pose frames -> identify static calibration region
3. Compute T-pose alignment -> sensor-to-segment rotational offset
4. Apply T-pose alignment -> transform to anatomical bone frame
5. Downsample to 120 Hz + Write BVH file

Usage:
    python imu_to_bvh.py -i capture.txt -o animation.bvh
    python imu_to_bvh.py -i capture.txt -o animation.bvh --fps 120 --verbose
"""

import argparse
import sys
from pathlib import Path

from cometa_parser import parse_cometa_file, detect_tpose_frames, validate_sensor_data
from imu_calibration import compute_tpose_calibration, apply_calibration
from bvh_writer import write_bvh_file


def main():
    parser = argparse.ArgumentParser(
        description='Convert Cometa IMU sensor data to BVH animation format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python imu_to_bvh.py -i imu_data/capture.txt -o output/anim.bvh
  python imu_to_bvh.py -i capture.txt -o anim.bvh --fps 120 --verbose

Notes:
  - Input file must be Cometa Systems .txt format (tab-separated)
  - Uses Cometa pre-fused quaternions (~139 Hz during motion)
  - T-pose calibration: Automatic sensor-to-segment alignment (first 1-2 seconds)
  - Output BVH uses CMU mocap full-body skeleton; only right arm + chest are animated
  - Static root position (suitable for upper body ADL animations)
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to Cometa IMU data file (.txt)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path to output BVH file (.bvh)'
    )

    parser.add_argument(
        '--fps',
        type=float,
        default=120.0,
        help='Target frame rate for BVH output (default: 120 Hz, matches CMU mocap)'
    )

    parser.add_argument(
        '--tpose-duration',
        type=float,
        default=1.0,
        help='Expected T-pose duration in seconds (default: 1.0)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run data validation checks (may be slow for long captures)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IMU TO BVH CONVERTER (Cometa Pre-Fused Quaternions)")
    print("=" * 70)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Target FPS: {args.fps}")
    print()

    calibrated_frames, source_frames, tpose_calib = _run_pipeline(args, input_path)

    if calibrated_frames is None:
        return 1

    # Write BVH
    print(f"\nWriting BVH file...")
    try:
        success = write_bvh_file(
            calibrated_frames,
            output_path,
            target_fps=args.fps,
            world_correction=tpose_calib.chest_offset,
            verbose=args.verbose
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
    print(f"Frames: {len(calibrated_frames)} (downsampled from {len(source_frames)})")
    print(f"Duration: {source_frames[-1].timestamp:.2f} seconds at {args.fps} FPS")
    print(f"\nYou can now use this BVH file with:")
    print(f"  - Blender: File -> Import -> Motion Capture (.bvh)")
    print(f"  - generate_human.py --rig-type cmu_mb --animation {output_path.name}")
    print()

    return 0


def _run_pipeline(args, input_path: Path):
    """
    Parse Cometa pre-fused quaternions and apply T-pose alignment.

    Steps:
    1. Parse pre-fused quaternions from Cometa TXT file
    2. Detect T-pose calibration region
    3. Compute T-pose alignment (sensor-to-bone rotational offset)
    4. Apply T-pose alignment to all frames

    Returns:
        (calibrated_frames, source_frames, tpose_calib) on success, (None, None, None) on failure
    """
    # Step 1: Parse pre-fused quaternions
    print("Step 1: Parsing Cometa pre-fused quaternions...")
    try:
        frames = parse_cometa_file(input_path, verbose=args.verbose)
        print(f"  Parsed {len(frames)} frames")
        print(f"  Duration: {frames[-1].timestamp:.2f} seconds")
    except Exception as e:
        print(f"  ERROR: Failed to parse Cometa quaternions: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None, None, None

    # Step 2: Detect T-pose region
    print("\nStep 2: Detecting T-pose calibration region...")
    try:
        tpose_start, tpose_end = detect_tpose_frames(
            frames,
            duration_seconds=args.tpose_duration,
            verbose=args.verbose
        )
        print(f"  T-pose region: frames {tpose_start}-{tpose_end}")
        print(f"  Timestamps: {frames[tpose_start].timestamp:.3f}s to {frames[tpose_end].timestamp:.3f}s")
    except Exception as e:
        print(f"  ERROR: Failed to detect T-pose: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None, None, None

    # Step 3 (optional): Validate sensor data
    if args.validate:
        print("\nStep 3: Validating sensor data...")
        is_valid = validate_sensor_data(frames, verbose=args.verbose)
        if not is_valid:
            print("  WARNING: Data validation found issues (see above)")
            print("  Continuing with conversion...")

    # Step 3: Compute T-pose alignment
    print("\nStep 3: Computing T-pose alignment...")
    try:
        tpose_calib = compute_tpose_calibration(
            frames, tpose_start, tpose_end, verbose=args.verbose
        )
        print("  T-pose calibration computed successfully")
    except Exception as e:
        print(f"  ERROR: Failed to compute T-pose calibration: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None, None, None

    # Step 4: Apply T-pose alignment
    print("\nStep 4: Applying T-pose alignment...")
    try:
        calibrated_frames = []
        chunk_size = 5000
        for i in range(0, len(frames), chunk_size):
            chunk = frames[i:i + chunk_size]
            calibrated_frames.extend(apply_calibration(f, tpose_calib) for f in chunk)
            if args.verbose and len(frames) > chunk_size:
                print(f"    Calibrated {min(i + chunk_size, len(frames))}/{len(frames)} frames...")
        print(f"  Applied T-pose alignment to {len(calibrated_frames)} frames")
    except Exception as e:
        print(f"  ERROR: Failed to apply T-pose calibration: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None, None, None

    return calibrated_frames, frames, tpose_calib


if __name__ == "__main__":
    sys.exit(main())
