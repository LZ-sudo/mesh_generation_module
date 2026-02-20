#!/usr/bin/env python3
"""
IMU to BVH Converter - Convert Cometa IMU sensor data to BVH animation format.

This script converts Cometa Systems IMU sensor data (tab-separated text format)
to BVH (Biovision Hierarchy) animation files compatible with motion capture
pipelines and 3D animation software like Blender.

The conversion pipeline:
1. Parse Cometa TXT file -> extract quaternion time series
2. Detect T-pose frames -> compute coordinate frame calibration
3. Apply calibration -> transform quaternions to BVH coordinate frame
4. Downsample to 120 Hz -> match CMU mocap frame rate
5. Write BVH file -> 4-bone right arm skeleton with static root

Usage:
    python imu_to_bvh.py --input imu_data/capture.txt --output output/animation.bvh
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
  python imu_to_bvh.py -i capture.txt -o anim.bvh --tpose-duration 2.0

Notes:
  - Input file must be Cometa Systems .txt format (tab-separated)
  - T-pose calibration is automatic (uses first 1-2 seconds)
  - Output BVH has 4-bone right arm skeleton (Chest -> Shoulder -> Elbow -> Wrist)
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

    # Convert paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IMU TO BVH CONVERTER")
    print("=" * 70)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Target FPS: {args.fps}")
    print()

    # Step 1: Parse Cometa file
    print("Step 1: Parsing Cometa IMU file...")
    try:
        frames = parse_cometa_file(input_path, verbose=args.verbose)
        print(f"  Parsed {len(frames)} frames")
        print(f"  Duration: {frames[-1].timestamp:.2f} seconds")
    except Exception as e:
        print(f"  ERROR: Failed to parse file: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Step 2: Validate data (optional)
    if args.validate:
        print("\nStep 2: Validating sensor data...")
        is_valid = validate_sensor_data(frames, verbose=args.verbose)
        if not is_valid:
            print("  WARNING: Data validation found issues (see above)")
            print("  Continuing with conversion...")

    # Step 3: Detect T-pose
    print(f"\n{'Step 3' if args.validate else 'Step 2'}: Detecting T-pose calibration frames...")
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
        return 1

    # Step 4: Compute calibration
    print(f"\n{'Step 4' if args.validate else 'Step 3'}: Computing T-pose calibration...")
    try:
        calib = compute_tpose_calibration(frames, tpose_start, tpose_end, verbose=args.verbose)
        print("  Calibration computed successfully")
    except Exception as e:
        print(f"  ERROR: Failed to compute calibration: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Step 5: Apply calibration
    print(f"\n{'Step 5' if args.validate else 'Step 4'}: Applying calibration to all frames...")
    try:
        calibrated_frames = []
        chunk_size = 5000
        for i in range(0, len(frames), chunk_size):
            chunk = frames[i:i + chunk_size]
            calibrated_chunk = [apply_calibration(f, calib) for f in chunk]
            calibrated_frames.extend(calibrated_chunk)
            if args.verbose and len(frames) > chunk_size:
                print(f"    Calibrated {min(i + chunk_size, len(frames))}/{len(frames)} frames...")

        print(f"  Calibrated {len(calibrated_frames)} frames")
    except Exception as e:
        print(f"  ERROR: Failed to apply calibration: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Step 6: Write BVH
    print(f"\n{'Step 6' if args.validate else 'Step 5'}: Writing BVH file...")
    try:
        success = write_bvh_file(
            calibrated_frames,
            output_path,
            target_fps=args.fps,
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

    # Success
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print(f"Frames: {len(calibrated_frames)} (downsampled from {len(frames)})")
    print(f"Duration: {frames[-1].timestamp:.2f} seconds at {args.fps} FPS")
    print("\nYou can now use this BVH file with:")
    print("  - Blender: File -> Import -> Motion Capture (.bvh)")
    print(f"  - generate_human.py --rig-type cmu_mb --animation {output_path.name}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
