#!/usr/bin/env python3
"""
IMU to BVH Converter - Convert Cometa IMU sensor data to BVH animation format.

This script converts Cometa Systems IMU sensor data (tab-separated text format)
to BVH (Biovision Hierarchy) animation files compatible with motion capture
pipelines and 3D animation software like Blender.

Pipeline (VQF sensor fusion + calibration):
1. Parse Cometa TXT file -> extract RAW IMU data (accel, gyro, mag)
2. Preliminary VQF fusion -> for T-pose detection
3. Detect T-pose frames -> identify static calibration region
4. Compute sensor calibration -> correct gyro bias, accel scale (Ferraris method)
5. Apply sensor calibration -> fix intrinsic sensor errors (bias, scale, drift)
6. Final VQF sensor fusion -> compute quaternions (2.9° RMSE, magnetic rejection)
7. Compute T-pose alignment -> sensor-to-segment rotational offset
8. Apply T-pose alignment -> transform to anatomical bone frame
9. Downsample to 120 Hz -> match CMU mocap frame rate
10. Write BVH file -> full-body CMU skeleton with static root

Usage:
    # Default: Full pipeline (sensor calibration + VQF + T-pose alignment)
    python imu_to_bvh.py -i capture.txt -o animation.bvh

    # Skip sensor calibration (if sensors are factory-calibrated)
    python imu_to_bvh.py -i capture.txt -o animation.bvh --skip-sensor-calibration

    # Skip T-pose alignment (preserves full motion range but may have errors)
    python imu_to_bvh.py -i capture.txt -o animation.bvh --skip-tpose-calibration

    # Verbose output for debugging
    python imu_to_bvh.py -i capture.txt -o animation.bvh --verbose
"""

import argparse
import sys
from pathlib import Path

from cometa_parser import parse_raw_imu_data, detect_tpose_frames, validate_sensor_data
from imu_calibration import compute_tpose_calibration, apply_calibration
from sensor_calibration import compute_sensor_calibration, apply_sensor_calibration
from bvh_writer import write_bvh_file
from vqf_fusion import fuse_sensor_data


def main():
    parser = argparse.ArgumentParser(
        description='Convert Cometa IMU sensor data to BVH animation format using VQF fusion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Full pipeline (sensor calibration + VQF + T-pose alignment)
  python imu_to_bvh.py -i imu_data/capture.txt -o output/anim.bvh

  # Skip sensor calibration (if sensors are factory-calibrated)
  python imu_to_bvh.py -i capture.txt -o anim.bvh --skip-sensor-calibration

  # Skip T-pose alignment (preserves full motion range)
  python imu_to_bvh.py -i capture.txt -o anim.bvh --skip-tpose-calibration

  # Custom FPS + verbose output
  python imu_to_bvh.py -i capture.txt -o anim.bvh --fps 120 --verbose

Notes:
  - Input file must be Cometa Systems .txt format (tab-separated)
  - VQF sensor fusion: 2.9 deg RMSE vs 5.3-16.7 deg for Madgwick/Mahony/EKF
  - Sensor calibration: Corrects gyro bias and accel scale (Ferraris method)
  - T-pose calibration: Automatic sensor-to-segment alignment (first 1-2 seconds)
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
        '--skip-tpose-calibration',
        action='store_true',
        help='Skip T-pose segment alignment (preserves full motion range)'
    )

    parser.add_argument(
        '--skip-sensor-calibration',
        action='store_true',
        help='Skip sensor intrinsic calibration (gyro bias, accel scale)'
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
    print("IMU TO BVH CONVERTER (VQF + Calibration Pipeline)")
    print("=" * 70)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Target FPS: {args.fps}")
    print()

    # Step 1: Parse raw IMU data
    print("Step 1: Parsing Cometa IMU file (RAW sensor data)...")
    try:
        raw_frames = parse_raw_imu_data(input_path, verbose=args.verbose)
        print(f"  Parsed {len(raw_frames)} raw IMU frames")
        print(f"  Duration: {raw_frames[-1].timestamp:.2f} seconds")
    except Exception as e:
        print(f"  ERROR: Failed to parse raw IMU data: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Step 2: Preliminary VQF fusion for T-pose detection
    print("\nStep 2: Preliminary VQF fusion for T-pose detection...")
    try:
        sample_rate = len(raw_frames) / raw_frames[-1].timestamp if raw_frames[-1].timestamp > 0 else 2000.0
        preliminary_frames = fuse_sensor_data(raw_frames, sample_rate=sample_rate, verbose=args.verbose)
        print(f"  Preliminary fusion complete ({len(preliminary_frames)} frames)")
    except Exception as e:
        print(f"  ERROR: Preliminary VQF fusion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Step 3: Detect T-pose region
    print("\nStep 3: Detecting T-pose calibration region...")
    try:
        tpose_start, tpose_end = detect_tpose_frames(
            preliminary_frames,
            duration_seconds=args.tpose_duration,
            verbose=args.verbose
        )
        print(f"  T-pose region: frames {tpose_start}-{tpose_end}")
        print(f"  Timestamps: {preliminary_frames[tpose_start].timestamp:.3f}s to {preliminary_frames[tpose_end].timestamp:.3f}s")
    except Exception as e:
        print(f"  ERROR: Failed to detect T-pose: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Step 4-5: Sensor intrinsic calibration
    if args.skip_sensor_calibration:
        print("\nStep 4: Skipping sensor intrinsic calibration...")
        print("  Using raw sensor data without bias/scale correction")
        calibrated_raw_frames = raw_frames
    else:
        print("\nStep 4: Computing sensor intrinsic calibration (Ferraris method)...")
        try:
            sensor_calib = compute_sensor_calibration(
                raw_frames,
                tpose_start,
                tpose_end,
                verbose=args.verbose
            )
            print("  Sensor calibration computed (gyro bias, accel scale)")
        except Exception as e:
            print(f"  ERROR: Failed to compute sensor calibration: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

        print("\nStep 5: Applying sensor calibration to raw IMU data...")
        try:
            calibrated_raw_frames = apply_sensor_calibration(
                raw_frames,
                sensor_calib,
                verbose=args.verbose
            )
            print(f"  Applied sensor calibration to {len(calibrated_raw_frames)} frames")
        except Exception as e:
            print(f"  ERROR: Failed to apply sensor calibration: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    # Step 6: Final VQF sensor fusion
    step_num = 6 if not args.skip_sensor_calibration else 5
    print(f"\nStep {step_num}: Final VQF sensor fusion...")
    try:
        frames = fuse_sensor_data(calibrated_raw_frames, sample_rate=sample_rate, verbose=args.verbose)
        print(f"  Fused {len(frames)} frames using VQF algorithm")
        print(f"  VQF achieves 2.9 deg RMSE vs 5.3-16.7 deg for Madgwick/Mahony/EKF")
        if not args.skip_sensor_calibration:
            print(f"  Sensor calibration corrected gyro bias and accel scale")
    except Exception as e:
        print(f"  ERROR: VQF sensor fusion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Step 7: Validate data (optional)
    if args.validate:
        step_num = 7 if not args.skip_sensor_calibration else 6
        print(f"\nStep {step_num}: Validating sensor data...")
        is_valid = validate_sensor_data(frames, verbose=args.verbose)
        if not is_valid:
            print("  WARNING: Data validation found issues (see above)")
            print("  Continuing with conversion...")

    # Step 8-10: T-pose segment alignment
    if args.skip_tpose_calibration:
        step_num = 8 if not args.skip_sensor_calibration else 7
        if args.validate:
            step_num += 1
        print(f"\nStep {step_num}: Skipping T-pose segment alignment...")
        print("  Using raw sensor quaternions with hierarchical forward kinematics")
        print("  Note: T-pose may not be perfectly aligned, but motion range will be preserved")
        calibrated_frames = frames
        step_offset = 1
    else:
        # Step 8: Detect T-pose again (from final fused frames)
        step_num = 8 if not args.skip_sensor_calibration else 7
        if args.validate:
            step_num += 1
        print(f"\nStep {step_num}: Detecting T-pose for segment alignment...")
        try:
            tpose_start_seg, tpose_end_seg = detect_tpose_frames(
                frames,
                duration_seconds=args.tpose_duration,
                verbose=args.verbose
            )
            print(f"  T-pose region: frames {tpose_start_seg}-{tpose_end_seg}")
            print(f"  Timestamps: {frames[tpose_start_seg].timestamp:.3f}s to {frames[tpose_end_seg].timestamp:.3f}s")
        except Exception as e:
            print(f"  ERROR: Failed to detect T-pose: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

        # Step 9: Compute T-pose calibration
        step_num += 1
        print(f"\nStep {step_num}: Computing T-pose segment alignment...")
        try:
            tpose_calib = compute_tpose_calibration(frames, tpose_start_seg, tpose_end_seg, verbose=args.verbose)
            print("  T-pose calibration computed successfully")
        except Exception as e:
            print(f"  ERROR: Failed to compute T-pose calibration: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

        # Step 10: Apply T-pose calibration
        step_num += 1
        print(f"\nStep {step_num}: Applying T-pose segment alignment...")
        try:
            calibrated_frames = []
            chunk_size = 5000
            for i in range(0, len(frames), chunk_size):
                chunk = frames[i:i + chunk_size]
                calibrated_chunk = [apply_calibration(f, tpose_calib) for f in chunk]
                calibrated_frames.extend(calibrated_chunk)
                if args.verbose and len(frames) > chunk_size:
                    print(f"    Calibrated {min(i + chunk_size, len(frames))}/{len(frames)} frames...")

            print(f"  Applied T-pose calibration to {len(calibrated_frames)} frames")
        except Exception as e:
            print(f"  ERROR: Failed to apply T-pose calibration: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

        step_offset = 3

    # Final step: Write BVH
    final_step = step_offset + (7 if not args.skip_sensor_calibration else 6) + (1 if args.validate else 0)
    print(f"\nStep {final_step}: Writing BVH file...")
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

    # Print pipeline summary
    print("\n" + "-" * 70)
    print("PIPELINE SUMMARY:")
    print("-" * 70)

    print("Sensor Fusion: VQF (2.9 deg RMSE)")
    print("  - Raw IMU data -> VQF sensor fusion -> Quaternions")
    print("  - Superior accuracy vs Madgwick/Mahony/EKF")
    print("  - Magnetic disturbance rejection enabled")

    if not args.skip_sensor_calibration:
        print("\nSensor Intrinsic Calibration: Ferraris method (simplified)")
        print("  - Gyroscope bias correction (critical for drift prevention)")
        print("  - Accelerometer scale normalization")
        print("  - Applied before VQF fusion for maximum accuracy")
    else:
        print("\nSensor Intrinsic Calibration: Skipped")
        print("  - Using raw sensor data (may have bias/scale errors)")

    if args.skip_tpose_calibration:
        print("\nT-pose Segment Alignment: Skipped (raw quaternions)")
        print("  - Full motion range preserved")
        print("  - T-pose may not be perfectly aligned")
    else:
        print("\nT-pose Segment Alignment: Hierarchical calibration")
        print("  - T-pose aligned to identity rotation")
        print("  - Sensor frame -> Anatomical bone frame")

    print("\nYou can now use this BVH file with:")
    print("  - Blender: File -> Import -> Motion Capture (.bvh)")
    print(f"  - generate_human.py --rig-type cmu_mb --animation {output_path.name}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
