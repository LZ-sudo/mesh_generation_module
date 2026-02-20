#!/usr/bin/env python3
"""
IMU Sensor Intrinsic Calibration - Correct sensor bias, scale, and misalignment.

This module implements simplified Ferraris calibration using static T-pose region
to correct IMU sensor intrinsic errors before applying VQF sensor fusion.

Corrections applied:
    - Gyroscope bias: Zero-rate offset (most critical for drift prevention)
    - Accelerometer bias/scale: Gravity vector alignment
    - Magnetometer: Not calibrated (requires tumbling test data)

Note: Full Ferraris calibration requires 6+ static orientations (tumbling test).
      This simplified version uses only the T-pose static region, providing
      basic bias correction. For best results, collect dedicated calibration data.

Functions:
    - compute_sensor_calibration(): Compute calibration from static T-pose region
    - apply_sensor_calibration(): Apply corrections to raw IMU data
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from cometa_parser import RawIMUFrame, RawIMUData


@dataclass
class SensorCalibrationParams:
    """
    Sensor intrinsic calibration parameters for one IMU.

    Simplified Ferraris calibration using static T-pose region:
    - Gyroscope bias: Average readings during static pose (should be zero)
    - Accelerometer bias: Offset from expected gravity vector
    - Accelerometer scale: Normalized to 1g magnitude
    """
    gyro_bias: np.ndarray  # [3] - Gyroscope bias in deg/s
    accel_bias: np.ndarray  # [3] - Accelerometer bias in g
    accel_scale: np.ndarray  # [3] - Accelerometer scale factors

    def __str__(self):
        return (
            f"Gyro bias:   [{self.gyro_bias[0]:+.4f}, {self.gyro_bias[1]:+.4f}, {self.gyro_bias[2]:+.4f}] deg/s\n"
            f"Accel bias:  [{self.accel_bias[0]:+.4f}, {self.accel_bias[1]:+.4f}, {self.accel_bias[2]:+.4f}] g\n"
            f"Accel scale: [{self.accel_scale[0]:.4f}, {self.accel_scale[1]:.4f}, {self.accel_scale[2]:.4f}]"
        )


@dataclass
class SensorCalibration:
    """
    Complete sensor calibration for all 4 IMUs.

    Contains calibration parameters for chest, upper_arm, forearm, hand sensors.
    Computed from static T-pose region to correct intrinsic sensor errors.
    """
    chest: SensorCalibrationParams
    upper_arm: SensorCalibrationParams
    forearm: SensorCalibrationParams
    hand: SensorCalibrationParams

    def __str__(self):
        return (
            f"Sensor Calibration:\n"
            f"\n  Chest:\n    {str(self.chest).replace(chr(10), chr(10) + '    ')}\n"
            f"\n  Upper Arm:\n    {str(self.upper_arm).replace(chr(10), chr(10) + '    ')}\n"
            f"\n  Forearm:\n    {str(self.forearm).replace(chr(10), chr(10) + '    ')}\n"
            f"\n  Hand:\n    {str(self.hand).replace(chr(10), chr(10) + '    ')}"
        )


def compute_sensor_calibration(
    raw_frames: List[RawIMUFrame],
    tpose_start: int,
    tpose_end: int,
    verbose: bool = False
) -> SensorCalibration:
    """
    Compute sensor intrinsic calibration from static T-pose region.

    This implements simplified Ferraris calibration using only the T-pose
    static region. Ideally, Ferraris calibration requires 6+ static orientations
    (tumbling test), but we approximate using available T-pose data.

    Calibration corrections:
    1. Gyroscope bias: Average readings during static T-pose (should be ~0)
       - Critical for preventing drift during integration
       - Typical bias: 0.01-0.1 deg/s for consumer IMUs

    2. Accelerometer bias/scale: Normalize to gravity vector
       - Assumes T-pose is relatively static (minimal linear acceleration)
       - Corrects bias and scales to 1g magnitude

    3. Magnetometer: Not calibrated (requires varied orientations)
       - VQF has built-in magnetic disturbance rejection
       - Full mag calibration needs ellipsoid fitting with 3D rotation data

    Args:
        raw_frames: List of all raw IMU frames
        tpose_start: Start frame index of static T-pose region
        tpose_end: End frame index of static T-pose region
        verbose: Print calibration details

    Returns:
        SensorCalibration with bias/scale parameters for all sensors
    """
    if verbose:
        print(f"Computing sensor calibration from T-pose frames {tpose_start}-{tpose_end}...")

    # Extract T-pose static region
    tpose_frames = raw_frames[tpose_start:tpose_end + 1]
    n_frames = len(tpose_frames)

    if verbose:
        print(f"  Using {n_frames} static frames for calibration")

    # Compute calibration for each sensor
    chest_cal = _compute_single_sensor_calibration(
        [f.chest for f in tpose_frames],
        verbose=verbose,
        sensor_name="Chest"
    )

    upper_arm_cal = _compute_single_sensor_calibration(
        [f.upper_arm for f in tpose_frames],
        verbose=verbose,
        sensor_name="Upper Arm"
    )

    forearm_cal = _compute_single_sensor_calibration(
        [f.forearm for f in tpose_frames],
        verbose=verbose,
        sensor_name="Forearm"
    )

    hand_cal = _compute_single_sensor_calibration(
        [f.hand for f in tpose_frames],
        verbose=verbose,
        sensor_name="Hand"
    )

    calibration = SensorCalibration(chest_cal, upper_arm_cal, forearm_cal, hand_cal)

    if verbose:
        print(f"\n  Calibration complete")
        print(f"\n{calibration}")

    return calibration


def _compute_single_sensor_calibration(
    sensor_data: List[RawIMUData],
    verbose: bool = False,
    sensor_name: str = "Sensor"
) -> SensorCalibrationParams:
    """
    Compute calibration parameters for a single IMU from static data.

    Args:
        sensor_data: List of RawIMUData from static T-pose region
        verbose: Print calibration info
        sensor_name: Name for verbose output

    Returns:
        SensorCalibrationParams with bias and scale corrections
    """
    if verbose:
        print(f"    Calibrating {sensor_name}...")

    n_samples = len(sensor_data)

    # Collect gyro and accel data
    gyro_data = np.array([s.gyro_array() for s in sensor_data])  # [N x 3] in deg/s
    accel_data = np.array([s.accel_array() for s in sensor_data])  # [N x 3] in g

    # 1. GYROSCOPE BIAS CALIBRATION
    # In static pose, gyroscope should read zero (no rotation)
    # Average readings give the bias (zero-rate offset)
    gyro_bias = np.mean(gyro_data, axis=0)  # [3] in deg/s

    # 2. ACCELEROMETER BIAS AND SCALE CALIBRATION
    # In static pose, accelerometer measures gravity (1g downward)
    # Strategy: Normalize to 1g magnitude

    # Average accelerometer reading
    accel_mean = np.mean(accel_data, axis=0)  # [3] in g

    # Compute magnitude (should be ~1g for static pose)
    accel_magnitude = np.linalg.norm(accel_mean)

    # Scale factor: normalize to 1g
    # Note: This assumes T-pose is truly static (no linear acceleration)
    # If accel_magnitude is far from 1g, there may be movement or mounting issues
    accel_scale = np.ones(3) / accel_magnitude if accel_magnitude > 0.01 else np.ones(3)

    # Bias: For simplicity, we don't estimate bias separately from scale
    # In full Ferraris calibration, bias would be estimated from multiple orientations
    # For now, assume factory bias correction is reasonable
    accel_bias = np.zeros(3)

    if verbose:
        print(f"      Gyro bias: [{gyro_bias[0]:+.4f}, {gyro_bias[1]:+.4f}, {gyro_bias[2]:+.4f}] deg/s")
        print(f"      Accel magnitude: {accel_magnitude:.4f}g (target: 1.0g)")
        print(f"      Accel scale: [{accel_scale[0]:.4f}, {accel_scale[1]:.4f}, {accel_scale[2]:.4f}]")

        # Check gyro bias magnitude (should be small)
        gyro_bias_mag = np.linalg.norm(gyro_bias)
        if gyro_bias_mag > 1.0:
            print(f"      WARNING: Large gyro bias ({gyro_bias_mag:.3f} deg/s) - sensor may need recalibration")

        # Check if accel magnitude is reasonable
        if abs(accel_magnitude - 1.0) > 0.2:
            print(f"      WARNING: Accel magnitude far from 1g - T-pose may not be static or sensor misaligned")

    return SensorCalibrationParams(gyro_bias, accel_bias, accel_scale)


def apply_sensor_calibration(
    raw_frames: List[RawIMUFrame],
    calibration: SensorCalibration,
    verbose: bool = False
) -> List[RawIMUFrame]:
    """
    Apply sensor intrinsic calibration to raw IMU data.

    Applies bias and scale corrections to all gyroscope and accelerometer data.
    Magnetometer data is passed through unchanged (VQF normalizes internally).

    Corrections applied:
    - Gyroscope: raw - bias
    - Accelerometer: (raw - bias) * scale
    - Magnetometer: unchanged (VQF handles normalization)

    Args:
        raw_frames: List of uncalibrated raw IMU frames
        calibration: SensorCalibration with bias/scale parameters
        verbose: Print calibration progress

    Returns:
        List of calibrated raw IMU frames
    """
    if verbose:
        print(f"Applying sensor calibration to {len(raw_frames)} frames...")

    calibrated_frames = []

    for frame in raw_frames:
        # Apply calibration to each sensor
        chest_cal = _apply_single_sensor_calibration(frame.chest, calibration.chest)
        upper_arm_cal = _apply_single_sensor_calibration(frame.upper_arm, calibration.upper_arm)
        forearm_cal = _apply_single_sensor_calibration(frame.forearm, calibration.forearm)
        hand_cal = _apply_single_sensor_calibration(frame.hand, calibration.hand)

        # Create calibrated frame
        calibrated_frame = RawIMUFrame(
            timestamp=frame.timestamp,
            chest=chest_cal,
            upper_arm=upper_arm_cal,
            forearm=forearm_cal,
            hand=hand_cal
        )
        calibrated_frames.append(calibrated_frame)

    if verbose:
        print(f"  Applied calibration to {len(calibrated_frames)} frames")

    return calibrated_frames


def _apply_single_sensor_calibration(
    sensor_data: RawIMUData,
    params: SensorCalibrationParams
) -> RawIMUData:
    """
    Apply calibration to a single sensor's data.

    Args:
        sensor_data: Uncalibrated raw IMU data
        params: Calibration parameters (bias, scale)

    Returns:
        Calibrated RawIMUData
    """
    # Apply gyroscope bias correction
    gyro = sensor_data.gyro_array() - params.gyro_bias  # deg/s

    # Apply accelerometer bias and scale correction
    accel = (sensor_data.accel_array() - params.accel_bias) * params.accel_scale  # g

    # Magnetometer: pass through unchanged (VQF normalizes)
    mag = sensor_data.mag_array()  # µT

    return RawIMUData(
        timestamp=sensor_data.timestamp,
        acc_x=accel[0],
        acc_y=accel[1],
        acc_z=accel[2],
        gyro_x=gyro[0],
        gyro_y=gyro[1],
        gyro_z=gyro[2],
        mag_x=mag[0],
        mag_y=mag[1],
        mag_z=mag[2]
    )


if __name__ == "__main__":
    print("IMU Sensor Calibration Module")
    print("=" * 70)
    print("\nThis is a library module. Import it in your scripts:")
    print("  from sensor_calibration import compute_sensor_calibration, apply_sensor_calibration")
    print("\n  # Compute calibration from static T-pose region")
    print("  calibration = compute_sensor_calibration(raw_frames, tpose_start, tpose_end)")
    print("\n  # Apply calibration to raw IMU data")
    print("  calibrated_frames = apply_sensor_calibration(raw_frames, calibration)")
    print("\nNote: This implements simplified Ferraris calibration using T-pose.")
    print("      For best results, collect dedicated tumbling test data (6+ orientations).")
