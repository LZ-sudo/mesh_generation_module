#!/usr/bin/env python3
"""
VQF Sensor Fusion - Estimate orientation quaternions from raw IMU data.

This module uses the VQF (Versatile Quaternion-based Filter) algorithm to fuse
raw accelerometer, gyroscope, and magnetometer data into orientation quaternions.

VQF achieves 2.9° RMSE vs 5.3-16.7° for other sensor fusion algorithms and
includes magnetic disturbance rejection (critical for indoor use).

Functions:
    - fuse_sensor_data(): Apply VQF sensor fusion to raw IMU frames
    - fuse_single_sensor(): Apply VQF to one sensor's data stream
"""

from pathlib import Path
from typing import List
import numpy as np
import vqf

from cometa_parser import RawIMUFrame, RawIMUData, IMUFrame, QuaternionFrame


def fuse_sensor_data(
    raw_frames: List[RawIMUFrame],
    sample_rate: float = 2000.0,
    mag_disturbance_rejection: bool = True,
    verbose: bool = False
) -> List[IMUFrame]:
    """
    Fuse raw IMU data into orientation quaternions using VQF algorithm.

    VQF (Versatile Quaternion-based Filter) is a state-of-the-art sensor fusion
    algorithm published in Information Fusion 2023. It outperforms Madgwick,
    Mahony, and EKF filters with 2.9° RMSE vs 5.3-16.7° for competitors.

    Args:
        raw_frames: List of RawIMUFrame objects with accel, gyro, mag data
        sample_rate: Sensor sample rate in Hz (default 2000.0 for Cometa)
        mag_disturbance_rejection: Enable magnetic disturbance rejection (default True)
        verbose: Print fusion progress

    Returns:
        List of IMUFrame objects with fused orientation quaternions

    Example:
        raw_frames = parse_raw_imu_data(Path('capture.txt'))
        fused_frames = fuse_sensor_data(raw_frames, verbose=True)
    """
    if not raw_frames:
        raise ValueError("No raw IMU frames provided")

    if verbose:
        print(f"Fusing {len(raw_frames)} frames using VQF...")
        print(f"  Sample rate: {sample_rate:.0f} Hz")
        print(f"  Magnetic disturbance rejection: {mag_disturbance_rejection}")

    # Fuse each sensor independently
    chest_quats = _fuse_single_sensor(
        [f.chest for f in raw_frames],
        sample_rate,
        mag_disturbance_rejection,
        verbose,
        sensor_name="Chest"
    )

    upper_arm_quats = _fuse_single_sensor(
        [f.upper_arm for f in raw_frames],
        sample_rate,
        mag_disturbance_rejection,
        verbose,
        sensor_name="Upper Arm"
    )

    forearm_quats = _fuse_single_sensor(
        [f.forearm for f in raw_frames],
        sample_rate,
        mag_disturbance_rejection,
        verbose,
        sensor_name="Forearm"
    )

    hand_quats = _fuse_single_sensor(
        [f.hand for f in raw_frames],
        sample_rate,
        mag_disturbance_rejection,
        verbose,
        sensor_name="Hand"
    )

    # Combine into IMUFrame objects
    fused_frames = []
    for i, raw_frame in enumerate(raw_frames):
        frame = IMUFrame(
            timestamp=raw_frame.timestamp,
            chest=QuaternionFrame(raw_frame.timestamp, *chest_quats[i]),
            upper_arm=QuaternionFrame(raw_frame.timestamp, *upper_arm_quats[i]),
            forearm=QuaternionFrame(raw_frame.timestamp, *forearm_quats[i]),
            hand=QuaternionFrame(raw_frame.timestamp, *hand_quats[i])
        )
        fused_frames.append(frame)

    if verbose:
        print(f"  Fusion complete: {len(fused_frames)} frames")

    return fused_frames


def _fuse_single_sensor(
    sensor_data: List[RawIMUData],
    sample_rate: float,
    mag_disturbance_rejection: bool,
    verbose: bool,
    sensor_name: str = "Sensor"
) -> np.ndarray:
    """
    Apply VQF sensor fusion to a single sensor's data stream.

    Args:
        sensor_data: List of RawIMUData for one sensor
        sample_rate: Sample rate in Hz
        mag_disturbance_rejection: Enable magnetic disturbance rejection
        verbose: Print fusion info
        sensor_name: Name for verbose output

    Returns:
        Numpy array of quaternions [N x 4] in [w, x, y, z] format
    """
    if verbose:
        print(f"    Fusing {sensor_name}...")

    # Create VQF instance
    # VQF(gyrTs, accTs=-1, magTs=-1)
    # Sampling time in seconds
    sampling_time = 1.0 / sample_rate

    # Initialize VQF with sampling time
    # accTs and magTs default to gyrTs if not provided
    vqf_instance = vqf.VQF(gyrTs=sampling_time)

    # Configure parameters
    if not mag_disturbance_rejection:
        vqf_instance.setMagDistRejectionEnabled(False)

    # Prepare data arrays
    n_samples = len(sensor_data)
    quaternions = np.zeros((n_samples, 4))

    # Process each sample
    for i, data in enumerate(sensor_data):
        # VQF expects:
        # - gyr: gyroscope in rad/s (we have deg/s, so convert)
        # - acc: accelerometer in m/s² (we have g, so convert: 1g = 9.81 m/s²)
        # - mag: magnetometer in arbitrary units (normalized internally)

        gyr = data.gyro_array() * np.pi / 180.0  # deg/s -> rad/s
        acc = data.accel_array() * 9.81  # g -> m/s²
        mag = data.mag_array()  # µT (arbitrary units, will be normalized)

        # Update VQF with new measurement
        vqf_instance.update(gyr, acc, mag)

        # Get 9D quaternion (uses magnetometer)
        # VQF returns quaternion in [w, x, y, z] format
        quat = vqf_instance.getQuat9D()
        quaternions[i] = quat

    if verbose:
        print(f"      Fused {n_samples} samples")

    return quaternions


if __name__ == "__main__":
    print("VQF Sensor Fusion Module")
    print("=" * 70)
    print("\nThis is a library module. Import it in your scripts:")
    print("  from vqf_fusion import fuse_sensor_data")
    print("  from cometa_parser import parse_raw_imu_data")
    print("\n  raw_frames = parse_raw_imu_data(Path('imu_data.txt'))")
    print("  fused_frames = fuse_sensor_data(raw_frames, verbose=True)")
    print("\nVQF Parameters:")
    print("  - Sample rate: 2000 Hz (Cometa default)")
    print("  - Magnetic disturbance rejection: Enabled (indoor use)")
    print("  - Algorithm: VQF (2.9° RMSE vs 5.3-16.7° for competitors)")
