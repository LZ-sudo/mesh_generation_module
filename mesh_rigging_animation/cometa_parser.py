#!/usr/bin/env python3
"""
Cometa IMU data parser for extracting quaternion time series.

This module parses Cometa Systems IMU sensor output files (tab-separated text
format) and extracts quaternion orientation data for conversion to BVH animation.

Functions:
    - parse_cometa_file(): Parse Cometa TXT file and extract quaternion time series
    - detect_tpose_frames(): Identify T-pose calibration frames at the start of recording
    - validate_sensor_data(): Check data quality and detect issues
"""

from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class QuaternionFrame:
    """Single frame of quaternion data from one IMU sensor."""
    timestamp: float
    w: float
    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        """Return quaternion as numpy array [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z])

    def normalize(self) -> 'QuaternionFrame':
        """Return normalized quaternion (unit magnitude)."""
        q = self.as_array()
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            raise ValueError(f"Quaternion has zero magnitude at t={self.timestamp}")
        q_norm = q / norm
        return QuaternionFrame(self.timestamp, q_norm[0], q_norm[1], q_norm[2], q_norm[3])


@dataclass
class IMUFrame:
    """Complete frame of IMU data from all sensors."""
    timestamp: float
    chest: QuaternionFrame
    upper_arm: QuaternionFrame
    forearm: QuaternionFrame
    hand: QuaternionFrame


def parse_cometa_file(
    file_path: Path,
    verbose: bool = False
) -> List[IMUFrame]:
    """
    Parse Cometa Systems IMU data file (tab-separated text format).

    The Cometa file format has:
    - Line 1: Filename header
    - Lines 2-4: Blank lines
    - Line 5: Tab-separated column headers
    - Lines 6+: Tab-separated numeric data (2000 Hz sample rate)

    Extracts quaternion data for 4 sensors: Chest, R.Right.Arm, R.Right.Forearm,
    R.Right.Hand. Each sensor has quaternion columns (W, X, Y, Z).

    Args:
        file_path: Path to Cometa .txt file
        verbose: Print parsing progress

    Returns:
        List of IMUFrame objects, one per timestamp

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or quaternion columns missing
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cometa file not found: {file_path}")

    if verbose:
        print(f"Parsing Cometa IMU file: {file_path.name}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError(f"File too short (expected header + data, got {len(lines)} lines)")

    # Line 5 (index 4) is the header
    header_line = lines[4].strip()
    headers = header_line.split('\t')

    if verbose:
        print(f"  Found {len(headers)} columns")

    # Find quaternion column indices for each sensor
    sensor_columns = _find_quaternion_columns(headers)

    if verbose:
        print(f"  Identified {len(sensor_columns)} sensors:")
        for sensor_name, cols in sensor_columns.items():
            print(f"    - {sensor_name}: W={cols['W']}, X={cols['X']}, Y={cols['Y']}, Z={cols['Z']}")

    # Parse data rows
    frames = []
    data_lines = lines[5:]  # Skip header and blank lines

    for line_num, line in enumerate(data_lines, start=6):
        line = line.strip()
        if not line:
            continue

        values = line.split('\t')
        if len(values) < len(headers):
            if verbose:
                print(f"  Warning: Line {line_num} has {len(values)} values, expected {len(headers)}. Skipping.")
            continue

        try:
            timestamp = float(values[0])

            # Extract quaternions for all sensors
            chest = QuaternionFrame(
                timestamp,
                float(values[sensor_columns['Chest']['W']]),
                float(values[sensor_columns['Chest']['X']]),
                float(values[sensor_columns['Chest']['Y']]),
                float(values[sensor_columns['Chest']['Z']])
            )

            upper_arm = QuaternionFrame(
                timestamp,
                float(values[sensor_columns['R.Right.Arm']['W']]),
                float(values[sensor_columns['R.Right.Arm']['X']]),
                float(values[sensor_columns['R.Right.Arm']['Y']]),
                float(values[sensor_columns['R.Right.Arm']['Z']])
            )

            forearm = QuaternionFrame(
                timestamp,
                float(values[sensor_columns['R.Right.Forearm']['W']]),
                float(values[sensor_columns['R.Right.Forearm']['X']]),
                float(values[sensor_columns['R.Right.Forearm']['Y']]),
                float(values[sensor_columns['R.Right.Forearm']['Z']])
            )

            hand = QuaternionFrame(
                timestamp,
                float(values[sensor_columns['R.Right.Hand']['W']]),
                float(values[sensor_columns['R.Right.Hand']['X']]),
                float(values[sensor_columns['R.Right.Hand']['Y']]),
                float(values[sensor_columns['R.Right.Hand']['Z']])
            )

            frame = IMUFrame(timestamp, chest, upper_arm, forearm, hand)
            frames.append(frame)

        except (ValueError, IndexError) as e:
            if verbose:
                print(f"  Warning: Failed to parse line {line_num}: {e}")
            continue

    if not frames:
        raise ValueError("No valid data frames found in file")

    if verbose:
        print(f"  Parsed {len(frames)} frames")
        print(f"  Duration: {frames[-1].timestamp:.2f} seconds")
        sample_rate = len(frames) / frames[-1].timestamp if frames[-1].timestamp > 0 else 0
        print(f"  Sample rate: ~{sample_rate:.0f} Hz")

    return frames


def _find_quaternion_columns(headers: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Find column indices for quaternion components (W, X, Y, Z) for each sensor.

    Expected column name patterns:
    - "Chest :W()", "Chest :X()", "Chest :Y()", "Chest :Z()"
    - "R.Right.Arm :W()", "R.Right.Arm :X()", ...
    - "R.Right.Forearm :W()", "R.Right.Forearm :X()", ...
    - "R.Right.Hand :W()", "R.Right.Hand :X()", ...

    Args:
        headers: List of column names from Cometa file

    Returns:
        Dictionary mapping sensor names to quaternion component indices

    Raises:
        ValueError: If required quaternion columns are missing
    """
    sensor_names = ['Chest', 'R.Right.Arm', 'R.Right.Forearm', 'R.Right.Hand']
    quaternion_components = ['W', 'X', 'Y', 'Z']

    result = {}

    for sensor in sensor_names:
        sensor_cols = {}
        for component in quaternion_components:
            # Pattern: "SensorName :Component():" (note trailing colon)
            pattern = f"{sensor} :{component}():"
            try:
                idx = headers.index(pattern)
                sensor_cols[component] = idx
            except ValueError:
                raise ValueError(
                    f"Missing quaternion column '{pattern}' for sensor '{sensor}'. "
                    f"Available headers: {headers[:10]}..."
                )

        result[sensor] = sensor_cols

    return result


def detect_tpose_frames(
    frames: List[IMUFrame],
    duration_seconds: float = 1.0,
    verbose: bool = False
) -> Tuple[int, int]:
    """
    Detect T-pose calibration frames at the start of the recording.

    Assumes the subject holds T-pose for the first 1-2 seconds. Detects the
    stable region by looking for low quaternion variance.

    Args:
        frames: List of IMU frames
        duration_seconds: Expected T-pose duration (default 1.0 second)
        verbose: Print detection info

    Returns:
        Tuple of (start_frame_idx, end_frame_idx) for T-pose region

    Raises:
        ValueError: If T-pose region cannot be reliably detected
    """
    if not frames:
        raise ValueError("No frames provided")

    # Estimate sample rate
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames to detect T-pose")

    dt = frames[1].timestamp - frames[0].timestamp
    sample_rate = 1.0 / dt if dt > 0 else 2000.0

    # Expected number of frames for T-pose duration
    expected_frames = int(duration_seconds * sample_rate)
    expected_frames = min(expected_frames, len(frames) - 1)

    if expected_frames < 10:
        raise ValueError(f"Too few frames ({expected_frames}) for reliable T-pose detection")

    # Extract chest quaternions for variance analysis
    chest_quats = np.array([f.chest.as_array() for f in frames[:expected_frames * 2]])

    # Compute rolling variance (window = expected_frames)
    window_size = min(expected_frames, 1000)  # Cap at 1000 frames for performance
    variances = []

    for i in range(len(chest_quats) - window_size):
        window = chest_quats[i:i + window_size]
        variance = np.var(window, axis=0).sum()
        variances.append(variance)

    # Find the window with minimum variance (most stable = T-pose)
    min_var_idx = np.argmin(variances)
    start_idx = min_var_idx
    end_idx = min_var_idx + window_size

    if verbose:
        print(f"T-pose detection:")
        print(f"  Sample rate: {sample_rate:.0f} Hz")
        print(f"  Expected T-pose frames: {expected_frames}")
        print(f"  Detected T-pose: frames {start_idx} to {end_idx}")
        print(f"  Timestamps: {frames[start_idx].timestamp:.3f}s to {frames[end_idx].timestamp:.3f}s")
        print(f"  Quaternion variance: {variances[min_var_idx]:.6f}")

    return start_idx, end_idx


def validate_sensor_data(
    frames: List[IMUFrame],
    verbose: bool = False
) -> bool:
    """
    Validate IMU sensor data quality and check for common issues.

    Checks for:
    - Quaternion magnitude (should be close to 1.0)
    - NaN or infinite values
    - Discontinuities (large frame-to-frame changes)

    Args:
        frames: List of IMU frames
        verbose: Print validation results

    Returns:
        True if data passes validation, False otherwise
    """
    if not frames:
        if verbose:
            print("Validation failed: No frames")
        return False

    issues = []

    # Check quaternion magnitudes
    for i, frame in enumerate(frames[::100]):  # Sample every 100th frame
        for sensor_name, quat in [
            ('Chest', frame.chest),
            ('Upper Arm', frame.upper_arm),
            ('Forearm', frame.forearm),
            ('Hand', frame.hand)
        ]:
            q_arr = quat.as_array()
            magnitude = np.linalg.norm(q_arr)

            if not np.isfinite(magnitude):
                issues.append(f"Frame {i*100}: {sensor_name} quaternion has non-finite values")
            elif abs(magnitude - 1.0) > 0.1:
                issues.append(f"Frame {i*100}: {sensor_name} quaternion magnitude {magnitude:.3f} (expected ~1.0)")

    # Check for discontinuities
    for i in range(min(1000, len(frames) - 1)):  # Check first 1000 frames
        dt = frames[i+1].timestamp - frames[i].timestamp
        if dt <= 0:
            issues.append(f"Frame {i}: Non-increasing timestamp")

        chest_diff = np.linalg.norm(frames[i+1].chest.as_array() - frames[i].chest.as_array())
        if chest_diff > 0.5:  # Large quaternion change between frames
            issues.append(f"Frame {i}: Large chest quaternion jump ({chest_diff:.3f})")

    if verbose:
        if issues:
            print(f"Validation found {len(issues)} issues:")
            for issue in issues[:10]:  # Print first 10
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("Validation passed: No issues detected")

    return len(issues) == 0


if __name__ == "__main__":
    print("Cometa IMU Parser")
    print("=" * 70)
    print("\nThis is a library module. Import it in your scripts:")
    print("  from cometa_parser import parse_cometa_file, detect_tpose_frames")
    print("  frames = parse_cometa_file(Path('imu_data.txt'), verbose=True)")
    print("  tpose_start, tpose_end = detect_tpose_frames(frames)")
