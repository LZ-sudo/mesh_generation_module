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
the C3D ANALOG section.  The chest IMU quaternion (also present in the C3D
file) is read separately to animate Spine1 with trunk orientation.

Angle-to-BVH channel mapping (ZYX rotation order, arm rest direction = -X):
  Chest (Spine1) uses _quat_to_zyx() with per-axis sign conventions in
  _CHEST_SIGNS.  The chest IMU quaternion is T-pose calibrated and converted
  directly to ZYX Euler angles.

  Shoulder uses _euler_shoulder_to_zyx() with sign conventions defined in
  _RIGHT_SHOULDER_SIGNS / _LEFT_SHOULDER_SIGNS.  Cometa's XZ'Y'' anatomical
  Euler sequence maps to ZX'Y'' in BVH world axes: Z=ABD, X'=VFLEX, Y''=HFLEX
  (Henschke et al. 2022, PMC9364332).  The full rotation matrix is reconstructed
  and decomposed into BVH ZYX, eliminating the cos(elev) spherical coupling.
  Output signs (z_sign, y_sign, x_sign) correspond directly to the old
  spherical model's elev_sign, azi_sign, axial_sign respectively.

  Elbow uses _spherical_to_zyx() with per-joint sign conventions defined in
  _RIGHT_ELBOW_SIGNS / _LEFT_ELBOW_SIGNS.  The general formulas are:
    theta_z = atan2(elev_sign * sin(elev), cos(elev) * cos(azi))
    theta_y = asin(azi_sign  * cos(elev) * sin(azi))
    theta_x = _axial_correction(elev, azi) + axial_sign * axial_angle
              (axial correction is zero when apply_correction=False)
  The axial correction compensates for palm-orientation drift introduced by
  the Rz*Ry compound rotation at large joint angles (see _axial_correction).

  Wrist uses _euler_wrist_to_zxy() with the intrinsic ZXY Euler sequence
  in BVH world frame: R = Rz(FE) x Rx(Rot) x Ry(Rad).  For the right forearm
  along -X, FE maps to the world +Z axis (Rz), axial rotation to +X (Rx), and
  radial/ulnar deviation to +Y (Ry).  BVH CHANNELS order: Zrotation Xrotation
  Yrotation.  Sign conventions in _RIGHT_WRIST_SIGNS / _LEFT_WRIST_SIGNS.

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
import numpy as np
from scipy.spatial.transform import Rotation

from bvh_writer import write_bvh_hierarchy
from imu_calibration import quaternion_inverse, quaternion_multiply, quaternion_to_euler


# ---------------------------------------------------------------------------
# Cometa channel label constants (Cometa EMG and Motion Tools naming)
# ---------------------------------------------------------------------------

_RIGHT_CHANNEL_LABELS = {
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

_LEFT_CHANNEL_LABELS = {
    'shoulder_horiz': 'Left Shoulder :Horizontal Flexion/Extension',
    'shoulder_vert':  'Left Shoulder :Vertical Flexion/Extension',
    'shoulder_abd':   'Left Shoulder :Abduction/Adduction',
    'elbow_fe':       'Left Elbow :Flexion/Extension',
    'elbow_ps':       'Left Elbow :Pronation/Supination',
    'elbow_dev':      'Left Elbow :Deviation',
    'wrist_fe':       'Left Wrist :Flexion/Extension',
    'wrist_rad':      'Left Wrist :Ulnar/Radial Deviation',
    'wrist_rot':      'Left Wrist :CW/CCW Rotation',
}

_STATIC_3 = "0.000000 0.000000 0.000000 "
_STATIC_6 = "0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 "

# Chest IMU quaternion component channels: W, X, Y, Z order (:1, :2, :3, :4)
_CHEST_QUAT_LABELS = ('Chest :1', 'Chest :2', 'Chest :3', 'Chest :4')


# ---------------------------------------------------------------------------
# CALIBRATION SETTINGS  (adjust here if needed)
# ---------------------------------------------------------------------------

# Shoulder: Cometa internally uses intrinsic YXZ (confirmed by IL decompilation).
# After the BVH axis mapping (Cometa X->BVH Z, Cometa Z->BVH X, Cometa Y->BVH Y)
# the sequence becomes intrinsic YZX in BVH frame:
#   R = Ry(horiz) x Rz(abd) x Rx(vert)
# _euler_shoulder_to_yzx() reconstructs this rotation and decomposes into BVH YZX,
# placing the singularity at ABD=+-90 deg (full coronal abduction) rather than
# at horiz=+-90 deg (arm forward) as the old ZYX decomposition did.
# abd_sign:   sign applied to shoulder_abd before Euler reconstruction
# vert_sign:  sign applied to shoulder_vert before Euler reconstruction
# horiz_sign: sign applied to shoulder_horiz before Euler reconstruction
# z_sign:     sign applied to output theta_z (ABD / lateral elevation channel)
# y_sign:     sign applied to output theta_y (horiz / forward channel)
# x_sign:     sign applied to output theta_x (axial rotation channel)
_RIGHT_SHOULDER_SIGNS = dict(abd_sign=1.0, vert_sign=1.0, horiz_sign=1.0, z_sign=-1.0, y_sign=1.0, x_sign=1.0)
_LEFT_SHOULDER_SIGNS  = dict(abd_sign=1.0, vert_sign=1.0, horiz_sign=1.0, z_sign=1.0, y_sign=-1.0, x_sign=-1.0)

# Elbow: spherical coordinate convention (unchanged).
# elev_sign:        sign of sin(elev) in the atan2 numerator for theta_z
# azi_sign:         sign of cos(elev)*sin(azi) in the asin argument for theta_y
# apply_correction: whether _axial_correction() is added to theta_x
# axial_sign:       sign applied to axial_deg when computing theta_x
# Left-arm equivalents: elev_sign and azi_sign are geometrically flipped because
# the left arm rests in +X (vs right arm -X).  axial_sign values are initial
# guesses -- validate against a left-arm recording in Blender and adjust if needed.
_RIGHT_ELBOW_SIGNS    = dict(elev_sign=-1.0, azi_sign=1.0, apply_correction=True,  axial_sign=-1.0)
_LEFT_ELBOW_SIGNS     = dict(elev_sign= 1.0, azi_sign=-1.0, apply_correction=True,  axial_sign= 1.0)

# Wrist: Cometa internally uses intrinsic YXZ (confirmed by IL decompilation).
# C3D channel mapping:
#   wrist_fe  = Y angle (1st rotation) -- Flexion/Extension
#   wrist_rot = X angle (2nd rotation) -- CW/CCW Rotation
#   wrist_rad = Z angle (3rd rotation) -- Ulnar/Radial Deviation
# For the right forearm along -X in BVH world, the axis mapping is:
#   FE  (mediolateral axis) -> BVH Z-axis -> Rz(wrist_fe)
#   Rot (forearm long axis) -> BVH X-axis -> Rx(wrist_rot)
#   Rad (dorsopalmar axis)  -> BVH Y-axis -> Ry(wrist_rad)
# Correct BVH reconstruction: R = Rz(FE) x Rx(-Rot) x Ry(Rad) = ZXY intrinsic.
# rot_sign=-1: Cometa's forearm long axis (X_cometa) points elbow->wrist = -X_bvh,
# so Rx_cometa(+Rot) = Rx_bvh(-Rot).  FE and Rad axes are unaffected.
# Singularity at wrist_rot=+-90 deg (full CW/CCW rotation), rarely reached.
# WristL negates all three outputs (IL confirmed).  With rot_sign=-1 for both:
#   WristR effective: (FE, -Rot, Rad); WristL effective: (-FE, +Rot, -Rad) ✓
# fe_sign:  sign applied to wrist_fe before reconstruction
# rot_sign: sign applied to wrist_rot before reconstruction (-1 = axis direction flip)
# rad_sign: sign applied to wrist_rad before reconstruction
# z_sign:   sign applied to output theta_z (FE channel)
# x_sign:   sign applied to output theta_x (CW/CCW rotation channel)
# y_sign:   sign applied to output theta_y (deviation channel)
_RIGHT_WRIST_SIGNS    = dict(fe_sign=1.0, rot_sign=-1.0, rad_sign=-1.0, z_sign=1.0, x_sign=1.0, y_sign=1.0)
_LEFT_WRIST_SIGNS     = dict(fe_sign=1.0, rot_sign=-1.0, rad_sign=1.0, z_sign=-1.0, x_sign=-1.0, y_sign=-1.0)

# Chest trunk sign conventions: change these if a trunk motion is inverted.
# z_sign: sign of theta_z (lateral lean:           +1 = right-side up)
# y_sign: sign of theta_y (axial rotation / twist: +1 = CCW when viewed from above)
# x_sign: sign of theta_x (sagittal bend:          +1 = forward)
_CHEST_SIGNS = dict(z_sign=1.0, y_sign=1.0, x_sign=1.0)

# Duration (seconds) of the initial T-pose segment used for the wrist_rad
# offset correction.  The subject must be in T-pose for at least this many
# seconds at the start of every recording.
TPOSE_DURATION_S: float = 1.0

# Wrist radial/ulnar manual bias (degrees). Applied after T-pose offset
# correction. Use to trim any residual radial/ulnar deviation visible in
# Blender after T-pose calibration. Positive = ulnar, negative = radial.
WRIST_RAD_BIAS_DEG: float = 0.0

# Wrist axial rotation bias (degrees). Applied after T-pose offset correction.
# Compensates for the static convention offset between CMU BVH's zero-rotation
# hand orientation and Cometa's T-pose palm direction.  At T-pose, CMU BVH with
# all-zero rotations has the hand local frame = world frame; if the palm in
# Blender faces backward (-Z) when it should face down (-Y), set this to -90.0.
# If it faces forward (+Z) instead, set to +90.0.
WRIST_ROT_BIAS_DEG: float = 0

# Shoulder abduction/adduction bias (degrees) added after T-pose offset correction.
# Positive = abduction (raises arm), negative = adduction (lowers arm).
# Adjust if the arm drifts above or below horizontal at rest after T-pose calibration.
# This offset also suppresses theta_z contamination at large shoulder_horiz angles,
# where small shoulder_abd residuals get amplified by the spherical-to-ZYX formula.
SHOULDER_ABD_BIAS_DEG: float = 0.0

# Elbow deviation bias (degrees) added after T-pose offset correction.
# Positive = deviation upward, negative = deviation downward.
# Adjust if the forearm is not straight at rest after T-pose calibration.
ELBOW_DEV_BIAS_DEG: float = 0.0


@dataclass
class JointAngleFrame:
    """One frame of pre-computed anatomical joint angles from Cometa C3D."""
    timestamp: float
    # Chest quaternion [W, X, Y, Z] calibrated so T-pose = identity (1, 0, 0, 0)
    chest_quat: Tuple[float, float, float, float]
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


def _detect_arm_side(label_to_idx: dict) -> str:
    """
    Detect which arm is instrumented from the C3D analog channel labels.

    Checks for the shoulder abduction/adduction channel of each side, which
    is always present in a Cometa joint-angle export regardless of recording
    configuration.

    Args:
        label_to_idx: Mapping of channel label string to analog array index.

    Returns:
        'right' or 'left'

    Raises:
        ValueError: If neither right nor left shoulder channels are present.
    """
    if 'Right Shoulder :Abduction/Adduction' in label_to_idx:
        return 'right'
    if 'Left Shoulder :Abduction/Adduction' in label_to_idx:
        return 'left'
    raise ValueError(
        "No arm joint angle channels found in C3D file. "
        "Expected 'Right Shoulder :Abduction/Adduction' or "
        "'Left Shoulder :Abduction/Adduction' in the analog channel labels."
    )


def parse_c3d_joint_angles(
    c3d_path: Path,
    tpose_duration: float = TPOSE_DURATION_S,
    wrist_rad_bias_deg: float = WRIST_RAD_BIAS_DEG,
    wrist_rot_bias_deg: float = WRIST_ROT_BIAS_DEG,
    shoulder_abd_bias_deg: float = SHOULDER_ABD_BIAS_DEG,
    elbow_dev_bias_deg: float = ELBOW_DEV_BIAS_DEG,
    verbose: bool = False,
) -> Tuple[List[JointAngleFrame], str]:
    """
    Parse pre-computed joint angles from a Cometa C3D file.

    Cometa stores all analog data at 2000 Hz but the actual IMU update rate
    is ~142.857 Hz (every 14th sample is unique). This function extracts the
    unique frames at the effective IMU rate by stepping through the 2000 Hz
    data at the IMU stride interval.

    T-pose offset corrections are applied to shoulder, elbow, and wrist
    channels: the mean value of each channel over the first tpose_duration
    seconds is subtracted from all frames to remove sensor bias at anatomical
    neutral.  Removing shoulder_abd bias is important because the
    spherical-to-ZYX formula amplifies small shoulder_abd residuals into large
    theta_z artifacts at large shoulder_horiz angles.  Removing wrist_fe,
    wrist_rot, and wrist_rad biases ensures the hand is in the correct rest
    pose at T-pose.  Additional per-channel bias constants can be added after
    T-pose correction to compensate for any residual visible in Blender.

    Args:
        c3d_path: Path to Cometa C3D file
        tpose_duration: Duration (seconds) of the initial T-pose segment used
            to compute calibration offsets for shoulder_abd, shoulder_vert, and
            elbow_dev (default: 1.0 s)
        wrist_rad_bias_deg: Extra offset (deg) added to all corrected wrist_rad
            values after T-pose correction (default: 15.0)
        shoulder_abd_bias_deg: Extra offset (deg) added to all corrected
            shoulder_abd values after T-pose correction (default: 0.0)
        elbow_dev_bias_deg: Extra offset (deg) added to all corrected
            elbow_dev values after T-pose correction (default: 0.0)
        verbose: Print parsing details

    Returns:
        Tuple of (frames, side) where frames is a List of JointAngleFrame at
        ~142.857 Hz with wrist_rad offset corrected, and side is 'right' or 'left'.

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

    side = _detect_arm_side(label_to_idx)
    channel_labels = _RIGHT_CHANNEL_LABELS if side == 'right' else _LEFT_CHANNEL_LABELS

    def _get_channel(key: str):
        label = channel_labels[key]
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

    chest_labels = _CHEST_QUAT_LABELS
    chest_ch = [analogs[0, label_to_idx[lbl], :] for lbl in chest_labels]

    n_total = analogs.shape[2]
    frames = []
    for i in range(0, n_total, step):
        frames.append(JointAngleFrame(
            timestamp=i / analog_rate,
            chest_quat=(
                float(chest_ch[0][i]),
                float(chest_ch[1][i]),
                float(chest_ch[2][i]),
                float(chest_ch[3][i]),
            ),
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

    # Subtract T-pose offsets for shoulder, elbow, and wrist channels.
    # Cometa reports non-zero values at anatomical neutral for these channels;
    # the mean over the initial T-pose segment is removed from every frame.
    # Zeroing shoulder_abd ensures the arm is horizontal at T-pose and prevents
    # contamination of the VFLEX and HFLEX channels in the Euler reconstruction.
    # Zeroing shoulder_vert removes the sensor-to-segment VFLEX bias so that
    # the arm lies in the coronal plane at T-pose.
    # Zeroing wrist_fe, wrist_rot, and wrist_rad removes palm-orientation drift
    # at anatomical neutral, ensuring the hand is in the correct rest pose.
    n_tpose = max(1, min(int(round(tpose_duration * imu_rate)), len(frames)))

    shoulder_abd_offset = sum(f.shoulder_abd for f in frames[:n_tpose]) / n_tpose
    for frame in frames:
        frame.shoulder_abd = frame.shoulder_abd - shoulder_abd_offset + shoulder_abd_bias_deg

    shoulder_vert_offset = sum(f.shoulder_vert for f in frames[:n_tpose]) / n_tpose
    for frame in frames:
        frame.shoulder_vert = frame.shoulder_vert - shoulder_vert_offset

    elbow_dev_offset = sum(f.elbow_dev for f in frames[:n_tpose]) / n_tpose
    for frame in frames:
        frame.elbow_dev = frame.elbow_dev - elbow_dev_offset + elbow_dev_bias_deg

    wrist_fe_offset = sum(f.wrist_fe for f in frames[:n_tpose]) / n_tpose
    for frame in frames:
        frame.wrist_fe = frame.wrist_fe - wrist_fe_offset

    wrist_rot_offset = sum(f.wrist_rot for f in frames[:n_tpose]) / n_tpose
    for frame in frames:
        frame.wrist_rot = frame.wrist_rot - wrist_rot_offset + wrist_rot_bias_deg

    wrist_rad_offset = sum(f.wrist_rad for f in frames[:n_tpose]) / n_tpose
    for frame in frames:
        frame.wrist_rad = frame.wrist_rad - wrist_rad_offset + wrist_rad_bias_deg

    # Apply chest quaternion T-pose calibration with world-frame correction.
    # The raw chest quaternion captures the sensor's orientation in Cometa's
    # world frame; at T-pose this is non-identity.  Left-multiplying by the
    # T-pose inverse (chest_offset * q_raw) is algebraically equivalent to
    # right-multiply calibration followed by the similarity transform
    # R_CB * q_cal * R_CB^{-1} used by the quaternion pipeline.  This maps
    # T-pose to identity AND re-expresses subsequent rotations in BVH's world
    # frame (Y-up, Z-forward), so that trunk twist maps to Y-rotation and
    # trunk lean maps to Z/X-rotation as expected.
    q_stack = np.array([list(f.chest_quat) for f in frames[:n_tpose]])
    q_ref = q_stack[0]
    for i in range(1, len(q_stack)):
        if np.dot(q_stack[i], q_ref) < 0:
            q_stack[i] = -q_stack[i]
    q_tpose_mean = q_stack.mean(axis=0)
    q_tpose_mean = q_tpose_mean / np.linalg.norm(q_tpose_mean)
    chest_offset = quaternion_inverse(q_tpose_mean)
    for frame in frames:
        q_raw = np.array(frame.chest_quat)
        if np.linalg.norm(q_raw) > 0.5:
            q_cal = quaternion_multiply(chest_offset, q_raw)
        else:
            # Zero-norm sample (dropped IMU frame in C3D): treat as no rotation.
            q_cal = np.array([1.0, 0.0, 0.0, 0.0])
        frame.chest_quat = (float(q_cal[0]), float(q_cal[1]), float(q_cal[2]), float(q_cal[3]))

    if verbose:
        print(f"  Arm side: {side}")
        print(f"  Analog rate: {analog_rate:.0f} Hz, IMU rate: {imu_rate:.3f} Hz (stride={step})")
        print(f"  Extracted {len(frames)} unique frames")
        print(f"  Duration: {frames[-1].timestamp:.2f}s")
        print(f"  shoulder_abd  T-pose offset removed: {shoulder_abd_offset:+.2f} deg")
        if shoulder_abd_bias_deg != 0.0:
            print(f"  shoulder_abd  extra bias applied:    {shoulder_abd_bias_deg:+.2f} deg")
        print(f"  shoulder_vert T-pose offset removed: {shoulder_vert_offset:+.2f} deg")
        print(f"  elbow_dev      T-pose offset removed: {elbow_dev_offset:+.2f} deg")
        if elbow_dev_bias_deg != 0.0:
            print(f"  elbow_dev      extra bias applied:    {elbow_dev_bias_deg:+.2f} deg")
        print(f"  wrist_fe       T-pose offset removed: {wrist_fe_offset:+.2f} deg")
        print(f"  wrist_rot      T-pose offset removed: {wrist_rot_offset:+.2f} deg")
        if wrist_rot_bias_deg != 0.0:
            print(f"  wrist_rot      extra bias applied:    {wrist_rot_bias_deg:+.2f} deg")
        print(f"  wrist_rad      T-pose offset removed: {wrist_rad_offset:+.2f} deg")
        if wrist_rad_bias_deg != 0.0:
            print(f"  wrist_rad      extra bias applied:    {wrist_rad_bias_deg:+.2f} deg")
        print(f"  Chest T-pose quat:  [{q_tpose_mean[0]:+.4f}, {q_tpose_mean[1]:+.4f}, "
              f"{q_tpose_mean[2]:+.4f}, {q_tpose_mean[3]:+.4f}]")

    return frames, side


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


def _axial_correction(dev_deg: float, fe_deg: float, side: str = 'right') -> float:
    """
    Compute the BVH theta_x bias introduced by the ZY compound rotation.

    When a limb is described by spherical coordinates (dev, fe) and mapped to
    BVH ZYX Euler angles, the Rz*Ry compound rotation moves the limb to the
    correct direction but rotates its axial reference away from the T-pose
    orientation.  This function returns the correction angle (degrees) that
    must be added to theta_x so that theta_x = 0 corresponds to the same palm
    orientation that Cometa's zero pronation/supination (ps = 0) represents
    (i.e. the rigid-body-rotation of the T-pose palm direction).

    Algorithm:
      1. Compute the new limb direction d_new from (dev, fe) in spherical coords.
      2. Find the rigid-body rotation R from d_tpose=(-1,0,0) to d_new (Rodrigues).
      3. Apply R to p_tpose=(0,-1,0) to get the expected palm direction.
      4. Transform expected_palm to limb-local frame by undoing the BVH ZY rotation.
      5. Return atan2(-p_local[2], -p_local[1]), the Rx angle that reproduces it.

    Returns 0 when dev=0 and fe=0 (T-pose identity check).

    Args:
        dev_deg: Deviation / carrying angle (deg)
        fe_deg:  Flexion / azimuth angle (deg)

    Returns:
        Correction offset in degrees to add to theta_x
    """
    dev = math.radians(dev_deg)
    fe  = math.radians(fe_deg)

    # Right arm rests along -X; left arm rests along +X.
    x_sign  = -1.0 if side == 'right' else 1.0
    d_tpose = (x_sign, 0.0, 0.0)
    p_tpose = (0.0, -1.0, 0.0)
    d_new   = (
        x_sign * math.cos(dev) * math.cos(fe),
        math.sin(dev),
        math.cos(dev) * math.sin(fe),
    )

    def _dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def _cross(a, b):
        return (
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0],
        )

    def _norm(v):
        n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        return (v[0]/n, v[1]/n, v[2]/n)

    def _rodrigues(v, k, angle):
        c, s = math.cos(angle), math.sin(angle)
        kxv = _cross(k, v)
        kd  = _dot(k, v)
        return (
            c*v[0] + s*kxv[0] + (1 - c)*kd*k[0],
            c*v[1] + s*kxv[1] + (1 - c)*kd*k[1],
            c*v[2] + s*kxv[2] + (1 - c)*kd*k[2],
        )

    cos_theta = max(-1.0, min(1.0, _dot(d_tpose, d_new)))
    if abs(cos_theta - 1.0) < 1e-9:
        expected_palm = p_tpose
    elif abs(cos_theta + 1.0) < 1e-9:
        expected_palm = _rodrigues(p_tpose, (0.0, 1.0, 0.0), math.pi)
    else:
        k_axis = _norm(_cross(d_tpose, d_new))
        expected_palm = _rodrigues(p_tpose, k_axis, math.acos(cos_theta))

    # BVH ZY angles for d_new — sign conventions mirror _spherical_to_zyx.
    # Right arm (x_sign=-1): elev_sign=-1, azi_sign=+1
    # Left arm  (x_sign=+1): elev_sign=+1, azi_sign=-1
    theta_y = math.asin(-x_sign * math.cos(dev) * math.sin(fe))
    theta_z = math.atan2( x_sign * math.sin(dev), math.cos(dev) * math.cos(fe))

    # Undo ZY to bring expected_palm into limb-local frame
    # p_local = Ry(-theta_y) * Rz(-theta_z) * expected_palm
    cy, sy = math.cos(-theta_y), math.sin(-theta_y)
    cz, sz = math.cos(-theta_z), math.sin(-theta_z)
    ex, ey, ez = expected_palm
    # Rz(-theta_z)
    rx = cz*ex - sz*ey
    ry = sz*ex + cz*ey
    rz = ez
    # Ry(-theta_y) — only Y and Z components needed for alpha
    py =  ry
    pz = -sy*rx + cy*rz

    # Rx(alpha)*(0,-1,0) = (0,-cos(alpha),-sin(alpha)), match py and pz
    # => alpha = atan2(-pz, -py)
    return math.degrees(math.atan2(-pz, -py))


def _spherical_to_zyx(
    elev_deg: float,
    azi_deg: float,
    axial_deg: float,
    *,
    elev_sign: float,
    azi_sign: float,
    apply_correction: bool,
    axial_sign: float,
    side: str = 'right',
) -> Tuple[float, float, float]:
    """
    Convert Cometa spherical joint angles to BVH ZYX Euler angles.

    All three arm joints (shoulder, elbow, wrist) share the same spherical
    coordinate convention where the limb rests along -X and is displaced by
    two angular coordinates (elevation, azimuth) plus an axial rotation.
    Sign conventions differ per joint and are controlled by the keyword
    arguments -- see _RIGHT_SHOULDER_SIGNS/_LEFT_SHOULDER_SIGNS etc.

    General formulas:
      theta_z = atan2(elev_sign * sin(elev), cos(elev) * cos(azi))
      theta_y = asin(azi_sign  * cos(elev) * sin(azi))
      theta_x = _axial_correction(elev, azi) + axial_sign * axial_deg
                (correction term is zero when apply_correction=False)

    Args:
        elev_deg:         Elevation angle [deg] -- shoulder abd, elbow dev, wrist fe
        azi_deg:          Azimuth angle [deg]   -- shoulder horiz, elbow fe, wrist rad
        axial_deg:        Axial rotation [deg]  -- shoulder vert, elbow ps, wrist rot
        elev_sign:        Sign of sin(elev) in the atan2 numerator for theta_z
        azi_sign:         Sign of cos(elev)*sin(azi) in the asin argument for theta_y
        apply_correction: If True, adds _axial_correction(elev_deg, azi_deg) to theta_x
        axial_sign:       Sign applied to axial_deg when computing theta_x

    Returns:
        (theta_z_deg, theta_y_deg, theta_x_deg) for BVH ZYX channels
    """
    elev = math.radians(elev_deg)
    azi  = math.radians(azi_deg)

    theta_y = math.asin(azi_sign * math.cos(elev) * math.sin(azi))
    theta_z = math.atan2(elev_sign * math.sin(elev), math.cos(elev) * math.cos(azi))
    correction = _axial_correction(elev_deg, azi_deg, side) if apply_correction else 0.0
    theta_x = math.radians(correction + axial_sign * axial_deg)

    return math.degrees(theta_z), math.degrees(theta_y), math.degrees(theta_x)


def _euler_shoulder_to_zyx(
    abd_deg: float,
    vert_deg: float,
    horiz_deg: float,
    *,
    abd_sign: float,
    vert_sign: float,
    horiz_sign: float,
    z_sign: float,
    y_sign: float,
    x_sign: float,
) -> Tuple[float, float, float]:
    """
    Convert Cometa shoulder Euler angles to BVH YZX Euler angles.

    Cometa internally uses intrinsic YXZ (confirmed by IL decompilation of
    EMGandMotionTools.exe: QuatToEulerAngles(q, 'yxz')).  After the BVH axis
    mapping (Cometa X->BVH Z, Cometa Z->BVH X, Cometa Y->BVH Y) the correct
    reconstruction in BVH world frame is intrinsic YZX:
      R = Ry(horiz) x Rz(abd) x Rx(vert)

    This is decomposed back into YZX for BVH output:
      Y (1st) = Horizontal Flex/Ext     (shoulder_horiz) -- sagittal/Y axis
      Z (2nd) = Abduction/Adduction     (shoulder_abd)   -- coronal/Z axis
      X (3rd) = Vertical Flex/Ext       (shoulder_vert)  -- humerus axial

    The singularity of YZX is at the middle angle Z = +-90 deg, which
    corresponds to full coronal abduction -- rarely reached in therapy.
    The arm-forward position (horiz ~90 deg) is singularity-free under YZX.

    Args:
        abd_deg:    Abduction/Adduction [deg], T-pose corrected
        vert_deg:   Vertical Flexion/Extension [deg], T-pose corrected
        horiz_deg:  Horizontal Flexion/Extension [deg]
        abd_sign:   Sign applied to abd_deg before Euler reconstruction
        vert_sign:  Sign applied to vert_deg before Euler reconstruction
        horiz_sign: Sign applied to horiz_deg before Euler reconstruction
        z_sign:     Sign applied to output theta_z (ABD channel)
        y_sign:     Sign applied to output theta_y (horiz channel)
        x_sign:     Sign applied to output theta_x (axial channel)

    Returns:
        (theta_y_deg, theta_z_deg, theta_x_deg) for BVH YZX channels
    """
    R = Rotation.from_euler(
        'yzx',
        [horiz_sign * horiz_deg, abd_sign * abd_deg, vert_sign * vert_deg],
        degrees=True,
    )
    ty, tz, tx = R.as_euler('yzx', degrees=True)
    return y_sign * ty, z_sign * tz, x_sign * tx


def _euler_wrist_to_zxy(
    fe_deg: float,
    rot_deg: float,
    rad_deg: float,
    *,
    fe_sign: float,
    rot_sign: float,
    rad_sign: float,
    z_sign: float,
    x_sign: float,
    y_sign: float,
) -> Tuple[float, float, float]:
    """
    Convert Cometa wrist Euler angles to BVH ZXY Euler angles.

    For the right forearm resting along -X in BVH world, the physical axis
    mapping is:
      FE  (mediolateral flexion axis) -> BVH +Z -> Rz rotation
      Rot (forearm long axis)         -> BVH +X -> Rx rotation
      Rad (dorsopalmar deviation axis)-> BVH +Y -> Ry rotation

    Correct BVH reconstruction: R = Rz(FE) x Rx(-Rot) x Ry(Rad) = ZXY intrinsic.
    The Rot sign is negated because Cometa's bone axis (elbow->wrist) is -X_bvh.

    This is decomposed back into ZXY for BVH output:
      Z (1st) = Flexion/Extension      (wrist_fe)  -- FE channel
      X (2nd) = CW/CCW Rotation        (wrist_rot) -- rotation channel
      Y (3rd) = Ulnar/Radial Deviation (wrist_rad) -- deviation channel

    The singularity of ZXY is at the middle angle X = +-90 deg, which
    corresponds to 90 deg of wrist CW/CCW rotation -- rarely reached in practice.

    WristL negates all three outputs (IL confirmed), captured in z_sign, x_sign,
    y_sign all set to -1.0 for the left wrist.

    Args:
        fe_deg:   Wrist Flexion/Extension [deg]
        rot_deg:  Wrist CW/CCW Rotation [deg]
        rad_deg:  Wrist Ulnar/Radial Deviation [deg]
        fe_sign:  Sign applied to fe_deg before Euler reconstruction
        rot_sign: Sign applied to rot_deg before Euler reconstruction
        rad_sign: Sign applied to rad_deg before Euler reconstruction
        z_sign:   Sign applied to output theta_z (FE channel)
        x_sign:   Sign applied to output theta_x (CW/CCW rotation channel)
        y_sign:   Sign applied to output theta_y (deviation channel)

    Returns:
        (theta_z_deg, theta_x_deg, theta_y_deg) for BVH ZXY channels
    """
    R = Rotation.from_euler(
        'zxy',
        [fe_sign * fe_deg, rot_sign * rot_deg, rad_sign * rad_deg],
        degrees=True,
    )
    tz, tx, ty = R.as_euler('zxy', degrees=True)
    return z_sign * tz, x_sign * tx, y_sign * ty


def _quat_to_zyx(
    chest_quat: Tuple[float, float, float, float],
    *,
    z_sign: float,
    y_sign: float,
    x_sign: float,
) -> Tuple[float, float, float]:
    """
    Convert a calibrated chest quaternion to BVH ZYX Euler angles.

    The quaternion must already be calibrated so that T-pose = identity
    (1, 0, 0, 0).  Sign conventions control the direction of each Euler
    channel and are defined in _CHEST_SIGNS in the CALIBRATION SETTINGS block.

    Args:
        chest_quat: Calibrated chest quaternion [W, X, Y, Z]
        z_sign:     Sign applied to theta_z (lateral lean)
        y_sign:     Sign applied to theta_y (axial rotation / twist)
        x_sign:     Sign applied to theta_x (sagittal bend)

    Returns:
        (theta_z_deg, theta_y_deg, theta_x_deg) for BVH ZYX channels
    """
    tz, ty, tx = quaternion_to_euler(np.array(chest_quat), order='ZYX')
    return z_sign * tz, y_sign * ty, x_sign * tx


def _map_angles_to_bvh(
    frame: JointAngleFrame,
    side: str,
) -> Tuple[
    Tuple[float, float, float],   # Spine1 (chest): Z, Y, X
    Tuple[float, float, float],   # Arm: Y, Z, X
    Tuple[float, float, float],   # ForeArm: Z, Y, X
    Tuple[float, float, float],   # Hand: Z, X, Y
]:
    """
    Map Cometa anatomical joint angles to BVH Euler channel values.

    The BVH skeleton uses CMU convention: +Y up, +Z forward, right arm at -X,
    left arm at +X.

    Chest uses _quat_to_zyx() with sign conventions defined in _CHEST_SIGNS.
    Shoulder uses _euler_shoulder_to_zyx() with the intrinsic YZX Euler
    sequence in BVH frame (HFLEX -> ABD -> VFLEX), derived from Cometa's
    internal YXZ sequence after BVH axis mapping.
    Elbow uses _spherical_to_zyx() with per-joint sign dicts selected by side:
    _RIGHT_ELBOW_SIGNS for right, _LEFT_ELBOW_SIGNS for left.
    Wrist uses _euler_wrist_to_zxy() with the intrinsic ZXY Euler sequence in
    BVH frame (Rz=FE -> Rx=Rot -> Ry=Rad), derived from the physical axis
    mapping of the right forearm along -X in BVH world.
    """
    if side == 'right':
        s_signs, e_signs, w_signs = _RIGHT_SHOULDER_SIGNS, _RIGHT_ELBOW_SIGNS, _RIGHT_WRIST_SIGNS
    else:
        s_signs, e_signs, w_signs = _LEFT_SHOULDER_SIGNS, _LEFT_ELBOW_SIGNS, _LEFT_WRIST_SIGNS

    chest   = _quat_to_zyx(frame.chest_quat, **_CHEST_SIGNS)
    arm     = _euler_shoulder_to_zyx(frame.shoulder_abd, frame.shoulder_vert, frame.shoulder_horiz, **s_signs)
    forearm = _spherical_to_zyx(frame.elbow_dev,    frame.elbow_fe,       frame.elbow_ps,      side=side, **e_signs)
    hand    = _euler_wrist_to_zxy(frame.wrist_fe,   frame.wrist_rot,      frame.wrist_rad,     **w_signs)

    return chest, arm, forearm, hand


def write_bvh_from_joint_angles(
    frames: List[JointAngleFrame],
    output_path: Path,
    side: str = 'right',
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
                chest, arm, forearm, hand = _map_angles_to_bvh(frame, side)
                # 96 channels total: 6 (Hips) + 30 joints x 3
                # Animated: Spine1 + the instrumented arm chain (left or right)
                arm_str     = f"{arm[0]:.6f} {arm[1]:.6f} {arm[2]:.6f} "
                forearm_str = f"{forearm[0]:.6f} {forearm[1]:.6f} {forearm[2]:.6f} "
                hand_str    = f"{hand[0]:.6f} {hand[1]:.6f} {hand[2]:.6f} "
                spine1_str  = f"{chest[0]:.6f} {chest[1]:.6f} {chest[2]:.6f} "
                if side == 'right':
                    line = (
                        _STATIC_6                   # Hips (pos + rot)
                        + _STATIC_3 * 5             # LHipJoint..LeftToeBase
                        + _STATIC_3 * 5             # RHipJoint..RightToeBase
                        + _STATIC_3 * 2             # LowerBack, Spine
                        + spine1_str                # Spine1
                        + _STATIC_3 * 3             # Neck, Neck1, Head
                        + _STATIC_3 * 7             # LeftShoulder..LThumb
                        + _STATIC_3                 # RightShoulder
                        + arm_str                   # RightArm
                        + forearm_str               # RightForeArm
                        + hand_str                  # RightHand
                        + _STATIC_3 * 2             # RightFingerBase, RightHandIndex1
                        + "0.000000 0.000000 0.000000\n"  # RThumb
                    )
                else:
                    line = (
                        _STATIC_6                   # Hips (pos + rot)
                        + _STATIC_3 * 5             # LHipJoint..LeftToeBase
                        + _STATIC_3 * 5             # RHipJoint..RightToeBase
                        + _STATIC_3 * 2             # LowerBack, Spine
                        + spine1_str                # Spine1
                        + _STATIC_3 * 3             # Neck, Neck1, Head
                        + _STATIC_3                 # LeftShoulder
                        + arm_str                   # LeftArm
                        + forearm_str               # LeftForeArm
                        + hand_str                  # LeftHand
                        + _STATIC_3 * 2             # LeftFingerBase, LeftHandIndex1
                        + _STATIC_3                 # LThumb
                        + _STATIC_3                 # RightShoulder
                        + _STATIC_3 * 3             # RightArm, RightForeArm, RightHand
                        + _STATIC_3 * 2             # RightFingerBase, RightHandIndex1
                        + "0.000000 0.000000 0.000000\n"  # RThumb
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
  - Arm side (left/right) is auto-detected from the C3D analog channel labels.
  - Required channels: Shoulder/Elbow/Wrist (3 DOF each) for the detected arm side.
  - Chest (Spine1) is animated from the chest IMU quaternion; the detected arm chain
    uses pre-computed joint angles. The non-instrumented arm remains static.
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
        frames, side = parse_c3d_joint_angles(
            input_path,
            verbose=args.verbose,
        )
        print(f"  Arm side: {side}")
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
            side=side,
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
