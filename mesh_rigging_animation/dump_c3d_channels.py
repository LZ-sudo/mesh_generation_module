#!/usr/bin/env python3
"""
Diagnostic tool: dump Cometa C3D joint angle channels to CSV.

Reads a Cometa C3D file and writes all shoulder/elbow/wrist angle channels
to a CSV file, along with a printed summary of T-pose offsets and per-channel
min/max values.  Useful for verifying sign conventions and T-pose corrections
before running the full c3d_to_bvh pipeline.

Usage:
    python dump_c3d_channels.py -i recording.c3d
    python dump_c3d_channels.py -i recording.c3d -o output.csv --tpose 1.0
"""

import argparse
import csv
import sys
from pathlib import Path

import ezc3d
import numpy as np


_RIGHT_CHANNELS = {
    'shoulder_abd':   'Right Shoulder :Abduction/Adduction',
    'shoulder_vert':  'Right Shoulder :Vertical Flexion/Extension',
    'shoulder_horiz': 'Right Shoulder :Horizontal Flexion/Extension',
    'elbow_fe':       'Right Elbow :Flexion/Extension',
    'elbow_ps':       'Right Elbow :Pronation/Supination',
    'elbow_dev':      'Right Elbow :Deviation',
    'wrist_fe':       'Right Wrist :Flexion/Extension',
    'wrist_rad':      'Right Wrist :Ulnar/Radial Deviation',
    'wrist_rot':      'Right Wrist :CW/CCW Rotation',
}

_LEFT_CHANNELS = {
    'shoulder_abd':   'Left Shoulder :Abduction/Adduction',
    'shoulder_vert':  'Left Shoulder :Vertical Flexion/Extension',
    'shoulder_horiz': 'Left Shoulder :Horizontal Flexion/Extension',
    'elbow_fe':       'Left Elbow :Flexion/Extension',
    'elbow_ps':       'Left Elbow :Pronation/Supination',
    'elbow_dev':      'Left Elbow :Deviation',
    'wrist_fe':       'Left Wrist :Flexion/Extension',
    'wrist_rad':      'Left Wrist :Ulnar/Radial Deviation',
    'wrist_rot':      'Left Wrist :CW/CCW Rotation',
}

_CHANNEL_ORDER = [
    'shoulder_abd', 'shoulder_vert', 'shoulder_horiz',
    'elbow_fe', 'elbow_ps', 'elbow_dev',
    'wrist_fe', 'wrist_rad', 'wrist_rot',
]


def main() -> int:
    """Entry point for dumping Cometa C3D joint angle channels to CSV for diagnostics."""
    parser = argparse.ArgumentParser(
        description='Dump Cometa C3D joint angle channels to CSV for diagnostics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dump_c3d_channels.py -i imu_data/recording.c3d
  python dump_c3d_channels.py -i recording.c3d -o angles.csv --tpose 1.0
        """,
    )
    parser.add_argument('-i', '--input', required=True, help='Path to Cometa C3D file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output CSV path (default: <input_stem>_channels.csv)')
    parser.add_argument('--tpose', type=float, default=1.0,
                        help='T-pose duration in seconds for offset calculation (default: 1.0)')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 1

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + '_channels.csv'
    )

    print(f"Loading: {input_path}")
    c = ezc3d.c3d(str(input_path))

    analog_rate = float(c['header']['analogs']['frame_rate'])
    imu_rate    = float(c['parameters']['ANALOG']['IMU_RATE']['value'][0])
    step        = int(round(analog_rate / imu_rate))

    analogs     = c['data']['analogs']
    an_labels   = c['parameters']['ANALOG']['LABELS']['value']
    label_to_idx = {lbl.strip(): i for i, lbl in enumerate(an_labels)}

    # Detect arm side
    if 'Right Shoulder :Abduction/Adduction' in label_to_idx:
        side = 'right'
        channel_map = _RIGHT_CHANNELS
    elif 'Left Shoulder :Abduction/Adduction' in label_to_idx:
        side = 'left'
        channel_map = _LEFT_CHANNELS
    else:
        print("ERROR: No shoulder angle channels found in C3D file.")
        return 1

    print(f"Arm side detected: {side}")
    print(f"Analog rate: {analog_rate:.0f} Hz  |  IMU rate: {imu_rate:.3f} Hz  |  stride: {step}")

    # Extract raw channel arrays (2000 Hz)
    raw = {}
    for key, label in channel_map.items():
        idx = label_to_idx.get(label)
        if idx is None:
            print(f"WARNING: Channel not found: '{label}'")
            raw[key] = None
        else:
            raw[key] = analogs[0, idx, :]

    # Sub-sample to IMU rate
    n_total = analogs.shape[2]
    indices = list(range(0, n_total, step))
    timestamps = [i / analog_rate for i in indices]

    sampled = {}
    for key in _CHANNEL_ORDER:
        if raw[key] is not None:
            sampled[key] = [float(raw[key][i]) for i in indices]
        else:
            sampled[key] = [float('nan')] * len(indices)

    n_frames  = len(indices)
    n_tpose   = max(1, min(int(round(args.tpose * imu_rate)), n_frames))

    # Compute T-pose offsets (mean of first n_tpose frames)
    offsets = {}
    for key in _CHANNEL_ORDER:
        vals = sampled[key][:n_tpose]
        offsets[key] = sum(vals) / len(vals) if vals else 0.0

    # Corrected values (T-pose offset subtracted)
    corrected = {}
    for key in _CHANNEL_ORDER:
        corrected[key] = [v - offsets[key] for v in sampled[key]]

    # Print summary
    print(f"\nT-pose window: first {n_tpose} frames ({args.tpose:.1f} s)")
    print(f"Total frames:  {n_frames}  (~{n_frames / imu_rate:.1f} s)\n")

    col_w = 14
    header = f"{'Channel':<22}  {'T-pose offset':>{col_w}}  {'Raw min':>{col_w}}  {'Raw max':>{col_w}}  {'Corr min':>{col_w}}  {'Corr max':>{col_w}}"
    print(header)
    print('-' * len(header))
    for key in _CHANNEL_ORDER:
        raw_vals  = sampled[key]
        cor_vals  = corrected[key]
        r_min, r_max = min(raw_vals), max(raw_vals)
        c_min, c_max = min(cor_vals), max(cor_vals)
        print(
            f"  {key:<20}  {offsets[key]:>{col_w}.3f}  "
            f"{r_min:>{col_w}.3f}  {r_max:>{col_w}.3f}  "
            f"{c_min:>{col_w}.3f}  {c_max:>{col_w}.3f}"
        )

    # Write CSV: timestamp + raw + corrected for all channels
    print(f"\nWriting CSV: {output_path}")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header row
        raw_headers  = [f"{k}_raw"  for k in _CHANNEL_ORDER]
        corr_headers = [f"{k}_corr" for k in _CHANNEL_ORDER]
        writer.writerow(['timestamp_s'] + raw_headers + corr_headers)
        # Data rows
        for fi in range(n_frames):
            row = [f"{timestamps[fi]:.6f}"]
            row += [f"{sampled[k][fi]:.4f}"   for k in _CHANNEL_ORDER]
            row += [f"{corrected[k][fi]:.4f}"  for k in _CHANNEL_ORDER]
            writer.writerow(row)

    print(f"Done. {n_frames} rows written.")
    print(
        "\nTip: open the CSV in a spreadsheet or plot with matplotlib to see "
        "how each channel changes over time during each movement."
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
