#!/usr/bin/env python3
"""
Test Macroparameter Inference

Validates the accuracy of the macroparameter inference model by:
1. Predicting macroparameters from input measurements using the trained model.
2. Generating a human mesh in Blender with only those macroparameters applied
   (no microparameter adjustment).
3. Extracting actual body measurements from the generated mesh.
4. Printing a target-vs-actual comparison table so you can see how closely
   the inferred macroparameters reproduce the input measurements before any
   microparameter correction is applied.

This is useful for assessing the standalone quality of the inverse mapping model
and for deciding whether microparameter adjustment is needed for a given subject.

The script operates in two modes selected automatically:

  Normal mode  (run directly with Python)
      Runs macroparameter inference, then launches itself inside Blender
      to perform mesh generation and measurement extraction.

  Blender mode (launched internally via run_blender.py --script <this file> -- --blender-mode)
      Creates the human mesh with the supplied macroparameters, extracts
      measurements, and writes the report JSON. Not intended for direct use.

Usage:
    python test_macroparameter_inference.py --input subject.json --models models.pkl
    python test_macroparameter_inference.py --input subject.json --models models.pkl --rig-type game_engine

Input JSON format (same as compute_all_parameters.py):
{
  "gender": "female",
  "race": "asian",
  "body_measurements": {
    "height_cm": 165.0,
    "shoulder_width_cm": 38.5,
    ...
  }
}
"""

import json
import sys
import subprocess
import argparse
import tempfile
import traceback
from pathlib import Path

# Resolve key directories relative to this file so the script works regardless
# of the working directory it is called from.
# _SCRIPT_DIR  : measurement_to_parameters_inference_models/  (this file's directory)
# _MODULE_DIR  : mesh_generation_module/  (parent, contains run_blender.py)
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _SCRIPT_DIR.parent

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

# Detect early whether we are executing inside Blender so we can conditionally
# import heavy inference libraries that are not available in Blender's Python.
_BLENDER_MODE = '--blender-mode' in sys.argv

if not _BLENDER_MODE:
    from infer_macroparameters import (
        load_models,
        find_macroparameters,
        MEASUREMENTS,
        parse_gender,
        parse_race,
    )


# ---------------------------------------------------------------------------
# Blender-side logic (runs inside Blender via run_blender.py)
# ---------------------------------------------------------------------------

def _blender_main():
    """
    Entry point when this script is executed inside Blender.

    Creates a human mesh with the supplied macroparameters (no microparameter
    adjustment), extracts body measurements, and writes a measurement report
    JSON file for the outer process to read.
    """
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--blender-mode', action='store_true')
    parser.add_argument('--macros',   type=str, required=True, help='Path to macros JSON')
    parser.add_argument('--target',   type=str, required=True, help='Path to target measurements JSON')
    parser.add_argument('--report',   type=str, required=True, help='Path to write measurement report JSON')
    parser.add_argument('--rig-type', type=str, default='default_no_toes')
    args = parser.parse_args(argv)

    # Import Blender-side helpers from adjust_microparameters.
    # adjust_microparameters.py lives in the same inference directory.
    from adjust_microparameters import (
        create_human_with_parameters,
        extract_measurements_with_cm_suffix,
        cleanup_mesh_and_armature,
        TOLERANCE_CM,
    )

    with open(args.macros, 'r') as f:
        macros = json.load(f)

    with open(args.target, 'r') as f:
        target_measurements = json.load(f)

    print("Creating human mesh with inferred macroparameters (no microparameter adjustment)...")
    mesh, armature = create_human_with_parameters(macros, {}, args.rig_type)
    actual_measurements = extract_measurements_with_cm_suffix(mesh, armature)
    cleanup_mesh_and_armature(mesh, armature)
    print("Mesh created and measured successfully.")

    # Build measurement report in the same format as adjust_microparameters.py
    report = {
        'measurements': {},
        'summary': {
            'total_measurements': 0,
            'converged_count':    0,
            'mean_absolute_error': 0.0,
            'max_absolute_error':  0.0,
            'all_converged':       True,
        },
    }

    errors_list = []
    for category, target in target_measurements.items():
        actual    = actual_measurements.get(category, 0.0)
        error     = actual - target
        abs_error = abs(error)
        converged = abs_error <= TOLERANCE_CM
        report['measurements'][category] = {
            'target':         round(target,    4),
            'actual':         round(actual,    4),
            'error':          round(error,     4),
            'absolute_error': round(abs_error, 4),
            'converged':      converged,
        }
        errors_list.append(abs_error)

    if errors_list:
        converged_count = sum(1 for m in report['measurements'].values() if m['converged'])
        report['summary']['total_measurements'] = len(errors_list)
        report['summary']['converged_count']    = converged_count
        report['summary']['mean_absolute_error'] = round(sum(errors_list) / len(errors_list), 4)
        report['summary']['max_absolute_error']  = round(max(errors_list), 4)
        report['summary']['all_converged']        = converged_count == len(errors_list)

    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Measurement report written to: {args.report}")


# ---------------------------------------------------------------------------
# Normal-mode helpers
# ---------------------------------------------------------------------------

def _run_blender_verification(
    full_macros: dict,
    target_measurements: dict,
    rig_type: str,
) -> dict:
    """
    Launch this script inside Blender to create a mesh and extract measurements.

    Args:
        full_macros:         Complete macro settings dict (gender, age, height, ...).
        target_measurements: Target measurements used to build the comparison report.
        rig_type:            Rig type passed to Blender.

    Returns:
        measurement_report dict with 'measurements' and 'summary' keys.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        macros_path = Path(f.name)
        json.dump(full_macros, f, indent=2)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        target_path = Path(f.name)
        json.dump(target_measurements, f, indent=2)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        report_path = Path(f.name)

    try:
        cmd = [
            'python',
            str(_MODULE_DIR / 'run_blender.py'),
            '--script', str(Path(__file__).resolve()),
            '--',
            '--blender-mode',
            '--macros',   str(macros_path),
            '--target',   str(target_path),
            '--report',   str(report_path),
            '--rig-type', rig_type,
        ]

        print("\nLaunching Blender to generate mesh and extract measurements...")
        result = subprocess.run(
            cmd,
            cwd=str(_MODULE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            print(f"\nERROR: Blender verification failed (exit code {result.returncode})",
                  file=sys.stderr)
            if result.stdout:
                print("--- stdout ---", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print("--- stderr ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            raise RuntimeError("Blender mesh verification failed.")

        if result.stdout:
            print(result.stdout)

        if not report_path.exists():
            raise RuntimeError("Blender did not produce a measurement report JSON.")

        with open(report_path, 'r') as f:
            return json.load(f)

    finally:
        for path in [macros_path, target_path, report_path]:
            if path.exists():
                path.unlink()

def print_measurement_report(measurement_report: dict) -> None:
    """
    Print a formatted target-vs-actual comparison table from a measurement report.

    Args:
        measurement_report: Dict with 'measurements' and 'summary' keys, as produced
                            by adjust_microparameters.py or test_macroparameter_inference.py.
    """
    measurements = measurement_report.get('measurements', {})
    summary      = measurement_report.get('summary', {})

    print("\n" + "=" * 80)
    print("MEASUREMENT REPORT")
    print("=" * 80)
    print(f"\n{'Measurement':<25s} {'Target':>12s} {'Actual':>12s} {'Error':>12s} {'Status':>10s}")
    print("-" * 80)

    for category, data in measurements.items():
        print(
            f"  {category:<23s}"
            f" {data['target']:>12.4f}"
            f" {data['actual']:>12.4f}"
            f" {data['error']:>+12.4f}"
        )

    print("-" * 80)
    if summary:
        converged     = summary.get('converged_count', 0)
        total         = summary.get('total_measurements', 0)
        mae           = summary.get('mean_absolute_error', 0.0)
        max_err       = summary.get('max_absolute_error', 0.0)
        print(f"  Converged          : {converged}/{total}")
        print(f"  Mean Absolute Error: {mae:.4f} cm")
        print(f"  Max Absolute Error : {max_err:.4f} cm")
    print("=" * 80)

# ---------------------------------------------------------------------------
# Normal-mode entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Test macroparameter inference accuracy via Blender mesh generation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to input JSON with subject details and measurements.',
    )
    parser.add_argument(
        '--models', type=str, required=True,
        help='Path to trained macroparameter inference models pickle file.',
    )
    parser.add_argument(
        '--rig-type', type=str, default='default_no_toes',
        choices=['default', 'default_no_toes', 'game_engine'],
        help='Rig type used when generating the verification mesh (default: default_no_toes).',
    )
    args = parser.parse_args()

    print("=" * 80)
    print("TEST MACROPARAMETER INFERENCE")
    print("=" * 80)

    try:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, 'r') as f:
            input_data = json.load(f)

        gender = parse_gender(input_data.get('gender', 'female'))
        race   = parse_race(input_data.get('race', 'asian'))

        measurements_src    = input_data.get('body_measurements', input_data)
        target_measurements = {k: v for k, v in measurements_src.items() if k in MEASUREMENTS}

        if not target_measurements:
            raise ValueError("No recognised measurements found in the input file.")

        print(f"\nSubject : {'Male' if gender > 0.5 else 'Female'}")
        print(f"Race    : {', '.join(f'{k}={v:.2f}' for k, v in race.items())}")
        print(f"\nTarget measurements ({len(target_measurements)}):")
        for name, value in target_measurements.items():
            print(f"  {name:<25s}: {value:.2f} cm")

        # Step 1: Infer macroparameters
        print("\n" + "-" * 80)
        print("STEP 1: Inferring macroparameters")
        print("-" * 80)

        models, macro_bounds = load_models(args.models)
        result        = find_macroparameters(models, macro_bounds, target_measurements, verbose=True)
        macroparameters = result['macroparameters']

        full_macros = {
            'gender':      gender,
            'age':         macroparameters['age'],
            'height':      macroparameters['height'],
            'proportions': macroparameters['proportions'],
            'cupsize':     0.5,
            'firmness':    0.5,
            'muscle':      0.5,
            'weight':      0.5,
            'race':        race,
        }

        # Step 2: Generate mesh in Blender and extract actual measurements
        print("\n" + "-" * 80)
        print("STEP 2: Generating mesh and extracting actual measurements")
        print("-" * 80)

        measurement_report = _run_blender_verification(
            full_macros, target_measurements, args.rig_type
        )

        # Step 3: Print comparison table
        print_measurement_report(measurement_report)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if _BLENDER_MODE:
        try:
            _blender_main()
        except Exception as e:
            print(f"\nError in Blender mode: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        sys.exit(main())
