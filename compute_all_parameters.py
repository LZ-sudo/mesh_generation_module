#!/usr/bin/env python3
"""
Compute All Parameters (Macroparameters + Microparameters)

This script takes target measurements and subject details (gender, race) and computes:
1. Macroparameters using the trained regression model
2. Microparameters using iterative adjustment

The output is a JSON file with both macroparameters and microparameters that can be
used directly with generate_human.py to create an accurate human mesh.

Usage:
    python compute_all_parameters.py --input subject_measurements.json --models models.pkl --output parameters.json
"""

import json
import sys
import subprocess
import argparse
import tempfile
from pathlib import Path

# Add measurement_to_parameters_inference_models to path
script_dir = Path(__file__).parent.absolute()
inference_dir = script_dir / 'measurement_to_parameters_inference_models'
if str(inference_dir) not in sys.path:
    sys.path.insert(0, str(inference_dir))

from measurement_to_parameters_inference_models.infer_macroparameters import load_models, find_macroparameters, MEASUREMENTS, parse_gender, parse_race


def validate_input_measurements(measurements: dict) -> dict:
    """
    Validate that required measurements are present.

    Args:
        measurements: Dictionary with measurement values

    Returns:
        Validated measurements dictionary

    Raises:
        ValueError: If required measurements are missing
    """
    # Extract only the measurements we need
    validated = {}
    missing = []

    for measure in MEASUREMENTS:
        if measure in measurements:
            validated[measure] = measurements[measure]
        else:
            missing.append(measure)

    if missing:
        raise ValueError(f"Missing required measurements: {', '.join(missing)}")

    return validated


def run_microparameter_adjustment(target_measurements, macroparameters, gender, race, rig_type='default_no_toes'):
    """
    Run iterative microparameter adjustment via Blender subprocess.

    Args:
        target_measurements: Dictionary of target measurements
        macroparameters: Dictionary of the 5 optimized macroparameters
        gender: Gender value (0.0 for female, 1.0 for male)
        race: Race dictionary with asian/caucasian/african values
        rig_type: Type of rig to use

    Returns:
        Dictionary of adjusted microparameters (with -incr/-decr suffixes)
    """
    # Build full macro settings
    full_macros = {
        'gender': gender,
        'age': macroparameters['age'],
        'muscle': macroparameters['muscle'],
        'weight': macroparameters['weight'],
        'height': macroparameters['height'],
        'proportions': macroparameters['proportions'],
        'cupsize': 0.5,       # Medium (default)
        'firmness': 0.5,      # Medium (default)
        'race': race
    }

    # Create temporary files for input/output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        target_path = Path(f.name)
        json.dump(target_measurements, f, indent=2)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        macros_path = Path(f.name)
        json.dump(full_macros, f, indent=2)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = Path(f.name)

    try:
        # Run Blender with adjust_microparameters.py
        adjust_script_path = inference_dir / 'adjust_microparameters.py'

        cmd = [
            'python',
            str(script_dir / 'run_blender.py'),
            '--script', str(adjust_script_path),
            '--',
            '--target', str(target_path),
            '--macros', str(macros_path),
            '--output', str(output_path),
            '--rig-type', rig_type,
            '--verbose'
        ]

        print("\nRunning microparameter adjustment...")
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"\nERROR: Microparameter adjustment failed with code {result.returncode}", file=sys.stderr)
            if result.stdout:
                print("--- Output ---", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print("--- Error ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"Microparameter adjustment failed")

        # Print output for visibility
        if result.stdout:
            print(result.stdout)

        # Load adjusted microparameters
        if not output_path.exists():
            raise RuntimeError("Microparameter output JSON not created")

        with open(output_path, 'r') as f:
            adjusted_micros = json.load(f)

        return adjusted_micros

    except subprocess.TimeoutExpired:
        raise RuntimeError("Microparameter adjustment timed out after 10 minutes")
    finally:
        # Clean up temporary files
        for path in [target_path, macros_path, output_path]:
            if path.exists():
                path.unlink()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Compute macroparameters and microparameters from target measurements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example input JSON format:
{
  "gender": "female",
  "race": "asian",
  "measurements": {
    "height_cm": 165.0,
    "shoulder_width_cm": 38.5,
    "hip_width_cm": 35.2,
    "head_width_cm": 14.8,
    "neck_length_cm": 10.5,
    "upper_arm_length_cm": 28.3,
    "forearm_length_cm": 24.7,
    "hand_length_cm": 18.2
  }
}

Example usage:
    python compute_all_parameters.py --input subject.json --models models.pkl --output parameters.json
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input JSON file with subject details and measurements'
    )

    parser.add_argument(
        '--models',
        type=str,
        required=True,
        help='Path to trained models pickle file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output JSON file with computed parameters'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='least_squares',
        choices=['least_squares', 'nelder_mead', 'powell'],
        help='Optimization method for macroparameter inference (default: least_squares)'
    )

    parser.add_argument(
        '--rig-type',
        type=str,
        default='default_no_toes',
        choices=['default', 'default_no_toes', 'game_engine'],
        help='Type of rig to use for microparameter adjustment (default: default_no_toes)'
    )

    parser.add_argument(
        '--no-micro-adjustment',
        action='store_true',
        help='Skip microparameter adjustment (compute macros only)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("COMPUTE ALL PARAMETERS")
    print("=" * 80)

    try:
        # Load input
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, 'r') as f:
            input_data = json.load(f)

        # Parse gender and race (supports both string and numeric formats)
        gender_input = input_data.get('gender', 'female')
        gender = parse_gender(gender_input)

        race_input = input_data.get('race', 'asian')
        race = parse_race(race_input)

        # Normalize race values if dict was provided
        if isinstance(race_input, dict):
            race_sum = sum(race.values())
            if abs(race_sum - 1.0) > 0.01:
                print(f"Warning: Race values sum to {race_sum}, normalizing...")
                race = {k: v/race_sum for k, v in race.items()}

        print(f"\nSubject Details:")
        print(f"  Gender: {'Male' if gender > 0.5 else 'Female'}")
        print(f"  Race: {', '.join([f'{k}={v:.2f}' for k, v in race.items()])}")

        # Extract and validate measurements
        measurements_dict = input_data.get('measurements', input_data)
        target_measurements = validate_input_measurements(measurements_dict)

        print(f"\nTarget Measurements:")
        for measure, value in target_measurements.items():
            print(f"  {measure:25s}: {value:.2f} cm")

        # Load models
        print(f"\nLoading models from: {args.models}")
        models, macro_bounds = load_models(args.models)

        # Step 1: Infer macroparameters
        print("\n" + "-" * 80)
        print("STEP 1: Inferring Macroparameters")
        print("-" * 80)

        result = find_macroparameters(
            models, macro_bounds, target_measurements,
            method=args.method, weights=None, verbose=True
        )

        macroparameters = result['macroparameters']
        print("\nInferred Macroparameters:")
        for param, value in macroparameters.items():
            print(f"  {param:12s}: {value:.4f}")

        # Build output
        output_data = {
            'macro_settings': {
                'gender': gender,
                'age': macroparameters['age'],
                'muscle': macroparameters['muscle'],
                'weight': macroparameters['weight'],
                'height': macroparameters['height'],
                'proportions': macroparameters['proportions'],
                'cupsize': 0.5,
                'firmness': 0.5,
                'race': race
            }
        }

        # Step 2: Adjust microparameters (if enabled)
        if not args.no_micro_adjustment:
            print("\n" + "-" * 80)
            print("STEP 2: Adjusting Microparameters")
            print("-" * 80)

            adjusted_micros = run_microparameter_adjustment(
                target_measurements,
                macroparameters,
                gender,
                race,
                rig_type=args.rig_type
            )

            output_data['micro_settings'] = adjusted_micros

            print(f"\nAdjusted {len(adjusted_micros)} microparameters")
        else:
            print("\n" + "-" * 80)
            print("STEP 2: Skipping Microparameter Adjustment")
            print("-" * 80)

        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print("\n" + "=" * 80)
        print("✓ PARAMETERS COMPUTED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nOutput saved to: {output_path}")
        print()

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
