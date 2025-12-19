"""
Test Model Accuracy with Real Mesh Generation (Batch Support)

This script validates the inverse mapping model with real-world measurements by:
1. Taking target measurements (single or batch)
2. Using the model to predict macroparameters
3. Generating actual meshes with those macroparameters
4. Measuring the generated meshes
5. Comparing actual measurements to predictions and targets
6. Producing aggregate statistics and detailed CSV results

This provides TRUE accuracy of the model's inverse mapping capability across
different demographics (gender, race, etc.).

JSON Input Formats:

Single measurement:
{
  "height_cm": 165.0,
  "shoulder_width_cm": 38.5,
  ...
}

Batch measurements:
{
  "category": "asian_male",
  "description": "Real measurements from Asian male subjects",
  "measurements": [
    {
      "subject_id": "AM001",
      "height_cm": 172.5,
      "shoulder_width_cm": 42.3,
      ...
    },
    {
      "subject_id": "AM002",
      ...
    }
  ]
}

Usage:
    # Single measurement
    python test_model_accuracy.py --input test_measurements.json --models model.pkl

    # Batch measurements
    python test_model_accuracy.py --input asian_male_batch.json --models model.pkl
"""

import json
import sys
import subprocess
import argparse
import csv
from pathlib import Path
import numpy as np
from statistics import mean, median, stdev

# Add parent directory to path for imports
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import from infer_macroparameters
from infer_macroparameters import load_models, find_macroparameters, MEASUREMENTS


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object that may contain numpy types

    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def generate_and_measure_mesh(macroparameters, rig_type='default_no_toes'):
    """
    Generate a mesh with given macroparameters and measure it.

    This function:
    1. Creates a config file in measure_batch.py format (single combination)
    2. Calls Blender with measure_batch.py to generate and measure
    3. Reads the resulting CSV
    4. Returns the measurements

    Args:
        macroparameters: Dictionary of macroparameter values
        rig_type: Type of rig to add

    Returns:
        Dictionary of actual measurements from the generated mesh
    """
    # Build config in measure_batch.py format
    # Fixed parameters match what was used in the lookup table training data
    config = {
        'fixed_params': {
            'gender': 0.0,        # Female
            'cupsize': 0.5,       # Medium
            'firmness': 0.5,      # Medium
            'race': {
                'asian': 1.0,
                'caucasian': 0.0,
                'african': 0.0
            }
        },
        'grid_params': {
            # Each grid param has a single value (min == max, step doesn't matter)
            'age': {
                'min': macroparameters['age'],
                'max': macroparameters['age'],
                'step': 1.0
            },
            'muscle': {
                'min': macroparameters['muscle'],
                'max': macroparameters['muscle'],
                'step': 1.0
            },
            'weight': {
                'min': macroparameters['weight'],
                'max': macroparameters['weight'],
                'step': 1.0
            },
            'height': {
                'min': macroparameters['height'],
                'max': macroparameters['height'],
                'step': 1.0
            },
            'proportions': {
                'min': macroparameters['proportions'],
                'max': macroparameters['proportions'],
                'step': 1.0
            }
        },
        'rig_type': rig_type
    }

    # Save config to temporary file
    config_path = parent_dir / 'temp_test_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Path for output CSV
    output_csv_path = parent_dir / 'temp_test_measurements.csv'

    # Run Blender with measure_batch.py using run_blender.py wrapper
    cmd = [
        'python',
        str(parent_dir / 'run_blender.py'),
        '--script', 'measurement_functions/measure_batch.py',
        '--',
        '--config', str(config_path),
        '--output', str(output_csv_path),
        '--rig-type', rig_type
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(parent_dir),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"\nERROR: Blender exited with code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print("--- Blender Error Output ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"Blender process failed with code {result.returncode}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Blender process timed out after 5 minutes")
    finally:
        # Clean up temp config
        if config_path.exists():
            config_path.unlink()

    # Load measurements from CSV
    if not output_csv_path.exists():
        raise RuntimeError("Measurements CSV not created by Blender")

    import pandas as pd
    df = pd.read_csv(output_csv_path)

    if len(df) == 0:
        raise RuntimeError("CSV is empty - no measurements recorded")

    # Extract first (and only) row of measurements
    row = df.iloc[0]
    measurements = {}
    for measure in MEASUREMENTS:
        if measure in row:
            measurements[measure] = float(row[measure])

    # Clean up temp CSV
    if output_csv_path.exists():
        output_csv_path.unlink()

    return measurements


def test_single_subject(subject_id, target_measurements, models, macro_bounds, method, rig_type):
    """
    Test model on a single subject's measurements.

    Returns:
        Dictionary with test results for this subject
    """
    print(f"\n{'='*80}")
    print(f"Testing Subject: {subject_id}")
    print(f"{'='*80}")

    # Find macroparameters
    result = find_macroparameters(
        models, macro_bounds, target_measurements,
        method=method, weights=None, verbose=False
    )

    print("\nPredicted Macroparameters:")
    for param, value in result['macroparameters'].items():
        print(f"  {param:12s}: {value:.4f}")

    # Generate and measure actual mesh
    print("\nGenerating and measuring mesh...")
    actual_measurements = generate_and_measure_mesh(
        result['macroparameters'],
        rig_type=rig_type
    )

    # Calculate errors
    errors_per_measurement = {}
    for measure in MEASUREMENTS:
        target = target_measurements.get(measure, 0.0)
        actual = actual_measurements.get(measure, 0.0)
        predicted = result['predicted_measurements'].get(measure, 0.0)

        errors_per_measurement[measure] = {
            'target': target,
            'predicted': predicted,
            'actual': actual,
            'prediction_error': abs(predicted - target),
            'actual_error': abs(actual - target)
        }

    # Calculate MAE
    actual_errors = [e['actual_error'] for e in errors_per_measurement.values()]
    mae = mean(actual_errors)
    max_error = max(actual_errors)

    print(f"\nResults:")
    print(f"  MAE: {mae:.4f} cm")
    print(f"  Max Error: {max_error:.4f} cm")

    return {
        'subject_id': subject_id,
        'macroparameters': result['macroparameters'],
        'measurements': errors_per_measurement,
        'mae': mae,
        'max_error': max_error
    }


def save_results_to_csv(results, category, output_path):
    """
    Save detailed results to CSV.

    Args:
        results: List of result dictionaries from test_single_subject
        category: Category name (e.g., "asian_male")
        output_path: Path to save CSV file
    """
    csv_rows = []

    for result in results:
        base_row = {
            'category': category,
            'subject_id': result['subject_id'],
            'mae': result['mae'],
            'max_error': result['max_error']
        }

        # Add macroparameters
        for param, value in result['macroparameters'].items():
            base_row[f'macro_{param}'] = value

        # Add per-measurement errors
        for measure, data in result['measurements'].items():
            base_row[f'{measure}_target'] = data['target']
            base_row[f'{measure}_predicted'] = data['predicted']
            base_row[f'{measure}_actual'] = data['actual']
            base_row[f'{measure}_error'] = data['actual_error']

        csv_rows.append(base_row)

    # Write CSV
    if csv_rows:
        with open(output_path, 'w', newline='') as f:
            fieldnames = list(csv_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


def print_aggregate_statistics(results, category):
    """
    Print aggregate statistics across all subjects.

    Args:
        results: List of result dictionaries
        category: Category name
    """
    if not results:
        print("No results to analyze")
        return

    print(f"\n{'='*80}")
    print(f"AGGREGATE STATISTICS - {category.upper()}")
    print(f"{'='*80}")

    # Overall MAE statistics
    maes = [r['mae'] for r in results]
    max_errors = [r['max_error'] for r in results]

    print(f"\nOverall Performance (n={len(results)} subjects):")
    print(f"  Mean MAE:       {mean(maes):.4f} cm")
    print(f"  Median MAE:     {median(maes):.4f} cm")
    if len(maes) > 1:
        print(f"  Std Dev MAE:    {stdev(maes):.4f} cm")
    print(f"  Min MAE:        {min(maes):.4f} cm")
    print(f"  Max MAE:        {max(maes):.4f} cm")
    print(f"\n  Mean Max Error: {mean(max_errors):.4f} cm")
    print(f"  Worst Error:    {max(max_errors):.4f} cm")

    # Per-measurement error statistics
    print(f"\nPer-Measurement Error Analysis:")
    print(f"  {'Measurement':<25s} {'Mean Error':<12s} {'Max Error':<12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")

    for measure in MEASUREMENTS:
        errors = [r['measurements'][measure]['actual_error'] for r in results]
        mean_err = mean(errors)
        max_err = max(errors)
        print(f"  {measure:<25s} {mean_err:<12.4f} {max_err:<12.4f}")

    # Identify best and worst subjects
    best = min(results, key=lambda x: x['mae'])
    worst = max(results, key=lambda x: x['mae'])

    print(f"\nBest Subject:  {best['subject_id']} (MAE: {best['mae']:.4f} cm)")
    print(f"Worst Subject: {worst['subject_id']} (MAE: {worst['mae']:.4f} cm)")


def main():
    parser = argparse.ArgumentParser(
        description='Test model accuracy with real mesh generation (supports batch testing)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input measurements JSON file (single or batch format)'
    )

    parser.add_argument(
        '--models',
        type=str,
        default='macroparameters_inference_models.pkl',
        help='Path to trained models (default: macroparameters_inference_models.pkl)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='differential_evolution',
        choices=['differential_evolution', 'both'],
        help='Optimization method (default: differential_evolution)'
    )

    parser.add_argument(
        '--rig-type',
        type=str,
        default='default_no_toes',
        choices=['default', 'default_no_toes', 'game_engine'],
        help='Type of rig to add (default: default_no_toes)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TEST MODEL ACCURACY WITH REAL MESH GENERATION")
    print("=" * 80)

    try:
        # Load input
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, 'r') as f:
            input_data = json.load(f)

        # Load models
        print("\nLoading models...")
        models, macro_bounds = load_models(args.models)

        # Determine if batch or single
        is_batch = 'measurements' in input_data and isinstance(input_data['measurements'], list)

        if is_batch:
            # Batch processing
            category = input_data.get('category', 'unknown')
            description = input_data.get('description', '')
            measurements_list = input_data['measurements']

            print(f"\nBatch Mode: {category}")
            print(f"Description: {description}")
            print(f"Number of subjects: {len(measurements_list)}")

            results = []
            for i, measurement_data in enumerate(measurements_list, 1):
                subject_id = measurement_data.get('subject_id', f'Subject_{i:03d}')

                # Extract measurement values (remove subject_id if present)
                target_measurements = {k: v for k, v in measurement_data.items()
                                     if k in MEASUREMENTS}

                result = test_single_subject(
                    subject_id, target_measurements,
                    models, macro_bounds,
                    args.method, args.rig_type
                )
                results.append(result)

            # Save CSV
            csv_path = str(input_path).replace('.json', '_batch_results.csv')
            save_results_to_csv(results, category, csv_path)
            print(f"\n\nDetailed results saved to: {csv_path}")

            # Print aggregate statistics
            print_aggregate_statistics(results, category)

        else:
            # Single measurement (backward compatibility)
            target_measurements = {k: v for k, v in input_data.items() if k in MEASUREMENTS}

            print("\nSingle Measurement Mode")
            print("\nTarget Measurements:")
            for measure, value in target_measurements.items():
                print(f"  {measure:25s}: {value:.2f} cm")

            result = test_single_subject(
                'single_test', target_measurements,
                models, macro_bounds,
                args.method, args.rig_type
            )

            # Save single result
            output_path = str(input_path).replace('.json', '_result.json')
            result_dict = convert_numpy_types(result)

            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=2)

            print(f"\n\nResults saved to: {output_path}")

        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
