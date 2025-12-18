"""
Test Model Accuracy with Real Mesh Generation

This script validates the inverse mapping model by:
1. Taking target measurements
2. Using the model to predict macroparameters
3. Generating an actual mesh with those macroparameters
4. Measuring the generated mesh
5. Comparing actual measurements to predictions and targets

This provides the TRUE accuracy of the model's inverse mapping capability.

Usage:
    python test_model_accuracy.py --input test_measurements.json --models model.pkl
    python test_model_accuracy.py --input test_measurements.json --models model.pkl --keep-mesh
"""

import json
import sys
import subprocess
import argparse
from pathlib import Path
import numpy as np

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
    print("\n" + "="*80)
    print("GENERATING AND MEASURING MESH")
    print("="*80)

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
        }
    }

    print("\nMacroparameters:")
    for param, value in macroparameters.items():
        print(f"  {param:12s}: {value:.4f}")

    print("\nFixed parameters (from training data):")
    print(f"  gender:      {config['fixed_params']['gender']}")
    print(f"  cupsize:     {config['fixed_params']['cupsize']}")
    print(f"  firmness:    {config['fixed_params']['firmness']}")
    print(f"  race:        asian={config['fixed_params']['race']['asian']}")

    # Save config to temporary file
    config_path = parent_dir / 'temp_test_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved config to: {config_path}")

    # Path for output CSV
    output_csv_path = parent_dir / 'temp_test_measurements.csv'

    # Run Blender with measure_batch.py using run_blender.py wrapper
    print("\nLaunching Blender to generate and measure mesh...")
    print("-" * 80)

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

        # Always print Blender output for debugging
        print("\n--- Blender Output ---")
        if result.stdout:
            print(result.stdout)
        print("--- End Blender Output ---\n")

        if result.returncode != 0:
            print(f"\nERROR: Blender exited with code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print("--- Blender Error Output ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                print("--- End Error Output ---", file=sys.stderr)
            raise RuntimeError(f"Blender process failed with code {result.returncode}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Blender process timed out after 5 minutes")

    # Load measurements from CSV
    if not output_csv_path.exists():
        raise RuntimeError("Measurements CSV not created by Blender")

    import pandas as pd
    df = pd.read_csv(output_csv_path)

    if len(df) == 0:
        raise RuntimeError("CSV is empty - no measurements recorded")

    # Extract first (and only) row of measurements
    row = df.iloc[0]
    actual_measurements = {
        'height_cm': row['height_cm'],
        'shoulder_width_cm': row['shoulder_width_cm'],
        'hip_width_cm': row['hip_width_cm'],
        'head_width_cm': row['head_width_cm'],
        'neck_length_cm': row['neck_length_cm'],
        'upper_arm_length_cm': row['upper_arm_length_cm'],
        'forearm_length_cm': row['forearm_length_cm'],
        'hand_length_cm': row['hand_length_cm']
    }

    # Clean up temporary files
    config_path.unlink()
    output_csv_path.unlink()

    print("-" * 80)
    print("\nMesh generated and measured successfully!")

    return actual_measurements


def compare_results(target_measurements, predicted_measurements, actual_measurements):
    """
    Compare target, predicted, and actual measurements.

    Args:
        target_measurements: Original target measurements
        predicted_measurements: Model's predicted measurements
        actual_measurements: Measurements from actual generated mesh

    Returns:
        Dictionary with comparison metrics
    """
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    print("\n" + "-"*95)
    print(f"{'Measurement':<25s} {'Target':>12s} {'Predicted':>12s} {'Actual':>12s} {'Pred Error':>12s} {'Actual Error':>12s}")
    print("-"*95)

    pred_errors = []
    actual_errors = []

    for measure in MEASUREMENTS:
        if measure in target_measurements:
            target = target_measurements[measure]
            predicted = predicted_measurements[measure]
            actual = actual_measurements.get(measure, None)

            if actual is not None:
                pred_error = predicted - target
                actual_error = actual - target

                pred_errors.append(abs(pred_error))
                actual_errors.append(abs(actual_error))

                print(f"{measure:<25s} {target:>12.2f} {predicted:>12.2f} {actual:>12.2f} {pred_error:>+12.2f} {actual_error:>+12.2f}")

    print("-"*95)

    pred_mae = np.mean(pred_errors)
    actual_mae = np.mean(actual_errors)

    pred_max = np.max(pred_errors)
    actual_max = np.max(actual_errors)

    print(f"{'MAE':<25s} {'':>12s} {pred_mae:>12.2f} {actual_mae:>12.2f}")
    print(f"{'Max Error':<25s} {'':>12s} {pred_max:>12.2f} {actual_max:>12.2f}")
    print("-"*95)

    print("\n" + "="*80)
    print("ACCURACY ASSESSMENT")
    print("="*80)

    print(f"\nModel Prediction Error (MAE): {pred_mae:.4f} cm")
    print(f"Actual Mesh Error (MAE):      {actual_mae:.4f} cm")

    improvement = pred_mae - actual_mae
    if improvement > 0:
        print(f"\nActual mesh is BETTER than predicted by {improvement:.4f} cm")
    elif improvement < 0:
        print(f"\nActual mesh is WORSE than predicted by {abs(improvement):.4f} cm")
    else:
        print(f"\nActual mesh matches prediction exactly")

    if actual_mae < 1.0:
        print("\n✓ EXCELLENT: Actual mesh error < 1.0 cm")
    elif actual_mae < 2.0:
        print("\n✓ GOOD: Actual mesh error < 2.0 cm")
    elif actual_mae < 3.0:
        print("\n~ ACCEPTABLE: Actual mesh error < 3.0 cm")
    else:
        print("\n✗ POOR: Actual mesh error >= 3.0 cm")

    print("="*80)

    return {
        'pred_mae': pred_mae,
        'actual_mae': actual_mae,
        'pred_max_error': pred_max,
        'actual_max_error': actual_max,
        'improvement': improvement
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test model accuracy with real mesh generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with a measurements JSON file
  python test_model_accuracy.py --input test_measurements.json --models model.pkl

  # Use hybrid optimization for potentially better results
  python test_model_accuracy.py --input test_measurements.json --models model.pkl --method both

This script will:
1. Load your target measurements
2. Use the model to predict macroparameters
3. Generate an actual mesh in Blender
4. Measure the generated mesh
5. Compare predictions vs reality
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input measurements JSON file'
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
        # Load target measurements
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, 'r') as f:
            target_measurements = json.load(f)

        print("\nTarget Measurements:")
        for measure, value in target_measurements.items():
            if measure in MEASUREMENTS:
                print(f"  {measure:25s}: {value:.2f} cm")

        # Load models and find macroparameters
        print("\n" + "="*80)
        models, macro_bounds = load_models(args.models)

        print("\nFinding optimal macroparameters...")
        result = find_macroparameters(
            models, macro_bounds, target_measurements,
            method=args.method, weights=None, verbose=False
        )

        print("\nPredicted Macroparameters:")
        for param, value in result['macroparameters'].items():
            print(f"  {param:12s}: {value:.4f}")

        print("\nPredicted Measurements:")
        for measure, value in result['predicted_measurements'].items():
            print(f"  {measure:25s}: {value:.2f} cm")

        # Generate and measure actual mesh
        actual_measurements = generate_and_measure_mesh(
            result['macroparameters'],
            rig_type=args.rig_type
        )

        # Compare results
        comparison = compare_results(
            target_measurements,
            result['predicted_measurements'],
            actual_measurements
        )

        # Save complete results
        output_path = str(input_path).replace('.json', '_accuracy_test_result.json')

        # Convert all numpy types to Python native types for JSON serialization
        results_dict = {
            'target_measurements': target_measurements,
            'predicted_macroparameters': result['macroparameters'],
            'predicted_measurements': result['predicted_measurements'],
            'actual_measurements': actual_measurements,
            'comparison': comparison
        }
        results_dict = convert_numpy_types(results_dict)

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nComplete results saved to: {output_path}")

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
