"""
Infer Macroparameters from Body Measurements

This script uses trained models to find the best macroparameter values
that produce a mesh matching target body measurements.

Usage:
    # From JSON file with measurements
    python infer_macroparameters.py --input measurements.json --models macroparameters_inference_models.pkl

    # From CSV file (batch mode)
    python infer_macroparameters.py --input measurements.csv --models macroparameters_inference_models.pkl --output results.csv

    # With custom optimization settings
    python infer_macroparameters.py --input measurements.json --models macroparameters_inference_models.pkl --method both
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
import sys
from pathlib import Path
from scipy.optimize import minimize, differential_evolution

# Configuration
# NOTE: This will be overridden by values loaded from the model file
# Default to full parameter set (matching train_model.py)
MACROPARAMETERS = ['age', 'muscle', 'weight', 'height', 'proportions']
EXCLUDED_MACROPARAMETERS = {}
MEASUREMENTS = [
    'height_cm', 'shoulder_width_cm', 'hip_width_cm', 'head_width_cm',
    'neck_length_cm', 'upper_arm_length_cm', 'forearm_length_cm', 'hand_length_cm'
]


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


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar to console."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def load_models(model_path):
    """
    Load trained models from pickle file.

    Args:
        model_path: Path to pickle file with trained models

    Returns:
        tuple: (models dict, macro_bounds dict, macroparameters list, excluded_macroparameters dict)
    """
    global MACROPARAMETERS, EXCLUDED_MACROPARAMETERS

    print(f"Loading models from: {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    models = data['models']
    macro_bounds = data['macro_bounds']

    # Load macroparameter configuration from model file (for compatibility)
    if 'macroparameters' in data:
        MACROPARAMETERS = data['macroparameters']
    if 'excluded_macroparameters' in data:
        EXCLUDED_MACROPARAMETERS = data['excluded_macroparameters']

    print(f"Loaded {len(models)} models")
    print(f"\nMacroparameter configuration:")
    print(f"  Predicted parameters: {MACROPARAMETERS}")
    if EXCLUDED_MACROPARAMETERS:
        print(f"  Excluded (fixed) parameters: {EXCLUDED_MACROPARAMETERS}")

    print(f"\nMacroparameter bounds (for optimization):")
    for param, (min_val, max_val) in macro_bounds.items():
        print(f"  {param:12s}: [{min_val:.3f}, {max_val:.3f}]")
    print("-" * 80)

    return models, macro_bounds


def predict_measurements(models, macroparameters):
    """
    Predict all measurements from given macroparameters.

    Args:
        models: Dictionary of trained models
        macroparameters: Array or dict of macroparameter values

    Returns:
        Dictionary of predicted measurements
    """
    # Convert to DataFrame with proper column names to avoid sklearn warning
    if isinstance(macroparameters, dict):
        macro_df = pd.DataFrame([macroparameters], columns=MACROPARAMETERS)
    else:
        macro_values = np.array(macroparameters).flatten()
        macro_df = pd.DataFrame([macro_values], columns=MACROPARAMETERS)

    # Predict each measurement
    predictions = {}
    for measure in MEASUREMENTS:
        predictions[measure] = models[measure].predict(macro_df)[0]

    return predictions


def objective_function(macroparameters, models, target_measurements, weights=None):
    """
    Objective function to minimize: weighted sum of squared errors.

    Args:
        macroparameters: Array of macroparameter values
        models: Dictionary of trained models
        target_measurements: Dictionary of target measurement values
        weights: Optional weights for each measurement

    Returns:
        Total weighted error
    """
    # Predict measurements from macroparameters
    predicted = predict_measurements(models, macroparameters)

    # Calculate weighted squared errors
    total_error = 0.0
    for measure in MEASUREMENTS:
        if measure in target_measurements:
            error = predicted[measure] - target_measurements[measure]
            weight = weights.get(measure, 1.0) if weights else 1.0
            total_error += weight * (error ** 2)

    return total_error


def find_macroparameters(models, macro_bounds, target_measurements,
                        method='differential_evolution', weights=None, verbose=True):
    """
    Find macroparameters that best match target measurements.

    Args:
        models: Dictionary of trained models
        macro_bounds: Dictionary of macroparameter bounds
        target_measurements: Dictionary of target measurements
        method: Optimization method ('differential_evolution' or 'both')
        weights: Optional weights for each measurement
        verbose: Print detailed results

    Returns:
        Dictionary with results
    """
    if verbose:
        print("\n" + "="*80)
        print("FINDING MACROPARAMETERS")
        print("="*80)
        print("\nTarget Measurements:")
        for measure, value in target_measurements.items():
            print(f"  {measure:25s}: {value:.2f} cm")

    # Prepare bounds for optimization
    bounds = [(macro_bounds[param][0], macro_bounds[param][1])
              for param in MACROPARAMETERS]

    # Optimize
    if method == 'differential_evolution':
        if verbose:
            print("\nOptimization method: Differential Evolution (global search)")
        result = differential_evolution(
            func=objective_function,
            bounds=bounds,
            args=(models, target_measurements, weights),
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            atol=0.01,
            tol=0.01,
            seed=42,
            workers=1,
            polish=True,
            updating='deferred'
        )
        macro_values = result.x

    elif method == 'both':
        if verbose:
            print("\nOptimization method: Hybrid (global + local)")
            print("  Step 1: Differential Evolution...")
        result_de = differential_evolution(
            func=objective_function,
            bounds=bounds,
            args=(models, target_measurements, weights),
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            atol=0.01,
            tol=0.01,
            seed=42,
            workers=1,
            polish=True,
            updating='deferred'
        )

        if verbose:
            print("  Step 2: L-BFGS-B refinement...")
        result = minimize(
            fun=objective_function,
            x0=result_de.x,
            args=(models, target_measurements, weights),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        macro_values = result.x

    else:
        raise ValueError(f"Unknown method: {method}")

    # Package results - add optimized params + excluded params with defaults
    macroparameters = {param: macro_values[i] for i, param in enumerate(MACROPARAMETERS)}
    macroparameters.update(EXCLUDED_MACROPARAMETERS)  # Add muscle=0.5, weight=0.5
    predicted_measurements = predict_measurements(models, macro_values)

    # Calculate errors
    errors = {}
    absolute_errors = []
    for measure in MEASUREMENTS:
        if measure in target_measurements:
            error = predicted_measurements[measure] - target_measurements[measure]
            errors[measure] = error
            absolute_errors.append(abs(error))

    mae = np.mean(absolute_errors)
    max_error = np.max(absolute_errors)

    result_dict = {
        'macroparameters': macroparameters,
        'predicted_measurements': predicted_measurements,
        'errors': errors,
        'mae': mae,
        'max_error': max_error
    }

    if verbose:
        print_results(result_dict, target_measurements, macro_bounds)

    return result_dict


def print_results(result, target_measurements, macro_bounds):
    """Print detailed optimization results."""
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print("\nFound Macroparameters:")
    for param, value in result['macroparameters'].items():
        min_val, max_val = macro_bounds[param]
        print(f"  {param:12s}: {value:.4f}  (range: [{min_val:.3f}, {max_val:.3f}])")

    print("\n" + "-"*80)
    print(f"{'Measurement':<25s} {'Target':>12s} {'Predicted':>12s} {'Error':>12s}")
    print("-"*80)

    for measure in MEASUREMENTS:
        if measure in result['errors']:
            target = target_measurements[measure]
            predicted = result['predicted_measurements'][measure]
            error = result['errors'][measure]
            print(f"{measure:<25s} {target:>12.2f} {predicted:>12.2f} {error:>+12.2f}")

    print("-"*80)
    print(f"Mean Absolute Error (MAE): {result['mae']:.4f} cm")
    print(f"Maximum Error:             {result['max_error']:.4f} cm")
    print("="*80)


def process_single(input_path, models, macro_bounds, method, weights):
    """Process a single measurement from JSON file."""
    print(f"\nProcessing: {input_path}")

    with open(input_path, 'r') as f:
        target_measurements = json.load(f)

    # Validate measurements
    for measure in target_measurements.keys():
        if measure not in MEASUREMENTS:
            print(f"WARNING: Unknown measurement '{measure}' will be ignored")

    # Find macroparameters
    result = find_macroparameters(
        models, macro_bounds, target_measurements,
        method=method, weights=weights, verbose=True
    )

    # Save result
    output_path = str(input_path).replace('.json', '_result.json')

    # Convert all numpy types to Python native types for JSON serialization
    result_dict = {
        'target_measurements': target_measurements,
        'macroparameters': result['macroparameters'],
        'predicted_measurements': result['predicted_measurements'],
        'errors': result['errors'],
        'mae': result['mae'],
        'max_error': result['max_error']
    }
    result_dict = convert_numpy_types(result_dict)

    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nResult saved to: {output_path}")

    return result


def process_batch(input_path, output_path, models, macro_bounds, method, weights):
    """Process multiple measurements from CSV file."""
    print(f"\nProcessing batch: {input_path}")

    df = pd.read_csv(input_path)

    print(f"Found {len(df)} entries to process")
    print("="*80)

    results = []

    for idx, row in df.iterrows():
        # Extract target measurements
        target_measurements = {}
        for measure in MEASUREMENTS:
            if measure in row and pd.notna(row[measure]):
                target_measurements[measure] = row[measure]

        if not target_measurements:
            print(f"\nWARNING: No valid measurements for row {idx}, skipping...")
            continue

        # Progress bar
        print_progress_bar(idx + 1, len(df),
                         prefix='Processing:',
                         suffix=f'Entry {idx + 1}/{len(df)}')

        # Find macroparameters
        result = find_macroparameters(
            models, macro_bounds, target_measurements,
            method=method, weights=weights, verbose=False
        )

        # Store results
        result_row = {'id': idx}
        result_row.update(result['macroparameters'])
        result_row['mae'] = result['mae']
        result_row['max_error'] = result['max_error']

        # Add target, predicted, and errors
        for measure in MEASUREMENTS:
            if measure in target_measurements:
                result_row[f'target_{measure}'] = target_measurements[measure]
                result_row[f'predicted_{measure}'] = result['predicted_measurements'][measure]
                result_row[f'error_{measure}'] = result['errors'][measure]

        results.append(result_row)

    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"\n\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Processed:         {len(results_df)} entries")
    print(f"Average MAE:       {results_df['mae'].mean():.4f} cm")
    print(f"Average Max Error: {results_df['max_error'].mean():.4f} cm")
    print(f"Best MAE:          {results_df['mae'].min():.4f} cm")
    print(f"Worst MAE:         {results_df['mae'].max():.4f} cm")
    print("="*80)

    return results_df


# ============================================================================
# MICROPARAMETER ADJUSTMENT FUNCTIONS
# ============================================================================

def load_sensitivity_matrix(sensitivity_file=None):
    """
    Load microparameter sensitivity matrix.

    Args:
        sensitivity_file: Path to sensitivity JSON file. If None, uses default location.

    Returns:
        Dictionary mapping microparameter names to sensitivity coefficients
    """
    if sensitivity_file is None:
        script_dir = Path(__file__).parent.absolute()
        parent_dir = script_dir.parent.absolute()
        sensitivity_file = parent_dir / 'macroparameters_generation_and_analysis' / 'micro_sensitivity_matrix.json'

    if not sensitivity_file.exists():
        print(f"\nWarning: Sensitivity matrix not found at {sensitivity_file}")
        print("Run build_micro_sensitivity_table.py first to generate it.")
        return None

    with open(sensitivity_file, 'r') as f:
        sensitivity_matrix = json.load(f)

    return sensitivity_matrix


def calculate_micro_adjustments(macro_measurements, target_measurements, sensitivity_matrix, verbose=True):
    """
    Calculate microparameter adjustments based on residuals and sensitivity matrix.

    Args:
        macro_measurements: Measurements from macroparameter-only mesh
        target_measurements: Target measurements to match
        sensitivity_matrix: Dictionary of microparameter sensitivities
        verbose: Print detailed results

    Returns:
        Dictionary of microparameter names to adjustment values (0.0-1.0)
    """
    if sensitivity_matrix is None:
        if verbose:
            print("\nNo sensitivity matrix available - skipping micro adjustments")
        return {}

    # Calculate residuals
    residuals = {
        measure: target_measurements.get(measure, 0) - macro_measurements.get(measure, 0)
        for measure in MEASUREMENTS
    }

    if verbose:
        print("\n" + "="*80)
        print("MICROPARAMETER ADJUSTMENT CALCULATION")
        print("="*80)
        print("\nMeasurement Residuals (Target - Macro):")
        for measure in MEASUREMENTS:
            residual = residuals[measure]
            print(f"  {measure:25s}: {residual:+.2f} cm")

    # Microparameter mapping
    # Format: {micro_name: primary_measurement}
    micro_mapping = {
        'measure-upperleg-height-incr': 'height_cm',
        'measure-lowerleg-height-incr': 'height_cm',
        'measure-napetowaist-dist-incr': 'height_cm',
        'measure-shoulder-dist-incr': 'shoulder_width_cm',
        'measure-hips-circ-incr': 'hip_width_cm',
        'head-scale-horiz-incr': 'head_width_cm',
        'neck-scale-vert-incr': 'neck_length_cm',
        'measure-upperarm-length-incr': 'upper_arm_length_cm',
        'measure-lowerarm-length-incr': 'forearm_length_cm',
        'l-hand-scale-incr': 'hand_length_cm',
    }

    # Calculate micro adjustments
    micro_adjustments = {}

    if verbose:
        print("\nMicroparameter Adjustments:")

    for micro_name, primary_measure in micro_mapping.items():
        if micro_name not in sensitivity_matrix:
            if verbose:
                print(f"  {micro_name:35s}: No sensitivity data")
            continue

        # Get sensitivity coefficient for this micro's primary measurement
        sensitivity = sensitivity_matrix[micro_name].get(primary_measure, 0)

        if abs(sensitivity) < 0.01:  # Skip if sensitivity too low
            micro_adjustments[micro_name] = 0.5  # Neutral value
            if verbose:
                print(f"  {micro_name:35s}: Neutral (low sensitivity)")
            continue

        # Calculate required adjustment: residual / sensitivity
        # Residual is in cm, sensitivity is in cm/unit
        # So adjustment is in units (0-1 range)
        required_adjustment = residuals[primary_measure] / sensitivity

        # For height, split adjustment across 3 components
        if primary_measure == 'height_cm':
            if 'upperleg' in micro_name:
                required_adjustment *= 0.4  # 40% to upper leg
            elif 'lowerleg' in micro_name:
                required_adjustment *= 0.4  # 40% to lower leg
            elif 'napetowaist' in micro_name:
                required_adjustment *= 0.2  # 20% to torso

        # Start from neutral (0.5) and add adjustment
        micro_value = 0.5 + required_adjustment

        # Clamp to valid range [0.0, 1.0]
        micro_value = max(0.0, min(1.0, micro_value))

        micro_adjustments[micro_name] = micro_value

        if verbose:
            print(f"  {micro_name:35s}: {micro_value:.3f} (Δ{required_adjustment:+.3f})")

    if verbose:
        print("="*80)

    return micro_adjustments


def apply_micro_adjustments(macroparameters, micro_adjustments):
    """
    Combine macroparameters with microparameter adjustments.

    Args:
        macroparameters: Dictionary of macroparameter values
        micro_adjustments: Dictionary of microparameter values

    Returns:
        Dictionary with both macros and micros combined
    """
    combined = {
        'macroparameters': macroparameters,
        'microparameters': micro_adjustments
    }

    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Infer macroparameters from body measurements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single measurement from JSON
  python infer_macroparameters.py --input measurements.json --models model.pkl

  # Batch processing from CSV
  python infer_macroparameters.py --input measurements.csv --models model.pkl --output results.csv

  # Use hybrid optimization for better accuracy
  python infer_macroparameters.py --input measurements.json --models model.pkl --method both
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input measurements file (JSON or CSV)'
    )

    parser.add_argument(
        '--models',
        type=str,
        default='macroparameters_inference_models.pkl',
        help='Path to trained models pickle file (default: macroparameters_inference_models.pkl)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='inference_results.csv',
        help='Output CSV file for batch mode (default: inference_results.csv)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='differential_evolution',
        choices=['differential_evolution', 'both'],
        help='Optimization method (default: differential_evolution)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("INFER MACROPARAMETERS FROM MEASUREMENTS")
    print("=" * 80)

    try:
        # Load models
        models, macro_bounds = load_models(args.models)

        # Determine input type and process
        input_path = Path(args.input)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path.suffix == '.json':
            # Single measurement
            process_single(input_path, models, macro_bounds, args.method, weights=None)

        elif input_path.suffix == '.csv':
            # Batch processing
            process_batch(input_path, args.output, models, macro_bounds, args.method, weights=None)

        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}. Use .json or .csv")

        print("\n" + "="*80)
        print("INFERENCE COMPLETE")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
