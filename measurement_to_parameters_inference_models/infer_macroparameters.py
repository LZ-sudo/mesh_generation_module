"""
Infer Macroparameters from Body Measurements using Inverse Mapping Models

This script uses trained models (TabM, TabPFN, or XGBoost) to directly predict
macroparameter values from body measurements (inverse mapping: measurements → macroparameters).

No optimization is required - predictions are instantaneous.

Supports:
- TabM (single multi-output model)
- TabPFN (5 independent models)
- XGBoost (5 independent models)

Usage:
    # From JSON file with measurements
    python infer_macroparameters.py --input measurements.json --models model.pkl

    # From CSV file (batch mode)
    python infer_macroparameters.py --input measurements.csv --models model.pkl --output results.csv
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
import sys
from pathlib import Path

# Try to import PyTorch for TabM models
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configuration
MACROPARAMETERS = ['age', 'muscle', 'weight', 'height', 'proportions']
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


def parse_gender(gender_input) -> float:
    """
    Parse gender input to numeric value.

    Args:
        gender_input: Either a string ("male"/"female") or float (0.0-1.0)

    Returns:
        Float value: 0.0 for female, 1.0 for male
    """
    if isinstance(gender_input, str):
        gender_lower = gender_input.lower()
        if gender_lower in ['female', 'f']:
            return 0.0
        elif gender_lower in ['male', 'm']:
            return 1.0
        else:
            raise ValueError(f"Invalid gender: {gender_input}. Use 'male' or 'female'")
    elif isinstance(gender_input, (int, float)):
        return float(gender_input)
    else:
        raise ValueError(f"Invalid gender type: {type(gender_input)}")


def parse_race(race_input) -> dict:
    """
    Parse race input to numeric dictionary.

    Args:
        race_input: Either a string ("asian"/"caucasian"/"african") or dict with race values

    Returns:
        Dictionary with race values: {'asian': float, 'caucasian': float, 'african': float}
    """
    if isinstance(race_input, str):
        race_lower = race_input.lower()
        if race_lower == 'asian':
            return {'asian': 1.0, 'caucasian': 0.0, 'african': 0.0}
        elif race_lower == 'caucasian':
            return {'asian': 0.0, 'caucasian': 1.0, 'african': 0.0}
        elif race_lower == 'african':
            return {'asian': 0.0, 'caucasian': 0.0, 'african': 1.0}
        else:
            raise ValueError(f"Invalid race: {race_input}. Use 'asian', 'caucasian', or 'african'")
    elif isinstance(race_input, dict):
        # Validate that all required keys are present
        required_keys = {'asian', 'caucasian', 'african'}
        if not required_keys.issubset(race_input.keys()):
            raise ValueError(f"Race dict must contain keys: {required_keys}")
        return race_input
    else:
        raise ValueError(f"Invalid race type: {type(race_input)}")


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

    Supports both formats:
    - Old format: data['models'] = dict of 5 models
    - TabM format: data['model'] = single multi-output model

    Args:
        model_path: Path to pickle file with trained models

    Returns:
        tuple: (model_data dict, macro_bounds dict)
    """
    print(f"Loading models from: {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    macro_bounds = data['macro_bounds']

    # Detect model format
    if 'model' in data:
        # TabM format: single multi-output model
        model_type = data.get('model_type', 'TabM_MultiOutput')
        print(f"Loaded TabM model (type: {model_type})")
        print(f"  Ensemble size: {data.get('ensemble_size', 'unknown')}")

        model_data = {
            'type': 'tabm',
            'model': data['model'],
            'scalers': data.get('scalers', {}),
            'ensemble_size': data.get('ensemble_size', 128)
        }
    elif 'models' in data:
        # Old format: multiple independent models
        models = data['models']
        print(f"Loaded {len(models)} independent models (XGBoost/TabPFN)")

        model_data = {
            'type': 'multi',
            'models': models
        }
    else:
        raise ValueError("Unknown model format: neither 'model' nor 'models' found in pickle file")

    print(f"\nMacroparameter bounds:")
    for param, (min_val, max_val) in macro_bounds.items():
        print(f"  {param:12s}: [{min_val:.3f}, {max_val:.3f}]")
    print("-" * 80)

    return model_data, macro_bounds


def predict_macroparameters(model_data, measurements):
    """
    Predict macroparameters from measurements using inverse mapping models.

    Supports both TabM (single model) and multi-model (TabPFN/XGBoost) formats.

    Args:
        model_data: Dictionary containing model info and trained model(s)
        measurements: Dictionary of measurement values

    Returns:
        Dictionary of predicted macroparameter values (as Python floats)
    """
    # Convert to DataFrame for consistency
    measurements_df = pd.DataFrame([measurements], columns=MEASUREMENTS)

    if model_data['type'] == 'tabm':
        # TabM: single multi-output model
        model = model_data['model']
        scalers = model_data['scalers']

        # Standardize input
        X_scaled = scalers['X_scaler'].transform(measurements_df.values)

        # Check for NaN after scaling
        if np.isnan(X_scaled).any():
            print(f"WARNING: NaN detected after input scaling!")
            print(f"  Input measurements: {measurements_df.values}")
            print(f"  Scaled values: {X_scaled}")

        # Predict with TabM model
        model.eval()
        if TORCH_AVAILABLE:
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                predictions = model(X_tensor, None)  # (1, k, 5)
                predictions_mean = predictions.mean(dim=1)  # Average ensemble: (1, 5)
                y_pred_scaled = predictions_mean.cpu().numpy()

                # Check for NaN in predictions
                if np.isnan(y_pred_scaled).any():
                    print(f"WARNING: NaN detected in model predictions!")
                    print(f"  Predictions (scaled): {y_pred_scaled}")
        else:
            raise RuntimeError("PyTorch is required for TabM models. Install with: pip install torch")

        # Inverse transform predictions
        y_pred = scalers['y_scaler'].inverse_transform(y_pred_scaled)[0]

        # Check for NaN after inverse transform
        if np.isnan(y_pred).any():
            print(f"WARNING: NaN detected after inverse transform!")
            print(f"  Predictions (unscaled): {y_pred}")

        # Convert to dictionary
        macroparameters = {}
        for i, param_name in enumerate(MACROPARAMETERS):
            macroparameters[param_name] = float(y_pred[i])

    elif model_data['type'] == 'multi':
        # Multiple independent models (TabPFN/XGBoost)
        models = model_data['models']

        macroparameters = {}
        for param_name in MACROPARAMETERS:
            prediction = models[param_name].predict(measurements_df.values)[0]
            # Convert numpy float32 to Python float for JSON serialization
            macroparameters[param_name] = float(prediction)

    else:
        raise ValueError(f"Unknown model type: {model_data['type']}")

    return macroparameters


def find_macroparameters(model_data, macro_bounds, target_measurements,
                        method='direct', weights=None, verbose=True):
    """
    Predict macroparameters directly from target measurements using inverse mapping.

    Supports TabM (single model) and TabPFN/XGBoost (multiple models).

    NOTE: The 'method' and 'weights' parameters are kept for backward compatibility
    but are ignored since direct inverse mapping is used.

    Args:
        model_data: Dictionary containing model info and trained model(s)
        macro_bounds: Dictionary of macroparameter bounds (used for validation)
        target_measurements: Dictionary of target measurements
        method: Ignored (kept for backward compatibility)
        weights: Ignored (kept for backward compatibility)
        verbose: Print detailed results

    Returns:
        Dictionary with results
    """
    model_type_name = "TabM" if model_data['type'] == 'tabm' else "Multi-Model"

    if verbose:
        print("\n" + "="*80)
        print(f"PREDICTING MACROPARAMETERS ({model_type_name} Direct Inverse Mapping)")
        print("="*80)
        print("\nInput Measurements:")
        for measure, value in target_measurements.items():
            print(f"  {measure:25s}: {value:.2f} cm")

    # Direct prediction using inverse mapping
    macroparameters = predict_macroparameters(model_data, target_measurements)

    # Clip to bounds
    for param in MACROPARAMETERS:
        min_val, max_val = macro_bounds[param]
        macroparameters[param] = np.clip(macroparameters[param], min_val, max_val)

    result_dict = {
        'macroparameters': macroparameters,
        'predicted_measurements': {},  # Not available with inverse mapping
        'errors': {},  # Not available with inverse mapping
        'mae': 0.0,  # Not available with inverse mapping
        'max_error': 0.0  # Not available with inverse mapping
    }

    if verbose:
        print("\nPredicted Macroparameters:")
        for param, value in macroparameters.items():
            min_val, max_val = macro_bounds[param]
            print(f"  {param:12s}: {value:.4f}  (range: [{min_val:.3f}, {max_val:.3f}])")
        print("="*80)

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
    """
    Process a single measurement from JSON file.

    Expected JSON format:
    {
      "gender": "female",  # Optional: "male" or "female" (default: "female")
      "race": "asian",     # Optional: "asian", "caucasian", or "african" (default: "asian")
      "measurements": {    # Can also be at top level
        "height_cm": 165.0,
        ...
      }
    }
    """
    print(f"\nProcessing: {input_path}")

    with open(input_path, 'r') as f:
        input_data = json.load(f)

    # Parse gender (supports both string and numeric formats)
    gender_input = input_data.get('gender', 'female')
    gender = parse_gender(gender_input)

    # Parse race (supports both string and dict formats)
    race_input = input_data.get('race', 'asian')
    race = parse_race(race_input)

    # Normalize race values if dict was provided
    if isinstance(race_input, dict):
        race_sum = sum(race.values())
        if abs(race_sum - 1.0) > 0.01:
            print(f"  Warning: Race values sum to {race_sum}, normalizing...")
            race = {k: v/race_sum for k, v in race.items()}

    print(f"\nSubject Details:")
    print(f"  Gender: {'Male' if gender > 0.5 else 'Female'}")
    print(f"  Race: {', '.join([f'{k}={v:.2f}' for k, v in race.items()])}")

    # Extract measurements (can be nested under 'measurements' or at top level)
    if 'measurements' in input_data:
        target_measurements = input_data['measurements']
    else:
        # Filter out non-measurement keys
        target_measurements = {k: v for k, v in input_data.items()
                             if k in MEASUREMENTS}

    # Validate measurements
    for measure in target_measurements.keys():
        if measure not in MEASUREMENTS:
            print(f"  WARNING: Unknown measurement '{measure}' will be ignored")

    # Find macroparameters
    result = find_macroparameters(
        models, macro_bounds, target_measurements,
        method=method, weights=weights, verbose=True
    )

    # Save result
    output_path = str(input_path).replace('.json', '_result.json')

    # Convert all numpy types to Python native types for JSON serialization
    result_dict = {
        'gender': gender,
        'race': race,
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
