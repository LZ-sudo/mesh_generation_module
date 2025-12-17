"""
Train Inverse Mapping Models

This script trains Random Forest models to map macroparameters to body measurements.
It reads from a lookup table CSV and saves trained models to a pickle file.

Usage:
    python train_model.py --input lookup_tables/lookup_table_female_asian.csv --output models/female_asian_model.pkl
    python train_model.py --input lookup_tables/lookup_table_female_asian.csv --output models/female_asian_model.pkl --n-estimators 300 --max-depth 20
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Configuration
MACROPARAMETERS = ['age', 'muscle', 'weight', 'height', 'proportions']
MEASUREMENTS = [
    'height_cm', 'shoulder_width_cm', 'hip_width_cm', 'head_width_cm',
    'neck_length_cm', 'upper_arm_length_cm', 'forearm_length_cm', 'hand_length_cm'
]


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar to console."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def load_data(csv_path):
    """
    Load training data from lookup table CSV.

    Args:
        csv_path: Path to lookup table CSV file

    Returns:
        tuple: (X DataFrame with macroparameters, y DataFrame with measurements, macro_bounds dict)
    """
    print(f"Loading data from: {csv_path}")

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate columns
    missing_macros = set(MACROPARAMETERS) - set(df.columns)
    if missing_macros:
        raise ValueError(f"Missing macroparameter columns: {missing_macros}")

    missing_measures = set(MEASUREMENTS) - set(df.columns)
    if missing_measures:
        raise ValueError(f"Missing measurement columns: {missing_measures}")

    # Extract features and targets
    X = df[MACROPARAMETERS]
    y = df[MEASUREMENTS]

    # Calculate bounds for macroparameters
    macro_bounds = {}
    for param in MACROPARAMETERS:
        macro_bounds[param] = (X[param].min(), X[param].max())

    print(f"Loaded {len(df)} samples")
    print(f"\nMacroparameter bounds:")
    for param, (min_val, max_val) in macro_bounds.items():
        print(f"  {param:12s}: [{min_val:.3f}, {max_val:.3f}]")
    print("-" * 80)

    return X, y, macro_bounds


def train_models(X, y, n_estimators=200, max_depth=15, test_size=0.2, random_state=42):
    """
    Train Random Forest models for each measurement.

    Args:
        X: DataFrame with macroparameter features
        y: DataFrame with measurement targets
        n_estimators: Number of trees in Random Forest
        max_depth: Maximum depth of trees
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        dict: Trained models for each measurement
    """
    print(f"\nTraining models...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  test_size: {test_size}")
    print("-" * 80)

    models = {}
    performance = []

    for idx, measure in enumerate(MEASUREMENTS):
        # Progress bar
        print_progress_bar(idx, len(MEASUREMENTS),
                         prefix='Training:',
                         suffix=f'{measure}')

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y[measure], test_size=test_size, random_state=random_state
        )

        # Train model (n_jobs=1 for Windows compatibility)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1
        )
        model.fit(X_train, y_train)

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        models[measure] = model
        performance.append({
            'measurement': measure,
            'train_r2': train_score,
            'test_r2': test_score
        })

    # Final progress
    print_progress_bar(len(MEASUREMENTS), len(MEASUREMENTS),
                     prefix='Training:',
                     suffix='Complete!')

    # Print performance summary
    print("\nModel Performance:")
    print("-" * 80)
    print(f"{'Measurement':<25s} {'Train R²':>12s} {'Test R²':>12s}")
    print("-" * 80)

    for perf in performance:
        print(f"{perf['measurement']:<25s} {perf['train_r2']:>12.4f} {perf['test_r2']:>12.4f}")

    print("-" * 80)

    # Check for overfitting
    avg_train_r2 = np.mean([p['train_r2'] for p in performance])
    avg_test_r2 = np.mean([p['test_r2'] for p in performance])

    print(f"{'Average':<25s} {avg_train_r2:>12.4f} {avg_test_r2:>12.4f}")

    if avg_train_r2 - avg_test_r2 > 0.05:
        print("\nWARNING: Possible overfitting detected (train R² >> test R²)")
    else:
        print("\nModels trained successfully!")

    print("-" * 80)

    return models, performance


def save_models(models, macro_bounds, output_path, performance=None):
    """
    Save trained models to pickle file.

    Args:
        models: Dictionary of trained models
        macro_bounds: Dictionary of macroparameter bounds
        output_path: Path to save pickle file
        performance: Optional performance metrics to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'models': models,
        'macro_bounds': macro_bounds,
        'macroparameters': MACROPARAMETERS,
        'measurements': MEASUREMENTS,
        'performance': performance
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nModels saved to: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Train inverse mapping models from lookup table',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default parameters
  python train_model.py --input lookup_tables/lookup_table_female_asian.csv

  # Train with custom parameters
  python train_model.py --input lookup_tables/lookup_table_female_asian.csv --n-estimators 300 --max-depth 20

  # Specify output location
  python train_model.py --input lookup_tables/lookup_table_female_asian.csv --output models/my_model.pkl
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to lookup table CSV file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='macroparameters_inference_models.pkl',
        help='Path to save trained models (default: macroparameters_inference_models.pkl)'
    )

    parser.add_argument(
        '--n-estimators',
        type=int,
        default=200,
        help='Number of trees in Random Forest (default: 200)'
    )

    parser.add_argument(
        '--max-depth',
        type=int,
        default=15,
        help='Maximum depth of trees (default: 15)'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TRAIN INVERSE MAPPING MODELS")
    print("=" * 80)

    try:
        # Load data
        X, y, macro_bounds = load_data(args.input)

        # Train models
        models, performance = train_models(
            X, y,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            test_size=args.test_size,
            random_state=args.random_seed
        )

        # Save models
        save_models(models, macro_bounds, args.output, performance)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\nTrained models saved to: {args.output}")
        print(f"Use 'infer_macroparameters.py' to find macroparameters from measurements")
        print(f"Use 'test_model_accuracy.py' to validate model performance")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
