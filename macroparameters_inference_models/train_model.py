"""
Train Inverse Mapping Models with Hybrid Bayesian Optimization

This script uses a two-stage approach:
1. Fast Bayesian optimization with cross-validation (50-100 trials, ~30-60 minutes)
2. Validation of top candidates with real mesh generation (~10-20 minutes)

This provides excellent results in a practical timeframe.

Usage:
    # Quick training (50 trials CV + top 3 validation)
    python train_model.py --input lookup_table.csv

    # Full training (100 trials CV + top 5 validation)
    python train_model.py --input lookup_table.csv --n-trials 100 --n-validate 5

    # Extended training (200 trials CV + top 5 validation)
    python train_model.py --input lookup_table.csv --n-trials 200 --n-validate 5
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import sys
import json
import subprocess
import time
import gc
from pathlib import Path
from sklearn.model_selection import cross_val_score
from scipy.optimize import differential_evolution

# Try to import optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: optuna not installed. Install with: pip install optuna")

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    # Check if GPU/CUDA is available for XGBoost by testing if 'gpu_hist' is a valid tree method
    try:
        # Try to create a DMatrix and train with gpu_hist
        # If XGBoost was compiled without GPU support, this will fail
        import numpy as np
        test_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        test_y = np.array([1.0, 2.0])
        dtrain = xgb.DMatrix(test_X, label=test_y)
        params = {'tree_method': 'gpu_hist', 'verbosity': 0}
        xgb.train(params, dtrain, num_boost_round=1)
        GPU_AVAILABLE = True
    except Exception as e:
        # If we get an error about invalid tree_method, GPU is not available
        GPU_AVAILABLE = False
except ImportError:
    XGBOOST_AVAILABLE = False
    GPU_AVAILABLE = False
    print("WARNING: xgboost not installed. Install with: pip install xgboost")

# Configuration
MACROPARAMETERS = ['age', 'muscle', 'weight', 'height', 'proportions']
MEASUREMENTS = [
    'height_cm', 'shoulder_width_cm', 'hip_width_cm', 'head_width_cm',
    'neck_length_cm', 'upper_arm_length_cm', 'forearm_length_cm', 'hand_length_cm'
]

# Adult measurement bounds (in cm) for generating random test cases
ADULT_MEASUREMENT_BOUNDS = {
    'height_cm': (145.0, 190.0),
    'shoulder_width_cm': (35.0, 50.0),
    'hip_width_cm': (30.0, 45.0),
    'head_width_cm': (13.0, 17.0),
    'neck_length_cm': (8.0, 13.0),
    'upper_arm_length_cm': (25.0, 35.0),
    'forearm_length_cm': (20.0, 30.0),
    'hand_length_cm': (15.0, 22.0)
}

# Get paths
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent.absolute()


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar to console."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def load_data(csv_path):
    """Load training data from lookup table CSV."""
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


def evaluate_with_cv(X, y, n_estimators, max_depth, learning_rate,
                     min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                     gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
                     cv_folds=5, random_state=42, verbose=False):
    """
    Evaluate XGBoost model using cross-validation with comprehensive hyperparameters.

    Returns average MAE across all measurements and folds.
    """
    if verbose:
        print(f"  Cross-validating: n_est={n_estimators}, depth={max_depth}, lr={learning_rate:.4f}")

    total_mae = 0.0

    # Determine tree method based on GPU availability
    tree_method = 'gpu_hist' if GPU_AVAILABLE else 'hist'
    device_to_use = 'cuda' if GPU_AVAILABLE else 'cpu'

    for idx, measure in enumerate(MEASUREMENTS):
        if verbose:
            print_progress_bar(idx, len(MEASUREMENTS),
                             prefix='    CV Progress:',
                             suffix=f'{measure}')

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method=tree_method,
            device=device_to_use,
            random_state=random_state,
            verbosity=0  # Suppress XGBoost output
        )

        # Cross-validation with negative MAE scoring
        scores = cross_val_score(
            model, X, y[measure],
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=1
        )

        # Convert negative MAE back to positive
        mae = -scores.mean()
        total_mae += mae

    if verbose:
        print_progress_bar(len(MEASUREMENTS), len(MEASUREMENTS),
                         prefix='    CV Progress:',
                         suffix='Complete!')

    # Average MAE across all measurements
    avg_mae = total_mae / len(MEASUREMENTS)

    return avg_mae


def train_models_with_params(X, y, hyperparams, random_state=42, verbose=True):
    """Train XGBoost models with specific hyperparameters on full dataset.

    Args:
        hyperparams: Dictionary containing all XGBoost hyperparameters
    """
    models = {}

    # Determine tree method based on GPU availability
    tree_method = 'gpu_hist' if GPU_AVAILABLE else 'hist'
    device_to_use = 'cuda' if GPU_AVAILABLE else 'cpu'

    for idx, measure in enumerate(MEASUREMENTS):
        if verbose:
            print_progress_bar(idx, len(MEASUREMENTS),
                             prefix='  Training models:',
                             suffix=f'{measure}')

        # Train on full dataset
        model = xgb.XGBRegressor(
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams['max_depth'],
            learning_rate=hyperparams['learning_rate'],
            min_child_weight=hyperparams.get('min_child_weight', 1),
            subsample=hyperparams.get('subsample', 1.0),
            colsample_bytree=hyperparams.get('colsample_bytree', 1.0),
            gamma=hyperparams.get('gamma', 0.0),
            reg_alpha=hyperparams.get('reg_alpha', 0.0),
            reg_lambda=hyperparams.get('reg_lambda', 1.0),
            tree_method=tree_method,
            device=device_to_use,
            random_state=random_state,
            verbosity=0  # Suppress XGBoost output
        )
        model.fit(X, y[measure])

        models[measure] = model

    if verbose:
        print_progress_bar(len(MEASUREMENTS), len(MEASUREMENTS),
                         prefix='  Training models:',
                         suffix='Complete!')

    return models


def generate_random_measurements():
    """Generate random adult measurements within realistic bounds."""
    measurements = {}
    for measure, (min_val, max_val) in ADULT_MEASUREMENT_BOUNDS.items():
        measurements[measure] = np.random.uniform(min_val, max_val)
    return measurements


def predict_measurements(models, macroparameters):
    """Predict measurements from macroparameters."""
    macro_df = pd.DataFrame([macroparameters], columns=MACROPARAMETERS)

    predictions = {}
    for measure in MEASUREMENTS:
        predictions[measure] = models[measure].predict(macro_df)[0]

    return predictions


def objective_function(macroparameters, models, target_measurements):
    """Objective function for optimization."""
    predicted = predict_measurements(models, macroparameters)

    total_error = 0.0
    for measure in MEASUREMENTS:
        if measure in target_measurements:
            error = predicted[measure] - target_measurements[measure]
            total_error += error ** 2

    return total_error


def find_macroparameters(models, macro_bounds, target_measurements):
    """Find macroparameters that best match target measurements."""
    bounds = [(macro_bounds[param][0], macro_bounds[param][1])
              for param in MACROPARAMETERS]

    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(models, target_measurements),
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

    macroparameters = {param: result.x[i] for i, param in enumerate(MACROPARAMETERS)}
    return macroparameters


def generate_and_measure_mesh(macroparameters, rig_type='default_no_toes'):
    """Generate mesh and return actual measurements."""
    # Build config
    config = {
        'fixed_params': {
            'gender': 0.0,
            'cupsize': 0.5,
            'firmness': 0.5,
            'race': {'asian': 1.0, 'caucasian': 0.0, 'african': 0.0}
        },
        'grid_params': {
            param: {'min': macroparameters[param], 'max': macroparameters[param], 'step': 1.0}
            for param in MACROPARAMETERS
        }
    }

    # Save config
    config_path = parent_dir / 'temp_optuna_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Output path
    output_csv_path = parent_dir / 'temp_optuna_measurements.csv'

    # Run Blender
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
            timeout=300
        )

        if result.returncode != 0:
            raise RuntimeError(f"Blender process failed with code {result.returncode}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Blender process timed out")

    # Load measurements
    if not output_csv_path.exists():
        raise RuntimeError("Measurements CSV not created by Blender")

    df = pd.read_csv(output_csv_path)
    if len(df) == 0:
        raise RuntimeError("CSV is empty")

    row = df.iloc[0]
    actual_measurements = {measure: row[measure] for measure in MEASUREMENTS}

    # Cleanup temporary files
    config_path.unlink()
    output_csv_path.unlink()

    # Force garbage collection
    del df, row, config
    gc.collect()

    return actual_measurements


def evaluate_model_with_mesh(models, macro_bounds):
    """
    Evaluate model by generating random measurements, inferring macroparameters,
    generating actual mesh, and comparing results.

    Returns MAE between predicted and actual measurements.
    """
    # Generate random target measurements
    target_measurements = generate_random_measurements()

    # Find macroparameters using trained models
    macroparameters = find_macroparameters(models, macro_bounds, target_measurements)

    # Predict what measurements these macroparameters should produce
    predicted_measurements = predict_measurements(models, macroparameters)

    # Generate actual mesh and measure it
    actual_measurements = generate_and_measure_mesh(macroparameters)

    # Calculate error between actual and target measurements
    errors = []
    for measure in MEASUREMENTS:
        error = abs(actual_measurements[measure] - target_measurements[measure])
        errors.append(error)

    mae = np.mean(errors)

    # Cleanup
    del target_measurements, macroparameters, predicted_measurements, actual_measurements
    gc.collect()

    return mae


def objective_optuna_cv(trial, X, y):
    """
    Optuna objective function for Stage 1: Cross-validation based optimization.

    Optimizes comprehensive set of XGBoost hyperparameters for best performance.
    """
    # Core tree parameters
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=50)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)

    # Regularization parameters (prevent overfitting)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    gamma = trial.suggest_float('gamma', 0.0, 5.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0.0, 10.0)  # L1 regularization
    reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0)  # L2 regularization

    # Evaluate with cross-validation
    mae = evaluate_with_cv(
        X, y,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        cv_folds=5,
        verbose=False
    )

    return mae


def save_models(models, macro_bounds, output_path, performance=None, hyperparameters=None):
    """Save trained models to pickle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'models': models,
        'macro_bounds': macro_bounds,
        'macroparameters': MACROPARAMETERS,
        'measurements': MEASUREMENTS,
        'performance': performance,
        'hyperparameters': hyperparameters
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nModels saved to: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Train models with hybrid Bayesian optimization (CV + mesh validation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training (50 trials + top 3 validation, ~40-60 minutes)
  python train_model.py --input lookup_table.csv

  # Full training (100 trials + top 5 validation, ~60-90 minutes)
  python train_model.py --input lookup_table.csv --n-trials 100 --n-validate 5

  # Extended training (200 trials + top 5 validation, ~2-3 hours)
  python train_model.py --input lookup_table.csv --n-trials 200 --n-validate 5
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
        '--n-trials',
        type=int,
        default=50,
        help='Number of CV trials in Stage 1 (default: 50)'
    )

    parser.add_argument(
        '--n-validate',
        type=int,
        default=3,
        help='Number of top candidates to validate with mesh generation in Stage 2 (default: 3)'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Check if required libraries are available
    if not OPTUNA_AVAILABLE:
        print("\nERROR: optuna is required for Bayesian optimization")
        print("Install with: pip install optuna")
        return 1

    if not XGBOOST_AVAILABLE:
        print("\nERROR: xgboost is required for training")
        print("Install with: pip install xgboost")
        return 1

    print("=" * 80)
    print("HYBRID BAYESIAN OPTIMIZATION TRAINING (XGBoost)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Stage 1 (CV): {args.n_trials} trials")
    print(f"  Stage 2 (Mesh): Top {args.n_validate} candidates")

    # GPU detection
    if GPU_AVAILABLE:
        print(f"\n  GPU: AVAILABLE (using tree_method='gpu_hist')")
        print(f"  Note: GPU acceleration can be 5-20x faster!")
    else:
        print(f"\n  GPU: Not available (using tree_method='hist' on CPU)")
        print(f"  Note: Install CUDA and compatible XGBoost for GPU acceleration")

    # Estimate time
    stage1_time_min = args.n_trials * 0.5  # ~30 seconds per trial
    stage1_time_max = args.n_trials * 1.2  # ~1.2 minutes per trial
    stage2_time_min = args.n_validate * 2  # ~2 minutes per mesh validation
    stage2_time_max = args.n_validate * 5  # ~5 minutes per mesh validation

    total_min = (stage1_time_min + stage2_time_min)
    total_max = (stage1_time_max + stage2_time_max)

    print(f"\nEstimated time:")
    print(f"  Stage 1 (CV): {stage1_time_min:.0f}-{stage1_time_max:.0f} minutes")
    print(f"  Stage 2 (Mesh): {stage2_time_min:.0f}-{stage2_time_max:.0f} minutes")
    print(f"  Total: {total_min:.0f}-{total_max:.0f} minutes")

    try:
        # Load data
        print("\n" + "=" * 80)
        X, y, macro_bounds = load_data(args.input)

        # ========================================================================
        # STAGE 1: FAST BAYESIAN OPTIMIZATION WITH CROSS-VALIDATION
        # ========================================================================
        print("\n" + "=" * 80)
        print("STAGE 1: BAYESIAN OPTIMIZATION WITH CROSS-VALIDATION")
        print("=" * 80)
        print(f"Running {args.n_trials} trials with 5-fold cross-validation...")
        print("Note: This stage is fast and explores the hyperparameter space efficiently\n")

        stage1_start = time.time()

        # Suppress Optuna's logging output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create study with sampler that prevents duplicate trials
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(
                seed=args.random_seed,
                n_startup_trials=10,  # Random trials before TPE kicks in
                multivariate=True,  # Consider parameter interactions
                warn_independent_sampling=False  # Suppress warnings
            )
        )

        # Custom callback to show progress
        def trial_callback(study, trial):
            # Show progress bar
            print_progress_bar(
                trial.number + 1,
                args.n_trials,
                prefix='Stage 1 Progress:',
                suffix=f'Trial {trial.number + 1}/{args.n_trials} | '
                       f'n_est={trial.params["n_estimators"]}, '
                       f'depth={trial.params["max_depth"]}, '
                       f'lr={trial.params["learning_rate"]:.3f}, '
                       f'MAE={trial.value:.4f}cm'
            )

        study.optimize(
            lambda trial: objective_optuna_cv(trial, X, y),
            n_trials=args.n_trials,
            callbacks=[trial_callback],
            show_progress_bar=False
        )

        print()  # New line after progress bar

        stage1_time = time.time() - stage1_start

        print(f"\nStage 1 completed in {stage1_time/60:.1f} minutes")

        # Get top N UNIQUE candidates (avoid duplicates)
        seen_params = set()
        top_trials = []
        for trial in sorted(study.trials, key=lambda t: t.value):
            # Create a hashable tuple of parameters
            param_tuple = tuple(sorted(trial.params.items()))
            if param_tuple not in seen_params:
                seen_params.add(param_tuple)
                top_trials.append(trial)
                if len(top_trials) >= args.n_validate:
                    break

        print(f"\n" + "-" * 80)
        print(f"TOP {len(top_trials)} UNIQUE CANDIDATES FROM STAGE 1:")
        print("-" * 80)
        for i, trial in enumerate(top_trials, 1):
            print(f"{i}. CV-MAE={trial.value:.4f} cm")
            print(f"   n_estimators={trial.params['n_estimators']}, max_depth={trial.params['max_depth']}, lr={trial.params['learning_rate']:.4f}")
            print(f"   min_child_weight={trial.params['min_child_weight']}, subsample={trial.params['subsample']:.3f}, colsample={trial.params['colsample_bytree']:.3f}")

        # ========================================================================
        # STAGE 2: VALIDATION WITH REAL MESH GENERATION
        # ========================================================================
        print("\n" + "=" * 80)
        print("STAGE 2: VALIDATION WITH REAL MESH GENERATION")
        print("=" * 80)
        print(f"Validating top {args.n_validate} candidates by generating actual meshes...")
        print("Note: This stage ensures real-world performance\n")

        stage2_start = time.time()

        validation_results = []

        for i, trial in enumerate(top_trials, 1):
            print(f"\nValidating candidate {i}/{args.n_validate}:")
            print(f"  Hyperparameters:")
            print(f"    n_estimators={trial.params['n_estimators']}, max_depth={trial.params['max_depth']}")
            print(f"    learning_rate={trial.params['learning_rate']:.4f}, min_child_weight={trial.params['min_child_weight']}")
            print(f"    subsample={trial.params['subsample']:.3f}, colsample_bytree={trial.params['colsample_bytree']:.3f}")
            print(f"    gamma={trial.params['gamma']:.3f}, reg_alpha={trial.params['reg_alpha']:.3f}, reg_lambda={trial.params['reg_lambda']:.3f}")
            print(f"  CV-MAE: {trial.value:.4f} cm")

            # Train models with these hyperparameters
            print("  Training models on full dataset...")
            models = train_models_with_params(
                X, y,
                trial.params,
                verbose=True
            )

            # Evaluate with mesh generation
            print("  Generating and measuring test mesh...")
            try:
                mesh_mae = evaluate_model_with_mesh(models, macro_bounds)
                print(f"  Mesh-MAE: {mesh_mae:.4f} cm")

                validation_results.append({
                    **trial.params,  # Include all hyperparameters
                    'cv_mae': trial.value,
                    'mesh_mae': mesh_mae,
                    'models': models
                })

            except Exception as e:
                print(f"  ERROR during mesh validation: {e}")
                validation_results.append({
                    **trial.params,  # Include all hyperparameters
                    'cv_mae': trial.value,
                    'mesh_mae': 999.9,
                    'models': models
                })

            # Cleanup
            if i < len(top_trials):  # Don't delete the last one yet
                del models
                gc.collect()

        stage2_time = time.time() - stage2_start

        print(f"\nStage 2 completed in {stage2_time/60:.1f} minutes")

        # Print Stage 2 validation results
        print(f"\n" + "-" * 100)
        print(f"STAGE 2 VALIDATION RESULTS:")
        print("-" * 100)
        print(f"{'#':<4s} {'n_est':<7s} {'depth':<7s} {'lr':<10s} {'subsample':<10s} {'CV-MAE':<10s} {'Mesh-MAE':<12s}")
        print("-" * 100)

        for i, result in enumerate(validation_results, 1):
            print(f"{i:<4d} {result['n_estimators']:<7d} {result['max_depth']:<7d} "
                  f"{result['learning_rate']:<10.4f} {result['subsample']:<10.3f} "
                  f"{result['cv_mae']:<10.4f} {result['mesh_mae']:<12.4f}")

        print("-" * 100)

        # ========================================================================
        # FINAL RESULTS
        # ========================================================================
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)

        # Sort by mesh MAE (real-world performance)
        validation_results.sort(key=lambda x: x['mesh_mae'])

        print("\n" + "-" * 100)
        print(f"{'Rank':<6s} {'n_est':<7s} {'depth':<7s} {'lr':<10s} {'CV-MAE':<10s} {'Mesh-MAE':<12s} {'Status':<10s}")
        print("-" * 100)

        for i, result in enumerate(validation_results, 1):
            status = "BEST" if i == 1 else ""
            print(f"{i:<6d} {result['n_estimators']:<7d} {result['max_depth']:<7d} "
                  f"{result['learning_rate']:<10.4f} "
                  f"{result['cv_mae']:<10.4f} {result['mesh_mae']:<12.4f} {status:<10s}")

        print("-" * 100)

        best_result = validation_results[0]

        print(f"\nBest hyperparameters (based on real mesh generation):")
        print(f"  Core parameters:")
        print(f"    n_estimators: {best_result['n_estimators']}")
        print(f"    max_depth: {best_result['max_depth']}")
        print(f"    learning_rate: {best_result['learning_rate']:.4f}")
        print(f"  Regularization:")
        print(f"    min_child_weight: {best_result['min_child_weight']}")
        print(f"    subsample: {best_result['subsample']:.3f}")
        print(f"    colsample_bytree: {best_result['colsample_bytree']:.3f}")
        print(f"    gamma: {best_result['gamma']:.3f}")
        print(f"    reg_alpha: {best_result['reg_alpha']:.3f}")
        print(f"    reg_lambda: {best_result['reg_lambda']:.3f}")
        print(f"  Performance:")
        print(f"    CV-MAE: {best_result['cv_mae']:.4f} cm")
        print(f"    Mesh-MAE: {best_result['mesh_mae']:.4f} cm")

        total_time = stage1_time + stage2_time
        print(f"\nTotal optimization time: {total_time/60:.1f} minutes")
        print(f"  Stage 1 (CV): {stage1_time/60:.1f} minutes")
        print(f"  Stage 2 (Mesh): {stage2_time/60:.1f} minutes")

        # Prepare hyperparameters dict (excluding non-hyperparameter keys)
        hyperparams_to_save = {k: v for k, v in best_result.items()
                              if k not in ['cv_mae', 'mesh_mae', 'models']}

        # Save best model
        save_models(
            best_result['models'],
            macro_bounds,
            args.output,
            performance={
                'cv_mae': best_result['cv_mae'],
                'mesh_mae': best_result['mesh_mae']
            },
            hyperparameters=hyperparams_to_save
        )

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\nBest model saved to: {args.output}")
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
