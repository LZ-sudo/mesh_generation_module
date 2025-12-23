"""
Train Inverse Mapping Model with TabM

TabM (Tabular Multiple predictions) is a state-of-the-art deep learning architecture
for tabular data that efficiently imitates an ensemble of MLPs through parallel training
and weight sharing. Published at ICLR 2025.

This script trains a SINGLE TabM model to perform multi-output regression, predicting
all 5 macroparameters simultaneously from 8 measurements. Unlike previous approaches
(5 independent models), TabM learns joint relationships between outputs.

Usage:
    # Train model on your data (with GPU)
    python train_model.py --input lookup_tables/lookup_table_female_asian_lhs.csv

    # Train with custom learning rate and ensemble size
    python train_model.py --input lookup_tables/lookup_table_female_asian_lhs.csv --learning-rate 5e-3 --ensemble-size 64

    # Train with more epochs
    python train_model.py --input lookup_tables/lookup_table_female_asian_lhs.csv --epochs 200

Key advantages of TabM over previous approaches:
- Single model learns correlations between all 5 macroparameters
- Built-in ensemble regularization prevents overfitting on synthetic data
- Efficient training on large datasets (100K+ samples)
- GPU acceleration for fast training
- No catastrophic forgetting issues (trains from scratch)
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import sys
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Try to import TabM and PyTorch utilities
try:
    from tabm import TabM
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import AdamW
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm
    TABM_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError as e:
    TABM_AVAILABLE = False
    CUDA_AVAILABLE = False
    print(f"ERROR: tabm or dependencies not installed: {e}")
    print("Install with: pip install tabm torch scikit-learn tqdm")

# Configuration
MACROPARAMETERS = ['age', 'muscle', 'weight', 'height', 'proportions']
MEASUREMENTS = [
    'height_cm', 'shoulder_width_cm', 'hip_width_cm', 'head_width_cm',
    'neck_length_cm', 'upper_arm_length_cm', 'forearm_length_cm', 'hand_length_cm'
]


def load_data(csv_path):
    """Load training data from lookup table CSV."""
    print(f"Loading data from: {csv_path}")

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Verify required columns exist
    missing_macros = set(MACROPARAMETERS) - set(df.columns)
    if missing_macros:
        raise ValueError(f"Missing macroparameter columns: {missing_macros}")

    missing_measures = set(MEASUREMENTS) - set(df.columns)
    if missing_measures:
        raise ValueError(f"Missing measurement columns: {missing_measures}")

    # Extract features (measurements) and targets (macroparameters)
    # INVERSE MAPPING: measurements -> macroparameters
    X = df[MEASUREMENTS]
    y = df[MACROPARAMETERS]

    # Calculate bounds for macroparameters
    macro_bounds = {}
    for param in MACROPARAMETERS:
        macro_bounds[param] = (y[param].min(), y[param].max())

    print(f"Loaded {len(df)} samples")
    print(f"\nMacroparameter bounds:")
    for param, (min_val, max_val) in macro_bounds.items():
        print(f"  {param:12s}: [{min_val:.3f}, {max_val:.3f}]")

    print(f"\nMeasurement statistics:")
    print(X.describe())

    print("-" * 80)

    return X, y, macro_bounds


def train_tabm_model(X_train, y_train, X_test, y_test, use_cuda=True,
                     learning_rate=1e-3, n_epochs=150, batch_size=256,
                     ensemble_size=128, weight_decay=5e-6):
    """
    Train TabM model for multi-output regression.

    Trains a SINGLE TabM model that predicts all 5 macroparameters simultaneously.
    Uses ensemble of MLPs with weight sharing for efficient, regularized training.

    Args:
        X_train: Training measurements (n_samples, 8) - DataFrame
        y_train: Training macroparameters (n_samples, 5) - DataFrame
        X_test: Test measurements - DataFrame
        y_test: Test macroparameters - DataFrame
        use_cuda: Whether to use CUDA acceleration (default: True)
        learning_rate: Learning rate for AdamW optimizer (default: 1e-3)
        n_epochs: Maximum number of epochs (default: 150)
        batch_size: Batch size for training (default: 256)
        ensemble_size: Number of ensemble members (k parameter) (default: 128)
        weight_decay: L2 regularization weight decay (default: 5e-6)

    Returns:
        Tuple of (model, scalers, performance metrics)
    """
    print("\n" + "=" * 80)
    print("TRAINING TabM MODEL (Multi-Output Regression)")
    print("=" * 80)

    # Determine device
    if use_cuda and CUDA_AVAILABLE:
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n[OK] CUDA available - Using GPU: {gpu_name}")
    else:
        device = 'cpu'
        if use_cuda and not CUDA_AVAILABLE:
            print("\n[WARNING] CUDA requested but not available - Using CPU")
        else:
            print("\n[INFO] Using CPU")

    print("\nTraining configuration:")
    print("  Model: TabM (ICLR 2025)")
    print("  Method: Ensemble of MLPs with weight sharing")
    print(f"  Device: {device.upper()}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Ensemble size (k): {ensemble_size}")
    print(f"  Architecture: Single model predicting all {y_train.shape[1]} macroparameters jointly")

    print(f"\nDataset:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Input features: {X_train.shape[1]} measurements")
    print(f"  Output targets: {y_train.shape[1]} macroparameters")

    start_time = time.time()

    # Standardize features and targets
    print("\nPreprocessing data...")
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train.values)
    y_train_scaled = y_scaler.fit_transform(y_train.values)

    # Split training into train/validation (80/20)
    from sklearn.model_selection import train_test_split as split_data
    X_train_fit, X_val_fit, y_train_fit, y_val_fit = split_data(
        X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
    )

    print(f"  Train samples: {len(X_train_fit)}, Validation samples: {len(X_val_fit)}")

    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_fit),
        torch.FloatTensor(y_train_fit)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_fit),
        torch.FloatTensor(y_val_fit)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create TabM model
    print("\nInitializing TabM model...")
    model = TabM.make(
        n_num_features=X_train.shape[1],  # 8 measurements
        cat_cardinalities=[],              # No categorical features
        d_out=y_train.shape[1],            # 5 macroparameters
        k=ensemble_size                    # Ensemble size
    )
    model = model.to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nStarting training for up to {n_epochs} epochs...")
    print("  Using early stopping based on validation loss (patience=15)")

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    best_state = None
    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        # Training phase
        model.train()
        train_losses = []

        for X_batch, y_batch in tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Forward pass: output shape (batch, k, d_out)
            predictions = model(X_batch, None)  # No categorical features

            # CRITICAL: Train k predictions independently (not loss of mean)
            # Expand y_batch to match ensemble dimension
            y_expanded = y_batch.unsqueeze(1).expand(-1, predictions.shape[1], -1)

            # Compute loss per ensemble member, then mean
            loss = criterion(predictions, y_expanded)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass
                predictions = model(X_batch, None)

                # Average across ensemble for final prediction
                predictions_mean = predictions.mean(dim=1)  # (batch, d_out)

                # Compute loss on averaged predictions
                loss = criterion(predictions_mean, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        # Learning rate scheduling
        scheduler.step()

        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"    Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}, LR={current_lr:.2e}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    → New best validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping triggered at epoch {epoch}")
                print(f"    Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"\n  Restored best model from epoch {best_epoch}")

    training_time = time.time() - start_time

    print(f"\n[OK] Training completed in {training_time:.1f} seconds")

    # Evaluate on test set
    print("\n" + "-" * 80)
    print("EVALUATING ON TEST SET")
    print("-" * 80)

    model.eval()
    X_test_scaled = X_scaler.transform(X_test.values)

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        test_preds = model(X_test_tensor, None)  # (n_test, k, 5)
        test_preds_mean = test_preds.mean(dim=1)  # Average ensemble: (n_test, 5)
        y_pred_scaled = test_preds_mean.cpu().numpy()

    # Inverse transform predictions
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    # Calculate metrics per macroparameter
    print("\nPer-Parameter Performance:")
    print(f"  {'Parameter':<15s} {'MAE':<10s} {'R²':<10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10}")

    maes = []
    r2s = []

    for i, param in enumerate(MACROPARAMETERS):
        y_true_param = y_test[param].values
        y_pred_param = y_pred[:, i]

        mae = mean_absolute_error(y_true_param, y_pred_param)
        r2 = r2_score(y_true_param, y_pred_param)

        maes.append(mae)
        r2s.append(r2)

        print(f"  {param:<15s} {mae:<10.4f} {r2:<10.4f}")

    overall_mae = np.mean(maes)
    overall_r2 = np.mean(r2s)

    print(f"  {'-'*15} {'-'*10} {'-'*10}")
    print(f"  {'Overall':<15s} {overall_mae:<10.4f} {overall_r2:<10.4f}")

    performance = {
        'per_parameter': {
            param: {'mae': mae, 'r2': r2}
            for param, mae, r2 in zip(MACROPARAMETERS, maes, r2s)
        },
        'overall_mae': overall_mae,
        'overall_r2': overall_r2,
        'training_time': training_time,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'ensemble_size': ensemble_size
    }

    scalers = {
        'X_scaler': X_scaler,
        'y_scaler': y_scaler
    }

    return model, scalers, performance


def save_model(model, scalers, macro_bounds, performance, output_path):
    """
    Save trained TabM model to pickle file.

    Args:
        model: Trained TabM model
        scalers: Dictionary with 'X_scaler' and 'y_scaler'
        macro_bounds: Dictionary of macroparameter bounds
        performance: Performance metrics
        output_path: Path to save model file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save original device
    original_device = next(model.parameters()).device

    # Move model to CPU for saving
    model.cpu()

    data = {
        'model': model,
        'scalers': scalers,
        'macro_bounds': macro_bounds,
        'macroparameters': MACROPARAMETERS,
        'measurements': MEASUREMENTS,
        'performance': performance,
        'model_type': 'TabM_MultiOutput',
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ensemble_size': performance.get('ensemble_size', 128)
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Model saved to: {output_path}")
    print(f"     File size: {file_size_mb:.1f} MB")
    print(f"     Contains: Single TabM model + input/output scalers")

    # Move model back to original device
    model.to(original_device)


def main():
    parser = argparse.ArgumentParser(
        description='Train TabM model for inverse mapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (uses GPU if available)
  python train_model.py --input lookup_tables/lookup_table_female_asian_lhs.csv

  # Train with custom hyperparameters
  python train_model.py --input lookup_tables/lookup_table_female_asian_lhs.csv --learning-rate 5e-3 --ensemble-size 64

  # Train with more epochs
  python train_model.py --input lookup_tables/lookup_table_female_asian_lhs.csv --epochs 200

  # Force CPU-only (not recommended for large datasets)
  python train_model.py --input lookup_tables/lookup_table_female_asian_lhs.csv --no-cuda
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
        default='macroparameters_inference_models_tabm.pkl',
        help='Path to save trained model (default: macroparameters_inference_models_tabm.pkl)'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing (default: 0.2)'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA acceleration and use CPU only'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate for AdamW optimizer (default: 1e-3)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
        help='Maximum number of training epochs (default: 150). Early stopping with patience=15.'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for training (default: 256)'
    )

    parser.add_argument(
        '--ensemble-size',
        type=int,
        default=128,
        help='Ensemble size (k parameter) - number of ensemble members (default: 128)'
    )

    parser.add_argument(
        '--weight-decay',
        type=float,
        default=5e-6,
        help='L2 regularization weight decay (default: 5e-6)'
    )

    args = parser.parse_args()

    # Check TabM availability
    if not TABM_AVAILABLE:
        print("\nERROR: tabm is required")
        print("Install with: pip install tabm torch")
        return 1

    print("=" * 80)
    print("TabM TRAINING FOR INVERSE MAPPING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Test split: {args.test_size * 100:.0f}%")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Ensemble size: {args.ensemble_size}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  CUDA: {'Disabled (CPU only)' if args.no_cuda else 'Enabled (if available)'}")

    try:
        # Load data
        print("\n" + "=" * 80)
        X, y, macro_bounds = load_data(args.input)

        print(f"\n[OK] Loaded {len(X)} samples - TabM can handle large datasets efficiently!")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_seed
        )

        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing:  {len(X_test)} samples")

        # Train model
        use_cuda = not args.no_cuda
        device = 'cuda' if (use_cuda and CUDA_AVAILABLE) else 'cpu'

        model, scalers, performance = train_tabm_model(
            X_train, y_train, X_test, y_test,
            use_cuda=use_cuda,
            learning_rate=args.learning_rate,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            ensemble_size=args.ensemble_size,
            weight_decay=args.weight_decay
        )

        # Save model
        save_model(model, scalers, macro_bounds, performance, args.output)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\nModel: {args.output}")
        print(f"Use 'infer_macroparameters.py' to predict macroparameters from measurements")
        print(f"Use 'test_model_accuracy.py' to validate model performance with realistic measurements")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
