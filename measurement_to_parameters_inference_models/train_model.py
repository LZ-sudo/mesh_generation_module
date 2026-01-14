"""
Train Inverse Mapping Model with TabM

TabM (Tabular Multiple predictions) is a state-of-the-art deep learning architecture
for tabular data that efficiently imitates an ensemble of MLPs through parallel training
and weight sharing. Published at ICLR 2025.

This script trains a SINGLE TabM model to perform multi-output regression, predicting
3 skeletal macroparameters (age, height, proportions) from 10 measurements. Unlike previous
approaches (5 independent models), this focuses only on skeletal structure parameters,
excluding muscle and weight which are conflated in length-based measurements.

Usage:
    # Train model on your data (with GPU)
    python train_model.py --input lookup_tables/lookup_table_female_asian_lhs.csv

    # Train with measurement noise for robustness (recommended)
    python train_model.py --input lookup_tables/lookup_table_female_asian_lhs.csv --measurement-noise 0.5

    # Train with custom hyperparameters
    python train_model.py --input lookup_tables/lookup_table_female_asian_lhs.csv --learning-rate 5e-3 --ensemble-size 64 --measurement-noise 0.5

    # Train with more epochs
    python train_model.py --input lookup_tables/lookup_table_female_asian_lhs.csv --epochs 200 --measurement-noise 0.5

Key advantages of TabM over previous approaches:
- Single model learns correlations between skeletal macroparameters
- Built-in ensemble regularization prevents overfitting on synthetic data
- Efficient training on large datasets (100K+ samples)
- GPU acceleration for fast training
- No catastrophic forgetting issues (trains from scratch)
- Focuses on skeletal structure (age, height, proportions) for better accuracy
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
import sys
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split as split_data
from sklearn.metrics import mean_absolute_error, r2_score
import traceback

# Try to import TabM and PyTorch utilities
try:
    from tabm import TabM
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import AdamW
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm

    # feature embedding imports
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
    from rtdl_num_embeddings import LinearReLUEmbeddings
    from rtdl_num_embeddings import PeriodicEmbeddings
    

    TABM_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError as e:
    TABM_AVAILABLE = False
    CUDA_AVAILABLE = False
    print(f"ERROR: tabm or dependencies not installed: {e}")
    print("Install with: pip install tabm torch scikit-learn tqdm")


def load_config(config_path):
    """
    Load training configuration from JSON file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Dictionary with configuration settings
    """
    print(f"Loading configuration from: {config_path}")

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"  Config version: {config.get('version', 'unknown')}")
    print(f"  Description: {config.get('description', 'N/A')}")

    return config


def create_embeddings(config, X_train, y_train):
    """
    Create feature embeddings based on configuration.

    Args:
        config: Configuration dictionary with 'embeddings' section
        X_train: Training features (DataFrame)
        y_train: Training targets (DataFrame)

    Returns:
        Embedding module or None
    """
    embedding_config = config.get('embeddings', {})
    embedding_type = embedding_config.get('type', 'piecewise_linear')

    print(f"\nCreating {embedding_type} embeddings...")

    if embedding_type == 'piecewise_linear':
        try:
            
            pl_config = embedding_config['piecewise_linear']
            bin_method = pl_config.get('bin_method', 'tree')

            # Compute bins
            print(f"  Computing bins using method: {bin_method}")
            if bin_method == 'tree':
                # Target-aware tree-based bins
                tree_kwargs = pl_config.get('tree_kwargs', {
                    'min_samples_leaf': 64,
                    'min_impurity_decrease': 1e-4
                })
                bin_target = pl_config.get('bin_target', 'height')

                if bin_target not in y_train.columns:
                    print(f"  WARNING: Target '{bin_target}' not found, using first target")
                    bin_target = y_train.columns[0]

                print(f"  Using target: {bin_target}")
                # Convert to PyTorch tensors (compute_bins requires tensors)
                X_tensor = torch.FloatTensor(X_train.values)
                y_tensor = torch.FloatTensor(y_train[bin_target].values)

                bins = compute_bins(
                    X_tensor,
                    y=y_tensor,
                    regression=True,
                    tree_kwargs=tree_kwargs
                )
            else:
                # Quantile-based bins
                X_tensor = torch.FloatTensor(X_train.values)
                bins = compute_bins(X_tensor)

            # Create embeddings
            d_embedding = pl_config.get('d_embedding', 12)
            activation = pl_config.get('activation', False)
            version = pl_config.get('version', 'B')

            print(f"  d_embedding: {d_embedding}, activation: {activation}, version: {version}")

            embeddings = PiecewiseLinearEmbeddings(
                bins,
                d_embedding=d_embedding,
                activation=activation,
                version=version
            )

            total_bins = sum(len(b) - 1 for b in bins)
            print(f"  Created embeddings with {total_bins} total bins across {len(bins)} features")

            return embeddings

        except ImportError as e:
            print(f"  ERROR: rtdl_num_embeddings not installed: {e}")
            print(f"  Install with: pip install rtdl_num_embeddings scikit-learn")
            return None

    elif embedding_type == 'periodic':
        try:
            per_config = embedding_config['periodic']
            d_embedding = per_config.get('d_embedding', 24)
            lite = per_config.get('lite', True)
            freq_scale = per_config.get('frequency_init_scale', None)

            kwargs = {
                'n_features': X_train.shape[1],
                'd_embedding': d_embedding,
                'lite': lite
            }
            if freq_scale is not None:
                kwargs['frequency_init_scale'] = freq_scale

            print(f"  d_embedding: {d_embedding}, lite: {lite}")

            embeddings = PeriodicEmbeddings(**kwargs)
            print(f"  Created periodic embeddings")

            return embeddings

        except ImportError as e:
            print(f"  ERROR: rtdl_num_embeddings not installed: {e}")
            print(f"  Install with: pip install rtdl_num_embeddings")
            return None

    elif embedding_type == 'linear_relu':
        try:
            lr_config = embedding_config['linear_relu']
            d_embedding = lr_config.get('d_embedding', 32)

            print(f"  d_embedding: {d_embedding}")

            embeddings = LinearReLUEmbeddings(
                n_features=X_train.shape[1],
                d_embedding=d_embedding
            )
            print(f"  Created linear+ReLU embeddings")

            return embeddings

        except ImportError as e:
            print(f"  ERROR: rtdl_num_embeddings not installed: {e}")
            print(f"  Install with: pip install rtdl_num_embeddings")
            return None

    elif embedding_type == 'none' or embedding_type is None:
        print(f"  No embeddings will be used (not recommended)")
        return None

    else:
        print(f"  WARNING: Unknown embedding type '{embedding_type}', using no embeddings")
        return None


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


def train_tabm_model(X_train, y_train, X_test, y_test, config, use_cuda=True):
    """
    Train TabM model for multi-output regression using configuration.

    Trains a SINGLE TabM model that predicts skeletal macroparameters simultaneously.
    Uses ensemble of MLPs with weight sharing for efficient, regularized training.

    Args:
        X_train: Training measurements (n_samples, n_features) - DataFrame
        y_train: Training macroparameters (n_samples, n_targets) - DataFrame
        X_test: Test measurements - DataFrame
        y_test: Test macroparameters - DataFrame
        config: Configuration dictionary with 'training', 'model', 'embeddings' sections
        use_cuda: Whether to use CUDA acceleration (default: True)

    Returns:
        Tuple of (model, scalers, performance metrics, config_used)
    """
    # Extract configuration
    train_cfg = config.get('training', {})
    model_cfg = config.get('model', {})

    learning_rate = train_cfg.get('learning_rate', 2e-3)
    weight_decay = train_cfg.get('weight_decay', 3e-4)
    n_epochs = train_cfg.get('n_epochs', 150)
    batch_size = train_cfg.get('batch_size', 128)
    patience = train_cfg.get('early_stopping_patience', 15)
    gradient_clip_norm = train_cfg.get('gradient_clip_norm', 1.0)
    measurement_noise_std = train_cfg.get('measurement_noise_std', 0.0)
    val_size = config.get('data', {}).get('val_size', 0.2)

    ensemble_size = model_cfg.get('ensemble_size', 64)
    n_blocks = model_cfg.get('n_blocks', None)
    d_block = model_cfg.get('d_block', None)
    dropout = model_cfg.get('dropout', None)
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
    print(f"  Measurement noise: ±{measurement_noise_std:.2f} cm (training only)")
    print(f"  Architecture: Single model predicting all {y_train.shape[1]} macroparameters jointly")

    print(f"\nDataset:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Input features: {X_train.shape[1]} measurements")
    print(f"  Output targets: {y_train.shape[1]} macroparameters")

    start_time = time.time()

    # Create feature embeddings BEFORE preprocessing
    # (embeddings need raw data for bin computation)
    num_embeddings = create_embeddings(config, X_train, y_train)

    # Standardize features and targets
    print("\nPreprocessing data...")
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train.values)
    y_train_scaled = y_scaler.fit_transform(y_train.values)

    # Split training into train/validation
    X_train_fit, X_val_fit, y_train_fit, y_val_fit = split_data(
        X_train_scaled, y_train_scaled, test_size=val_size, random_state=42
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

    # Create TabM model with embeddings and optional architecture params
    print("\nInitializing TabM model...")

    # Build TabM.make() arguments
    tabm_kwargs = {
        'n_num_features': X_train.shape[1],
        'cat_cardinalities': [],
        'd_out': y_train.shape[1],
        'k': ensemble_size
    }

    # Add embeddings if created
    if num_embeddings is not None:
        tabm_kwargs['num_embeddings'] = num_embeddings

    # Add optional architecture params if specified
    if n_blocks is not None:
        tabm_kwargs['n_blocks'] = n_blocks
    if d_block is not None:
        tabm_kwargs['d_block'] = d_block
    if dropout is not None:
        tabm_kwargs['dropout'] = dropout

    model = TabM.make(**tabm_kwargs)
    model = model.to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nStarting training for up to {n_epochs} epochs...")
    print(f"  Using early stopping based on validation loss (patience={patience})")

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        # Training phase
        model.train()
        train_losses = []

        for X_batch, y_batch in tqdm(train_loader, desc=f"  Epoch {epoch}/{n_epochs}", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Add Gaussian noise to measurements for robustness (data augmentation)
            if measurement_noise_std > 0:
                noise = torch.randn_like(X_batch) * measurement_noise_std
                X_batch = X_batch + noise

            optimizer.zero_grad()

            # Forward pass: output shape (batch, k, d_out)
            predictions = model(X_batch, None)  # No categorical features

            # CRITICAL: Train k predictions independently (not loss of mean)
            # Expand y_batch to match ensemble dimension
            y_expanded = y_batch.unsqueeze(1).expand(-1, predictions.shape[1], -1)

            # Compute loss per ensemble member, then mean
            loss = criterion(predictions, y_expanded)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
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

    # Process test set in batches to avoid OOM errors
    with torch.no_grad():
        test_batch_size = min(batch_size, len(X_test_scaled))
        y_pred_scaled_list = []

        for i in range(0, len(X_test_scaled), test_batch_size):
            batch_end = min(i + test_batch_size, len(X_test_scaled))
            X_batch = X_test_scaled[i:batch_end]

            X_test_tensor = torch.FloatTensor(X_batch).to(device)
            test_preds = model(X_test_tensor, None)  # (batch, k, 5)
            test_preds_mean = test_preds.mean(dim=1)  # Average ensemble: (batch, 5)
            y_pred_scaled_list.append(test_preds_mean.cpu().numpy())

            # Free GPU memory immediately
            del X_test_tensor, test_preds, test_preds_mean
            if device == 'cuda':
                torch.cuda.empty_cache()

        y_pred_scaled = np.vstack(y_pred_scaled_list)

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

    return model, scalers, performance, config


def save_model(model, scalers, macro_bounds, performance, config, output_path):
    """
    Save trained TabM model to pickle file.

    Args:
        model: Trained TabM model
        scalers: Dictionary with 'X_scaler' and 'y_scaler'
        macro_bounds: Dictionary of macroparameter bounds
        performance: Performance metrics
        config: Configuration dictionary used for training
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
        'ensemble_size': performance.get('ensemble_size', 128),
        'config': config  # Save configuration for reproducibility
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
  # Train with default config (uses tabm_config.json)
  python measurement_to_parameters_inference_models/train_model.py --input lookup_tables/lookup_table_female_asian_lhs.csv

  # Train with custom config file
  python measurement_to_parameters_inference_models/train_model.py --input data.csv --config my_config.json

  # Train with custom output path
  python measurement_to_parameters_inference_models/train_model.py --input data.csv --output my_model.pkl

  # Force CPU-only (not recommended for large datasets)
  python measurement_to_parameters_inference_models/train_model.py --input data.csv --no-cuda

Configuration:
  All training hyperparameters are specified in the JSON config file.
  Edit tabm_config.json to change learning rate, ensemble size, embeddings, etc.
  See tabm_config.json for detailed parameter descriptions and tuning ranges.
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
        '--config',
        type=str,
        default='_tabm_config.json',
        help='Path to JSON configuration file (default: _tabm_config.json in same directory)'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA acceleration and use CPU only'
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

    try:
        # Load configuration
        print("\n" + "=" * 80)
        config_path = Path(args.config)
        if not config_path.is_absolute():
            # Look for config in same directory as script
            script_dir = Path(__file__).parent
            config_path = script_dir / args.config

        config = load_config(config_path)

        # Extract key settings for display
        data_cfg = config.get('data', {})
        train_cfg = config.get('training', {})
        model_cfg = config.get('model', {})
        embed_cfg = config.get('embeddings', {})

        print(f"\nConfiguration Summary:")
        print(f"  Input data: {args.input}")
        print(f"  Output model: {args.output}")
        print(f"  Config file: {config_path}")
        print(f"  Test split: {data_cfg.get('test_size', 0.2) * 100:.0f}%")
        print(f"  Random seed: {data_cfg.get('random_seed', 42)}")
        print(f"  Embeddings: {embed_cfg.get('type', 'none')}")
        print(f"  Learning rate: {train_cfg.get('learning_rate', 0.002)}")
        print(f"  Weight decay: {train_cfg.get('weight_decay', 0.0003)}")
        print(f"  Epochs (max): {train_cfg.get('n_epochs', 150)}")
        print(f"  Batch size: {train_cfg.get('batch_size', 128)}")
        print(f"  Ensemble size (k): {model_cfg.get('ensemble_size', 64)}")
        print(f"  Measurement noise: ±{train_cfg.get('measurement_noise_std', 0.0):.2f} cm")
        print(f"  CUDA: {'Disabled (CPU only)' if args.no_cuda else 'Enabled (if available)'}")

        # Load data
        print("\n" + "=" * 80)
        # Update MACROPARAMETERS and MEASUREMENTS from config if specified
        global MACROPARAMETERS, MEASUREMENTS
        MACROPARAMETERS = data_cfg.get('macroparameters')
        MEASUREMENTS = data_cfg.get('measurements')

        X, y, macro_bounds = load_data(args.input)

        print(f"\n[OK] Loaded {len(X)} samples - TabM can handle large datasets efficiently!")

        # Split data
        test_size = data_cfg.get('test_size', 0.2)
        random_seed = data_cfg.get('random_seed', 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_seed
        )

        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing:  {len(X_test)} samples")

        # Train model
        use_cuda = not args.no_cuda

        model, scalers, performance, config = train_tabm_model(
            X_train, y_train, X_test, y_test,
            config=config,
            use_cuda=use_cuda
        )

        # Save model
        save_model(model, scalers, macro_bounds, performance, config, args.output)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\nModel: {args.output}")
        print(f"Use 'infer_macroparameters.py' to predict macroparameters from measurements")
        print(f"Use 'test_model_accuracy.py' to validate model performance with realistic measurements")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
