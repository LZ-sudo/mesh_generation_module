"""
Hyperparameter Optimization for TabM Model using Optuna TPE Sampler

This script uses Optuna's Tree-structured Parzen Estimator (TPE) sampler to find
optimal hyperparameters for the TabM model. It searches over the 6 core hyperparameters
recommended by the TabM paper:
- Model architecture: n_blocks, d_block
- Training: learning_rate, weight_decay
- Feature embeddings (PiecewiseLinear): d_embedding, min_samples_leaf (n_bins)

Other parameters (batch_size, dropout, measurement_noise_std) are fixed from base config.
Embedding type is fixed to piecewise_linear for height-prioritized inference.

Usage:
    # Run optimization with default settings (50 trials)
    python optimize_hyperparameters.py --input lookup_table.csv --config base_config.json

    # Run with custom number of trials (30-50 recommended for large datasets)
    python optimize_hyperparameters.py --input data.csv --config base_config.json --n-trials 30

    # Resume previous study
    python optimize_hyperparameters.py --input data.csv --config base_config.json --study-name my_study --resume

Output:
    - best_config.json: Best hyperparameter configuration found
    - optuna_study.db: SQLite database with all trial results
    - optimization_history.png: Visualization of optimization progress
    - param_importances.png: Parameter importance analysis
"""

import pandas as pd
import numpy as np
import json
import argparse
import sys
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import traceback

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("ERROR: optuna not installed")
    print("Install with: pip install optuna")

try:
    from tabm import TabM
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import AdamW
    from tqdm import tqdm
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings, PeriodicEmbeddings, compute_bins
    TABM_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError as e:
    TABM_AVAILABLE = False
    CUDA_AVAILABLE = False
    print(f"ERROR: Required dependencies not installed: {e}")
    print("Install with: pip install tabm torch rtdl_num_embeddings scikit-learn")


def load_base_config(config_path):
    """Load base configuration to use as starting point."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_data(csv_path, config):
    """Load and prepare dataset."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    macroparams = config['data']['macroparameters']
    measurements = config['data']['measurements']

    # Extract features and targets
    X = df[measurements].copy()
    y = df[macroparams].copy()

    # Compute bounds for macroparameters
    macro_bounds = {param: (y[param].min(), y[param].max()) for param in macroparams}

    print(f"  Loaded {len(X)} samples")
    print(f"  Features: {X.shape[1]} measurements")
    print(f"  Targets: {y.shape[1]} macroparameters")

    return X, y, macro_bounds


def create_embeddings(embedding_config, X_train, y_train):
    """Create feature embeddings based on configuration."""
    embedding_type = embedding_config.get('type', 'piecewise_linear')

    if embedding_type == 'piecewise_linear':
        pl_config = embedding_config['piecewise_linear']

        # Compute tree-based bins
        tree_kwargs = pl_config.get('tree_kwargs', {
            'min_samples_leaf': 64,
            'min_impurity_decrease': 1e-4
        })
        bin_target = pl_config.get('bin_target', 'height')

        if bin_target not in y_train.columns:
            bin_target = y_train.columns[0]

        X_tensor = torch.FloatTensor(X_train.values)
        y_tensor = torch.FloatTensor(y_train[bin_target].values)

        bins = compute_bins(
            X_tensor,
            y=y_tensor,
            regression=True,
            tree_kwargs=tree_kwargs
        )

        embeddings = PiecewiseLinearEmbeddings(
            bins,
            d_embedding=pl_config['d_embedding'],
            activation=pl_config.get('activation', False),
            version=pl_config.get('version', 'B')
        )

        return embeddings

    elif embedding_type == 'periodic':
        per_config = embedding_config['periodic']

        kwargs = {
            'n_features': X_train.shape[1],
            'd_embedding': per_config['d_embedding'],
            'lite': per_config.get('lite', True)
        }

        freq_scale = per_config.get('frequency_init_scale')
        if freq_scale is not None:
            kwargs['frequency_init_scale'] = freq_scale

        embeddings = PeriodicEmbeddings(**kwargs)
        return embeddings

    else:
        return None


def train_trial_model(X_train, y_train, X_val, y_val, trial_config, device='cuda'):
    """
    Train a single model with given hyperparameters and return validation loss.

    This is a simplified training loop optimized for fast hyperparameter search.
    """
    # Extract configuration
    train_cfg = trial_config['training']
    model_cfg = trial_config['model']
    embed_cfg = trial_config['embeddings']

    learning_rate = train_cfg['learning_rate']
    weight_decay = train_cfg['weight_decay']
    n_epochs = train_cfg.get('n_epochs', 50)  # Reduced for tuning
    batch_size = train_cfg['batch_size']
    patience = train_cfg.get('early_stopping_patience', 10)
    gradient_clip_norm = train_cfg.get('gradient_clip_norm', 1.0)
    measurement_noise_std = train_cfg.get('measurement_noise_std', 0.0)

    ensemble_size = model_cfg['ensemble_size']
    n_blocks = model_cfg.get('n_blocks')
    d_block = model_cfg.get('d_block')
    dropout = model_cfg.get('dropout')

    # Create embeddings BEFORE preprocessing
    num_embeddings = create_embeddings(embed_cfg, X_train, y_train)

    # Standardize data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train.values)
    y_train_scaled = y_scaler.fit_transform(y_train.values)
    X_val_scaled = X_scaler.transform(X_val.values)
    y_val_scaled = y_scaler.transform(y_val.values)

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_scaled)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_scaled)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    tabm_kwargs = {
        'n_num_features': X_train.shape[1],
        'cat_cardinalities': [],
        'd_out': y_train.shape[1],
        'k': ensemble_size
    }

    if num_embeddings is not None:
        tabm_kwargs['num_embeddings'] = num_embeddings
    if n_blocks is not None:
        tabm_kwargs['n_blocks'] = n_blocks
    if d_block is not None:
        tabm_kwargs['d_block'] = d_block
    if dropout is not None:
        tabm_kwargs['dropout'] = dropout

    model = TabM.make(**tabm_kwargs).to(device)

    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Add measurement noise for robustness
            if measurement_noise_std > 0:
                noise = torch.randn_like(X_batch) * measurement_noise_std
                X_batch = X_batch + noise

            optimizer.zero_grad()
            predictions = model(X_batch, None)

            # Expand targets for ensemble
            y_expanded = y_batch.unsqueeze(1).expand(-1, predictions.shape[1], -1)
            loss = criterion(predictions, y_expanded)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch, None)
                y_expanded = y_batch.unsqueeze(1).expand(-1, predictions.shape[1], -1)
                loss = criterion(predictions, y_expanded)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Compute per-target validation metrics
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch, None).mean(dim=1)  # Average ensemble
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Inverse transform to original scale
    all_preds_orig = y_scaler.inverse_transform(all_preds)
    all_targets_orig = y_scaler.inverse_transform(all_targets)

    # Compute MAE per target
    per_target_mae = {}
    for i, target_name in enumerate(y_train.columns):
        mae = mean_absolute_error(all_targets_orig[:, i], all_preds_orig[:, i])
        per_target_mae[target_name] = mae

    return best_val_loss, per_target_mae


def objective(trial, base_config, X_train, y_train, X_val, y_val, device):
    """
    Optuna objective function: suggest hyperparameters and return validation loss.

    Optimizes the 6 core hyperparameters recommended by TabM paper:
    - n_blocks, d_block (model architecture)
    - learning_rate, weight_decay (optimization)
    - d_embedding, min_samples_leaf (n_bins for embeddings)
    """
    # Create trial configuration by modifying base config
    trial_config = json.loads(json.dumps(base_config))  # Deep copy

    # === Model Architecture (TabM paper recommendations) ===
    trial_config['model']['n_blocks'] = trial.suggest_int('n_blocks', 1, 4)
    trial_config['model']['d_block'] = trial.suggest_int('d_block', 64, 1024, step=16)

    # === Training Hyperparameters (TabM paper recommendations) ===
    trial_config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)

    # Weight decay: either 0 or log-uniform (TabM paper)
    use_weight_decay = trial.suggest_categorical('use_weight_decay', [True, False])
    if use_weight_decay:
        trial_config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    else:
        trial_config['training']['weight_decay'] = 0.0

    # === Feature Embeddings (PiecewiseLinear only) ===
    # Fix embedding type to piecewise_linear as recommended
    trial_config['embeddings']['type'] = 'piecewise_linear'

    # Tune d_embedding (TabM paper: [8, 32, step=4])
    d_embedding = trial.suggest_int('d_embedding', 8, 32, step=4)
    trial_config['embeddings']['piecewise_linear']['d_embedding'] = d_embedding

    # Tune n_bins via min_samples_leaf (controls tree granularity)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 32, 128, step=16)
    trial_config['embeddings']['piecewise_linear']['tree_kwargs']['min_samples_leaf'] = min_samples_leaf

    # Train model and get validation loss
    try:
        val_loss, per_target_mae = train_trial_model(
            X_train, y_train, X_val, y_val,
            trial_config,
            device=device
        )

        # Log per-target metrics as user attributes
        for target_name, mae in per_target_mae.items():
            trial.set_user_attr(f'mae_{target_name}', mae)

        return val_loss

    except Exception as e:
        print(f"\nTrial {trial.number} failed: {e}")
        traceback.print_exc()
        raise optuna.TrialPruned()


def run_optimization(
    input_csv,
    base_config_path,
    n_trials=50,
    study_name='tabm_optimization',
    storage=None,
    resume=False,
    device='cuda'
):
    """
    Run Optuna hyperparameter optimization.

    Args:
        input_csv: Path to training data CSV
        base_config_path: Path to base configuration JSON
        n_trials: Number of optimization trials
        study_name: Name for the Optuna study
        storage: Database URL for study persistence (default: SQLite)
        resume: Whether to resume existing study
        device: Device to use for training ('cuda' or 'cpu')

    Returns:
        Best configuration dictionary
    """
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA TPE SAMPLER")
    print("=" * 80)

    # Load base configuration and data
    base_config = load_base_config(base_config_path)
    X, y, macro_bounds = load_data(input_csv, base_config)

    # Split data: train/val/test
    test_size = base_config['data'].get('test_size', 0.2)
    val_size = base_config['data'].get('val_size', 0.2)
    random_seed = base_config['data'].get('random_seed', 42)

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=random_seed
    )

    print(f"\nData split:")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:       {len(X_test)} samples (held out)")

    print(f"\nOptimization settings:")
    print(f"  Number of trials: {n_trials}")
    print(f"  Study name: {study_name}")
    print(f"  Device: {device}")
    print(f"  Sampler: TPE (Tree-structured Parzen Estimator)")

    # Create or load study
    if storage is None:
        storage = f"sqlite:///{study_name}.db"

    load_if_exists = resume

    sampler = TPESampler(seed=random_seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction='minimize',
        load_if_exists=load_if_exists
    )

    print(f"\nStarting optimization...")
    start_time = time.time()

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config, X_train, y_train, X_val, y_val, device),
        n_trials=n_trials,
        show_progress_bar=True
    )

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    print(f"\nOptimization time: {elapsed_time / 60:.1f} minutes")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")

    # Display best hyperparameters
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Display per-target MAE from best trial
    print("\nBest trial per-target MAE:")
    for key, value in study.best_trial.user_attrs.items():
        if key.startswith('mae_'):
            target_name = key.replace('mae_', '')
            print(f"  {target_name}: {value:.4f}")

    # Create best configuration
    best_config = json.loads(json.dumps(base_config))  # Deep copy
    best_params = study.best_params

    # Update with best parameters (6 core hyperparameters)
    best_config['model']['n_blocks'] = best_params['n_blocks']
    best_config['model']['d_block'] = best_params['d_block']

    best_config['training']['learning_rate'] = best_params['learning_rate']

    if best_params['use_weight_decay']:
        best_config['training']['weight_decay'] = best_params['weight_decay']
    else:
        best_config['training']['weight_decay'] = 0.0

    # Always use piecewise_linear embeddings
    best_config['embeddings']['type'] = 'piecewise_linear'
    best_config['embeddings']['piecewise_linear']['d_embedding'] = best_params['d_embedding']
    best_config['embeddings']['piecewise_linear']['tree_kwargs']['min_samples_leaf'] = best_params['min_samples_leaf']

    # Add optimization metadata
    best_config['optimization'] = {
        'study_name': study_name,
        'n_trials': len(study.trials),
        'best_trial': study.best_trial.number,
        'best_validation_loss': study.best_value,
        'optimization_time_minutes': elapsed_time / 60,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    return best_config, study


def save_best_config(config, output_path):
    """Save best configuration to JSON file."""
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n[OK] Best configuration saved to: {output_path}")


def visualize_optimization(study, output_dir='.'):
    """Create visualization plots for the optimization process."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

        # Optimization history
        ax1 = plot_optimization_history(study)
        fig1 = ax1.figure
        fig1.savefig(output_dir / 'optimization_history.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Saved optimization history plot")

        # Parameter importances
        ax2 = plot_param_importances(study)
        fig2 = ax2.figure
        fig2.savefig(output_dir / 'param_importances.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Saved parameter importances plot")

        plt.close('all')

    except ImportError:
        print("\n[INFO] matplotlib not installed, skipping visualization")
        print("      Install with: pip install matplotlib")


def main():
    parser = argparse.ArgumentParser(
        description='Optimize TabM hyperparameters using Optuna TPE sampler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run optimization with default settings (50 trials)
  python optimize_hyperparameters.py --input lookup_table.csv --config _tabm_config.json

  # Run with 30 trials (recommended for large datasets)
  python optimize_hyperparameters.py --input data.csv --config _tabm_config.json --n-trials 30

  # Resume previous optimization study
  python optimize_hyperparameters.py --input data.csv --config _tabm_config.json --study-name my_study --resume

  # Force CPU-only
  python optimize_hyperparameters.py --input data.csv --config _tabm_config.json --no-cuda

Output:
  - best_config.json: Best hyperparameter configuration
  - <study_name>.db: SQLite database with all trial results
  - optimization_history.png: Optimization progress visualization
  - param_importances.png: Parameter importance analysis
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to training data CSV file'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to base configuration JSON file'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50, recommend 30-50 for large datasets)'
    )

    parser.add_argument(
        '--study-name',
        type=str,
        default='tabm_optimization',
        help='Name for the Optuna study (default: tabm_optimization)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='best_config.json',
        help='Path to save best configuration (default: best_config.json)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume existing study with same name'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA and use CPU only'
    )

    args = parser.parse_args()

    # Check dependencies
    if not OPTUNA_AVAILABLE:
        print("\nERROR: optuna is required")
        print("Install with: pip install optuna")
        return 1

    if not TABM_AVAILABLE:
        print("\nERROR: tabm and dependencies are required")
        print("Install with: pip install tabm torch rtdl_num_embeddings scikit-learn")
        return 1

    try:
        # Determine device
        device = 'cpu' if args.no_cuda else ('cuda' if CUDA_AVAILABLE else 'cpu')

        if device == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU (optimization will be slower)")

        # Run optimization
        best_config, study = run_optimization(
            input_csv=args.input,
            base_config_path=args.config,
            n_trials=args.n_trials,
            study_name=args.study_name,
            resume=args.resume,
            device=device
        )

        # Save best configuration
        save_best_config(best_config, args.output)

        # Create visualizations
        visualize_optimization(study, output_dir='.')

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print(f"\n1. Review best configuration in: {args.output}")
        print(f"2. Train full model with best config:")
        print(f"   python train_model.py --input {args.input} --config {args.output}")
        print(f"\n3. View optimization details:")
        print(f"   - Study database: {args.study_name}.db")
        print(f"   - History plot: optimization_history.png")
        print(f"   - Importances: param_importances.png")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
