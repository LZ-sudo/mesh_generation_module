"""
Shared utilities for TabM model training and hyperparameter optimization.

This module contains common functions used by both train_model.py and
optimize_hyperparameters.py to avoid code duplication.
"""

import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

try:
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
    from rtdl_num_embeddings import LinearReLUEmbeddings
    from rtdl_num_embeddings import PeriodicEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


def load_config(config_path, verbose=True):
    """
    Load training configuration from JSON file.

    Args:
        config_path: Path to JSON config file (str or Path)
        verbose: Whether to print config details (default: True)

    Returns:
        Dictionary with configuration settings

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)

    if verbose:
        print(f"Loading configuration from: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    if verbose:
        print(f"  Config version: {config.get('version', 'unknown')}")
        print(f"  Description: {config.get('description', 'N/A')}")

    return config


def create_embeddings(embedding_config, X_train, y_train, verbose=True):
    """
    Create feature embeddings based on configuration.

    Args:
        embedding_config: Dictionary with 'type' and embedding-specific params
        X_train: Training features (DataFrame)
        y_train: Training targets (DataFrame)
        verbose: Whether to print embedding details (default: True)

    Returns:
        Embedding module or None
    """
    if not EMBEDDINGS_AVAILABLE:
        if verbose:
            print("  WARNING: rtdl_num_embeddings not available")
        return None

    embedding_type = embedding_config.get('type', 'piecewise_linear')

    if verbose:
        print(f"\nCreating {embedding_type} embeddings...")

    if embedding_type == 'piecewise_linear':
        pl_config = embedding_config['piecewise_linear']
        bin_method = pl_config.get('bin_method', 'tree')

        if verbose:
            print(f"  Computing bins using method: {bin_method}")

        # Compute bins
        if bin_method == 'tree':
            tree_kwargs = pl_config.get('tree_kwargs', {
                'min_samples_leaf': 64,
                'min_impurity_decrease': 1e-4
            })
            bin_target = pl_config.get('bin_target', 'height')

            if bin_target not in y_train.columns:
                if verbose:
                    print(f"  WARNING: Target '{bin_target}' not found, using first target")
                bin_target = y_train.columns[0]

            if verbose:
                print(f"  Using target: {bin_target}")

            X_tensor = torch.FloatTensor(X_train.values)
            y_tensor = torch.FloatTensor(y_train[bin_target].values)

            bins = compute_bins(
                X_tensor,
                y=y_tensor,
                regression=True,
                tree_kwargs=tree_kwargs
            )
        else:
            X_tensor = torch.FloatTensor(X_train.values)
            bins = compute_bins(X_tensor)

        # Create embeddings
        d_embedding = pl_config.get('d_embedding', 12)
        activation = pl_config.get('activation', False)
        version = pl_config.get('version', 'B')

        if verbose:
            print(f"  d_embedding: {d_embedding}, activation: {activation}, version: {version}")

        embeddings = PiecewiseLinearEmbeddings(
            bins,
            d_embedding=d_embedding,
            activation=activation,
            version=version
        )

        if verbose:
            total_bins = sum(len(b) - 1 for b in bins)
            print(f"  Created embeddings with {total_bins} total bins across {len(bins)} features")

        return embeddings

    elif embedding_type == 'periodic':
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

        if verbose:
            print(f"  d_embedding: {d_embedding}, lite: {lite}")

        embeddings = PeriodicEmbeddings(**kwargs)

        if verbose:
            print(f"  Created periodic embeddings")

        return embeddings

    elif embedding_type == 'linear_relu':
        lr_config = embedding_config['linear_relu']
        d_embedding = lr_config.get('d_embedding', 32)

        if verbose:
            print(f"  d_embedding: {d_embedding}")

        embeddings = LinearReLUEmbeddings(
            n_features=X_train.shape[1],
            d_embedding=d_embedding
        )

        if verbose:
            print(f"  Created linear+ReLU embeddings")

        return embeddings

    elif embedding_type == 'none' or embedding_type is None:
        if verbose:
            print(f"  No embeddings will be used (not recommended)")
        return None

    else:
        if verbose:
            print(f"  WARNING: Unknown embedding type '{embedding_type}', using no embeddings")
        return None


def load_data(csv_path, macroparameters, measurements, verbose=True):
    """
    Load and validate training data from CSV.

    Args:
        csv_path: Path to CSV file
        macroparameters: List of macroparameter column names
        measurements: List of measurement column names
        verbose: Whether to print data details (default: True)

    Returns:
        Tuple of (X, y, macro_bounds) where:
        - X: DataFrame with measurement features
        - y: DataFrame with macroparameter targets
        - macro_bounds: Dictionary with (min, max) for each macroparameter

    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If required columns are missing
    """
    csv_path = Path(csv_path)

    if verbose:
        print(f"Loading data from: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Verify required columns exist
    missing_macros = set(macroparameters) - set(df.columns)
    if missing_macros:
        raise ValueError(f"Missing macroparameter columns: {missing_macros}")

    missing_measures = set(measurements) - set(df.columns)
    if missing_measures:
        raise ValueError(f"Missing measurement columns: {missing_measures}")

    # Extract features and targets
    X = df[measurements].copy()
    y = df[macroparameters].copy()

    # Calculate bounds for macroparameters
    macro_bounds = {}
    for param in macroparameters:
        macro_bounds[param] = (y[param].min(), y[param].max())

    if verbose:
        print(f"  Loaded {len(df)} samples")
        print(f"  Features: {X.shape[1]} measurements")
        print(f"  Targets: {y.shape[1]} macroparameters")

        print(f"\n  Macroparameter bounds:")
        for param, (min_val, max_val) in macro_bounds.items():
            print(f"    {param:12s}: [{min_val:.3f}, {max_val:.3f}]")

    return X, y, macro_bounds


def build_tabm_kwargs(n_features, n_targets, ensemble_size,
                      num_embeddings=None, n_blocks=None,
                      d_block=None, dropout=None):
    """
    Build keyword arguments for TabM.make() with optional parameters.

    Args:
        n_features: Number of input features
        n_targets: Number of output targets
        ensemble_size: Number of ensemble members (k parameter)
        num_embeddings: Optional embedding module
        n_blocks: Optional number of residual blocks
        d_block: Optional hidden dimension per block
        dropout: Optional dropout rate

    Returns:
        Dictionary of kwargs for TabM.make()
    """
    kwargs = {
        'n_num_features': n_features,
        'cat_cardinalities': [],
        'd_out': n_targets,
        'k': ensemble_size
    }

    if num_embeddings is not None:
        kwargs['num_embeddings'] = num_embeddings
    if n_blocks is not None:
        kwargs['n_blocks'] = n_blocks
    if d_block is not None:
        kwargs['d_block'] = d_block
    if dropout is not None:
        kwargs['dropout'] = dropout

    return kwargs


def standardize_data(X_train, y_train, X_val=None, y_val=None):
    """
    Standardize features and targets using StandardScaler.

    Args:
        X_train: Training features (DataFrame or array)
        y_train: Training targets (DataFrame or array)
        X_val: Optional validation features
        y_val: Optional validation targets

    Returns:
        If X_val is None:
            (X_train_scaled, y_train_scaled, X_scaler, y_scaler)
        If X_val is provided:
            (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler)
    """
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Convert to numpy if DataFrame
    X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train

    X_train_scaled = X_scaler.fit_transform(X_train_arr)
    y_train_scaled = y_scaler.fit_transform(y_train_arr)

    if X_val is not None:
        X_val_arr = X_val.values if hasattr(X_val, 'values') else X_val
        y_val_arr = y_val.values if hasattr(y_val, 'values') else y_val

        X_val_scaled = X_scaler.transform(X_val_arr)
        y_val_scaled = y_scaler.transform(y_val_arr)

        return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler

    return X_train_scaled, y_train_scaled, X_scaler, y_scaler


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size, shuffle_train=True):
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        X_train: Training features (array)
        y_train: Training targets (array)
        X_val: Validation features (array)
        y_val: Validation targets (array)
        batch_size: Batch size for DataLoader
        shuffle_train: Whether to shuffle training data (default: True)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
