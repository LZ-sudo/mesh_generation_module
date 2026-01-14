# TabM Training Configuration Guide

## Overview

The TabM training script now uses a JSON configuration file for all hyperparameters and settings. This makes it easy to experiment with different configurations, track what was used for each model, and reproduce results.

## Configuration File Structure

The configuration is stored in `tabm_config.json` with the following sections:

### 1. Data Configuration
```json
"data": {
  "macroparameters": ["age", "height", "proportions"],
  "measurements": [...],
  "test_size": 0.2,
  "val_size": 0.2,
  "random_seed": 42
}
```

### 2. Feature Embeddings (CRITICAL!)
```json
"embeddings": {
  "type": "piecewise_linear",  // Options: "piecewise_linear", "periodic", "linear_relu", "none"
  "piecewise_linear": {
    "d_embedding": 12,           // Embedding dimension (tune: 8-32)
    "activation": false,         // Use ReLU activation
    "version": "B",              // TabM paper version
    "bin_method": "tree",        // "tree" (target-aware) or "quantile"
    "tree_kwargs": {
      "min_samples_leaf": 64,
      "min_impurity_decrease": 0.0001
    },
    "bin_target": "height"       // Which target to use for tree-based bins
  }
}
```

**Important**: Feature embeddings are essential for good performance. Without them, TabM will have significantly degraded accuracy.

### 3. Model Configuration
```json
"model": {
  "ensemble_size": 64,   // k parameter: number of ensemble members
  "n_blocks": null,      // null = use TabM default
  "d_block": null,       // null = use TabM default
  "dropout": null        // null = use TabM default
}
```

### 4. Training Configuration
```json
"training": {
  "learning_rate": 0.002,          // TabM paper default
  "weight_decay": 0.0003,          // TabM paper default (was 5e-6, now corrected!)
  "n_epochs": 150,
  "batch_size": 64,
  "early_stopping_patience": 15,
  "gradient_clip_norm": 1.0,
  "measurement_noise_std": 0.5     // Gaussian noise for robustness (cm)
}
```
## Embedding Types

### PiecewiseLinearEmbeddings (RECOMMENDED)
- **Best average performance**
- Target-aware bins optimize for important measurements like height
- Less sensitive to data preprocessing
- Requires sklearn for tree-based bins

### PeriodicEmbeddings
- Good alternative, no preprocessing needed
- Fully end-to-end trainable
- May require tuning `frequency_init_scale`

### LinearReLUEmbeddings
- Simple baseline
- Lower performance than advanced embeddings
- Use for quick testing only

## Hyperparameter Tuning

For optimal results, tune hyperparameters using Optuna TPE sampler (50-100 trials):

| Parameter | Range |
|-----------|-------|
| `k` | Usually fixed at 32 or 64 |
| `n_blocks` | UniformInt[1, 4] |
| `d_block` | UniformInt[64, 1024, step=16] |
| `lr` | LogUniform[1e-4, 5e-3] |
| `weight_decay` | {0, LogUniform[1e-4, 1e-1]} |
| `d_embedding` | UniformInt[8, 32, step=4] |
| `frequency_init_scale` | LogUniform[0.01, 1.0] (periodic only) |

## Saved Model Contents

The trained model pickle file now includes:
- Trained TabM model
- Input/output scalers
- Macroparameter bounds
- Performance metrics
- **Configuration used** (for reproducibility!)

## Troubleshooting

### Missing rtdl_num_embeddings
```bash
pip install rtdl_num_embeddings
pip install "scikit-learn>=1.0,<2"
```

### Out of Memory
- Reduce `batch_size` in config
- Reduce `ensemble_size` (k parameter)
- Use smaller `d_block` if specified

## References

- TabM Paper: https://arxiv.org/abs/2410.24210
- rtdl-num-embeddings: https://github.com/yandex-research/rtdl-num-embeddings
