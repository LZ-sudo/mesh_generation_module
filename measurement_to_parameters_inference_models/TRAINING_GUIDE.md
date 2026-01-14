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

## Automated Hyperparameter Optimization

The `optimize_hyperparameters.py` script uses Optuna's TPE (Tree-structured Parzen Estimator) sampler to find optimal hyperparameters.

### Quick Start

```bash
# Step 1: Run hyperparameter optimization
python optimize_hyperparameters.py \
  --input lookup_table.csv \
  --config _tabm_config.json \
  --n-trials 30 \
  --output best_config.json

# Step 2: Train model with optimized config
python train_model.py \
  --input lookup_table.csv \
  --config best_config.json \
  --output model.pkl

# Resume interrupted optimization
python optimize_hyperparameters.py \
  --input lookup_table.csv \
  --config _tabm_config.json \
  --n-trials 50 \
  --study-name my_study \
  --resume

# Direct training without optimization (if you already have good hyperparameters)
python train_model.py \
  --input lookup_table.csv \
  --config _tabm_config.json \
  --output model.pkl
```

### What Gets Optimized

The optimization searches over the **6 core hyperparameters** recommended by the TabM paper:

**Model Architecture:**
- `n_blocks`: [1, 4] - Number of residual blocks
- `d_block`: [64, 1024, step=16] - Hidden dimension per block

**Training:**
- `learning_rate`: [1e-4, 5e-3] log-uniform
- `weight_decay`: {0} ∪ [1e-4, 1e-1] log-uniform

**Feature Embeddings (PiecewiseLinear only):**
- `d_embedding`: [8, 32, step=4] - Embedding dimension
- `min_samples_leaf`: [32, 128, step=16] - Tree granularity (n_bins)

**Fixed in base config (not optimized):**
- `ensemble_size` (k) - Usually 32 or 64 (TabM recommendation: don't tune)
- `embedding_type` - Fixed to piecewise_linear for height-prioritized inference
- `batch_size` - From base config
- `dropout` - From base config
- `measurement_noise_std` - From base config

### Number of Trials

Follow TabM paper recommendations:
- **Small datasets (<50k samples)**: 50-100 trials
- **Large datasets (150k+ samples)**: 30-50 trials

More trials = better optimization but longer runtime. Each trial trains a model from scratch.

### Output Files

The optimization script creates the following files in the current directory:

```
<study_name>.db              # SQLite database with all trial results (default: tabm_optimization.db)
best_config.json             # Best hyperparameter configuration (or custom --output path)
optimization_history.png     # Optimization progress visualization
param_importances.png        # Which parameters mattered most
```

### Workflow

The training workflow consists of two separate steps:

1. **Optimization phase** (1-2 hours for 30 trials):
   ```bash
   python optimize_hyperparameters.py --input data.csv --config _tabm_config.json --n-trials 30
   ```
   - Runs N trials with reduced epochs (50) for fast evaluation
   - Saves best_config.json with optimal hyperparameters
   - Creates visualization plots

2. **Training phase** (10-20 minutes):
   ```bash
   python train_model.py --input data.csv --config best_config.json --output model.pkl
   ```
   - Trains final model with best config
   - Uses full epochs (150) for production quality
   - Saves trained model

### Tips for Best Results

1. **Organize by demographic**: Use separate study names for different populations:
   ```bash
   python optimize_hyperparameters.py --input female_asian.csv --study-name female_asian_opt --n-trials 30
   python optimize_hyperparameters.py --input male_asian.csv --study-name male_asian_opt --n-trials 30
   ```

2. **Fix ensemble_size**: Keep `k` at 32 or 64 in base config (TabM recommendation: don't tune during optimization)

3. **Height prioritization**: Base config already has `bin_target: "height"` for height-aware feature embeddings

4. **Monitor per-target MAE**: The script logs MAE for each macroparameter (age, height, proportions) during optimization

5. **Resume if interrupted**: Use `--resume` flag to continue from last completed trial:
   ```bash
   python optimize_hyperparameters.py --input data.csv --study-name my_study --n-trials 50 --resume
   ```

6. **Start conservative**: Begin with 30 trials, extend to 50 if needed (each trial ~2-3 min on GPU)

### Manual Hyperparameter Tuning (Alternative)

If you prefer manual tuning, here are the recommended search ranges from the TabM paper:

| Parameter | Range |
|-----------|-------|
| `k` (ensemble_size) | Usually fixed at 32 or 64 |
| `n_blocks` | UniformInt[1, 4] |
| `d_block` | UniformInt[64, 1024, step=16] |
| `lr` | LogUniform[1e-4, 5e-3] |
| `weight_decay` | {0, LogUniform[1e-4, 1e-1]} |
| `d_embedding` | UniformInt[8, 32, step=4] |
| `frequency_init_scale` | LogUniform[0.01, 1.0] (periodic only) |
| `min_samples_leaf` | UniformInt[32, 128] (tree bins) |
| `batch_size` | Categorical[64, 128, 256] |

**Note:** The automated optimization uses these exact ranges from the TabM paper.

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
