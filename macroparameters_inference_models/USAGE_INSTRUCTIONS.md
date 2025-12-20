# Macroparameter Inference Models

This directory contains modularized scripts for training, using, and validating inverse mapping models that find macroparameters from body measurements.

## Overview

The inverse mapping problem: Given body measurements → Find macroparameters that produce a mesh matching those measurements.

**Solution:** Train Random Forest models on lookup table data, then use optimization to find the best macroparameters.

## Scripts

### 1. `train_model.py` - Train Models

Trains Random Forest models from a lookup table CSV file.

**Usage:**
```bash
# Basic training
python train_model.py --input ../macroparameters_generation_and_analysis/lookup_tables/lookup_table_female_asian.csv

# Custom output location
python train_model.py --input lookup_table.csv --output models/my_model.pkl

# Adjust model complexity
python train_model.py --input lookup_table.csv --n-estimators 300 --max-depth 20
```

**Parameters:**
- `--input`: Path to lookup table CSV (required)
- `--output`: Path to save trained models (default: `macroparameters_inference_models.pkl`)
- `--n-estimators`: Number of trees in Random Forest (default: 200)
- `--max-depth`: Maximum depth of trees (default: 15)
- `--test-size`: Fraction of data for testing (default: 0.2)
- `--random-seed`: Random seed for reproducibility (default: 42)

**Output:**
- `.pkl` file containing trained models and metadata
- Console output showing R² scores for each measurement

### 2. `infer_macroparameters.py` - Find Macroparameters

Uses trained models to find macroparameters for given measurements.

**Usage:**
```bash
# Single measurement from JSON
python infer_macroparameters.py --input test_measurements.json --models macroparameters_inference_models.pkl

# Batch processing from CSV
python infer_macroparameters.py --input measurements.csv --models model.pkl --output results.csv

# Use hybrid optimization for better accuracy
python infer_macroparameters.py --input measurements.json --models model.pkl --method both
```

**Input Format (JSON):**
```json
{
  "height_cm": 165.0,
  "shoulder_width_cm": 33.0,
  "hip_width_cm": 23.0,
  "head_width_cm": 12.5,
  "neck_length_cm": 7.5,
  "upper_arm_length_cm": 24.0,
  "forearm_length_cm": 22.0,
  "hand_length_cm": 16.5
}
```

**Parameters:**
- `--input`: Input measurements file (JSON or CSV, required)
- `--models`: Path to trained models (default: `macroparameters_inference_models.pkl`)
- `--output`: Output CSV file for batch mode (default: `inference_results.csv`)
- `--method`: Optimization method
  - `differential_evolution` (default): Global search, more thorough
  - `both`: Global + local refinement, potentially more accurate

**Output:**
- For JSON: `*_result.json` with macroparameters and predicted measurements
- For CSV: Results file with macroparameters for each entry

### 3. `test_model_accuracy.py` - Validate with Real Meshes

**THE CRUCIAL VALIDATION SCRIPT**

This script tests the TRUE accuracy of the model by:
1. Taking target measurements
2. Using the model to predict macroparameters
3. **Generating an actual mesh in Blender** with those macroparameters
4. **Measuring the generated mesh**
5. Comparing actual measurements to predictions

**Usage:**
```bash
# Test with a measurements file
python test_model_accuracy.py --input test_measurements.json --models macroparameters_inference_models.pkl

# Use hybrid optimization
python test_model_accuracy.py --input test_measurements.json --models model.pkl --method both

# Specify rig type
python test_model_accuracy.py --input test_measurements.json --models model.pkl --rig-type default
```

**Parameters:**
- `--input`: Input measurements JSON file (required)
- `--models`: Path to trained models (default: `macroparameters_inference_models.pkl`)
- `--method`: Optimization method (`differential_evolution` or `both`)
- `--rig-type`: Type of rig to add (`default`, `default_no_toes`, `game_engine`)

**Output:**
- Console output with comprehensive comparison
- `*_accuracy_test_result.json` with complete results
- Shows three columns of measurements:
  - **Target**: What you wanted
  - **Predicted**: What the model predicted
  - **Actual**: What the mesh really measured

**Interpretation:**
- **Predicted Error**: How well the RF models learned the lookup table
- **Actual Error**: TRUE performance of the inverse mapping
- If Actual MAE < 1.0 cm: **Excellent**
- If Actual MAE < 2.0 cm: **Good**
- If Actual MAE < 3.0 cm: **Acceptable**

## Workflow

### Complete Pipeline:

```bash
# Step 1: Train models (done once)
python train_model.py --input lookup_tables/lookup_table_female_asian.csv

# Step 2: Test accuracy with real mesh generation
python test_model_accuracy.py --input test_measurements.json --models macroparameters_inference_models.pkl

# Step 3: Use in production
python infer_macroparameters.py --input panellist_measurements.json --models macroparameters_inference_models.pkl
```

### For Production Use:

Once validated, use `infer_macroparameters.py` to get macroparameters for any measurements, then use those macroparameters with `generate_human.py` to create the final mesh.

## Files in This Directory

- `train_model.py` - Model training script
- `infer_macroparameters.py` - Inference script (production-ready)
- `test_model_accuracy.py` - Validation script (generates real meshes)
- `macroparameters_inference_models.pkl` - Trained models (4.2GB)
- `README.md` - This file

## Technical Details

### Model Architecture:
- **8 separate Random Forest models** (one per measurement)
- Each model: `macroparameters [age, muscle, weight, height, proportions] → one measurement`
- Random Forest automatically learns:
  - Non-linear relationships
  - Parameter interactions (e.g., height × proportions)
  - Complex patterns from lookup table data

### Inverse Mapping:
- Uses optimization (Differential Evolution) to search parameter space
- Objective: Minimize error between predicted and target measurements
- Ensures all 5 macroparameters work together to match all 8 measurements

### Why This Works:
1. Each model considers ALL 5 macroparameters as input
2. Random Forest captures complex non-linear relationships
3. Optimization finds the best balance across all measurements
4. No need for explicit mathematical formulas

## Performance

From your training output:
- **R² ≈ 0.999** for all measurements
- Models explain 99.9% of the variance
- Very high accuracy in forward prediction (macros → measurements)

**Next step:** Run `test_model_accuracy.py` to validate the inverse mapping with real mesh generation!

## Troubleshooting

**Models file too large?**
- 4.2GB is normal for 200 trees × 8 models
- Reduce with `--n-estimators 100` if needed

**Optimization too slow?**
- Use `--method differential_evolution` (default, faster)
- Or increase tolerance in the code

**Poor accuracy?**
- Try `--method both` for hybrid optimization
- Train with more trees: `--n-estimators 300`
- Check if lookup table covers the target measurement ranges

## Next Steps

1. **Validate:** Run `test_model_accuracy.py` on representative test cases
2. **Compare:** Try different model configurations (n_estimators, max_depth)
3. **Experiment:** Test XGBoost or other models (requires code modifications)
4. **Deploy:** Use `infer_macroparameters.py` in your production pipeline
