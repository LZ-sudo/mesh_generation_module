# Human Mesh Generation Module

Automated tool for generating customizable 3D human meshes using Blender and MPFB2 (MakeHuman for Blender). Features ML-powered measurement-to-parameter inference, batch processing for lookup table generation, and precise bone-based body measurements.

## Module Structure

```
mesh_generation_module/
├── measurement_functions/          # Body measurement extraction
│   ├── measurements.py             # Bone-based measurement functions
│   └── measure_batch.py            # Batch measurement processor (for build_lookup_table.py)
├── measurement_to_parameters_inference_models/  # ML inference system
│   ├── train_model.py              # Train TabM regression model
│   ├── test_generation_accuracy.py # Model accuracy testing
│   ├── infer_macroparameters.py    # Infer macros from measurements
│   └── adjust_microparameters.py   # Fine-tune microparameters    
├── mesh_data_generation_scripts/   # Data generation utilities
│   ├── generate_realistic_test_measurements.py  # Test data generator
│   └── build_lookup_table.py       # Lookup table generator (For training TabM model)
├── configs/                        # Configuration files
│   └── lookup_table_config_*.json  # Lookup table configurations
├── lookup_tables/                  # Generated measurement databases
│   └── lookup_table_*.csv          # Parameter → measurement mappings
├── run_blender.py                  # Blender launcher utility
├── generate_human.py               # Single character generation
├── compute_all_parameters.py       # End-to-end parameter computation
└── utils.py                        # Shared utility functions
```

## Features Overview

### Mesh Generation
- **Parametric Humans**: Generate diverse characters from macroparameters (age, muscle, weight, height, proportions, gender, race)
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Headless Operation**: No GUI required for batch processing
- **Rigging Support**: Automatic skeletal rig with multiple rig types (default, default_no_toes, game_engine)
- **FBX Export**: Compatible with Unity, Unreal Engine, and other 3D applications

### Measurement System
- **Bone-Based Measurements**: 10 precise measurements extracted from armature
- **Default Pose**: All measurements taken in anatomically consistent default pose
- **CV-Compatible**: Measurements designed to match Mediapipe pose estimation landmarks

### Machine Learning Pipeline
- **TabM Regression**: Neural network for measurements → macroparameters inference
- **Two-Phase Microparameter Adjustment**: Iterative refinement for accurate mesh recreation
- **Batch Training**: Generate lookup tables with thousands of samples for training

### Measurements Extracted

| Measurement | Description | Method |
|-------------|-------------|--------|
| `height_cm` | Total height | Head top to feet |
| `shoulder_width_cm` | Shoulder breadth | Distance between shoulder01.L/R tail bones |
| `hip_width_cm` | Hip width | Distance between upperleg01.L/R head bones (hip joints) |
| `head_width_cm` | Head width | Distance between temporalis02.L/R bones |
| `upper_arm_length_cm` | Upper arm | Bone chain: upperarm01 → upperarm02 |
| `forearm_length_cm` | Forearm | Bone chain: lowerarm01 → lowerarm02 |
| `hand_length_cm` | Hand | Bone chain: wrist → finger3-3 (middle finger) |
| `upper_leg_length_cm` | Upper leg | Bone chain: upperleg01 → upperleg02 |
| `lower_leg_length_cm` | Lower leg | Bone chain: lowerleg01 → lowerleg02 |
| `shoulder_to_waist_cm` | Torso length | Perpendicular distance between shoulder and hip lines |
| `neck_length_cm`* | Neck length | Bone chain: neck01 → neck02 (measured but not used for inference) |

*Note: `neck_length_cm` is measured but not used as input for macroparameter inference because CV (Mediapipe) cannot reliably measure it. It's used only for microparameter adjustment (Phase 2 height reconciliation).

## Installation

### Prerequisites

1. **Blender 4.5.3 LTS** - Download from [blender.org](https://download.blender.org/release/Blender4.5/blender-4.5.3-windows-x64.msi)
2. **MPFB2 Addon** - Install from Blender Extensions:
   - Open Blender → Edit → Preferences → Extensions
   - Search for "MPFB" and click Install
   - Restart Blender
3. **Python 3.11** (Required for Blender API compatibility)

### Python Dependencies

```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
myenv/Scripts/activate  # Windows
# source myenv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch (for ML models)
- tabm (TabM regression)
- pandas, numpy (data processing)
- scikit-learn (preprocessing)
- tqdm (progress bars)

## Quick Start

### 1. Generate Single Character

Create a configuration file (e.g., `human_male_test.json`):

```json
{
  "macro_settings": {
    "gender": 1.0,
    "age": 0.48862963914871216,
    "muscle": 0.6260563731193542,
    "weight": 0.7747020721435547,
    "height": 0.5371153950691223,
    "proportions": 0.7396003007888794,
    "cupsize": 0.5,
    "firmness": 0.5,
    "race": {
      "asian": 1.0,
      "caucasian": 0.0,
      "african": 0.0
    }
  },
  "micro_settings": {
    "measure-lowerleg-height-incr": 0.02744101061206962,
    "measure-upperleg-height-incr": 0.02744101061206962,
    "measure-napetowaist-dist-incr": 0.02744101061206962,
    "measure-waisttohip-dist-incr": 0.02744101061206962,
    "torso-scale-horiz-incr": 0.026564937015151894,
    "measure-waist-circ-incr": 0.026564937015151894,
    "head-scale-horiz-incr": 0.016596781655837403,
    "head-scale-depth-incr": 0.016596781655837403,
    "measure-neck-height-incr": 0.005397638885732131,
    "measure-upperarm-length-incr": 0.02246926840901563,
    "measure-lowerarm-length-incr": 0.021840181286682137
  }
}
```

Generate the mesh:

```bash
python run_blender.py --script generate_human.py -- --config human_test.json
```

### 2. Generate Lookup Table

Create a lookup table configuration (e.g., `configs/lookup_table_config_female_asian.json`):

```json
{
  "fixed_params": {
    "gender": 0.0,
    "cupsize": 0.5,
    "firmness": 0.5,
    "race": {
      "asian": 1.0,
      "caucasian": 0.0,
      "african": 0.0
    }
  },
  "grid_params": {
    "age": {
      "min": 0.3,
      "max": 1.0,
      "step": 0.05
    },
    "muscle": {
      "min": 0.0,
      "max": 1.0,
      "step": 0.10
    },
    "weight": {
      "min": 0.0,
      "max": 1.0,
      "step": 0.10
    },
    "height": {
      "min": 0.25,
      "max": 0.80,
      "step": 0.05
    },
    "proportions": {
      "min": 0.2,
      "max": 0.8,
      "step": 0.05
    }
  }
}
```

Generate the lookup table (LHS method recommended for parameter space coverage):

```bash
python build_lookup_table.py --config configs/lookup_table_config_female_asian.json --method lhs --n-samples 150000 --output lookup_tables/lookup_table_female_asian.csv
```

Output: `lookup_tables/lookup_table_female_asian.csv`

### 3. Train ML Model

Train a TabM regression model on the lookup table:

```bash
python measurement_to_parameters_inference_models/train_model.py \
  --data lookup_tables/lookup_table_female_asian.csv \
  --output models/female_asian_models.pkl \
  --epochs 100
```

### 4. Infer Parameters from Measurements

Create a measurements JSON file (e.g., `subject_measurements.json`):

```json
{
  "gender": "female",
  "race": "asian",
  "measurements": {
    "height_cm": 165.0,
    "shoulder_width_cm": 38.5,
    "hip_width_cm": 35.2,
    "head_width_cm": 14.8,
    "upper_arm_length_cm": 28.3,
    "forearm_length_cm": 24.7,
    "hand_length_cm": 18.2,
    "upper_leg_length_cm": 42.5,
    "lower_leg_length_cm": 38.0,
    "shoulder_to_waist_cm": 45.8
  }
}
```

Compute both macroparameters and microparameters:

```bash
python compute_all_parameters.py \
  --input subject_measurements.json \
  --models models/female_asian_models.pkl \
  --output parameters.json
```

Generate the mesh from computed parameters:

```bash
python run_blender.py --script generate_human.py -- --config parameters.json
```

## Advanced Usage

### Parameter Ranges

All macroparameters use values between **0.0 and 1.0**:

| Parameter | 0.0 | 0.5 (default) | 1.0 |
|-----------|-----|---------------|-----|
| `gender` | Female | Androgynous | Male |
| `age` | Child/Young | Adult | Elderly |
| `muscle` | Minimal | Average | Maximum |
| `weight` | Underweight | Average | Overweight |
| `height` | Short (~1.4m) | Average (~1.7m) | Tall (~2.1m) |
| `proportions` | Stylized | Realistic | Stylized |
| `cupsize` | Small | Medium | Large |
| `firmness` | Soft | Medium | Firm |

**Race values** must sum to 1.0:
- `asian`, `caucasian`, `african`: Each 0.0 to 1.0

### Rig Types

```bash
# Default rig with toes
python run_blender.py --script generate_human.py -- --config human.json --rig-type default

# Simplified rig without toes (recommended for games)
python run_blender.py --script generate_human.py -- --config human.json --rig-type default_no_toes

# Optimized for game engines
python run_blender.py --script generate_human.py -- --config human.json --rig-type game_engine
```

### Two-Phase Microparameter Adjustment

The system uses a two-phase adjustment strategy for accurate mesh recreation:

**Phase 1 - Anchor Adjustments** (9 reliable CV measurements):
- Body dimensions: shoulder_width_cm, hip_width_cm, head_width_cm
- Arms: upper_arm_length_cm, forearm_length_cm, hand_length_cm
- Legs: upper_leg_length_cm, lower_leg_length_cm
- Torso: shoulder_to_waist_cm (uses measure-napetowaist-dist, measure-waisttohip-dist)

**Phase 2 - Height Reconciliation**:
- height_cm (uses measure-neck-height to adjust final height after other proportions are locked)

This ensures that CV-measurable proportions are prioritized, with neck length adjusting to reconcile overall height.

### Export Settings

Customize FBX export for different applications:

**For Unity:**
```json
"export_settings": {
  "global_scale": 1.0,
  "axis_forward": "-Z",
  "axis_up": "Y"
}
```

**For Unreal Engine:**
```json
"export_settings": {
  "global_scale": 1.0,
  "axis_forward": "X",
  "axis_up": "Z"
}
```

### Testing Generation Accuracy

Test how accurately the system can recreate meshes from measurements:

```bash
# Generate test measurements
python mesh_data_generation_scripts/generate_realistic_test_measurements.py \
  --csv lookup_tables/lookup_table_female_asian.csv \
  --num-samples 10 \
  --output test_measurements.json

# Test accuracy
python test_generation_accuracy.py \
  --input test_measurements.json \
  --models models/female_asian_models.pkl \
  --rig-type default_no_toes
```

## Output Files

### Lookup Table CSV

```csv
age,muscle,weight,height,proportions,height_cm,shoulder_width_cm,hip_width_cm,head_width_cm,upper_arm_length_cm,forearm_length_cm,hand_length_cm,upper_leg_length_cm,lower_leg_length_cm,shoulder_to_waist_cm,neck_length_cm
0.0,0.0,0.0,0.0,0.5,140.5,32.1,24.5,14.2,26.8,22.3,16.7,35.2,30.1,38.4,9.1
0.0,0.0,0.0,0.1,0.5,145.2,33.4,25.1,14.5,27.5,23.1,17.1,36.8,31.5,40.1,9.3
```

### Parameters JSON

After running `compute_all_parameters.py`:

```json
{
  "macro_settings": {
    "gender": 0.0,
    "age": 0.3524,
    "muscle": 0.6012,
    "weight": 0.4489,
    "height": 0.6523,
    "proportions": 0.5,
    "cupsize": 0.5,
    "firmness": 0.5,
    "race": {
      "asian": 1.0,
      "caucasian": 0.0,
      "african": 0.0
    }
  },
  "micro_settings": {
    "measure-napetowaist-dist-incr": 0.23,
    "measure-upperarm-length-decr": 0.15,
    ...
  }
}
```

## Configuration

### Blender Path Detection

The script automatically finds Blender. To manually specify:

```bash
# Set environment variable
export BLENDER_PATH="/path/to/blender"

# Or use command line
python run_blender.py --blender-path "/path/to/blender" --script generate_human.py -- --config human.json
```

### Lookup Table Configuration

Configuration files specify fixed and grid parameters:

- **fixed_params**: Held constant (gender, race, cupsize, firmness)
- **grid_params**: Varied with min, max, step values
- Total combinations = product of all grid dimensions

Example: 11 × 11 × 11 × 11 × 1 = 14,641 combinations

### Performance Tuning

```bash
# Adjust checkpoint frequency (default: every 50 models)
# Edit build_lookup_table.py: CHECKPOINT_INTERVAL = 100

# Dry run to validate config
python build_lookup_table.py --config config.json --dry-run

# Keep models for debugging
python build_lookup_table.py --config config.json --no-delete
```

## Troubleshooting

### "Blender not found"

1. Set environment variable:
   ```bash
   export BLENDER_PATH="/Applications/Blender.app/Contents/MacOS/Blender"  # macOS
   # $env:BLENDER_PATH="C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"  # Windows
   ```

2. Or specify manually:
   ```bash
   python run_blender.py --blender-path "/path/to/blender" --script generate_human.py -- --config human.json
   ```

### "MPFB2 addon not found"

1. Open Blender GUI
2. Edit → Preferences → Extensions
3. Search "MPFB" and click Install
4. Restart Blender

### Missing measurements in CSV

Ensure all 10 required measurements are present in input JSON:
- height_cm, shoulder_width_cm, hip_width_cm, head_width_cm
- upper_arm_length_cm, forearm_length_cm, hand_length_cm
- upper_leg_length_cm, lower_leg_length_cm, shoulder_to_waist_cm

### Poor reconstruction accuracy

1. **Train with more data**: Increase grid granularity (smaller step sizes)
2. **Match demographics**: Use models trained on same gender/race as target
3. **Check measurement quality**: Ensure CV measurements are accurate
4. **Use microparameter adjustment**: Don't skip `--no-micro-adjustment` flag

## References

### Software & Libraries
- [Blender](https://www.blender.org) - 3D creation suite (GPL v3)
- [MPFB2](http://www.makehumancommunity.org/) - MakeHuman for Blender (AGPL v3)
- [TabM](https://github.com/yandex-research/tabm) - Tabular regression model
- [PyTorch](https://pytorch.org/) - Machine learning framework
- [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide) - CV landmark detection

### Related Work
- [MakeHuman](http://www.makehumancommunity.org/) - Open-source character creation
- [SMPL](https://smpl.is.tue.mpg.de/) - Skinned Multi-Person Linear model

## License

This tool uses GPL v3 (Blender) and AGPL v3 (MPFB2) licensed software. Generated meshes are yours to use freely.

## Credits

- **Blender Foundation** - [blender.org](https://www.blender.org)
- **MPFB2 Team** - [makehumancommunity.org](http://www.makehumancommunity.org/)
- **Yandex Research** - TabM regression model
