# Human Mesh Generation Module

Automated tool for generating customizable human 3D meshes using Blender and MPFB2 (MakeHuman for Blender), with advanced body measurement extraction and lookup table generation.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration File Format](#configuration-file-format)
- [Command Line Options](#command-line-options)
- [Example Configurations](#example-configurations)
- [Lookup Table Generation](#lookup-table-generation)
  - [Quick Start](#quick-start-1)
  - [Configuration](#lookup-table-configuration)
  - [Measurements](#measurements-extracted)
  - [Performance](#performance--scalability)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Performance](#performance)
- [Quick Reference](#quick-reference)

## Features

- **Cross-platform**: Works on Windows, macOS, and Linux
- **Automatic Blender detection**: Finds Blender installation automatically
- **Headless operation**: No GUI required
- **Full customization**: Control body proportions, age, gender, race, and more
- **Rigging support**: Automatically adds skeletal rig for animation
- **FBX export**: Standard format compatible with Unity, Unreal Engine, and other 3D applications
- **Body measurements**: Precise bone-based measurements extracted from generated models
- **Lookup table generation**: Batch process thousands of parameter combinations efficiently
- **Memory-efficient processing**: Generate millions of combinations without running out of memory

## Prerequisites

1. **Blender 4.5.3 LTS for Windows** - Download from [blender.org](https://download.blender.org/release/Blender4.5/blender-4.5.3-windows-x64.msi)
2. **MPFB2 addon** - Install from Blender Extensions:
   - Open Blender
   - Go to Edit → Preferences → Extensions
   - Search for "MPFB" and click Install
   - Restart Blender

3. **Python 3.11** (Specifically required for Blender API to work)

## Installation

1. Clone or download this repository
2. No additional Python packages needed - the script uses Blender's built-in Python

## Quick Start

### Basic Usage

```bash
python run_blender.py --script generate_human.py -- --config human_test.json
```

This will:
1. Automatically find your Blender installation
2. Generate a human mesh based on `human_test.json`
3. Add rigging
4. Export to `output/human.fbx`

### First Run

On the first run, the script will search for Blender and cache its location in `.blender_config.json` for faster subsequent runs.

## Configuration File Format

Create a JSON file (e.g., `my_character.json`) with the following structure:

```json
{
  "macro_settings": {
    "gender": 0.0,
    "age": 0.35,
    "muscle": 0.6,
    "weight": 0.45,
    "proportions": 0.5,
    "height": 0.65,
    "cupsize": 0.5,
    "firmness": 0.5,
    "race": {
      "asian": 0.33,
      "caucasian": 0.33,
      "african": 0.34
    }
  },
  "output": {
    "directory": "./output",
    "filename": "my_character.fbx"
  },
  "export_settings": {
    "global_scale": 1.0,
    "axis_forward": "-Z",
    "axis_up": "Y"
  }
}
```

### Parameter Reference

All macro settings use values between **0.0 and 1.0**:

| Parameter | 0.0 | 0.5 (default) | 1.0 |
|-----------|-----|---------------|-----|
| `gender` | Female | Androgynous | Male |
| `age` | Child/Young | Adult | Elderly |
| `muscle` | Minimal muscle | Average | Maximum muscle |
| `weight` | Underweight | Average | Overweight |
| `height` | Short (~1.4m) | Average (~1.7m) | Tall (~2.1m) |
| `proportions` | Stylized | Realistic | Stylized |
| `cupsize` | Small | Medium | Large |
| `firmness` | Soft | Medium | Firm |

**Race values** must sum to approximately 1.0:
- `asian`: 0.0 to 1.0
- `caucasian`: 0.0 to 1.0
- `african`: 0.0 to 1.0

### Export Settings (Optional)

Customize FBX export for different target applications:

#### For Unity:
```json
"export_settings": {
  "global_scale": 1.0,
  "axis_forward": "-Z",
  "axis_up": "Y"
}
```

#### For Unreal Engine:
```json
"export_settings": {
  "global_scale": 1.0,
  "axis_forward": "X",
  "axis_up": "Z"
}
```

#### For generic applications (larger scale):
```json
"export_settings": {
  "global_scale": 100.0,
  "bake_space_transform": true
}
```

## Command Line Options

### Basic Options

```bash
# Generate with default rig
python run_blender.py --script generate_human.py -- --config my_character.json

# Generate without rigging
python run_blender.py --script generate_human.py -- --config my_character.json --no-rig

# Use specific rig type
python run_blender.py --script generate_human.py -- --config my_character.json --rig-type default_no_toes

# Enable verbose output
python run_blender.py --script generate_human.py -- --config my_character.json --verbose
```

### Rig Types

- `default`: Standard rig with toes (default)
- `default_no_toes`: Standard rig without toe bones (simpler, recommended for games)
- `game_engine`: Optimized rig for game engines

### Advanced Options

```bash
# Manually specify Blender path
python run_blender.py --blender-path "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --script generate_human.py -- --config my_character.json

# Find Blender location only (diagnostic)
python run_blender.py --find-only

# Run with GUI (for debugging)
python run_blender.py --gui --script generate_human.py -- --config my_character.json
```

## Example Configurations

### Athletic Female
```json
{
  "macro_settings": {
    "gender": 0.0,
    "age": 0.25,
    "muscle": 0.7,
    "weight": 0.4,
    "height": 0.6,
    "cupsize": 0.5,
    "firmness": 0.7,
    "race": {
      "asian": 0.0,
      "caucasian": 1.0,
      "african": 0.0
    }
  },
  "output": {
    "directory": "./output",
    "filename": "athletic_female.fbx"
  }
}
```

### Large Male
```json
{
  "macro_settings": {
    "gender": 1.0,
    "age": 0.5,
    "muscle": 0.8,
    "weight": 0.8,
    "height": 0.85,
    "race": {
      "asian": 0.0,
      "caucasian": 0.0,
      "african": 1.0
    }
  },
  "output": {
    "directory": "./output",
    "filename": "large_male.fbx"
  }
}
```

### Elderly Character
```json
{
  "macro_settings": {
    "gender": 0.5,
    "age": 0.9,
    "muscle": 0.3,
    "weight": 0.4,
    "height": 0.45,
    "race": {
      "asian": 0.5,
      "caucasian": 0.5,
      "african": 0.0
    }
  },
  "output": {
    "directory": "./output",
    "filename": "elderly_character.fbx"
  }
}
```

## Output

Generated files are saved to the `output/` directory (or your specified directory):

- **FBX file**: Contains mesh geometry, rigging, and vertex weights
- **File size**: Typically 1-2 MB with rigging
- **Mesh details**: ~19,000 vertices, ~163 bones (default rig)

## Troubleshooting

### "Blender not found"

If the script can't find Blender:

1. **Set environment variable**:
   ```bash
   # Windows (PowerShell)
   $env:BLENDER_PATH="C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"

   # macOS/Linux
   export BLENDER_PATH="/Applications/Blender.app/Contents/MacOS/Blender"
   ```

2. **Or specify path manually**:
   ```bash
   python run_blender.py --blender-path "path/to/blender.exe" --script generate_human.py -- --config my_character.json
   ```

### "MPFB2 addon not found"

1. Open Blender normally (GUI)
2. Go to Edit → Preferences → Extensions
3. Search for "MPFB" and click Install
4. Restart Blender and try again

### Mesh looks default despite custom settings

This was a known issue that has been fixed. Make sure you're using the latest version of `utils.py` which includes the `reapply_macro_details()` call.

### Rigging appears displaced

Try different export settings in your config file:

```json
"export_settings": {
  "global_scale": 100.0,
  "bake_space_transform": true
}
```

Also check your target application's FBX import settings - ensure scale is set correctly (usually 1.0 or 100.0).

## Lookup Table Generation

Generate comprehensive measurement lookup tables by batch processing parameter combinations.

### Quick Start

```bash
# Generate lookup table from configuration
python build_lookup_table.py --config configs/lookup_table_config_female_asian.json
```

This will:
1. Load the configuration file
2. Generate all parameter combinations on-the-fly (memory-efficient)
3. Create models in Blender and extract measurements
4. Save results to `output/lookup_table_female_asian.csv`

### Lookup Table Configuration

Create a configuration file (e.g., `lookup_table_config_female_asian.json`):

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
      "min": 0.0,
      "max": 1.0,
      "step": 0.1
    },
    "muscle": {
      "min": 0.0,
      "max": 1.0,
      "step": 0.1
    },
    "weight": {
      "min": 0.0,
      "max": 1.0,
      "step": 0.1
    },
    "height": {
      "min": 0.0,
      "max": 1.0,
      "step": 0.1
    },
    "proportions": {
      "min": 0.5,
      "max": 0.5,
      "step": 1.0
    }
  }
}
```

**Configuration explanation:**
- `fixed_params`: Parameters held constant for all combinations
- `grid_params`: Parameters varied across a grid with min, max, and step values
- The example above generates 11 × 11 × 11 × 11 × 1 = 14,641 combinations

### Dynamic File Naming

Output files are automatically named based on the config filename:

```bash
# Input: configs/lookup_table_config_female_asian.json
# Outputs: output/lookup_table_female_asian.csv

# Input: configs/lookup_table_config_male_caucasian.json
# Outputs: output/lookup_table_male_caucasian.csv
```

### Measurements Extracted

Each model is measured using precise bone-based methods:

| Measurement | Description | Method |
|-------------|-------------|--------|
| `height_cm` | Total height | Head top to feet |
| `shoulder_width_cm` | Shoulder breadth | Joint-based |
| `chest_width_cm` | Chest depth | Mesh depth (front-to-back) |
| `head_width_cm` | Head width | Mesh width (side-to-side) |
| `neck_length_cm` | Neck length | Bone chain: neck01 → neck02 |
| `upper_arm_length_cm` | Upper arm length | Bone chain: upperarm01 → upperarm02 |
| `forearm_length_cm` | Forearm length | Bone chain: lowerarm01 → lowerarm02 |
| `hand_length_cm` | Hand length | Bone chain: wrist → finger3-3 (middle finger tip) |

All measurements use the left side of the body for consistency.

### Command Line Options

```bash
# Basic usage
python build_lookup_table.py --config configs/lookup_table_config.json

# Custom output path
python build_lookup_table.py --config configs/lookup_table_config.json --output custom_path.csv

# Dry run (validate config without processing)
python build_lookup_table.py --config configs/lookup_table_config.json --dry-run

# Keep models after measurement (for debugging)
python build_lookup_table.py --config configs/lookup_table_config.json --no-delete
```

### Output Format

The generated CSV includes both input parameters and measurements:

```csv
age,muscle,weight,height,proportions,height_cm,shoulder_width_cm,chest_width_cm,head_width_cm,neck_length_cm,upper_arm_length_cm,forearm_length_cm,hand_length_cm
0.0,0.0,0.0,0.0,0.5,140.5,32.1,21.3,14.2,9.1,26.8,22.3,16.7
0.0,0.0,0.0,0.1,0.5,145.2,33.4,22.1,14.5,9.3,27.5,23.1,17.1
...
```

### Performance & Scalability

- **Processing speed**: ~1-2 seconds per model
- **Memory usage**: Constant (generates combinations on-the-fly)
- **Config file size**: ~2 KB regardless of combination count
- **Checkpoint saving**: Progress saved every 50 models by default
- **Scalability**: Can handle millions of combinations without memory issues

**Example processing times:**
- 1,000 combinations: ~20-30 minutes
- 10,000 combinations: ~3-5 hours
- 100,000 combinations: ~30-50 hours

### Memory-Efficient Design

The system uses an intelligent on-the-fly generation approach:

**Traditional approach (problematic):**
- Generate all combinations → Store in JSON (can be 100+ MB for millions of combinations) → Load all into memory → Process

**Our approach (memory-efficient):**
- Store only grid parameters (always <5 KB) → Generate one combination at a time → Process → Repeat

This allows processing of millions of combinations without running out of memory.

## Project Structure

```
mesh_generation_module/
├── run_blender.py              # Main launcher script
├── generate_human.py           # Single human generation script
├── build_lookup_table.py       # Lookup table builder (orchestrator)
├── measure_batch.py            # Batch measurement processor (runs in Blender)
├── measurements.py             # Body measurement extraction functions
├── utils.py                    # Utility functions
├── human_test.json             # Example single character config
├── configs/
│   └── lookup_table_config_*.json  # Lookup table configurations
├── output/
│   ├── *.fbx                   # Generated FBX files
│   └── lookup_table_*.csv      # Generated lookup tables
├── .blender_config.json        # Cached Blender path (auto-generated)
└── README.md                   # This file
```

## How It Works

### Single Character Generation

1. **Blender Detection**: `run_blender.py` searches common installation locations and caches the result
2. **Headless Execution**: Blender runs in background mode without GUI
3. **Mesh Generation**: MPFB2 creates base human mesh
4. **Parameter Application**: Macro settings are applied and baked into mesh geometry
5. **Rigging**: Skeletal armature is fitted to the mesh
6. **Export**: Final mesh and rig exported as FBX

### Lookup Table Generation

1. **Configuration Loading**: Load grid parameters from config file
2. **On-the-fly Generation**: Generate parameter combinations one at a time (memory-efficient)
3. **For each combination**:
   - Create human mesh with MPFB2
   - Apply parameters and bake into geometry
   - Add rigging
   - **Extract measurements** using bone-based methods
   - Record to CSV
   - Delete model and continue
4. **Checkpoint Saving**: CSV is flushed to disk periodically to prevent data loss

### Measurement System

The measurement extraction uses a hybrid approach combining bone-based and mesh-based methods:

**Bone-based measurements** (most accurate):
- Uses the actual skeletal rig bone positions
- Measures from extreme points of bone chains
- Examples: neck_length (neck01→neck02), upper_arm_length (upperarm01→upperarm02), forearm_length (lowerarm01→lowerarm02), hand_length (wrist→finger3-3)

**Joint-based measurements**:
- Finds anatomical joints from mesh geometry
- Measures distances between identified joint positions
- Example: shoulder_width (distance between shoulder joints)

**Mesh-based measurements**:
- Analyzes mesh vertex positions directly
- Calculates widths and depths from vertex spans
- Examples: chest_width (front-to-back depth), head_width (side-to-side width)

This approach provides high accuracy and consistency across all generated models.

## Performance

### Single Character Generation
- **Generation time**: ~5-10 seconds per character (including Blender startup)
- **File size**: 1-2 MB FBX with rigging
- **Mesh complexity**: ~19,000 vertices, ~163 bones (default rig)

### Lookup Table Generation
- **Processing speed**: ~1-2 seconds per model (after Blender startup)
- **Memory usage**: Constant, regardless of combination count
- **Throughput**: ~30-60 models per minute
- **Scalability**: Successfully tested with 100,000+ combinations

## License

This tool uses:
- **Blender**: GPL v3
- **MPFB2**: AGPL v3
- Your generated meshes are yours to use freely

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify MPFB2 is installed in Blender
3. Try running with `--verbose` flag for detailed output

## Quick Reference

### Generate Single Character
```bash
python run_blender.py --script generate_human.py -- --config human_test.json
```

### Generate Lookup Table
```bash
python build_lookup_table.py --config configs/lookup_table_config_female_asian.json
```

### Key Files
- **Configuration files**: `configs/lookup_table_config_*.json`
- **Generated CSVs**: `output/lookup_table_*.csv`
- **Generated FBX**: `output/*.fbx`
- **Blender cache**: `.blender_config.json`

### Measurement Columns
`height_cm`, `shoulder_width_cm`, `chest_width_cm`, `head_width_cm`, `neck_length_cm`, `upper_arm_length_cm`, `forearm_length_cm`, `hand_length_cm`

### Parameter Ranges
All parameters: `0.0` to `1.0` (except race which must sum to ~1.0)

## Credits

- **Blender Foundation** - [blender.org](https://www.blender.org)
- **MPFB2 Team** - [makehumancommunity.org](http://www.makehumancommunity.org/)
