# Human Mesh Generation Module

Automated tool for generating customizable human 3D meshes using Blender and MPFB2 (MakeHuman for Blender).

## Features

- **Cross-platform**: Works on Windows, macOS, and Linux
- **Automatic Blender detection**: Finds Blender installation automatically
- **Headless operation**: No GUI required
- **Full customization**: Control body proportions, age, gender, race, and more
- **Rigging support**: Automatically adds skeletal rig for animation
- **FBX export**: Standard format compatible with Unity, Unreal Engine, and other 3D applications

## Prerequisites

1. **Blender 4.2+** - Download from [blender.org](https://www.blender.org/download/)
2. **MPFB2 addon** - Install from Blender Extensions:
   - Open Blender
   - Go to Edit → Preferences → Extensions
   - Search for "MPFB" and click Install
   - Restart Blender

3. **Python 3.8+** (usually comes with your system)

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

## Project Structure

```
mesh_generation_module/
├── run_blender.py           # Main launcher script
├── generate_human.py        # Human generation script
├── utils.py                 # Utility functions
├── human_test.json          # Example configuration
├── output/                  # Generated FBX files
├── .blender_config.json     # Cached Blender path (auto-generated)
└── README.md               # This file
```

## How It Works

1. **Blender Detection**: `run_blender.py` searches common installation locations and caches the result
2. **Headless Execution**: Blender runs in background mode without GUI
3. **Mesh Generation**: MPFB2 creates base human mesh
4. **Parameter Application**: Macro settings are applied and baked into mesh geometry
5. **Rigging**: Skeletal armature is fitted to the mesh
6. **Export**: Final mesh and rig exported as FBX

## Performance

- **Generation time**: ~5-10 seconds per character (including Blender startup)
- **Batch processing**: Run multiple configs sequentially for batch generation

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

## Credits

- **Blender Foundation** - [blender.org](https://www.blender.org)
- **MPFB2 Team** - [makehumancommunity.org](http://www.makehumancommunity.org/)
