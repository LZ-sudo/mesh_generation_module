#!/usr/bin/env python3
"""
Batch Measurement Script - Runs INSIDE Blender

This script processes a batch of parameter combinations:
1. Loads configuration file
2. Generates parameter combinations on-the-fly
3. For each parameter combination:
   - Generate human mesh
   - Add rig
   - Extract measurements
   - Record to CSV
   - Delete model
4. Save final CSV

Usage (via run_blender.py):
    python run_blender.py --script measure_batch.py -- --config configs/lookup_table_config_test.json 
"""

import sys
import os
import csv
import json
import argparse
import gc
from pathlib import Path
from itertools import product
import numpy as np

# Add parent directory to path to import utils
# (since measure_batch.py is now in measurement_functions subdirectory)
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import utilities from parent directory
import utils

# Import measurements from same directory (measurement_functions)
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
import measurements


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    # Find where Blender arguments end and script arguments begin
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(
        description='Batch process human models and extract measurements'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/lookup_table.csv',
        help='Path for output CSV file'
    )
    
    parser.add_argument(
        '--rig-type',
        type=str,
        default='default_no_toes',
        choices=['default', 'default_no_toes', 'game_engine'],
        help='Type of rig to add'
    )
    
    parser.add_argument(
        '--no-delete',
        action='store_true',
        help='Do not delete models after measurement (for debugging)'
    )
    
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=50,
        help='Save CSV every N models (default: 50)'
    )
    
    return parser.parse_args(argv)


def load_config(config_path: str) -> dict:
    """
    Load configuration file.

    Args:
        config_path: Path to configuration JSON

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Calculate total combinations
    total = 1
    for param_config in config["grid_params"].values():
        min_val = param_config["min"]
        max_val = param_config["max"]
        step = param_config["step"]
        n_values = len(np.arange(min_val, max_val + step/2, step))
        total *= n_values

    print(f"✓ Loaded configuration: {total:,} combinations")
    return config


def generate_parameter_combinations(config: dict):
    """
    Generate parameter combinations on-the-fly from grid configuration.

    This avoids loading millions of combinations into memory at once.

    Args:
        config: Configuration dictionary with fixed_params and grid_params

    Yields:
        Parameter dictionary for each combination
    """
    fixed_params = config["fixed_params"]
    grid_params = config["grid_params"]

    # Generate value lists for each grid parameter
    param_names = []
    param_values = []

    for param_name, param_config in grid_params.items():
        min_val = param_config["min"]
        max_val = param_config["max"]
        step = param_config["step"]

        # Generate values using numpy for precision
        values = np.arange(min_val, max_val + step/2, step)
        values = np.clip(values, 0.0, 1.0)  # Ensure within bounds
        values = [round(float(v), 3) for v in values]  # Round to 3 decimals

        param_names.append(param_name)
        param_values.append(values)

    # Generate combinations one at a time
    for combo in product(*param_values):
        params = fixed_params.copy()

        # Add grid parameter values
        for i, param_name in enumerate(param_names):
            params[param_name] = combo[i]

        yield params


def initialize_csv(output_path: str, config: dict) -> tuple:
    """
    Initialize CSV file with headers.

    Args:
        output_path: Path for CSV file
        config: Configuration dictionary (used for validation)

    Returns:
        Tuple of (file_handle, csv_writer)
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Validate config has required keys
    if "fixed_params" not in config or "grid_params" not in config:
        raise ValueError("Config must contain 'fixed_params' and 'grid_params'")

    # Define parameter columns
    # Include ALL parameters: both fixed and grid
    param_columns = [
        # Fixed parameters
        "gender",
        "cupsize",
        "firmness",
        "asian",
        "caucasian",
        "african",
        # Grid parameters
        "age",
        "muscle",
        "weight",
        "height",
        "proportions"
    ]

    measurement_columns = [
        "height_cm",
        "shoulder_width_cm",
        "chest_width_cm",
        "head_width_cm",
        "neck_length_cm",
        "upper_arm_length_cm",
        "forearm_length_cm",
        "hand_length_cm"
    ]

    columns = param_columns + measurement_columns

    # Open CSV file
    csv_file = open(output_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=columns)
    csv_writer.writeheader()

    print(f"✓ Initialized CSV: {output_path}")
    print(f"  Columns: {len(columns)}")
    print(f"    Parameters: {len(param_columns)} (6 fixed + 5 grid)")
    print(f"    Measurements: {len(measurement_columns)}")

    return csv_file, csv_writer


def process_single_model(params: dict, rig_type: str) -> dict:
    """
    Generate model, measure, and return measurements.
    
    Args:
        params: Parameter dictionary
        rig_type: Type of rig to add
        
    Returns:
        Dictionary with measurements
    """
    import bpy
    import importlib
    
    # Get MPFB module path
    mpfb_path = utils._get_mpfb_module_path()
    HumanService = importlib.import_module(f'{mpfb_path}.services.humanservice').HumanService
    
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Create human
    basemesh = HumanService.create_human()
    
    # Apply macro settings
    utils.apply_macro_settings_to_human(basemesh, params)
    
    # Add rig
    armature, _ = utils.add_standard_rig(basemesh, rig_type)
    
    # Extract measurements
    measured = measurements.extract_all_measurements(basemesh, armature)
    
    return measured, basemesh, armature


def write_measurement_row(csv_writer, params: dict, measured: dict):
    """
    Write a single row to CSV.

    Args:
        csv_writer: CSV writer object
        params: Parameter dictionary (includes both fixed and grid params)
        measured: Measurements dictionary
    """
    row = {
        # Fixed parameters
        "gender": params["gender"],
        "cupsize": params["cupsize"],
        "firmness": params["firmness"],
        "asian": params["race"]["asian"],
        "caucasian": params["race"]["caucasian"],
        "african": params["race"]["african"],

        # Grid parameters
        "age": params["age"],
        "muscle": params["muscle"],
        "weight": params["weight"],
        "height": params["height"],
        "proportions": params["proportions"],

        # Measurements (rename keys to match CSV columns)
        "height_cm": measured["height"],
        "shoulder_width_cm": measured["shoulder_width"],
        "chest_width_cm": measured["chest_width"],
        "head_width_cm": measured["head_width"],
        "neck_length_cm": measured["neck_length"],
        "upper_arm_length_cm": measured["upper_arm_length"],
        "forearm_length_cm": measured["forearm_length"],
        "hand_length_cm": measured["hand_length"]
    }

    csv_writer.writerow(row)


def cleanup_scene(basemesh, armature):
    """
    Delete generated objects from scene and clear orphaned data blocks.

    This is critical for batch processing to prevent memory leaks.
    Blender doesn't automatically remove mesh/armature data blocks when
    objects are deleted, causing RAM to fill up during long batch runs.

    Args:
        basemesh: Mesh object to delete
        armature: Armature object to delete
    """
    import bpy

    # Store references to data blocks before deleting objects
    mesh_data = basemesh.data if basemesh else None
    armature_data = armature.data if armature else None

    # Delete objects
    bpy.ops.object.select_all(action='DESELECT')

    if basemesh:
        basemesh.select_set(True)
    if armature:
        armature.select_set(True)

    bpy.ops.object.delete()

    # Manually remove orphaned data blocks
    # This is essential to prevent memory leaks in batch processing
    if mesh_data and mesh_data.users == 0:
        bpy.data.meshes.remove(mesh_data)

    if armature_data and armature_data.users == 0:
        bpy.data.armatures.remove(armature_data)

    # Clear orphaned data blocks globally
    # This catches any other orphaned data (materials, textures, etc.)
    for block_type in [bpy.data.meshes, bpy.data.armatures, bpy.data.materials,
                       bpy.data.textures, bpy.data.images]:
        for block in block_type:
            if block.users == 0:
                block_type.remove(block)

    # Force Python garbage collection to free memory immediately
    gc.collect()


def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("BATCH MEASUREMENT PROCESSOR")
    print("="*70 + "\n")
    
    # Check if running in Blender
    try:
        import bpy
        print(f"✓ Running in Blender {bpy.app.version_string}")
    except ImportError:
        print("✗ ERROR: This script must be run with Blender!")
        sys.exit(1)
    
    # Check MPFB2
    if not utils.check_mpfb2_installed():
        sys.exit(1)
    
    # Parse arguments
    try:
        args = parse_arguments()
    except SystemExit as e:
        if e.code != 0:
            print("\n✗ Error parsing arguments")
        sys.exit(e.code)
    
    print(f"\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"  Output: {args.output}")
    print(f"  Rig type: {args.rig_type}")
    print(f"  Delete models: {not args.no_delete}")
    print(f"  Checkpoint interval: {args.checkpoint_interval}")

    try:
        # Load configuration
        print("\n" + "-"*70)
        print("LOADING CONFIGURATION")
        print("-"*70)
        config = load_config(args.config)

        # Calculate total combinations
        total = 1
        for param_config in config["grid_params"].values():
            min_val = param_config["min"]
            max_val = param_config["max"]
            step = param_config["step"]
            n_values = len(np.arange(min_val, max_val + step/2, step))
            total *= n_values

        print(f"\nWill process {total:,} models (generated on-the-fly)")
        
        # Initialize CSV
        print("\n" + "-"*70)
        print("INITIALIZING OUTPUT")
        print("-"*70)
        csv_file, csv_writer = initialize_csv(args.output, config)
        
        # Set up Blender scene
        print("\n" + "-"*70)
        print("SETTING UP BLENDER SCENE")
        print("-"*70)
        utils.setup_blender_scene()
        
        # Process each model
        print("\n" + "="*70)
        print("PROCESSING MODELS")
        print("="*70)

        successful = 0
        failed = 0

        # Generate parameter combinations on-the-fly
        param_generator = generate_parameter_combinations(config)

        for i, params in enumerate(param_generator, 1):
            print(f"\n[{i}/{total}] Processing model {i:,}/{total:,}")
            print("-"*70)
            
            try:
                # Generate and measure
                measured, basemesh, armature = process_single_model(params, args.rig_type)
                
                # Write to CSV
                write_measurement_row(csv_writer, params, measured)
                
                # Cleanup (if not debugging)
                if not args.no_delete:
                    cleanup_scene(basemesh, armature)
                
                successful += 1
                
                # Checkpoint
                if i % args.checkpoint_interval == 0:
                    csv_file.flush()
                    # Force garbage collection periodically to prevent memory buildup
                    gc.collect()
                    print(f"\n✓ Checkpoint: {i}/{total} models processed ({successful} successful, {failed} failed)")
                    print(f"  Memory cleanup performed")
                
            except Exception as e:
                print(f"\n✗ Error processing model {i}: {e}")
                failed += 1
                
                # Write empty row or continue?
                # For now, just continue
                
                if not args.no_delete:
                    # Try to clean up anyway
                    try:
                        import bpy
                        bpy.ops.object.select_all(action='SELECT')
                        bpy.ops.object.delete()
                    except:
                        pass
        
        # Close CSV
        csv_file.close()
        
        # Final summary
        print("\n" + "="*70)
        print("BATCH PROCESSING COMPLETE")
        print("="*70)
        print(f"\nTotal models: {total:,}")
        print(f"Successful: {successful:,}")
        print(f"Failed: {failed:,}")
        print(f"Success rate: {100*successful/total:.1f}%")
        print(f"\nOutput saved to: {args.output}")
        
        # Check file size
        output_path = Path(args.output)
        if output_path.exists():
            file_size = output_path.stat().st_size / 1024  # KB
            print(f"File size: {file_size:.1f} KB")
            
            # Count rows
            with open(args.output, 'r') as f:
                row_count = sum(1 for line in f) - 1  # Exclude header
            print(f"Rows: {row_count:,}")
        
        print("="*70 + "\n")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to close CSV if it was opened
        try:
            csv_file.close()
        except:
            pass
        
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(130)