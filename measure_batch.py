#!/usr/bin/env python3
"""
Batch Measurement Script - Runs INSIDE Blender

This script processes a batch of parameter combinations:
1. Loads batch configuration
2. For each parameter combination:
   - Generate human mesh
   - Add rig
   - Extract measurements
   - Record to CSV
   - Delete model
3. Save final CSV

Usage (via run_blender.py):
    python run_blender.py --script measure_batch.py -- --batch-config batch_config.json --output lookup_table.csv
"""

import sys
import os
import csv
import json
import argparse
from pathlib import Path

# Add script directory to path
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Import utilities
import utils
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
        '--batch-config',
        type=str,
        required=True,
        help='Path to batch configuration JSON file'
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


def load_batch_config(config_path: str) -> dict:
    """
    Load batch configuration file.
    
    Args:
        config_path: Path to batch configuration JSON
        
    Returns:
        Batch configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Batch configuration not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✓ Loaded batch configuration: {config['total_combinations']:,} combinations")
    return config


def initialize_csv(output_path: str) -> tuple:
    """
    Initialize CSV file with headers.
    
    Args:
        output_path: Path for CSV file
        
    Returns:
        Tuple of (file_handle, csv_writer)
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Define CSV columns
    param_columns = ["age", "muscle", "weight", "height", "proportions"]
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
        params: Parameter dictionary
        measured: Measurements dictionary
    """
    row = {
        # Parameters
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
    Delete generated objects from scene.
    
    Args:
        basemesh: Mesh object to delete
        armature: Armature object to delete
    """
    import bpy
    
    bpy.ops.object.select_all(action='DESELECT')
    
    if basemesh:
        basemesh.select_set(True)
    if armature:
        armature.select_set(True)
    
    bpy.ops.object.delete()


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
    print(f"  Batch config: {args.batch_config}")
    print(f"  Output: {args.output}")
    print(f"  Rig type: {args.rig_type}")
    print(f"  Delete models: {not args.no_delete}")
    print(f"  Checkpoint interval: {args.checkpoint_interval}")
    
    try:
        # Load batch configuration
        print("\n" + "-"*70)
        print("LOADING BATCH CONFIGURATION")
        print("-"*70)
        batch_config = load_batch_config(args.batch_config)
        
        param_list = batch_config["parameters"]
        total = len(param_list)
        
        print(f"\nWill process {total:,} models")
        
        # Initialize CSV
        print("\n" + "-"*70)
        print("INITIALIZING OUTPUT")
        print("-"*70)
        csv_file, csv_writer = initialize_csv(args.output)
        
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
        
        for i, params in enumerate(param_list, 1):
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
                    print(f"\n✓ Checkpoint: {i}/{total} models processed ({successful} successful, {failed} failed)")
                
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