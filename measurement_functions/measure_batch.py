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
import time
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
import utils as utils

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
        '--param-list',
        type=str,
        required=True,
        help='Path to JSON file containing list of parameter combinations'
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


def load_parameter_list(param_list_path: str) -> list:
    """
    Load pre-generated parameter combinations from JSON file.

    Args:
        param_list_path: Path to JSON file containing parameter list

    Returns:
        List of parameter dictionaries
    """
    if not os.path.exists(param_list_path):
        raise FileNotFoundError(f"Parameter list not found: {param_list_path}")

    with open(param_list_path, 'r') as f:
        param_list = json.load(f)

    print(f"✓ Loaded {len(param_list):,} parameter combinations")
    return param_list


def initialize_csv(output_path: str, param_list: list) -> tuple:
    """
    Initialize CSV file with headers.

    Args:
        output_path: Path for CSV file
        param_list: List of parameter dictionaries (used for validation)

    Returns:
        Tuple of (file_handle, csv_writer)
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Validate param_list is not empty
    if not param_list:
        raise ValueError("Parameter list cannot be empty")

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
        "hip_width_cm",
        "head_width_cm",
        "neck_length_cm",
        "upper_arm_length_cm",
        "forearm_length_cm",
        "hand_length_cm",
        "upper_leg_length_cm",
        "lower_leg_length_cm",
        "foot_length_cm"
    ]

    columns = param_columns + measurement_columns

    # Open CSV file with larger buffer (8MB) to reduce I/O operations
    # buffering=-1 uses default system buffer size (typically 4-8KB)
    # We use a larger 8MB buffer for better performance with large files
    csv_file = open(output_path, 'w', newline='', buffering=8*1024*1024)
    csv_writer = csv.DictWriter(csv_file, fieldnames=columns)
    csv_writer.writeheader()

    print(f"✓ Initialized CSV: {output_path}")
    print(f"  Columns: {len(columns)}")
    print(f"    Parameters: {len(param_columns)} (6 fixed + 5 grid)")
    print(f"    Measurements: {len(measurement_columns)} (8 upper body + 3 lower body)")

    return csv_file, csv_writer


def process_single_model(params: dict, rig_type: str, quiet: bool = True) -> dict:
    """
    Generate model, measure, and return measurements.

    Args:
        params: Parameter dictionary
        rig_type: Type of rig to add
        quiet: If True, suppress verbose output from measurements

    Returns:
        Dictionary with measurements
    """
    import bpy
    import importlib

    # Get MPFB module path
    mpfb_path = utils._get_mpfb_module_path()
    HumanService = importlib.import_module(f'{mpfb_path}.services.humanservice').HumanService

    # Suppress output from Blender operations and measurements if in quiet mode
    if quiet:
        # Save original streams and file descriptors
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_stdout_fd = os.dup(1)  # Duplicate stdout file descriptor
        old_stderr_fd = os.dup(2)  # Duplicate stderr file descriptor

        # Open devnull and redirect at OS level
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)  # Redirect stdout to devnull
        os.dup2(devnull_fd, 2)  # Redirect stderr to devnull

        # Also redirect Python's sys.stdout/stderr
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull

    try:
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

    finally:
        if quiet:
            # Restore file descriptors first (OS level)
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            os.close(devnull_fd)

            # Restore Python streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            devnull.close()

    return measured, basemesh, armature


def format_time(seconds):
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_progress_bar(current, total, start_time, successful, failed, bar_length=50):
    """
    Print a simple text-based progress bar.

    Args:
        current: Current iteration number
        total: Total number of iterations
        start_time: Start time (from time.time())
        successful: Number of successful models
        failed: Number of failed models
        bar_length: Length of the progress bar in characters
    """
    # Calculate progress
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)

    # Calculate timing
    elapsed = time.time() - start_time
    if current > 0:
        avg_time_per_model = elapsed / current
        remaining_models = total - current
        estimated_remaining = avg_time_per_model * remaining_models
    else:
        estimated_remaining = 0.9 * total  # Initial estimate: 0.9 seconds per model

    # Format the progress bar
    percentage = progress * 100
    elapsed_str = format_time(elapsed)
    remaining_str = format_time(estimated_remaining)

    # Build progress text
    progress_text = (f'Progress: |{bar}| {current}/{total} ({percentage:.1f}%) '
                     f'[Elapsed: {elapsed_str}, ETA: {remaining_str}] '
                     f'[Success: {successful}, Failed: {failed}]')

    # Clear the entire line first, then write progress (no newline!)
    # Use a consistent line width to prevent flickering
    line_width = 150
    padded_text = progress_text.ljust(line_width)

    sys.stdout.write(f'\r{padded_text}')
    sys.stdout.flush()


def write_measurement_row(csv_writer, params: dict, measured: dict, csv_file=None, row_number=None):
    """
    Write a single row to CSV.

    Args:
        csv_writer: CSV writer object
        params: Parameter dictionary (includes both fixed and grid params)
        measured: Measurements dictionary
        csv_file: Optional file handle for flushing
        row_number: Optional row number for debugging
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
        "hip_width_cm": measured["hip_width"],
        "head_width_cm": measured["head_width"],
        "neck_length_cm": measured["neck_length"],
        "upper_arm_length_cm": measured["upper_arm_length"],
        "forearm_length_cm": measured["forearm_length"],
        "hand_length_cm": measured["hand_length"],
        "upper_leg_length_cm": measured["upper_leg_length"],
        "lower_leg_length_cm": measured["lower_leg_length"],
        "foot_length_cm": measured["foot_length"]
    }

    csv_writer.writerow(row)

    # Debug: verify write every 1000 rows
    if row_number and row_number % 1000 == 0 and csv_file:
        csv_file.flush()
        # Write to stderr so it's not suppressed
        sys.stderr.write(f"\n[DEBUG] Wrote row {row_number} to CSV\n")
        sys.stderr.flush()


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
    print(f"  Parameter list: {args.param_list}")
    print(f"  Output: {args.output}")
    print(f"  Rig type: {args.rig_type}")
    print(f"  Delete models: {not args.no_delete}")
    print(f"  Checkpoint interval: {args.checkpoint_interval}")

    try:
        # Load parameter list
        print("\n" + "-"*70)
        print("LOADING PARAMETER COMBINATIONS")
        print("-"*70)
        param_list = load_parameter_list(args.param_list)

        print(f"\nWill process {len(param_list):,} models")

        # Initialize CSV
        print("\n" + "-"*70)
        print("INITIALIZING OUTPUT")
        print("-"*70)
        csv_file, csv_writer = initialize_csv(args.output, param_list)

        # Set up Blender scene
        print("\n" + "-"*70)
        print("SETTING UP BLENDER SCENE")
        print("-"*70)
        utils.setup_blender_scene()

        # Process each model
        print("\n" + "="*70)
        print("PROCESSING MODELS")
        print("="*70 + "\n")

        successful = 0
        failed = 0
        start_time = time.time()
        total = len(param_list)

        # Process each model with progress bar
        for i, params in enumerate(param_list, 1):
            try:
                # Generate and measure (quiet mode with OS-level suppression)
                measured, basemesh, armature = process_single_model(params, args.rig_type, quiet=True)

                # Write to CSV
                write_measurement_row(csv_writer, params, measured, csv_file, i)

                # Cleanup (if not debugging)
                if not args.no_delete:
                    cleanup_scene(basemesh, armature)

                successful += 1

                # Update progress bar on every iteration
                print_progress_bar(i, total, start_time, successful, failed)

                # Flush CSV every 100 rows
                if i % 100 == 0:
                    csv_file.flush()

                # Memory cleanup every 100 models
                if i % 100 == 0:
                    # Force garbage collection to prevent memory buildup
                    gc.collect()

                    # Clear orphaned Blender data blocks (safe in headless mode)
                    import bpy
                    for block_type in [bpy.data.meshes, bpy.data.armatures, bpy.data.materials,
                                       bpy.data.textures, bpy.data.images]:
                        for block in list(block_type):  # Use list() to avoid iterator issues
                            if block.users == 0:
                                try:
                                    block_type.remove(block)
                                except:
                                    pass

                    # Clear current line, print checkpoint message on new line
                    sys.stdout.write('\r' + ' ' * 150 + '\r')  # Clear line
                    sys.stdout.write(f"✓ Memory cleanup: {i}/{total} models processed\n")
                    sys.stdout.flush()

            except Exception as e:
                failed += 1
                # Clear current line, print error message on new line
                sys.stdout.write('\r' + ' ' * 150 + '\r')  # Clear line
                sys.stdout.write(f"✗ Error processing model {i}: {e}\n")
                sys.stdout.flush()

                if not args.no_delete:
                    # Try to clean up anyway
                    try:
                        import bpy
                        bpy.ops.object.select_all(action='SELECT')
                        bpy.ops.object.delete()
                    except:
                        pass

        # Print newline after progress bar is complete
        print()
        
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