"""
Build Microparameter Sensitivity Table

This script generates a sensitivity analysis table for microparameters (MPFB2 targets)
to determine how each microparameter affects the 8 body measurements.

Approach:
1. Set all macroparameters to baseline (0.5)
2. For each microparameter, test at 2001 values [0.0000, 0.0005, ..., 1.0000]
3. Extract all 8 measurements for each configuration
4. Calculate sensitivity: Δmeasurement / Δmicroparameter
5. Save results to CSV with streaming writes (memory efficient)

Total samples: 10 microparameters × 2001 values = 20,010 meshes
Output: micro_sensitivity_table.csv, micro_sensitivity_matrix.json
"""

import sys
import os
import json
import csv
import argparse
import importlib
import gc
import time
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from measurement_functions.measurements import (
    measure_height,
    measure_shoulder_width,
    measure_hip_width,
    measure_head_width,
    measure_neck_length,
    measure_upper_arm_length,
    measure_forearm_length,
    measure_hand_length
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# 8 measurements to track
MEASUREMENTS = [
    'height_cm',
    'shoulder_width_cm',
    'hip_width_cm',
    'head_width_cm',
    'neck_length_cm',
    'upper_arm_length_cm',
    'forearm_length_cm',
    'hand_length_cm'
]

# Baseline macroparameters (configurable via command line)
# Defaults: Neutral values for body, customizable gender/race
DEFAULT_BASELINE_MACROS = {
    'age': 0.5,
    'height': 0.5,
    'muscle': 0.5,
    'weight': 0.5,
    'proportions': 0.5,
    'gender': 0.0,  # 0.0 = female, 1.0 = male
    'race': {
        'asian': 0.33,
        'caucasian': 0.34,
        'african': 0.33
    }
}

# Microparameters to test (mapping to your 8 measurements)
MICROPARAMETERS = {
    'measure-upperleg-height-incr': 'height_cm_component_1',
    'measure-lowerleg-height-incr': 'height_cm_component_2',
    'measure-napetowaist-dist-incr': 'height_cm_component_3',
    'measure-shoulder-dist-incr': 'shoulder_width_cm',
    'measure-hips-circ-incr': 'hip_width_cm',
    'head-scale-horiz-incr': 'head_width_cm',
    'neck-scale-vert-incr': 'neck_length_cm',
    'measure-upperarm-length-incr': 'upper_arm_length_cm',
    'measure-lowerarm-length-incr': 'forearm_length_cm',
    'l-hand-scale-incr': 'hand_length_cm',
}

# Test values for each microparameter (2001 values from 0.0 to 1.0)
MICRO_TEST_VALUES = [i / 2000.0 for i in range(2001)]  # 0.0000, 0.0005, ..., 1.0000

# Output directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / 'micro_sensitivity_table.csv'

# ============================================================================
# PROGRESS BAR FUNCTIONS
# ============================================================================

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


# ============================================================================
# BLENDER/MPFB2 SETUP
# ============================================================================

def get_mpfb_module_path():
    """Determine the correct MPFB module path for the current Blender version."""
    if 'bl_ext.user_default.mpfb' in sys.modules:
        return 'bl_ext.user_default.mpfb'
    if 'mpfb' in sys.modules:
        return 'mpfb'

    try:
        import bl_ext.user_default.mpfb
        return 'bl_ext.user_default.mpfb'
    except ImportError:
        return 'mpfb'


def create_human_with_macro_and_micro(macros: Dict[str, float],
                                       micros: Dict[str, float]) -> Tuple['bpy.types.Object', 'bpy.types.Object']:
    """
    Create a human mesh with specified macro and microparameters.

    Args:
        macros: Dictionary of macroparameter values
        micros: Dictionary of microparameter names and values

    Returns:
        Tuple of (mesh object, armature object)
    """
    import bpy

    mpfb_path = get_mpfb_module_path()

    # Import MPFB2 services
    HumanService = importlib.import_module(f'{mpfb_path}.services.humanservice').HumanService
    TargetService = importlib.import_module(f'{mpfb_path}.services.targetservice').TargetService
    HumanObjectProperties = importlib.import_module(f'{mpfb_path}.entities.objectproperties').HumanObjectProperties

    # Create base human
    basemesh = HumanService.create_human()

    # Apply macroparameters
    for param, value in macros.items():
        # Handle race specially - it's a dict that needs to be flattened
        if param == 'race' and isinstance(value, dict):
            for race_component, race_value in value.items():
                HumanObjectProperties.set_value(race_component, race_value, entity_reference=basemesh)
        else:
            HumanObjectProperties.set_value(param, value, entity_reference=basemesh)

    # Apply macro details (creates shape keys for macros)
    TargetService.reapply_macro_details(basemesh)

    # Apply microparameters (targets)
    for micro_name, micro_value in micros.items():
        try:
            # Get full path for target
            full_path = TargetService.target_full_path(micro_name)

            if full_path:
                # Load target with specified value
                TargetService.load_target(basemesh, full_path, weight=micro_value, name=micro_name)
                print(f"  ✓ Loaded {micro_name} = {micro_value}")
            else:
                print(f"  ✗ Warning: Could not find target {micro_name}")
        except Exception as e:
            print(f"  ✗ Error loading {micro_name}: {e}")

    # Bake all targets to apply them permanently
    TargetService.bake_targets(basemesh)

    # Add rig for measurements
    armature = HumanService.add_builtin_rig(basemesh, rig_name="default", import_weights=True)

    return basemesh, armature


def extract_measurements(armature) -> Dict[str, float]:
    """
    Extract all 8 measurements from an armature object.

    Args:
        armature: Blender armature object

    Returns:
        Dictionary of measurement name to value (in cm)
    """
    measurements = {}

    try:
        measurements['height_cm'] = measure_height(armature)
        measurements['shoulder_width_cm'] = measure_shoulder_width(armature)
        measurements['hip_width_cm'] = measure_hip_width(armature)
        measurements['head_width_cm'] = measure_head_width(armature)
        measurements['neck_length_cm'] = measure_neck_length(armature)
        measurements['upper_arm_length_cm'] = measure_upper_arm_length(armature)
        measurements['forearm_length_cm'] = measure_forearm_length(armature)
        measurements['hand_length_cm'] = measure_hand_length(armature)
    except Exception as e:
        print(f"Error extracting measurements: {e}")
        raise

    return measurements


def cleanup_mesh_and_armature(mesh_obj, armature_obj):
    """
    Delete mesh and armature objects from scene and clear orphaned data blocks.

    This is critical for batch processing to prevent memory leaks.
    Blender doesn't automatically remove mesh/armature data blocks when
    objects are deleted, causing RAM to fill up during long batch runs.

    Args:
        mesh_obj: Mesh object to delete
        armature_obj: Armature object to delete
    """
    import bpy

    # Store references to data blocks before deleting objects
    mesh_data = mesh_obj.data if mesh_obj else None
    armature_data = armature_obj.data if armature_obj else None

    # Delete objects
    bpy.ops.object.select_all(action='DESELECT')

    if mesh_obj:
        mesh_obj.select_set(True)
    if armature_obj:
        armature_obj.select_set(True)

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


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def build_sensitivity_table(baseline_macros):
    """
    Build microparameter sensitivity table.

    Args:
        baseline_macros: Dictionary of macroparameter baseline values

    Process:
    1. Generate baseline mesh (all macros at baseline, no micros)
    2. For each microparameter:
       - Test at 2001 values from 0.0 to 1.0
       - Extract all 8 measurements
       - Calculate sensitivity (Δmeasurement / Δmicro)
    3. Save results to CSV (streaming write to avoid memory issues)
    """
    import bpy

    print("\n" + "="*80)
    print("MICROPARAMETER SENSITIVITY ANALYSIS")
    print("="*80)

    # Step 1: Generate baseline measurements (micro=0.5 for all)
    print("\nStep 1: Generating baseline measurements...")
    print(f"  Macroparameters: {baseline_macros}")

    baseline_micros = {micro: 0.5 for micro in MICROPARAMETERS.keys()}
    baseline_mesh, baseline_armature = create_human_with_macro_and_micro(baseline_macros, baseline_micros)
    baseline_measurements = extract_measurements(baseline_armature)

    print("\n  Baseline measurements:")
    for measure, value in baseline_measurements.items():
        print(f"    {measure}: {value:.2f} cm")

    cleanup_mesh_and_armature(baseline_mesh, baseline_armature)

    # Step 2: Setup CSV file for streaming writes
    print(f"\nStep 2: Setting up CSV file at {OUTPUT_FILE}")

    fieldnames = ['microparameter', 'target_measurement', 'micro_value'] + MEASUREMENTS + \
                 [f'{m}_delta' for m in MEASUREMENTS]

    # Step 3: Test each microparameter with progress tracking
    print(f"\nStep 3: Testing {len(MICROPARAMETERS)} microparameters × {len(MICRO_TEST_VALUES)} values = {len(MICROPARAMETERS) * len(MICRO_TEST_VALUES)} total samples")
    print("")  # Blank line for progress bar

    total_samples = len(MICROPARAMETERS) * len(MICRO_TEST_VALUES)
    sample_count = 0
    successful = 0
    failed = 0
    start_time = time.time()

    # Open CSV file for streaming writes
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()  # Flush header immediately

        for micro_name, target_measurement in MICROPARAMETERS.items():
            for micro_value in MICRO_TEST_VALUES:
                sample_count += 1

                try:
                    # Suppress verbose output from Blender operations
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    old_stdout_fd = os.dup(1)
                    old_stderr_fd = os.dup(2)

                    devnull_fd = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull_fd, 1)
                    os.dup2(devnull_fd, 2)

                    devnull = open(os.devnull, 'w')
                    sys.stdout = devnull
                    sys.stderr = devnull

                    try:
                        # Create micros dict with only this micro active
                        test_micros = {micro_name: micro_value}

                        # Generate mesh and armature (suppressed output)
                        test_mesh, test_armature = create_human_with_macro_and_micro(baseline_macros, test_micros)

                        # Extract measurements (suppressed output)
                        measurements = extract_measurements(test_armature)

                    finally:
                        # Restore output streams
                        os.dup2(old_stdout_fd, 1)
                        os.dup2(old_stderr_fd, 2)
                        os.close(old_stdout_fd)
                        os.close(old_stderr_fd)
                        os.close(devnull_fd)

                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                        devnull.close()

                    # Calculate deltas from baseline
                    deltas = {
                        measure: measurements[measure] - baseline_measurements[measure]
                        for measure in MEASUREMENTS
                    }

                    # Write result directly to CSV (streaming)
                    result = {
                        'microparameter': micro_name,
                        'target_measurement': target_measurement,
                        'micro_value': micro_value,
                        **measurements,
                        **{f'{m}_delta': deltas[m] for m in MEASUREMENTS}
                    }
                    writer.writerow(result)

                    # Cleanup
                    cleanup_mesh_and_armature(test_mesh, test_armature)

                    successful += 1

                    # Update progress bar
                    print_progress_bar(sample_count, total_samples, start_time, successful, failed)

                    # Flush CSV every 100 rows
                    if sample_count % 100 == 0:
                        csvfile.flush()

                    # Memory cleanup every 100 models
                    if sample_count % 100 == 0:
                        # Clear orphaned Blender data blocks
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
                        sys.stdout.write(f"✓ Memory cleanup: {sample_count}/{total_samples} samples processed\n")
                        sys.stdout.flush()

                except Exception as e:
                    failed += 1
                    # Clear current line, print error message on new line
                    sys.stdout.write('\r' + ' ' * 150 + '\r')  # Clear line
                    sys.stdout.write(f"✗ Error processing sample {sample_count} ({micro_name}={micro_value}): {e}\n")
                    sys.stdout.flush()

                    # Try to clean up anyway
                    try:
                        bpy.ops.object.select_all(action='SELECT')
                        bpy.ops.object.delete()
                        gc.collect()
                    except:
                        pass

    # Final newline after progress bar
    print("")

    # Step 4: Calculate and display sensitivity coefficients
    print(f"\n✓ Saved {successful} samples to {OUTPUT_FILE}")

    if successful == 0:
        print("  ✗ No results to process")
        return

    print("\nStep 4: Calculating sensitivity coefficients from CSV...")
    print("\nSensitivity Matrix (Δcm per Δmicro unit):")
    print("-" * 80)
    print(f"{'Microparameter':<35} {'Target Meas.':<20} {'Sensitivity':<15}")
    print("-" * 80)

    sensitivity_matrix = {}

    # Read CSV file to calculate sensitivities
    with open(OUTPUT_FILE, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        all_results = list(reader)

    for micro_name in MICROPARAMETERS.keys():
        # Get results for this micro at 0.0 and 1.0
        micro_results = [r for r in all_results if r['microparameter'] == micro_name]
        result_0 = next((r for r in micro_results if float(r['micro_value']) == 0.0), None)
        result_1 = next((r for r in micro_results if float(r['micro_value']) == 1.0), None)

        if not result_0 or not result_1:
            print(f"⚠ Warning: Missing data for {micro_name}, skipping sensitivity calculation")
            continue

        # Calculate sensitivity for each measurement
        sensitivities = {}
        for measure in MEASUREMENTS:
            delta_measurement = float(result_1[measure]) - float(result_0[measure])
            delta_micro = 1.0 - 0.0
            sensitivity = delta_measurement / delta_micro
            sensitivities[measure] = sensitivity

        sensitivity_matrix[micro_name] = sensitivities

        # Show primary target sensitivity
        target_measure = MICROPARAMETERS[micro_name].split('_component')[0]
        primary_sensitivity = sensitivities[target_measure]
        print(f"{micro_name:<35} {target_measure:<20} {primary_sensitivity:+.2f} cm/unit")

    # Save sensitivity matrix to JSON
    sensitivity_file = OUTPUT_DIR / 'micro_sensitivity_matrix.json'
    with open(sensitivity_file, 'w') as f:
        json.dump(sensitivity_matrix, f, indent=2)

    print(f"\n✓ Sensitivity matrix saved to {sensitivity_file}")

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. {OUTPUT_FILE} - Raw measurement data ({successful} samples)")
    print(f"  2. {sensitivity_file} - Sensitivity coefficients")

    # Calculate total time
    total_time = time.time() - start_time
    print(f"\nTotal time: {format_time(total_time)}")
    print(f"Average time per sample: {total_time/successful:.2f}s") if successful > 0 else None


# ============================================================================
# MAIN
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    # Find where Blender arguments end and script arguments begin
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description='Build microparameter sensitivity table',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Asian female baseline
    python run_blender.py --script build_micro_sensitivity_table.py -- --gender 0.0 --asian 1.0

    # Caucasian male baseline
    python run_blender.py --script build_micro_sensitivity_table.py -- --gender 1.0 --caucasian 1.0

    # Mixed race female
    python run_blender.py --script build_micro_sensitivity_table.py -- --gender 0.0 --asian 0.5 --caucasian 0.5
        """
    )

    parser.add_argument('--gender', type=float, default=0.0,
                        help='Gender baseline (0.0=female, 1.0=male, default: 0.0)')
    parser.add_argument('--asian', type=float, default=0.33,
                        help='Asian race proportion (0.0-1.0, default: 0.33)')
    parser.add_argument('--caucasian', type=float, default=0.34,
                        help='Caucasian race proportion (0.0-1.0, default: 0.34)')
    parser.add_argument('--african', type=float, default=0.33,
                        help='African race proportion (0.0-1.0, default: 0.33)')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Suffix for output files (e.g., "_asian_female")')

    return parser.parse_args(argv)


def main():
    """Main execution function."""
    import bpy

    # Parse arguments
    args = parse_arguments()

    # Build baseline macros from arguments
    baseline_macros = {
        'age': 0.5,
        'height': 0.5,
        'muscle': 0.5,
        'weight': 0.5,
        'proportions': 0.5,
        'gender': args.gender,
        'race': {
            'asian': args.asian,
            'caucasian': args.caucasian,
            'african': args.african
        }
    }

    # Update output file names with suffix if provided
    global OUTPUT_FILE
    if args.output_suffix:
        OUTPUT_FILE = OUTPUT_DIR / f'micro_sensitivity_table{args.output_suffix}.csv'

    print("\n" + "="*80)
    print("MICROPARAMETER SENSITIVITY TABLE BUILDER")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Gender: {args.gender} ({'female' if args.gender < 0.5 else 'male'})")
    print(f"  Race: Asian={args.asian}, Caucasian={args.caucasian}, African={args.african}")
    print(f"  Microparameters to test: {len(MICROPARAMETERS)}")
    print(f"  Test values per micro: {len(MICRO_TEST_VALUES)}")
    print(f"  Total samples: {len(MICROPARAMETERS) * len(MICRO_TEST_VALUES)}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Verify we're in Blender
    try:
        import bpy
        print(f"\n✓ Running in Blender {bpy.app.version_string}")
    except ImportError:
        print("\n✗ ERROR: This script must be run inside Blender")
        print("Usage: python run_blender.py --script build_micro_sensitivity_table.py -- [options]")
        sys.exit(1)

    # Verify MPFB2 is available
    mpfb_path = get_mpfb_module_path()
    print(f"✓ MPFB2 module path: {mpfb_path}")

    # Run sensitivity analysis
    build_sensitivity_table(baseline_macros)


if __name__ == "__main__":
    main()
