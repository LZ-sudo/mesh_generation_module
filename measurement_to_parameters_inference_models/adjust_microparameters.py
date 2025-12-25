#!/usr/bin/env python3
"""
Iterative microparameter adjustment to match target measurements.

This script takes target measurements and macroparameters, then iteratively
adjusts microparameters to match the target measurements as closely as possible.

Usage:
    Run via Blender subprocess from test_model_accuracy.py
"""

import sys
import json
import gc
from pathlib import Path
from typing import Dict, Tuple, Callable
import importlib

# Add parent directory to path for imports
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


# ============================================================================
# CONFIGURATION
# ============================================================================

# Microparameter adjustment settings
MAX_ITERATIONS_PER_CATEGORY = 20
TOLERANCE_CM = 0.01  # 0.01 cm = within 2 decimal places
MICRO_MIN = -1.0  # Microparameters can be negative (use -decr targets)
MICRO_MAX = 1.0   # Microparameters can be positive (use -incr targets)
INITIAL_DAMPING = 0.5
MIN_DAMPING = 0.1
DAMPING_DECAY = 0.9

# Microparameter categories
# Each measurement is associated with specific microparameters (base names only)
MICROPARAMETER_CATEGORIES = {
    'height_cm': [
        'measure-lowerleg-height',
        'measure-upperleg-height',
        'measure-napetowaist-dist',
        'measure-waisttohip-dist'
    ],
    'shoulder_width_cm': [
        'torso-scale-horiz',
        'measure-waist-circ'
    ],
    'hip_width_cm': [
        'measure-waist-circ',
        'measure-waisttohip-dist'
    ],
    'head_width_cm': [
        'head-scale-horiz',
        'head-scale-depth'
    ],
    'neck_length_cm': [
        'measure-neck-height'
    ],
    'upper_arm_length_cm': [
        'measure-upperarm-length'
    ],
    'forearm_length_cm': [
        'measure-lowerarm-length'
    ],
    'hand_length_cm': [
        'measure-hand-length'
    ]
}

# Category processing order (adjust height first, then widths, then extremities)
CATEGORY_ORDER = [
    'height_cm',
    'shoulder_width_cm',
    'hip_width_cm',
    'head_width_cm',
    'neck_length_cm',
    'upper_arm_length_cm',
    'forearm_length_cm',
    'hand_length_cm'
]


# ============================================================================
# MEASUREMENT FUNCTIONS
# ============================================================================

def extract_measurements_with_cm_suffix(mesh, armature) -> Dict[str, float]:
    """
    Extract all measurements and add _cm suffix to match model format.

    Args:
        mesh: Blender mesh object
        armature: Blender armature object

    Returns:
        Dictionary with measurements like {'height_cm': 165.0, 'shoulder_width_cm': 38.5, ...}
    """
    from measurement_functions.measurements import extract_all_measurements

    # Extract measurements (returns keys like 'height', 'shoulder_width', etc.)
    raw_measurements = extract_all_measurements(mesh, armature)

    # Convert to _cm suffix format for consistency with model
    measurements_cm = {}
    for key, value in raw_measurements.items():
        measurements_cm[f"{key}_cm"] = value

    return measurements_cm


def create_measurement_extractor(measurement_name: str) -> Callable:
    """
    Create a measurement extraction function for a specific measurement.

    Args:
        measurement_name: Name like 'height_cm', 'shoulder_width_cm', etc.

    Returns:
        Function that takes an armature and returns the measurement value
    """
    def extract_single_measurement(armature):
        # Get the basemesh from the armature
        import bpy

        # Find the human mesh (should be parented to armature)
        mesh = None
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.parent == armature:
                mesh = obj
                break

        if not mesh:
            raise RuntimeError("Could not find mesh object for measurement")

        # Extract all measurements
        all_measurements = extract_measurements_with_cm_suffix(mesh, armature)

        # Return the specific measurement
        return all_measurements.get(measurement_name, 0.0)

    return extract_single_measurement


# ============================================================================
# MPFB2 UTILITIES
# ============================================================================

def get_mpfb_module_path():
    """
    Determine the correct MPFB module path for the current Blender version.

    Returns:
        str: Either 'bl_ext.user_default.mpfb' for Blender 4.2+ or 'mpfb' for older versions
    """
    import sys

    # Check if using new extension system (Blender 4.2+)
    if 'bl_ext.user_default.mpfb' in sys.modules:
        return 'bl_ext.user_default.mpfb'

    # Check if using legacy addon system
    if 'mpfb' in sys.modules:
        return 'mpfb'

    # Try importing to detect which format is available
    try:
        import bl_ext.user_default.mpfb
        return 'bl_ext.user_default.mpfb'
    except ImportError:
        # Fall back to legacy format
        return 'mpfb'


def apply_microparameters_to_mesh(basemesh, micros: Dict[str, float], verbose: bool = False):
    """
    Apply microparameters to mesh using MPFB2 target system.

    Args:
        basemesh: The Blender mesh object
        micros: Dictionary of base microparameter names to values in [-1.0, 1.0]
                Example: {'measure-shoulder-dist': 0.5, 'measure-lowerleg-height': -0.3}
        verbose: Whether to print progress
    """
    mpfb_path = get_mpfb_module_path()
    TargetService = importlib.import_module(f'{mpfb_path}.services.targetservice').TargetService

    for micro_name, micro_value in micros.items():
        # Skip zero values (no effect)
        if abs(micro_value) < 0.001:
            continue

        # Determine target suffix based on sign
        if micro_value >= 0:
            target_name = f"{micro_name}-incr"
            target_weight = micro_value
        else:
            target_name = f"{micro_name}-decr"
            target_weight = abs(micro_value)

        # Get full path and load target
        full_path = TargetService.target_full_path(target_name)
        if full_path:
            TargetService.load_target(basemesh, full_path, weight=target_weight, name=target_name)
            if verbose:
                print(f"  Applied {target_name}: {target_weight:.3f}")
        else:
            if verbose:
                print(f"  ✗ Warning: Could not find target {target_name}")


def convert_micros_to_output_format(micros: Dict[str, float]) -> Dict[str, float]:
    """
    Convert internal microparameter format to output format for generate_human.py.

    Internal format: {'measure-shoulder-dist': 0.5, 'measure-lowerleg-height': -0.3}
    Output format: {'measure-shoulder-dist-incr': 0.5, 'measure-lowerleg-height-decr': 0.3}

    Args:
        micros: Dictionary with base names and signed values

    Returns:
        Dictionary with suffixed names and positive values
    """
    output = {}
    for micro_name, micro_value in micros.items():
        # Skip zero values
        if abs(micro_value) < 0.001:
            continue

        if micro_value >= 0:
            output[f"{micro_name}-incr"] = micro_value
        else:
            output[f"{micro_name}-decr"] = abs(micro_value)

    return output


# ============================================================================
# MESH CREATION AND CLEANUP
# ============================================================================

def create_human_with_parameters(
    macros: Dict[str, float],
    micros: Dict[str, float],
    rig_type: str = 'default'
) -> Tuple:
    """
    Create a human mesh with specified macro and microparameters.

    Args:
        macros: Macroparameter dictionary
        micros: Microparameter dictionary (base names with signed values)
        rig_type: Type of rig to add

    Returns:
        Tuple of (mesh object, armature object)
    """
    import bpy

    mpfb_path = get_mpfb_module_path()
    HumanService = importlib.import_module(f'{mpfb_path}.services.humanservice').HumanService
    TargetService = importlib.import_module(f'{mpfb_path}.services.targetservice').TargetService
    HumanObjectProperties = importlib.import_module(f'{mpfb_path}.entities.objectproperties').HumanObjectProperties

    # Create base human
    basemesh = HumanService.create_human()

    # Apply macroparameters via HumanObjectProperties
    for param, value in macros.items():
        if param == 'race' and isinstance(value, dict):
            for race_component, race_value in value.items():
                HumanObjectProperties.set_value(race_component, race_value, entity_reference=basemesh)
        else:
            HumanObjectProperties.set_value(param, value, entity_reference=basemesh)

    # Apply macro details (creates shape keys for macros)
    TargetService.reapply_macro_details(basemesh)

    # Apply microparameters if provided
    if micros:
        apply_microparameters_to_mesh(basemesh, micros, verbose=False)

    # Bake all targets (macros + micros) to apply them permanently
    TargetService.bake_targets(basemesh)

    # Update scene
    bpy.context.view_layer.update()
    basemesh.data.update()

    # Add rig for measurements
    armature = HumanService.add_builtin_rig(basemesh, rig_name=rig_type, import_weights=True)

    return basemesh, armature


def cleanup_mesh_and_armature(mesh_obj, armature_obj):
    """Clean up mesh and armature to prevent memory leaks."""
    import bpy

    # Store references to data blocks
    mesh_data = mesh_obj.data if mesh_obj else None
    armature_data = armature_obj.data if armature_obj else None

    # Delete objects
    bpy.ops.object.select_all(action='DESELECT')
    if mesh_obj:
        mesh_obj.select_set(True)
    if armature_obj:
        armature_obj.select_set(True)
    bpy.ops.object.delete()

    # Remove orphaned data blocks
    if mesh_data and mesh_data.users == 0:
        bpy.data.meshes.remove(mesh_data)
    if armature_data and armature_data.users == 0:
        bpy.data.armatures.remove(armature_data)

    # Clear orphaned data
    for block_type in [bpy.data.meshes, bpy.data.armatures, bpy.data.materials,
                       bpy.data.textures, bpy.data.images]:
        for block in list(block_type):
            if block.users == 0:
                block_type.remove(block)

    gc.collect()


# ============================================================================
# MICROPARAMETER ADJUSTMENT LOGIC
# ============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max bounds."""
    return max(min_val, min(max_val, value))


def adjust_category(
    category_name: str,
    target_measurement: float,
    current_micros: Dict[str, float],
    macros: Dict[str, float],
    rig_type: str = 'default',
    verbose: bool = True
) -> Tuple[Dict[str, float], bool, int]:
    """
    Adjust microparameters for a single measurement category.

    Args:
        category_name: Name of the measurement (e.g., 'height_cm')
        target_measurement: Target value for this measurement
        current_micros: Current microparameter values (base names with signed values)
        macros: Macroparameter values
        rig_type: Type of rig to use
        verbose: Whether to print detailed progress

    Returns:
        Tuple of (updated_micros, converged, iterations)
    """
    if verbose:
        print(f"\n{'-'*80}")
        print(f"Adjusting: {category_name}")
        print(f"Target: {target_measurement:.2f} cm")
        print(f"{'-'*80}")

    micro_names = MICROPARAMETER_CATEGORIES[category_name]
    measure_func = create_measurement_extractor(category_name)

    # Initialize microparameters for this category if not present
    for micro_name in micro_names:
        if micro_name not in current_micros:
            current_micros[micro_name] = 0.0

    iteration = 0
    damping = INITIAL_DAMPING
    converged = False
    previous_error = None

    while iteration < MAX_ITERATIONS_PER_CATEGORY:
        iteration += 1

        # Generate mesh with current microparameters
        mesh, armature = create_human_with_parameters(macros, current_micros, rig_type)

        # Measure
        current_measurement = measure_func(armature)
        error = current_measurement - target_measurement

        if verbose:
            print(f"  Iteration {iteration}: Current={current_measurement:.4f} cm, "
                  f"Error={error:+.4f} cm, Damping={damping:.2f}")

        # Check convergence
        if abs(error) <= TOLERANCE_CM:
            if verbose:
                print(f"  ✓ Converged! Final measurement: {current_measurement:.4f} cm")
            converged = True
            cleanup_mesh_and_armature(mesh, armature)
            break

        # Estimate sensitivity using empirical gradient
        test_micros = current_micros.copy()
        test_delta = 0.1  # Small test increment

        for micro_name in micro_names:
            test_micros[micro_name] = clamp(
                current_micros[micro_name] + test_delta,
                MICRO_MIN, MICRO_MAX
            )

        # Cleanup current mesh before generating test mesh
        cleanup_mesh_and_armature(mesh, armature)

        # Generate test mesh
        test_mesh, test_armature = create_human_with_parameters(macros, test_micros, rig_type)
        test_measurement = measure_func(test_armature)
        cleanup_mesh_and_armature(test_mesh, test_armature)

        # Calculate sensitivity (gradient)
        sensitivity = (test_measurement - current_measurement) / test_delta

        if abs(sensitivity) < 0.0001:
            if verbose:
                print(f"  ✗ Sensitivity too low ({sensitivity:.6f}), cannot adjust further")
            break

        # Calculate adjustment using damped proportional control
        adjustment = -damping * error / sensitivity

        # Apply adjustment to all microparameters in this category equally
        for micro_name in micro_names:
            new_value = clamp(
                current_micros[micro_name] + adjustment,
                MICRO_MIN, MICRO_MAX
            )
            current_micros[micro_name] = new_value

        # Adaptive damping
        if previous_error is not None and abs(error) > abs(previous_error):
            # Error increased, reduce damping
            damping = max(MIN_DAMPING, damping * DAMPING_DECAY)

        previous_error = error

    if not converged and verbose:
        print(f"  ⚠ Did not converge after {MAX_ITERATIONS_PER_CATEGORY} iterations")

    return current_micros, converged, iteration


def adjust_all_microparameters(
    target_measurements: Dict[str, float],
    macroparameters: Dict[str, float],
    rig_type: str = 'default',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Adjust all microparameters to match target measurements.

    Args:
        target_measurements: Dictionary of target measurements
        macroparameters: Dictionary of macroparameters
        rig_type: Type of rig to use
        verbose: Whether to print progress

    Returns:
        Dictionary of adjusted microparameters in output format (with -incr/-decr suffixes)
    """
    if verbose:
        print("\n" + "="*80)
        print("MICROPARAMETER ADJUSTMENT")
        print("="*80)

    # Initialize microparameters (base names with signed values)
    microparameters = {}

    # Convergence tracking
    convergence_summary = {}

    # Process each category in order
    for category in CATEGORY_ORDER:
        if category not in target_measurements:
            if verbose:
                print(f"\nSkipping {category} (no target measurement provided)")
            continue

        target = target_measurements[category]
        microparameters, converged, iterations = adjust_category(
            category, target, microparameters, macroparameters, rig_type, verbose
        )
        convergence_summary[category] = {
            'converged': converged,
            'iterations': iterations
        }

    # Final verification
    if verbose:
        print("\n" + "="*80)
        print("FINAL VERIFICATION")
        print("="*80)

    final_mesh, final_armature = create_human_with_parameters(macroparameters, microparameters, rig_type)
    final_measurements = extract_measurements_with_cm_suffix(final_mesh, final_armature)
    cleanup_mesh_and_armature(final_mesh, final_armature)

    if verbose:
        print("\nFinal Results:")
        print(f"  {'Measurement':<25s} {'Target':>12s} {'Actual':>12s} {'Error':>12s} {'Status':>10s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

        all_converged = True
        for category in CATEGORY_ORDER:
            if category not in target_measurements:
                continue

            target = target_measurements[category]
            actual = final_measurements[category]
            error = actual - target
            status = "✓" if abs(error) <= TOLERANCE_CM else "✗"

            if abs(error) > TOLERANCE_CM:
                all_converged = False

            print(f"  {category:<25s} {target:>12.4f} {actual:>12.4f} {error:>+12.4f} {status:>10s}")

        print(f"\nOverall Status: {'✓ All measurements converged' if all_converged else '⚠ Some measurements did not converge'}")

    # Convert to output format (with -incr/-decr suffixes and positive values)
    output_micros = convert_micros_to_output_format(microparameters)

    return output_micros


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution when run via Blender subprocess."""
    import argparse

    # Parse arguments
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description='Adjust microparameters to match target measurements')
    parser.add_argument('--target', type=str, required=True, help='Path to target measurements JSON')
    parser.add_argument('--macros', type=str, required=True, help='Path to macroparameters JSON')
    parser.add_argument('--output', type=str, required=True, help='Path to output microparameters JSON')
    parser.add_argument('--rig-type', type=str, default='default_no_toes', help='Rig type to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args(argv)

    # Check Blender environment
    try:
        import bpy
        print("✓ Running in Blender environment")
    except ImportError:
        print("✗ ERROR: This script must be run with Blender!")
        sys.exit(1)

    # Load inputs
    with open(args.target, 'r') as f:
        target_measurements = json.load(f)

    with open(args.macros, 'r') as f:
        macroparameters = json.load(f)

    # Adjust microparameters
    adjusted_micros = adjust_all_microparameters(
        target_measurements,
        macroparameters,
        rig_type=args.rig_type,
        verbose=args.verbose
    )

    # Save output
    with open(args.output, 'w') as f:
        json.dump(adjusted_micros, f, indent=2)

    print(f"\n✓ Microparameters saved to: {args.output}")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
