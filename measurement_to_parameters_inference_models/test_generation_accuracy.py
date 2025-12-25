"""
Test Model Accuracy with Real Mesh Generation (Batch Support)

This script validates the inverse mapping model with real-world measurements by:
1. Taking target measurements (single or batch)
2. Using the model to predict macroparameters
3. Generating actual meshes with those macroparameters
4. Measuring the generated meshes
5. Comparing actual measurements to predictions and targets
6. Producing aggregate statistics and detailed CSV results

This provides TRUE accuracy of the model's inverse mapping capability across
different demographics (gender, race, etc.).

JSON Input Formats:

Single measurement:
{
  "height_cm": 165.0,
  "shoulder_width_cm": 38.5,
  ...
}

Batch measurements:
{
  "category": "asian_male",
  "description": "Real measurements from Asian male subjects",
  "measurements": [
    {
      "subject_id": "AM001",
      "height_cm": 172.5,
      "shoulder_width_cm": 42.3,
      ...
    },
    {
      "subject_id": "AM002",
      ...
    }
  ]
}

Usage:
    # Single measurement
    python test_model_accuracy.py --input test_measurements.json --models model.pkl

    # Batch measurements
    python test_model_accuracy.py --input asian_male_batch.json --models model.pkl
"""

import json
import sys
import subprocess
import argparse
import csv
from pathlib import Path
import numpy as np
from statistics import mean, median, stdev

# Add parent directory to path for imports
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import from infer_macroparameters
from infer_macroparameters import load_models, find_macroparameters, MEASUREMENTS, parse_gender, parse_race


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object that may contain numpy types

    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def generate_and_measure_mesh(macroparameters, microparameters=None, rig_type='default_no_toes', gender=0.0, race=None):
    """
    Generate a mesh with given macroparameters (and optionally microparameters) and measure it.

    This function:
    1. Creates a config file for generate_human.py (with macro and optionally micro settings)
    2. Calls Blender with generate_human.py to generate the model
    3. Calls Blender to measure the generated FBX
    4. Returns the measurements

    Args:
        macroparameters: Dictionary of macroparameter values (age, muscle, weight, height, proportions)
        microparameters: Optional dictionary of microparameter values
        rig_type: Type of rig to add
        gender: Gender value (0.0 for female, 1.0 for male), default 0.0
        race: Race dictionary (e.g., {'asian': 1.0, 'caucasian': 0.0, 'african': 0.0}), default Asian

    Returns:
        Dictionary of actual measurements from the generated mesh
    """
    # Default race if not provided
    if race is None:
        race = {'asian': 1.0, 'caucasian': 0.0, 'african': 0.0}

    # Build config for generate_human.py
    config = {
        'macro_settings': {
            'gender': gender,
            'age': macroparameters['age'],
            'muscle': macroparameters['muscle'],
            'weight': macroparameters['weight'],
            'height': macroparameters['height'],
            'proportions': macroparameters['proportions'],
            'cupsize': 0.5,       # Medium
            'firmness': 0.5,      # Medium
            'race': race
        },
        'output': {
            'directory': str(parent_dir / 'output'),
            'filename': 'temp_test_model.fbx'
        },
        'export_settings': {
            'use_mesh_modifiers': True,
            'add_leaf_bones': True
        }
    }

    # Add microparameters if provided
    if microparameters:
        config['micro_settings'] = microparameters

    # Save config to temporary file
    config_path = parent_dir / 'temp_test_human_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Run Blender with generate_human.py
    cmd = [
        'python',
        str(parent_dir / 'run_blender.py'),
        '--script', 'generate_human.py',
        '--',
        '--config', str(config_path),
        '--rig-type', rig_type
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(parent_dir),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"\nERROR: Blender exited with code {result.returncode}", file=sys.stderr)
            if result.stdout:
                print("--- Blender Output ---", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print("--- Blender Error Output ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"Blender process failed with code {result.returncode}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Blender process timed out after 5 minutes")
    finally:
        # Clean up temp config
        if config_path.exists():
            config_path.unlink()

    # Create a temporary Blender script to measure the generated mesh
    # This script imports the FBX and measures it
    measurements_json_path = parent_dir / 'temp_test_measurements.json'
    fbx_path = parent_dir / 'output' / 'temp_test_model.fbx'

    measure_script = f"""
import sys
import json
from pathlib import Path

# Add parent to path for measurements module
parent = Path(__file__).parent.absolute()
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

try:
    import bpy
    from measurement_functions import measurements

    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Import FBX
    bpy.ops.import_scene.fbx(filepath=r'{fbx_path}')

    # Find mesh and armature
    basemesh = None
    armature = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and basemesh is None:
            basemesh = obj
        elif obj.type == 'ARMATURE' and armature is None:
            armature = obj

    if not basemesh:
        print("ERROR: No mesh found in FBX")
        sys.exit(1)
    if not armature:
        print("ERROR: No armature found in FBX")
        sys.exit(1)

    # Extract measurements
    measured = measurements.extract_all_measurements(basemesh, armature)

    # Convert keys to add _cm suffix for compatibility
    measurements_with_suffix = {{}}
    for key, value in measured.items():
        if not key.endswith('_cm'):
            measurements_with_suffix[f"{{key}}_cm"] = value
        else:
            measurements_with_suffix[key] = value

    # Save to JSON
    with open(r'{measurements_json_path}', 'w') as f:
        json.dump(measurements_with_suffix, f, indent=2)

    print("Measurements saved successfully")
    sys.exit(0)

except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

    # Write temporary measurement script
    temp_measure_script = parent_dir / 'temp_measure_script.py'
    with open(temp_measure_script, 'w') as f:
        f.write(measure_script)

    # Run measurement script
    measure_cmd = [
        'python',
        str(parent_dir / 'run_blender.py'),
        '--script', str(temp_measure_script)
    ]

    try:
        result = subprocess.run(
            measure_cmd,
            cwd=str(parent_dir),
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"\nERROR: Measurement failed with code {result.returncode}", file=sys.stderr)
            if result.stdout:
                print("--- Measurement Output ---", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print("--- Measurement Error Output ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"Measurement failed with code {result.returncode}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Measurement process timed out after 5 minutes")
    finally:
        # Clean up temp script
        if temp_measure_script.exists():
            temp_measure_script.unlink()

    # Load measurements from JSON
    if not measurements_json_path.exists():
        raise RuntimeError("Measurements JSON not created")

    with open(measurements_json_path, 'r') as f:
        measurements = json.load(f)

    # Clean up temp JSON and FBX
    if measurements_json_path.exists():
        measurements_json_path.unlink()

    if fbx_path.exists():
        fbx_path.unlink()

    return measurements


def run_iterative_microparameter_adjustment(target_measurements, macroparameters, rig_type='default_no_toes', gender=0.0, race=None):
    """
    Run iterative microparameter adjustment via Blender subprocess.

    This function calls adjust_microparameters.py via Blender to iteratively
    adjust microparameters until measurements match targets.

    Args:
        target_measurements: Dictionary of target measurements
        macroparameters: Dictionary of macroparameters (only the 5 optimized params: age, muscle, weight, height, proportions)
        rig_type: Type of rig to use
        gender: Gender value (0.0 for female, 1.0 for male), default 0.0
        race: Race dictionary (e.g., {'asian': 1.0, 'caucasian': 0.0, 'african': 0.0}), default Asian

    Returns:
        Dictionary of adjusted microparameters (in output format with -incr/-decr suffixes)
    """
    import tempfile

    # Default race if not provided
    if race is None:
        race = {'asian': 1.0, 'caucasian': 0.0, 'african': 0.0}

    # Build full macro settings (flat dict for adjust_microparameters.py)
    full_macros = {
        'gender': gender,
        'age': macroparameters['age'],
        'muscle': macroparameters['muscle'],
        'weight': macroparameters['weight'],
        'height': macroparameters['height'],
        'proportions': macroparameters['proportions'],
        'cupsize': 0.5,       # Medium
        'firmness': 0.5,      # Medium
        'race': race
    }

    # Create temporary files for input/output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        target_path = Path(f.name)
        json.dump(target_measurements, f, indent=2)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        macros_path = Path(f.name)
        json.dump(full_macros, f, indent=2)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = Path(f.name)

    try:
        # Run Blender with adjust_microparameters.py
        adjust_script_path = script_dir / 'adjust_microparameters.py'

        cmd = [
            'python',
            str(parent_dir / 'run_blender.py'),
            '--script', str(adjust_script_path),
            '--',
            '--target', str(target_path),
            '--macros', str(macros_path),
            '--output', str(output_path),
            '--rig-type', rig_type,
            '--verbose'
        ]

        result = subprocess.run(
            cmd,
            cwd=str(parent_dir),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout (iterative process can take longer)
        )

        if result.returncode != 0:
            print(f"\nERROR: Microparameter adjustment failed with code {result.returncode}", file=sys.stderr)
            if result.stdout:
                print("--- Adjustment Output ---", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print("--- Adjustment Error Output ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"Microparameter adjustment failed with code {result.returncode}")

        # Print output for visibility
        if result.stdout:
            print(result.stdout)

        # Load adjusted microparameters
        if not output_path.exists():
            raise RuntimeError("Microparameter output JSON not created")

        with open(output_path, 'r') as f:
            adjusted_micros = json.load(f)

        return adjusted_micros

    except subprocess.TimeoutExpired:
        raise RuntimeError("Microparameter adjustment process timed out after 10 minutes")
    finally:
        # Clean up temporary files
        for path in [target_path, macros_path, output_path]:
            if path.exists():
                path.unlink()


def test_single_subject(subject_id, target_measurements, models, macro_bounds, method, rig_type, use_micro_adjustment=True, gender=0.0, race=None):
    """
    Test model on a single subject's measurements.

    Args:
        subject_id: Identifier for this subject
        target_measurements: Target measurements to match
        models: Trained regression models
        macro_bounds: Bounds for macroparameters
        method: Optimization method
        rig_type: Type of rig to use
        use_micro_adjustment: Whether to apply iterative microparameter adjustment (default True)
        gender: Gender value (0.0 for female, 1.0 for male), default 0.0
        race: Race dictionary (e.g., {'asian': 1.0, 'caucasian': 0.0, 'african': 0.0}), default Asian

    Returns:
        Dictionary with test results for this subject
    """
    # Default race if not provided
    if race is None:
        race = {'asian': 1.0, 'caucasian': 0.0, 'african': 0.0}
    print(f"\n{'='*80}")
    print(f"Testing Subject: {subject_id}")
    print(f"{'='*80}")

    # Find macroparameters
    result = find_macroparameters(
        models, macro_bounds, target_measurements,
        method=method, weights=None, verbose=False
    )

    print("\nPredicted Macroparameters:")
    for param, value in result['macroparameters'].items():
        print(f"  {param:12s}: {value:.4f}")

    # Generate and measure actual mesh (macros only)
    print("\nGenerating and measuring mesh (macroparameters only)...")
    actual_measurements_macro = generate_and_measure_mesh(
        result['macroparameters'],
        rig_type=rig_type,
        gender=gender,
        race=race
    )

    # Calculate errors with macros only
    errors_macro = {}
    for measure in MEASUREMENTS:
        target = target_measurements.get(measure, 0.0)
        actual = actual_measurements_macro.get(measure, 0.0)
        predicted = result['predicted_measurements'].get(measure, 0.0)

        errors_macro[measure] = {
            'target': target,
            'predicted': predicted,
            'actual': actual,
            'prediction_error': abs(predicted - target),
            'actual_error': abs(actual - target),
            'error_value': actual - target
        }

    # Calculate MAE with macros only
    actual_errors_macro = [e['actual_error'] for e in errors_macro.values()]
    mae_macro = mean(actual_errors_macro)
    max_error_macro = max(actual_errors_macro)

    print(f"\nResults (Macroparameters only):")
    print(f"  MAE: {mae_macro:.4f} cm")
    print(f"  Max Error: {max_error_macro:.4f} cm")

    # Build result data
    result_data = {
        'subject_id': subject_id,
        'macroparameters': result['macroparameters'],
        'measurements_macro_only': errors_macro,
        'mae_macro_only': mae_macro,
        'max_error_macro_only': max_error_macro
    }

    # Apply iterative microparameter adjustment if enabled
    if use_micro_adjustment:
        print("\n" + "-"*80)
        print("Applying Iterative Microparameter Adjustment")
        print("-"*80)

        # Run iterative adjustment
        adjusted_micros = run_iterative_microparameter_adjustment(
            target_measurements,
            result['macroparameters'],
            rig_type=rig_type,
            gender=gender,
            race=race
        )

        print("\nAdjusted Microparameters:")
        for micro_name, value in adjusted_micros.items():
            print(f"  {micro_name:40s}: {value:+.4f}")

        # Generate and measure with macros + adjusted micros
        print("\nGenerating and measuring mesh (macroparameters + microparameters)...")
        actual_measurements_with_micro = generate_and_measure_mesh(
            result['macroparameters'],
            microparameters=adjusted_micros,
            rig_type=rig_type,
            gender=gender,
            race=race
        )

        # Calculate errors with macros + micros
        errors_with_micro = {}
        for measure in MEASUREMENTS:
            target = target_measurements.get(measure, 0.0)
            actual = actual_measurements_with_micro.get(measure, 0.0)

            errors_with_micro[measure] = {
                'target': target,
                'actual': actual,
                'actual_error': abs(actual - target)
            }

        # Calculate MAE with macros + micros
        actual_errors_with_micro = [e['actual_error'] for e in errors_with_micro.values()]
        mae_with_micro = mean(actual_errors_with_micro)
        max_error_with_micro = max(actual_errors_with_micro)

        print(f"\nResults (Macroparameters + Microparameters):")
        print(f"  MAE: {mae_with_micro:.4f} cm")
        print(f"  Max Error: {max_error_with_micro:.4f} cm")

        improvement = ((mae_macro - mae_with_micro) / mae_macro) * 100 if mae_macro > 0 else 0.0
        print(f"\n  Improvement: {improvement:.2f}%")

        # Add micro results to return data
        result_data['microparameters'] = adjusted_micros
        result_data['measurements_with_micro'] = errors_with_micro
        result_data['mae_with_micro'] = mae_with_micro
        result_data['max_error_with_micro'] = max_error_with_micro
        result_data['improvement_percent'] = improvement
    else:
        # No microparameter adjustment - return macro-only results as final results
        result_data['measurements'] = errors_macro
        result_data['mae'] = mae_macro
        result_data['max_error'] = max_error_macro

    return result_data


def save_results_to_csv(results, category, output_path):
    """
    Save detailed results to CSV.

    Args:
        results: List of result dictionaries from test_single_subject
        category: Category name (e.g., "asian_male")
        output_path: Path to save CSV file
    """
    csv_rows = []

    for result in results:
        # Check if microparameters were used
        has_micro = 'microparameters' in result

        base_row = {
            'category': category,
            'subject_id': result['subject_id'],
            'mae_macro_only': result.get('mae_macro_only', result.get('mae', 0.0)),
            'max_error_macro_only': result.get('max_error_macro_only', result.get('max_error', 0.0))
        }

        # Add microparameter results if available
        if has_micro:
            base_row['mae_with_micro'] = result['mae_with_micro']
            base_row['max_error_with_micro'] = result['max_error_with_micro']
            base_row['improvement_percent'] = result['improvement_percent']

        # Add macroparameters
        for param, value in result['macroparameters'].items():
            base_row[f'macro_{param}'] = value

        # Add microparameters if present
        if has_micro:
            for param, value in result['microparameters'].items():
                base_row[f'micro_{param}'] = value

        # Add per-measurement errors (use macro_only measurements)
        measurements = result.get('measurements_macro_only', result.get('measurements', {}))
        for measure, data in measurements.items():
            base_row[f'{measure}_target'] = data['target']
            if 'predicted' in data:
                base_row[f'{measure}_predicted'] = data['predicted']
            base_row[f'{measure}_actual_macro'] = data['actual']
            base_row[f'{measure}_error_macro'] = data['actual_error']

        # Add microparameter measurement results if available
        if has_micro:
            for measure, data in result['measurements_with_micro'].items():
                base_row[f'{measure}_actual_micro'] = data['actual']
                base_row[f'{measure}_error_micro'] = data['actual_error']

        csv_rows.append(base_row)

    # Write CSV
    if csv_rows:
        # Collect all unique fieldnames from all rows (handles variable microparameters)
        all_fieldnames = set()
        for row in csv_rows:
            all_fieldnames.update(row.keys())

        # Sort fieldnames for consistent column ordering
        # Keep important columns first
        priority_fields = ['category', 'subject_id', 'mae_macro_only', 'max_error_macro_only',
                          'mae_with_micro', 'max_error_with_micro', 'improvement_percent']
        fieldnames = [f for f in priority_fields if f in all_fieldnames]

        # Add remaining fields (macros, micros, measurements) alphabetically
        remaining_fields = sorted([f for f in all_fieldnames if f not in priority_fields])
        fieldnames.extend(remaining_fields)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


def print_aggregate_statistics(results, category):
    """
    Print aggregate statistics across all subjects.

    Args:
        results: List of result dictionaries
        category: Category name
    """
    if not results:
        print("No results to analyze")
        return

    print(f"\n{'='*80}")
    print(f"AGGREGATE STATISTICS - {category.upper()}")
    print(f"{'='*80}")

    # Check if microparameters were used
    has_micro = 'microparameters' in results[0]

    # Overall MAE statistics (macro only)
    maes_macro = [r.get('mae_macro_only', r.get('mae', 0.0)) for r in results]
    max_errors_macro = [r.get('max_error_macro_only', r.get('max_error', 0.0)) for r in results]

    print(f"\nOverall Performance - Macroparameters Only (n={len(results)} subjects):")
    print(f"  Mean MAE:       {mean(maes_macro):.4f} cm")
    print(f"  Median MAE:     {median(maes_macro):.4f} cm")
    if len(maes_macro) > 1:
        print(f"  Std Dev MAE:    {stdev(maes_macro):.4f} cm")
    print(f"  Min MAE:        {min(maes_macro):.4f} cm")
    print(f"  Max MAE:        {max(maes_macro):.4f} cm")
    print(f"\n  Mean Max Error: {mean(max_errors_macro):.4f} cm")
    print(f"  Worst Error:    {max(max_errors_macro):.4f} cm")

    if has_micro:
        # Statistics with microparameters
        maes_micro = [r['mae_with_micro'] for r in results]
        max_errors_micro = [r['max_error_with_micro'] for r in results]
        improvements = [r['improvement_percent'] for r in results]

        print(f"\nOverall Performance - With Microparameters (n={len(results)} subjects):")
        print(f"  Mean MAE:       {mean(maes_micro):.4f} cm")
        print(f"  Median MAE:     {median(maes_micro):.4f} cm")
        if len(maes_micro) > 1:
            print(f"  Std Dev MAE:    {stdev(maes_micro):.4f} cm")
        print(f"  Min MAE:        {min(maes_micro):.4f} cm")
        print(f"  Max MAE:        {max(maes_micro):.4f} cm")
        print(f"\n  Mean Max Error: {mean(max_errors_micro):.4f} cm")
        print(f"  Worst Error:    {max(max_errors_micro):.4f} cm")

        print(f"\nImprovement from Microparameters:")
        print(f"  Mean Improvement:   {mean(improvements):.2f}%")
        print(f"  Median Improvement: {median(improvements):.2f}%")
        print(f"  Min Improvement:    {min(improvements):.2f}%")
        print(f"  Max Improvement:    {max(improvements):.2f}%")

    # Per-measurement error statistics
    print(f"\nPer-Measurement Error Analysis (Macroparameters Only):")
    print(f"  {'Measurement':<25s} {'Mean Error':<12s} {'Max Error':<12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")

    for measure in MEASUREMENTS:
        measurements = [r.get('measurements_macro_only', r.get('measurements', {})) for r in results]
        errors = [m[measure]['actual_error'] for m in measurements if measure in m]
        if errors:
            mean_err = mean(errors)
            max_err = max(errors)
            print(f"  {measure:<25s} {mean_err:<12.4f} {max_err:<12.4f}")

    # Identify best and worst subjects
    best_macro = min(results, key=lambda x: x.get('mae_macro_only', x.get('mae', float('inf'))))
    worst_macro = max(results, key=lambda x: x.get('mae_macro_only', x.get('mae', 0.0)))

    print(f"\nBest Subject (Macros):  {best_macro['subject_id']} (MAE: {best_macro.get('mae_macro_only', best_macro.get('mae', 0.0)):.4f} cm)")
    print(f"Worst Subject (Macros): {worst_macro['subject_id']} (MAE: {worst_macro.get('mae_macro_only', worst_macro.get('mae', 0.0)):.4f} cm)")

    if has_micro:
        best_micro = min(results, key=lambda x: x['mae_with_micro'])
        worst_micro = max(results, key=lambda x: x['mae_with_micro'])
        print(f"\nBest Subject (With Micros):  {best_micro['subject_id']} (MAE: {best_micro['mae_with_micro']:.4f} cm)")
        print(f"Worst Subject (With Micros): {worst_micro['subject_id']} (MAE: {worst_micro['mae_with_micro']:.4f} cm)")


def main():
    parser = argparse.ArgumentParser(
        description='Test model accuracy with real mesh generation (supports batch testing)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input measurements JSON file (single or batch format)'
    )

    parser.add_argument(
        '--models',
        type=str,
        default='macroparameters_inference_models.pkl',
        help='Path to trained models (default: macroparameters_inference_models.pkl)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='differential_evolution',
        choices=['differential_evolution', 'both'],
        help='Optimization method (default: differential_evolution)'
    )

    parser.add_argument(
        '--rig-type',
        type=str,
        default='default_no_toes',
        choices=['default', 'default_no_toes', 'game_engine'],
        help='Type of rig to add (default: default_no_toes)'
    )

    parser.add_argument(
        '--no-micro-adjustment',
        action='store_true',
        help='Disable iterative microparameter adjustment (test macros only)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TEST MODEL ACCURACY WITH REAL MESH GENERATION")
    print("=" * 80)

    try:
        # Load input
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, 'r') as f:
            input_data = json.load(f)

        # Load models
        print("\nLoading models...")
        models, macro_bounds = load_models(args.models)

        # Determine whether to use microparameter adjustment
        use_micro_adjustment = not args.no_micro_adjustment
        if use_micro_adjustment:
            print("Iterative microparameter adjustment: ENABLED")
        else:
            print("Iterative microparameter adjustment: DISABLED (testing macros only)")

        # Determine if batch or single
        is_batch = 'measurements' in input_data and isinstance(input_data['measurements'], list)

        if is_batch:
            # Batch processing
            category = input_data.get('category', 'unknown')
            description = input_data.get('description', '')
            measurements_list = input_data['measurements']

            # Parse gender and race from batch (applies to all subjects in batch)
            gender_input = input_data.get('gender', 'female')
            gender = parse_gender(gender_input)
            race_input = input_data.get('race', 'asian')
            race = parse_race(race_input)

            print(f"\nBatch Mode: {category}")
            print(f"Description: {description}")
            print(f"Gender: {'Male' if gender > 0.5 else 'Female'}")
            print(f"Race: {', '.join([f'{k}={v:.2f}' for k, v in race.items()])}")
            print(f"Number of subjects: {len(measurements_list)}")

            results = []
            for i, measurement_data in enumerate(measurements_list, 1):
                subject_id = measurement_data.get('subject_id', f'Subject_{i:03d}')

                # Extract measurement values (remove subject_id if present)
                target_measurements = {k: v for k, v in measurement_data.items()
                                     if k in MEASUREMENTS}

                result = test_single_subject(
                    subject_id, target_measurements,
                    models, macro_bounds,
                    args.method, args.rig_type,
                    use_micro_adjustment,
                    gender=gender,
                    race=race
                )
                results.append(result)

            # Save CSV
            csv_path = str(input_path).replace('.json', '_batch_results.csv')
            save_results_to_csv(results, category, csv_path)
            print(f"\n\nDetailed results saved to: {csv_path}")

            # Print aggregate statistics
            print_aggregate_statistics(results, category)

        else:
            # Single measurement (backward compatibility)
            target_measurements = {k: v for k, v in input_data.items() if k in MEASUREMENTS}

            # Parse gender and race from single measurement
            gender_input = input_data.get('gender', 'female')
            gender = parse_gender(gender_input)
            race_input = input_data.get('race', 'asian')
            race = parse_race(race_input)

            print("\nSingle Measurement Mode")
            print(f"Gender: {'Male' if gender > 0.5 else 'Female'}")
            print(f"Race: {', '.join([f'{k}={v:.2f}' for k, v in race.items()])}")
            print("\nTarget Measurements:")
            for measure, value in target_measurements.items():
                print(f"  {measure:25s}: {value:.2f} cm")

            result = test_single_subject(
                'single_test', target_measurements,
                models, macro_bounds,
                args.method, args.rig_type,
                use_micro_adjustment,
                gender=gender,
                race=race
            )

            # Save single result
            output_path = str(input_path).replace('.json', '_result.json')
            result_dict = convert_numpy_types(result)

            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=2)

            print(f"\n\nResults saved to: {output_path}")

        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
