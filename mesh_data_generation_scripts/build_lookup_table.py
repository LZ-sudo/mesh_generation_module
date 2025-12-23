#!/usr/bin/env python3
"""
Lookup Table Builder - Orchestrator Script

This script runs OUTSIDE of Blender and:
1. Loads configuration (fixed parameters + grid ranges or sampling config)
2. Generates parameter combinations (grid search or Latin Hypercube Sampling)
3. Creates a batch configuration file
4. Launches Blender to process the batch
5. Reports results

Usage:
    # Grid search (traditional)
    python build_lookup_table.py --config configs/lookup_table_config_test.json --method grid

    # Latin Hypercube Sampling 
    python build_lookup_table.py --config configs/lookup_table_config_test.json --method lhs --n-samples 100000
"""

import json
import sys
import subprocess
import argparse
from pathlib import Path
from itertools import product
import numpy as np
from scipy.stats import qmc

# Get script directory
script_dir = Path(__file__).parent.absolute()


def extract_config_suffix(config_path: str) -> str:
    """
    Extract suffix from config filename for naming output files.

    Example:
        lookup_table_config_female_asian.json -> female_asian
        lookup_table_config.json -> (empty string)

    Args:
        config_path: Path to configuration file

    Returns:
        Suffix string (empty if no suffix)
    """
    filename = Path(config_path).stem  # Get filename without extension

    # Look for pattern: lookup_table_config_SUFFIX
    if "lookup_table_config_" in filename:
        suffix = filename.replace("lookup_table_config_", "")
        return suffix

    return ""


def load_config(config_path: str) -> dict:
    """
    Load configuration file.

    Args:
        config_path: Path to configuration JSON

    Returns:
        Configuration dictionary
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"[OK] Loaded configuration from: {config_path}")
    return config


def validate_config(config: dict) -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError if configuration is invalid
    """
    # Check required keys
    if "fixed_params" not in config:
        raise ValueError("Configuration must contain 'fixed_params'")
    
    if "grid_params" not in config:
        raise ValueError("Configuration must contain 'grid_params'")
    
    # Validate fixed parameters
    required_fixed = {"gender", "cupsize", "firmness", "race"}
    missing_fixed = required_fixed - set(config["fixed_params"].keys())
    if missing_fixed:
        raise ValueError(f"Missing fixed parameters: {missing_fixed}")
    
    # Validate grid parameters
    required_grid = {"age", "muscle", "weight", "height", "proportions"}
    missing_grid = required_grid - set(config["grid_params"].keys())
    if missing_grid:
        raise ValueError(f"Missing grid parameters: {missing_grid}")
    
    # Validate grid parameter structure
    for param, values in config["grid_params"].items():
        if not isinstance(values, dict):
            raise ValueError(f"Grid parameter '{param}' must be a dictionary with min, max, step")
        
        required_keys = {"min", "max", "step"}
        missing_keys = required_keys - set(values.keys())
        if missing_keys:
            raise ValueError(f"Grid parameter '{param}' missing keys: {missing_keys}")
        
        # Validate ranges
        if not (0.0 <= values["min"] <= 1.0):
            raise ValueError(f"Grid parameter '{param}' min must be in [0.0, 1.0]")
        
        if not (0.0 <= values["max"] <= 1.0):
            raise ValueError(f"Grid parameter '{param}' max must be in [0.0, 1.0]")
        
        if values["min"] > values["max"]:
            raise ValueError(f"Grid parameter '{param}' min must be <= max")
        
        if values["step"] <= 0:
            raise ValueError(f"Grid parameter '{param}' step must be > 0")
    
    print("[OK] Configuration validated")
    return True


def generate_parameter_grid(config: dict) -> list:
    """
    Generate all parameter combinations from grid configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of parameter dictionaries
    """
    print("\nGenerating parameter grid...")

    grid_params = config["grid_params"]

    # Generate value lists for each parameter
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

        print(f"  {param_name}: {len(values)} values - {values}")

    # Generate all combinations
    combinations = list(product(*param_values))

    print(f"\n[OK] Generated {len(combinations):,} parameter combinations")

    # Create list of full parameter dictionaries
    param_list = []
    for combo in combinations:
        params = config["fixed_params"].copy()

        # Add grid parameter values
        for i, param_name in enumerate(param_names):
            params[param_name] = combo[i]

        param_list.append(params)

    return param_list


def generate_parameter_lhs(config: dict, n_samples: int = 9900, seed: int = 42) -> list:
    """
    Generate parameter combinations using Latin Hypercube Sampling.

    Latin Hypercube Sampling ensures better coverage of the parameter space
    with fewer samples compared to grid search. Ideal for TabPFN which has
    a 10,000 sample limit.

    Args:
        config: Configuration dictionary with grid_params defining bounds
        n_samples: Number of samples to generate (default: 9900, max: 10000 for TabPFN)
        seed: Random seed for reproducibility

    Returns:
        List of parameter dictionaries
    """
    print(f"\nGenerating {n_samples:,} samples using Latin Hypercube Sampling...")

    grid_params = config["grid_params"]

    # Extract parameter names and bounds in consistent order
    param_names = ['age', 'muscle', 'weight', 'height', 'proportions']
    bounds = []

    for param_name in param_names:
        if param_name not in grid_params:
            raise ValueError(f"Missing grid parameter: {param_name}")

        param_config = grid_params[param_name]
        min_val = param_config["min"]
        max_val = param_config["max"]
        bounds.append((min_val, max_val))

        print(f"  {param_name}: [{min_val:.3f}, {max_val:.3f}]")

    # Create Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=len(param_names), seed=seed)

    # Generate samples in unit hypercube [0, 1]^d
    unit_samples = sampler.random(n=n_samples)

    # Scale samples to actual parameter bounds
    scaled_samples = np.zeros_like(unit_samples)
    for i, (min_val, max_val) in enumerate(bounds):
        scaled_samples[:, i] = min_val + unit_samples[:, i] * (max_val - min_val)

    # Round to 3 decimal places for consistency
    scaled_samples = np.round(scaled_samples, 3)

    print(f"\n[OK] Generated {n_samples:,} parameter combinations using LHS")

    # Create list of full parameter dictionaries
    param_list = []
    for sample in scaled_samples:
        params = config["fixed_params"].copy()

        # Add sampled parameter values
        for i, param_name in enumerate(param_names):
            params[param_name] = float(sample[i])

        param_list.append(params)

    # Print coverage statistics
    print("\nParameter coverage statistics:")
    for i, param_name in enumerate(param_names):
        values = scaled_samples[:, i]
        print(f"  {param_name}: min={values.min():.3f}, max={values.max():.3f}, "
              f"mean={values.mean():.3f}, std={values.std():.3f}")

    return param_list


def calculate_total_combinations(config: dict) -> int:
    """
    Calculate total number of parameter combinations from grid configuration.

    Args:
        config: Configuration dictionary with grid_params

    Returns:
        Total number of combinations
    """
    total = 1
    for param_config in config["grid_params"].values():
        min_val = param_config["min"]
        max_val = param_config["max"]
        step = param_config["step"]
        n_values = len(np.arange(min_val, max_val + step/2, step))
        total *= n_values

    return total


def save_parameter_list(param_list: list, output_path: str):
    """
    Save parameter list to JSON file for measure_batch.py to consume.

    Args:
        param_list: List of parameter dictionaries
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(param_list, f)
    print(f"[OK] Saved {len(param_list):,} parameter combinations to: {output_path}")


def launch_blender(param_list_path: str, output_csv_path: str, delete_models: bool = False):
    """
    Launch Blender with measure_batch.py script.

    Args:
        param_list_path: Path to JSON file containing parameter combinations
        output_csv_path: Path for output CSV file
        delete_models: Whether to delete models after measurement
    """
    print("\n" + "="*70)
    print("LAUNCHING BLENDER")
    print("="*70)

    # Build command - pass parameter list file instead of config
    cmd = [
        "python", "run_blender.py",
        "--script", "measurement_functions/measure_batch.py",
        "--",
        "--param-list", param_list_path,
        "--output", output_csv_path
    ]

    if not delete_models:
        cmd.append("--no-delete")

    print(f"\nCommand: {' '.join(cmd)}\n")
    print("="*70 + "\n")

    # Execute
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Blender process failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\n[ERROR] Interrupted by user")
        return 130


def print_summary(param_list: list, method: str = 'grid'):
    """
    Print summary of parameter space exploration.

    Args:
        param_list: List of parameter dictionaries
        method: Sampling method used ('grid' or 'lhs')
    """
    print("\n" + "="*70)
    print("PARAMETER SPACE SUMMARY")
    print("="*70)

    # Collect fixed parameters
    fixed_params = {}
    for key in ["gender", "cupsize", "firmness", "race"]:
        fixed_params[key] = param_list[0][key]

    print("\nFixed Parameters:")
    for key, value in fixed_params.items():
        if key == "race":
            print(f"  {key}:")
            for race, val in value.items():
                print(f"    {race}: {val}")
        else:
            print(f"  {key}: {value}")

    # Collect parameter ranges
    macro_params = ["age", "muscle", "weight", "height", "proportions"]

    param_label = "Sampled Parameters:" if method == 'lhs' else "Grid Parameters:"
    print(f"\n{param_label}")
    for param in macro_params:
        values = sorted(set(p[param] for p in param_list))
        print(f"  {param}: {len(values)} unique values - [{min(values):.2f}, {max(values):.2f}]")

    print(f"\nTotal Samples: {len(param_list):,}")

    # Estimate processing time (rough estimate: 0.9 seconds per model)
    estimated_time = len(param_list) * 0.9 / 60  # minutes
    print(f"Estimated Processing Time: {estimated_time:.1f} - {estimated_time*1.5:.1f} minutes")

    print("="*70 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Build lookup table for body measurements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Grid search (traditional approach - generates many samples)
    python build_lookup_table.py --config configs/lookup_table_config.json --method grid

    # Latin Hypercube Sampling (for TabPFN - max 10k samples)
    python build_lookup_table.py --config configs/lookup_table_config.json --method lhs --n-samples 9900

    # Dry run to see how many samples will be generated
    python build_lookup_table.py --config configs/lookup_table_config.json --method lhs --dry-run
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/lookup_table_config.json',
        help='Path to configuration file (default: configs/lookup_table_config.json)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output/lookup_table.csv',
        help='Path for output CSV file (default: output/lookup_table.csv)'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['grid', 'lhs'],
        default='grid',
        help='Sampling method: "grid" for grid search, "lhs" for Latin Hypercube Sampling (default: grid)'
    )

    parser.add_argument(
        '--n-samples',
        type=int,
        default=9900,
        help='Number of samples for LHS method (default: 9900, max: 10000 for TabPFN)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for LHS sampling (default: 42)'
    )

    parser.add_argument(
        '--no-delete',
        action='store_true',
        help='Do not delete models after measurement (for debugging)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate batch config but do not launch Blender'
    )

    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("LOOKUP TABLE BUILDER")
    print("="*70 + "\n")

    try:
        # Load and validate configuration
        config = load_config(args.config)
        validate_config(config)

        # Extract suffix from config filename for dynamic naming
        suffix = extract_config_suffix(args.config)

        # Generate dynamic output path
        if suffix:
            # Override output path if user didn't specify custom one
            if args.output == 'output/lookup_table.csv':
                # Add method suffix to differentiate grid vs lhs
                method_suffix = f"_{args.method}" if args.method == 'lhs' else ""
                output_csv_path = f"lookup_tables/lookup_table_{suffix}{method_suffix}.csv"
            else:
                output_csv_path = args.output
        else:
            output_csv_path = args.output

        print(f"\nConfiguration:")
        print(f"  Config file: {args.config}")
        print(f"  Sampling method: {args.method.upper()}")
        print(f"  Output CSV: {output_csv_path}")

        # Generate parameter combinations based on method
        if args.method == 'grid':
            # Grid search - calculate total combinations
            total_combinations = calculate_total_combinations(config)
            print(f"  Grid combinations: {total_combinations:,}")

            # Generate parameter grid
            param_list = generate_parameter_grid(config)

        elif args.method == 'lhs':
            # Latin Hypercube Sampling
            if args.n_samples > 10000:
                print(f"\nWARNING: n_samples={args.n_samples} exceeds TabPFN's 10,000 limit!")
                print(f"  Recommended: Use --n-samples 9900 or less")

            print(f"  LHS samples: {args.n_samples:,}")
            print(f"  Random seed: {args.seed}")

            # Generate LHS samples
            param_list = generate_parameter_lhs(config, n_samples=args.n_samples, seed=args.seed)

        # Print summary
        print_summary(param_list, method=args.method)

        # Launch Blender (unless dry-run)
        if args.dry_run:
            print("\n[OK] Dry run complete. Configuration validated.")
            print(f"  To process: python build_lookup_table.py --config {args.config} --method {args.method}")
            if args.method == 'lhs':
                print(f"             (with --n-samples {args.n_samples})")
        else:
            # Confirm before processing
            if len(param_list) > 100:
                response = input(f"\nProcess {len(param_list):,} samples? This may take a while. (yes/no): ")
                if response.lower() not in ['yes', 'y']:
                    print("Aborted.")
                    return 0

            # Save parameter list to temporary JSON file
            param_list_path = script_dir / 'temp_param_list.json'
            save_parameter_list(param_list, str(param_list_path))

            try:
                # Launch Blender with parameter list
                exit_code = launch_blender(
                    str(param_list_path),
                    output_csv_path,
                    delete_models=not args.no_delete
                )
            finally:
                # Clean up temporary parameter list file
                if param_list_path.exists():
                    param_list_path.unlink()
                    print(f"[OK] Cleaned up temporary parameter list")
            
            if exit_code == 0:
                print("\n" + "="*70)
                print("[SUCCESS] LOOKUP TABLE GENERATION COMPLETE!")
                print("="*70)
                print(f"\nOutput saved to: {output_csv_path}")

                # Check output file
                output_path = Path(output_csv_path)
                if output_path.exists():
                    file_size = output_path.stat().st_size / 1024  # KB
                    print(f"File size: {file_size:.1f} KB")

                    # Count rows
                    with open(output_csv_path, 'r') as f:
                        row_count = sum(1 for line in f) - 1  # Exclude header
                    print(f"Rows: {row_count:,}")

                print("="*70 + "\n")
            else:
                print("\n[ERROR] Lookup table generation failed")
                return exit_code
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())