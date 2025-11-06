#!/usr/bin/env python3
"""
Lookup Table Builder - Orchestrator Script

This script runs OUTSIDE of Blender and:
1. Loads configuration (fixed parameters + grid ranges)
2. Generates all parameter combinations
3. Creates a batch configuration file
4. Launches Blender to process the batch
5. Reports results

Usage:
    python build_lookup_table.py --config lookup_table_config.json
"""

import json
import sys
import subprocess
import argparse
from pathlib import Path
from itertools import product
import numpy as np


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
    
    print(f"✓ Loaded configuration from: {config_path}")
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
    
    print("✓ Configuration validated")
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
    
    print(f"\n✓ Generated {len(combinations):,} parameter combinations")
    
    # Create list of full parameter dictionaries
    param_list = []
    for combo in combinations:
        params = config["fixed_params"].copy()
        
        # Add grid parameter values
        for i, param_name in enumerate(param_names):
            params[param_name] = combo[i]
        
        param_list.append(params)
    
    return param_list


def create_batch_config(param_list: list, output_path: str = "configs/batch_config.json"):
    """
    Create batch configuration file for Blender processing.
    
    Args:
        param_list: List of parameter dictionaries
        output_path: Path to save batch configuration
    """
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    batch_config = {
        "total_combinations": len(param_list),
        "parameters": param_list
    }
    
    with open(output_path, 'w') as f:
        json.dump(batch_config, f, indent=2)
    
    print(f"✓ Batch configuration saved to: {output_path}")
    
    # Report file size
    file_size = Path(output_path).stat().st_size / 1024  # KB
    print(f"  File size: {file_size:.1f} KB")


def launch_blender(batch_config_path: str, output_csv_path: str, delete_models: bool = False):
    """
    Launch Blender with measure_batch.py script.
    
    Args:
        batch_config_path: Path to batch configuration file
        output_csv_path: Path for output CSV file
        delete_models: Whether to delete models after measurement
    """
    print("\n" + "="*70)
    print("LAUNCHING BLENDER")
    print("="*70)
    
    # Build command
    cmd = [
        "python", "run_blender.py",
        "--script", "measure_batch.py",
        "--",
        "--batch-config", batch_config_path,
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
        print(f"\n✗ Error: Blender process failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        return 130


def print_summary(param_list: list):
    """
    Print summary of parameter space exploration.
    
    Args:
        param_list: List of parameter dictionaries
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
    
    # Collect grid parameter ranges
    grid_params = ["age", "muscle", "weight", "height", "proportions"]
    
    print("\nGrid Parameters:")
    for param in grid_params:
        values = sorted(set(p[param] for p in param_list))
        print(f"  {param}: {len(values)} values - [{min(values):.2f}, {max(values):.2f}]")
    
    print(f"\nTotal Combinations: {len(param_list):,}")
    
    # Estimate processing time (rough estimate: 1-2 seconds per model)
    estimated_time = len(param_list) * 1.5 / 60  # minutes
    print(f"Estimated Processing Time: {estimated_time:.1f} - {estimated_time*1.5:.1f} minutes")
    
    print("="*70 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Build lookup table for body measurements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python build_lookup_table.py --config lookup_table_config.json
    
    python build_lookup_table.py --config lookup_table_config.json --no-delete
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
        
        # Generate parameter grid
        param_list = generate_parameter_grid(config)
        
        # Print summary
        print_summary(param_list)
        
        # Create batch configuration
        batch_config_path = "configs/batch_config.json"
        create_batch_config(param_list, batch_config_path)
        
        # Launch Blender (unless dry-run)
        if args.dry_run:
            print("\n✓ Dry run complete. Batch configuration ready.")
            print(f"  To process: python build_lookup_table.py --config {args.config}")
        else:
            # Confirm before processing
            if len(param_list) > 100:
                response = input(f"\nProcess {len(param_list):,} combinations? This may take a while. (yes/no): ")
                if response.lower() not in ['yes', 'y']:
                    print("Aborted.")
                    return 0
            
            exit_code = launch_blender(
                batch_config_path,
                args.output,
                delete_models=not args.no_delete
            )
            
            if exit_code == 0:
                print("\n" + "="*70)
                print("✓ LOOKUP TABLE GENERATION COMPLETE!")
                print("="*70)
                print(f"\nOutput saved to: {args.output}")
                
                # Check output file
                output_path = Path(args.output)
                if output_path.exists():
                    file_size = output_path.stat().st_size / 1024  # KB
                    print(f"File size: {file_size:.1f} KB")
                    
                    # Count rows
                    with open(output_path, 'r') as f:
                        row_count = sum(1 for line in f) - 1  # Exclude header
                    print(f"Rows: {row_count:,}")
                
                print("="*70 + "\n")
            else:
                print("\n✗ Lookup table generation failed")
                return exit_code
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())