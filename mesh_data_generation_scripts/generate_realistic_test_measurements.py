"""
Generate realistic test measurements from training data distribution.

This script samples measurements from the actual training data to ensure:
1. Measurements are within the model's training range
2. Proportions are anatomically consistent
3. Correlations between measurements are preserved
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Measurements
MEASUREMENTS = [
    'height_cm', 'shoulder_width_cm', 'hip_width_cm', 'head_width_cm',
    'neck_length_cm', 'upper_arm_length_cm', 'forearm_length_cm', 'hand_length_cm',
    'upper_leg_length_cm', 'lower_leg_length_cm', 'foot_length_cm'
]


def analyze_measurement_bounds(csv_path):
    """Analyze measurement bounds from training data."""
    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"\nLoaded {len(df)} samples")
    print("\n" + "=" * 100)
    print("MEASUREMENT BOUNDS FROM TRAINING DATA")
    print("=" * 100)
    print(f"{'Measurement':<30s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'Std':>8s} {'Median':>8s}")
    print("-" * 100)

    bounds = {}
    for measure in MEASUREMENTS:
        min_val = df[measure].min()
        max_val = df[measure].max()
        mean_val = df[measure].mean()
        std_val = df[measure].std()
        median_val = df[measure].median()

        bounds[measure] = {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'median': median_val
        }

        print(f"{measure:<30s} {min_val:8.2f} {max_val:8.2f} {mean_val:8.2f} {std_val:8.2f} {median_val:8.2f}")

    print("-" * 100)

    return df, bounds


def sample_from_training_data(df, n_samples=20, method='stratified'):
    """
    Sample measurements from training data.

    Args:
        df: Training data DataFrame
        n_samples: Number of test cases to generate
        method: 'random', 'stratified', or 'percentiles'

    Returns:
        List of measurement dictionaries
    """
    test_cases = []

    if method == 'random':
        # Simple random sampling
        sampled_indices = np.random.choice(len(df), size=n_samples, replace=False)
        sampled_rows = df.iloc[sampled_indices]

    elif method == 'stratified':
        # Stratified sampling based on height (small, medium, tall)
        height_col = 'height_cm'

        # Define height bins
        height_33 = df[height_col].quantile(0.33)
        height_67 = df[height_col].quantile(0.67)

        short = df[df[height_col] <= height_33]
        medium = df[(df[height_col] > height_33) & (df[height_col] <= height_67)]
        tall = df[df[height_col] > height_67]

        # Sample equally from each stratum
        n_per_stratum = n_samples // 3
        remainder = n_samples % 3

        short_samples = short.sample(n=n_per_stratum, replace=False)
        medium_samples = medium.sample(n=n_per_stratum, replace=False)
        tall_samples = tall.sample(n=n_per_stratum + remainder, replace=False)

        sampled_rows = pd.concat([short_samples, medium_samples, tall_samples])

    elif method == 'percentiles':
        # Sample at specific percentiles to cover the distribution
        percentiles = np.linspace(5, 95, n_samples)
        indices = []

        for p in percentiles:
            # Find index closest to this percentile based on height
            height_val = df['height_cm'].quantile(p / 100)
            idx = (df['height_cm'] - height_val).abs().idxmin()
            indices.append(idx)

        sampled_rows = df.loc[indices]

    else:
        raise ValueError(f"Unknown sampling method: {method}")

    # Convert to list of dictionaries
    for idx, row in sampled_rows.iterrows():
        test_case = {measure: float(row[measure]) for measure in MEASUREMENTS}
        test_cases.append(test_case)

    return test_cases


def generate_test_json(test_cases, output_path, description=""):
    """Generate JSON file with test measurements."""
    data = {
        "description": description,
        "test_cases": test_cases
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Generated {len(test_cases)} test cases")
    print(f"✓ Saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate realistic test measurements from training data'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Path to training data CSV'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Path to output JSON file'
    )

    parser.add_argument(
        '--n-samples',
        type=int,
        default=20,
        help='Number of test cases to generate (default: 20)'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['random', 'stratified', 'percentiles'],
        default='stratified',
        help='Sampling method (default: stratified by height)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    print("=" * 100)
    print("REALISTIC TEST MEASUREMENT GENERATOR")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Number of samples: {args.n_samples}")
    print(f"  Sampling method: {args.method}")
    print(f"  Random seed: {args.seed}")

    # Analyze training data bounds
    df, bounds = analyze_measurement_bounds(args.input)

    # Generate test cases
    print(f"\n" + "=" * 100)
    print(f"GENERATING TEST CASES ({args.method.upper()} SAMPLING)")
    print("=" * 100)

    test_cases = sample_from_training_data(df, n_samples=args.n_samples, method=args.method)

    # Print sample of generated cases
    print(f"\nSample of generated test cases:")
    print("-" * 100)
    for i, case in enumerate(test_cases[:3], 1):
        print(f"\nTest Case {i}:")
        for measure, value in case.items():
            bound_info = bounds[measure]
            within_range = bound_info['min'] <= value <= bound_info['max']
            status = "✓" if within_range else "✗"
            print(f"  {status} {measure:<30s}: {value:6.2f} cm")

    if len(test_cases) > 3:
        print(f"\n... and {len(test_cases) - 3} more test cases")

    # Generate JSON
    description = (
        f"Realistic test measurements for Asian female body models. "
        f"Sampled from training data using {args.method} sampling method. "
        f"All measurements are within the training distribution bounds."
    )

    generate_test_json(test_cases, args.output, description)

    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
    print(f"\nYou can now use this file to test your model:")
    print(f"  python test_model_accuracy.py --model <model.pkl> --test-measurements {args.output}")


if __name__ == "__main__":
    main()
