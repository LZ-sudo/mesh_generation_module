#!/usr/bin/env python3
"""
Hair Physics Rigging File Generator

This script generates MPFB2-compatible rigging files (.mpfbskel and .mhw) for
hair assets to enable physics simulation in game engines like Unreal Engine.

The generated files are placed alongside the hair asset's .mhclo file and
MPFB2 will automatically load them when the hair asset is applied to a human.

Output Files:
    - {asset_name}.mpfbskel: Skeleton definition with bone chains
    - {asset_name}.mhw: Vertex weight assignments
    - {asset_name}.mhmask-subrig (vertex group data embedded in weights)

The algorithm:
1. Load the hair asset OBJ file
2. Analyze mesh geometry using spatial segmentation
3. Identify hair regions (front, sides, back, top)
4. Determine flow direction per region (root to tip)
5. Generate bone chain definitions
6. Calculate vertex weights
7. Save as MPFB2-compatible JSON files

Usage:
    # Run via Blender headless
    python run_blender.py --script mesh_hair_generation/generate_hair_rigging.py -- \\
        --asset mpfb_hair_assets/Long_Hair_A --verbose

    # Process all hair assets
    python run_blender.py --script mesh_hair_generation/generate_hair_rigging.py -- \\
        --all --verbose
"""

import json
import math
import argparse
import sys
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HairPhysicsConfig:
    """Configuration for hair physics bone generation."""

    # Region detection
    min_region_vertices: int = 50
    region_angle_threshold: float = 45.0  # degrees

    # Bone chain parameters
    min_extent_for_bones: float = 3.0  # cm - minimum hair length for physics
    bones_per_10cm: float = 2.0
    max_bones_per_chain: int = 6
    min_bones_per_chain: int = 2

    # Weight parameters
    weight_falloff: float = 0.5
    root_head_influence: float = 0.3

    # Physics hints (stored as custom properties)
    stiffness: float = 0.8
    damping: float = 0.5
    gravity_scale: float = 1.0

    # Head bone reference
    head_bone_name: str = "head"


class HairRegionType(Enum):
    """Types of hair regions based on position relative to head."""
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    BACK = "back"
    TOP = "top"


@dataclass
class HairRegion:
    """Represents a detected hair region with its properties."""
    region_type: HairRegionType
    vertex_indices: List[int]
    root_position: Tuple[float, float, float]
    tip_position: Tuple[float, float, float]
    flow_direction: Tuple[float, float, float]
    extent: float  # distance from root to tip in cm

    @property
    def bone_count(self) -> int:
        """Calculate optimal bone count based on extent."""
        if self.extent < 3.0:
            return 2
        elif self.extent < 10.0:
            return 3
        elif self.extent < 20.0:
            return 4
        elif self.extent < 30.0:
            return 5
        else:
            return 6


# ============================================================================
# Geometry Analysis (Works with raw OBJ data or Blender mesh)
# ============================================================================

def load_obj_vertices(obj_path: Path) -> List[Tuple[float, float, float]]:
    """
    Load vertices from an OBJ file.

    Args:
        obj_path: Path to the .obj file

    Returns:
        List of (x, y, z) vertex coordinates
    """
    vertices = []

    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((x, y, z))

    return vertices


def estimate_head_position(vertices: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """
    Estimate head center position from hair vertices.

    For hair, the centroid of the closest-to-center vertices gives a good
    approximation of where the scalp/head is.

    Args:
        vertices: List of vertex coordinates

    Returns:
        Estimated head center position
    """
    if not vertices:
        return (0.0, 0.0, 1.6)  # Default head height

    # Calculate centroid XY of all vertices
    cx = sum(v[0] for v in vertices) / len(vertices)
    cy = sum(v[1] for v in vertices) / len(vertices)

    # Find minimum Z (bottom of hair, closest to scalp)
    # Head center is approximately at the centroid XY but at the base of hair
    min_z = min(v[2] for v in vertices)

    return (cx, cy, min_z - 0.05)  # Slightly below hair base


def analyze_vertices(
    vertices: List[Tuple[float, float, float]],
    head_position: Tuple[float, float, float]
) -> List[Dict]:
    """
    Analyze vertices relative to head position.

    Args:
        vertices: List of vertex coordinates
        head_position: Estimated head center

    Returns:
        List of vertex analysis dictionaries
    """
    vertices_data = []

    hx, hy, hz = head_position

    for idx, (vx, vy, vz) in enumerate(vertices):
        # Calculate relative position
        rx, ry, rz = vx - hx, vy - hy, vz - hz

        # Distance from head
        distance = math.sqrt(rx*rx + ry*ry + rz*rz)
        if distance < 0.001:
            continue

        # Spherical coordinates
        # Theta: angle from vertical (Z-up)
        theta = math.acos(max(-1, min(1, rz / distance)))

        # Phi: angle in XY plane
        phi = math.atan2(ry, rx)

        vertices_data.append({
            'index': idx,
            'world_pos': (vx, vy, vz),
            'relative': (rx, ry, rz),
            'distance': distance,
            'theta': math.degrees(theta),
            'phi': math.degrees(phi)
        })

    return vertices_data


def segment_into_regions(
    vertices_data: List[Dict],
    config: HairPhysicsConfig
) -> List[HairRegion]:
    """
    Segment hair vertices into regions based on angular position.

    Args:
        vertices_data: Output from analyze_vertices()
        config: Hair physics configuration

    Returns:
        List of HairRegion objects
    """
    if not vertices_data:
        return []

    # Define region boundaries based on phi angle (XY plane)
    top_threshold = 45.0  # degrees from vertical

    # Bin vertices into regions
    region_vertices = {rt: [] for rt in HairRegionType}

    for vert_data in vertices_data:
        phi = vert_data['phi']
        theta = vert_data['theta']

        # Check if top region
        if theta < top_threshold:
            region_vertices[HairRegionType.TOP].append(vert_data)
            continue

        # Determine horizontal region
        if -45 <= phi < 45:
            region_vertices[HairRegionType.FRONT].append(vert_data)
        elif 45 <= phi < 135:
            region_vertices[HairRegionType.RIGHT].append(vert_data)
        elif phi >= 135 or phi < -135:
            region_vertices[HairRegionType.BACK].append(vert_data)
        else:  # -135 <= phi < -45
            region_vertices[HairRegionType.LEFT].append(vert_data)

    # Build HairRegion objects for valid regions
    regions = []

    for region_type, verts in region_vertices.items():
        if len(verts) < config.min_region_vertices:
            continue

        # Find root vertices (closest to head) and tip vertices (furthest)
        sorted_by_dist = sorted(verts, key=lambda v: v['distance'])

        # Take bottom 20% as roots, top 20% as tips
        n_boundary = max(1, len(sorted_by_dist) // 5)
        root_verts = sorted_by_dist[:n_boundary]
        tip_verts = sorted_by_dist[-n_boundary:]

        # Calculate centroids
        root_centroid = [0.0, 0.0, 0.0]
        for v in root_verts:
            for i in range(3):
                root_centroid[i] += v['world_pos'][i]
        root_centroid = tuple(c / len(root_verts) for c in root_centroid)

        tip_centroid = [0.0, 0.0, 0.0]
        for v in tip_verts:
            for i in range(3):
                tip_centroid[i] += v['world_pos'][i]
        tip_centroid = tuple(c / len(tip_verts) for c in tip_centroid)

        # Calculate flow direction and extent
        flow_vec = [tip_centroid[i] - root_centroid[i] for i in range(3)]
        extent_m = math.sqrt(sum(c*c for c in flow_vec))
        extent_cm = extent_m * 100  # convert to cm

        if extent_cm < config.min_extent_for_bones:
            continue

        # Normalize flow direction
        flow_direction = tuple(c / extent_m for c in flow_vec) if extent_m > 0 else (0, 0, -1)

        region = HairRegion(
            region_type=region_type,
            vertex_indices=[v['index'] for v in verts],
            root_position=root_centroid,
            tip_position=tip_centroid,
            flow_direction=flow_direction,
            extent=extent_cm
        )

        regions.append(region)

    return regions


# ============================================================================
# MPFB2 File Generation
# ============================================================================

def generate_mpfbskel(
    regions: List[HairRegion],
    config: HairPhysicsConfig,
    asset_name: str
) -> Dict:
    """
    Generate MPFB2 skeleton file content (.mpfbskel).

    Args:
        regions: List of HairRegion objects
        config: Hair physics configuration
        asset_name: Name of the hair asset

    Returns:
        Dictionary structure for .mpfbskel JSON file
    """
    skeleton = {
        "name": f"{asset_name}_physics",
        "version": 110,  # Required by MPFB2 rig.py validation
        "is_subrig": True,
        "scale_factor": 1.0,
        "bones": {},
        "rigify_ui": None,
        "extra_bones": [],
        "hair_physics_metadata": {
            "version": "1.0",
            "stiffness": config.stiffness,
            "damping": config.damping,
            "gravity_scale": config.gravity_scale,
            "regions": [r.region_type.value for r in regions]
        }
    }

    for region in regions:
        bone_count = min(region.bone_count, config.max_bones_per_chain)
        bone_count = max(bone_count, config.min_bones_per_chain)

        region_prefix = f"hair_{region.region_type.value}"

        # Calculate bone positions along flow direction
        root = list(region.root_position)
        flow = list(region.flow_direction)
        total_length = region.extent / 100  # convert to meters
        bone_length = total_length / bone_count

        for i in range(bone_count):
            bone_name = f"{region_prefix}_{i:02d}"

            # Calculate bone head and tail
            head = [root[j] + flow[j] * (i * bone_length) for j in range(3)]
            tail = [root[j] + flow[j] * ((i + 1) * bone_length) for j in range(3)]

            # Determine parent bone
            # For subrigs, root bones have no parent within the subrig
            # MPFB2 handles connecting the subrig to the main rig
            if i == 0:
                parent = ""  # No parent for root bones in subrig
            else:
                parent = f"{region_prefix}_{i-1:02d}"

            # MPFB2 bone structure (fields required by rig.py update_edit_bone_metadata)
            skeleton["bones"][bone_name] = {
                "head": {
                    "strategy": "MEAN",
                    "vertex_indices": [],  # Will be computed at load time
                    "default_position": head
                },
                "tail": {
                    "strategy": "MEAN",
                    "vertex_indices": [],
                    "default_position": tail
                },
                "parent": parent,
                "roll": 0.0,
                "use_connect": i > 0,
                "use_deform": True,
                "use_local_location": True,
                "use_inherit_rotation": True,
                "inherit_scale": "FULL",
                "constraints": [],
                "rigify": {}
            }

            # Add to extra_bones list
            skeleton["extra_bones"].append(bone_name)

    return skeleton


def generate_mhw(
    regions: List[HairRegion],
    vertices: List[Tuple[float, float, float]],
    config: HairPhysicsConfig
) -> Dict:
    """
    Generate MPFB2 weight file content (.mhw).

    Args:
        regions: List of HairRegion objects
        vertices: All vertex coordinates
        config: Hair physics configuration

    Returns:
        Dictionary structure for .mhw JSON file
    """
    weights = {
        "copyright": "Generated by generate_hair_rigging.py",
        "description": "Hair physics bone weights",
        "license": "CC0",
        "name": "hair_physics_weights",
        "version": 1,
        "weights": {}
    }

    # Also track mhmask-subrig weights
    subrig_mask_weights = []

    for region in regions:
        bone_count = min(region.bone_count, config.max_bones_per_chain)
        bone_count = max(bone_count, config.min_bones_per_chain)

        region_prefix = f"hair_{region.region_type.value}"

        root = list(region.root_position)
        flow = list(region.flow_direction)
        total_length = region.extent / 100
        bone_length = total_length / bone_count if bone_count > 0 else 0.1

        # Initialize weight lists for each bone
        bone_weights = {f"{region_prefix}_{i:02d}": [] for i in range(bone_count)}

        for vert_idx in region.vertex_indices:
            if vert_idx >= len(vertices):
                continue

            vx, vy, vz = vertices[vert_idx]

            # Project vertex onto bone chain axis
            vert_vec = [vx - root[0], vy - root[1], vz - root[2]]
            projection = sum(vert_vec[j] * flow[j] for j in range(3))

            # Clamp to chain length
            projection = max(0, min(projection, total_length))

            # Determine primary bone
            bone_index = int(projection / bone_length)
            bone_index = min(bone_index, bone_count - 1)

            # Calculate blend factor
            local_pos = projection - (bone_index * bone_length)
            blend = local_pos / bone_length if bone_length > 0 else 0

            # Primary bone weight
            primary_weight = 1.0 - blend * config.weight_falloff
            bone_name = f"{region_prefix}_{bone_index:02d}"
            bone_weights[bone_name].append([vert_idx, round(primary_weight, 4)])

            # Secondary bone weight (for smooth blending)
            if bone_index + 1 < bone_count and blend > 0.2:
                secondary_weight = blend * config.weight_falloff
                next_bone = f"{region_prefix}_{bone_index + 1:02d}"
                bone_weights[next_bone].append([vert_idx, round(secondary_weight, 4)])

            # Add to subrig mask (full influence for hair vertices)
            subrig_mask_weights.append([vert_idx, 1.0])

        # Add bone weights to output
        for bone_name, vert_weights in bone_weights.items():
            if vert_weights:
                weights["weights"][bone_name] = vert_weights

    # Add subrig mask
    weights["weights"]["mhmask-subrig"] = subrig_mask_weights

    return weights


def save_rigging_files(
    asset_folder: Path,
    asset_name: str,
    skeleton: Dict,
    weights: Dict,
    verbose: bool = False
) -> Tuple[Path, Path]:
    """
    Save generated rigging files to the asset folder.

    Args:
        asset_folder: Path to the hair asset folder
        asset_name: Name of the hair asset
        skeleton: Skeleton dictionary from generate_mpfbskel()
        weights: Weights dictionary from generate_mhw()
        verbose: Enable verbose output

    Returns:
        Tuple of (skel_path, weights_path)
    """
    # Determine output filenames
    skel_path = asset_folder / f"{asset_name}.mpfbskel"
    weights_path = asset_folder / f"{asset_name}.mhw"

    # Save skeleton file
    with open(skel_path, 'w') as f:
        json.dump(skeleton, f, indent=2)

    if verbose:
        print(f"  Saved skeleton: {skel_path.name}")

    # Save weights file
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)

    if verbose:
        print(f"  Saved weights: {weights_path.name}")

    return skel_path, weights_path


# ============================================================================
# Main Processing Functions
# ============================================================================

def process_hair_asset(
    asset_folder: Path,
    config: Optional[HairPhysicsConfig] = None,
    verbose: bool = False
) -> bool:
    """
    Process a single hair asset folder and generate rigging files.

    Args:
        asset_folder: Path to the hair asset folder
        config: Hair physics configuration (uses defaults if None)
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = HairPhysicsConfig()

    asset_name = asset_folder.name

    if verbose:
        print(f"\nProcessing hair asset: {asset_name}")

    try:
        # Find OBJ file
        obj_files = list(asset_folder.glob("*.obj"))
        if not obj_files:
            print(f"  Error: No .obj file found in {asset_folder}")
            return False

        obj_path = obj_files[0]

        if verbose:
            print(f"  Loading mesh: {obj_path.name}")

        # Load vertices from OBJ
        vertices = load_obj_vertices(obj_path)

        if verbose:
            print(f"  Loaded {len(vertices)} vertices")

        if len(vertices) < config.min_region_vertices:
            print(f"  Skipping: Too few vertices ({len(vertices)})")
            return False

        # Estimate head position
        head_position = estimate_head_position(vertices)

        if verbose:
            print(f"  Estimated head position: ({head_position[0]:.3f}, "
                  f"{head_position[1]:.3f}, {head_position[2]:.3f})")

        # Analyze vertices
        vertices_data = analyze_vertices(vertices, head_position)

        # Segment into regions
        regions = segment_into_regions(vertices_data, config)

        if not regions:
            print(f"  Skipping: No valid hair regions found (hair may be too short)")
            return False

        if verbose:
            print(f"  Found {len(regions)} hair regions:")
            for region in regions:
                print(f"    {region.region_type.value}: {len(region.vertex_indices)} verts, "
                      f"{region.bone_count} bones, extent={region.extent:.1f}cm")

        # Generate MPFB2 files
        skeleton = generate_mpfbskel(regions, config, asset_name)
        weights = generate_mhw(regions, vertices, config)

        # Count total bones
        total_bones = sum(
            min(r.bone_count, config.max_bones_per_chain)
            for r in regions
        )

        if verbose:
            print(f"  Generated {total_bones} physics bones")

        # Save files
        save_rigging_files(asset_folder, asset_name, skeleton, weights, verbose)

        print(f"  Successfully generated rigging files for {asset_name}")
        return True

    except Exception as e:
        print(f"  Error processing {asset_name}: {e}")
        if verbose:
            traceback.print_exc()
        return False


def process_all_hair_assets(
    assets_dir: Path,
    config: Optional[HairPhysicsConfig] = None,
    verbose: bool = False
) -> Dict[str, bool]:
    """
    Process all hair assets in the mpfb_hair_assets folder.

    Args:
        assets_dir: Path to mpfb_hair_assets directory
        config: Hair physics configuration
        verbose: Enable verbose output

    Returns:
        Dictionary mapping asset names to success status
    """
    results = {}

    if not assets_dir.exists():
        print(f"Error: Assets directory not found: {assets_dir}")
        return results

    # Find all asset folders (those containing .mhclo files)
    for folder in sorted(assets_dir.iterdir()):
        if not folder.is_dir():
            continue

        mhclo_files = list(folder.glob("*.mhclo"))
        if not mhclo_files:
            continue

        success = process_hair_asset(folder, config, verbose)
        results[folder.name] = success

    return results


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(
        description='Generate MPFB2 rigging files for hair assets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single hair asset
  python run_blender.py --script mesh_hair_generation/generate_hair_rigging.py -- \\
      --asset mpfb_hair_assets/Long_Hair_A --verbose

  # Process all hair assets
  python run_blender.py --script mesh_hair_generation/generate_hair_rigging.py -- \\
      --all --verbose

  # Custom configuration
  python run_blender.py --script mesh_hair_generation/generate_hair_rigging.py -- \\
      --asset mpfb_hair_assets/Long_Hair_A --min-extent 5.0 --stiffness 0.7
        """
    )

    parser.add_argument(
        '--asset',
        type=str,
        help='Path to a specific hair asset folder to process'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all hair assets in mpfb_hair_assets folder'
    )

    parser.add_argument(
        '--assets-dir',
        type=str,
        default=None,
        help='Custom path to hair assets directory (default: ../mpfb_hair_assets)'
    )

    parser.add_argument(
        '--min-extent',
        type=float,
        default=3.0,
        help='Minimum hair extent (cm) for physics bones (default: 3.0)'
    )

    parser.add_argument(
        '--stiffness',
        type=float,
        default=0.8,
        help='Physics stiffness hint (0.0-1.0, default: 0.8)'
    )

    parser.add_argument(
        '--damping',
        type=float,
        default=0.5,
        help='Physics damping hint (0.0-1.0, default: 0.5)'
    )

    parser.add_argument(
        '--head-bone',
        type=str,
        default='head',
        help='Name of head bone to parent to (default: head)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    # Handle Blender's argument passing
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]

    args = parser.parse_args(argv)

    print("=" * 70)
    print("HAIR PHYSICS RIGGING FILE GENERATOR")
    print("=" * 70)

    # Build configuration
    config = HairPhysicsConfig(
        min_extent_for_bones=args.min_extent,
        stiffness=args.stiffness,
        damping=args.damping,
        head_bone_name=args.head_bone
    )

    # Determine assets directory
    script_dir = Path(__file__).parent.absolute()
    parent_dir = script_dir.parent.absolute()

    if args.assets_dir:
        assets_dir = Path(args.assets_dir)
    else:
        assets_dir = parent_dir / "mpfb_hair_assets"

    try:
        if args.all:
            # Process all assets
            print(f"\nProcessing all hair assets in: {assets_dir}")
            results = process_all_hair_assets(assets_dir, config, args.verbose)

            # Summary
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            success_count = sum(1 for v in results.values() if v)
            total_count = len(results)
            print(f"Processed {success_count}/{total_count} hair assets successfully")

            if args.verbose:
                for name, success in results.items():
                    status = "OK" if success else "FAILED"
                    print(f"  {name}: {status}")

        elif args.asset:
            # Process single asset
            asset_path = Path(args.asset)
            if not asset_path.is_absolute():
                asset_path = parent_dir / args.asset

            if not asset_path.exists():
                print(f"Error: Asset folder not found: {asset_path}")
                return 1

            success = process_hair_asset(asset_path, config, args.verbose)
            if not success:
                return 1

        else:
            print("Error: Specify --asset <path> or --all")
            parser.print_help()
            return 1

        print("\n" + "=" * 70)
        print("COMPLETE")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
