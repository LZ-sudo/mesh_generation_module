#!/usr/bin/env python3
"""
Hair Physics Rigging Generator

This module provides functions to dynamically generate hair physics bones
and weights at runtime during human mesh generation.

The bones are added directly to the main armature (parented to the head bone)
and vertex weights are calculated based on strand detection.

Detection Method:
    GEODESIC-BASED: Uses geodesic distance computation (Dijkstra on mesh edges)
    to detect hair strand paths on continuous mesh hair assets.

    Algorithm:
    1. Extract scalp reference from MPFB human mesh (head bone position)
    2. Find hair vertices in scalp region
    3. Compute geodesic distance from scalp using Dijkstra on mesh edges
    4. Find hair tip candidates (high geodesic distance)
    5. Trace strand paths from tips back to scalp
    6. Generate bones along detected strand paths
    7. Calculate vertex weights based on proximity to strands

Requirements:
    - numpy (included with Blender)
    - scipy (optional, for KDTree clustering - has greedy fallback)

Usage:
    from generate_hair_rigging import (
        add_hair_bones_to_armature,
        calculate_hair_vertex_weights,
    )

    # Add bones and get weight info
    bones, weight_info = add_hair_bones_to_armature(
        armature_obj, hair_obj, human_obj, verbose=True
    )

    # Apply weights to hair mesh
    calculate_hair_vertex_weights(hair_obj, weight_info, verbose=True)
"""

import sys
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

# Add script directory and parent to path for imports when running through Blender
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
for _path in [str(_SCRIPT_DIR), str(_PROJECT_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

if TYPE_CHECKING:
    from geodesic_strand_detection import HairStrand, GeodesicConfig

# Import geodesic-based hair detection
GEODESIC_IMPORT_ERROR = None
try:
    from geodesic_strand_detection import (
        HairStrand, GeodesicConfig,
        extract_mesh_for_geodesic, detect_strands_geodesic,
        assign_vertices_to_strands, check_dependencies
    )
    GEODESIC_AVAILABLE = True
except ImportError as e:
    GEODESIC_AVAILABLE = False
    GEODESIC_IMPORT_ERROR = str(e)
    HairStrand = None
    GeodesicConfig = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HairPhysicsConfig:
    """Configuration for hair physics bone generation."""

    # Geodesic detection parameters
    scalp_percentile: float = 12.0
    tip_percentile: float = 92.0
    tip_cluster_distance: float = 0.025
    max_strands: int = 60

    # Bone chain parameters
    min_strand_length: float = 0.03  # meters
    bones_per_10cm: float = 3.0
    max_bones_per_strand: int = 8
    min_bones_per_strand: int = 2

    # Weight parameters
    weight_falloff: float = 2.0

    # Bone containment (Issue 1: bones outside mesh)
    bone_inward_offset: float = 0.002

    # Path overlap filtering (Issue 2: overlapping chains)
    path_overlap_distance: float = 0.015
    path_overlap_ratio: float = 0.5

    # Physics hints (stored as custom properties)
    stiffness: float = 0.8
    damping: float = 0.5
    gravity_scale: float = 1.0

    # Head bone reference
    head_bone_name: str = "head"

    def to_geodesic_config(self) -> 'GeodesicConfig':
        """Convert to GeodesicConfig for the detection module."""
        if not GEODESIC_AVAILABLE:
            return None
        return GeodesicConfig(
            scalp_percentile=self.scalp_percentile,
            tip_percentile=self.tip_percentile,
            tip_cluster_distance=self.tip_cluster_distance,
            min_strand_length=self.min_strand_length,
            max_strands=self.max_strands,
            bones_per_10cm=self.bones_per_10cm,
            min_bones_per_strand=self.min_bones_per_strand,
            max_bones_per_strand=self.max_bones_per_strand,
            weight_falloff=self.weight_falloff,
            bone_inward_offset=self.bone_inward_offset,
            path_overlap_distance=self.path_overlap_distance,
            path_overlap_ratio=self.path_overlap_ratio
        )


# ============================================================================
# Scalp Reference Extraction
# ============================================================================

def extract_scalp_reference(
    human_obj,
    armature_obj,
    head_bone_name: str = "head",
    verbose: bool = False
) -> Optional[Tuple[Tuple[float, float, float], float]]:
    """
    Extract scalp reference point and radius from MPFB human mesh.

    Uses the head bone position and vertices weighted to the head to determine
    the scalp center and approximate radius for hair strand detection.

    Args:
        human_obj: The MPFB human mesh object
        armature_obj: The armature object containing the head bone
        head_bone_name: Name of the head bone (default: "head")
        verbose: Enable verbose output

    Returns:
        Tuple of ((x, y, z) scalp center, scalp_radius) or None if extraction fails
    """
    try:
        # Get head bone position from armature
        if armature_obj is None or armature_obj.type != 'ARMATURE':
            if verbose:
                print("  No valid armature provided for scalp reference")
            return None

        armature_data = armature_obj.data
        head_bone = armature_data.bones.get(head_bone_name)

        if head_bone is None:
            if verbose:
                print(f"  Head bone '{head_bone_name}' not found in armature")
            return None

        # Get head bone world position (use head of bone as scalp center)
        # The head bone's head position is at the neck, tail is at top of head
        bone_head_local = head_bone.head_local
        bone_tail_local = head_bone.tail_local

        # Transform to world coordinates
        armature_matrix = armature_obj.matrix_world
        bone_head_world = armature_matrix @ bone_head_local
        bone_tail_world = armature_matrix @ bone_tail_local

        # Scalp center is approximately at the top of the head (bone tail)
        # but slightly lower to account for where hair attaches
        scalp_center = (
            (bone_head_world.x + bone_tail_world.x) / 2,
            (bone_head_world.y + bone_tail_world.y) / 2,
            bone_tail_world.z  # Top of head
        )

        # Calculate scalp radius from head bone length
        bone_length = (bone_tail_world - bone_head_world).length
        scalp_radius = bone_length * 0.6  # Approximate scalp radius

        if verbose:
            print(f"  Scalp reference extracted:")
            print(f"    Center: ({scalp_center[0]:.3f}, {scalp_center[1]:.3f}, {scalp_center[2]:.3f})")
            print(f"    Radius: {scalp_radius*100:.1f}cm")

        return scalp_center, scalp_radius

    except Exception as e:
        if verbose:
            print(f"  Error extracting scalp reference: {e}")
        return None


def find_scalp_vertices_from_reference(
    vertices: 'np.ndarray',
    scalp_center: Tuple[float, float, float],
    scalp_radius: float,
    z_tolerance: float = 0.05,
    verbose: bool = False
) -> 'np.ndarray':
    """
    Find hair mesh vertices that are in the scalp region based on external reference.

    Uses the scalp center from the human mesh to identify which hair vertices
    are at the root (scalp attachment) region.

    Args:
        vertices: (N, 3) hair mesh vertex positions
        scalp_center: (x, y, z) scalp center from human mesh
        scalp_radius: Approximate radius of scalp region
        z_tolerance: Z-height tolerance for scalp region (meters)
        verbose: Enable verbose output

    Returns:
        Array of vertex indices in the scalp region
    """
    import numpy as np

    scalp_center_arr = np.array(scalp_center)

    # Find vertices near the scalp center in XY plane and at similar Z height
    xy_distances = np.sqrt(
        (vertices[:, 0] - scalp_center_arr[0])**2 +
        (vertices[:, 1] - scalp_center_arr[1])**2
    )

    z_coords = vertices[:, 2]
    z_max = z_coords.max()

    # Scalp vertices are:
    # 1. Within scalp_radius * 1.5 in XY plane (generous to catch all hair roots)
    # 2. In the upper portion of the hair mesh (top 20% by Z or within z_tolerance of scalp)
    z_threshold = max(scalp_center_arr[2] - z_tolerance, z_max - (z_max - z_coords.min()) * 0.2)

    scalp_mask = (xy_distances <= scalp_radius * 1.5) & (z_coords >= z_threshold)
    scalp_indices = np.where(scalp_mask)[0]

    # If we got very few vertices, fall back to just using Z-height
    if len(scalp_indices) < 10:
        if verbose:
            print(f"    Only {len(scalp_indices)} vertices near scalp center, using Z-height fallback")
        z_percentile = np.percentile(z_coords, 85)  # Top 15%
        scalp_mask = z_coords >= z_percentile
        scalp_indices = np.where(scalp_mask)[0]

    if verbose:
        print(f"    Found {len(scalp_indices)} scalp vertices from reference")

    return scalp_indices


# ============================================================================
# Geodesic-Based Hair Analysis
# ============================================================================

def analyze_hair_mesh_geodesic(
    hair_obj,
    config: Optional[HairPhysicsConfig] = None,
    scalp_reference: Optional[Tuple[Tuple[float, float, float], float]] = None,
    verbose: bool = False
) -> Optional[List['HairStrand']]:
    """
    Analyze hair mesh using geodesic distance computation.

    This detects hair strand paths on continuous mesh hair assets by:
    1. Finding scalp vertices using external reference (preferred) or Z-height
    2. Computing geodesic distance from scalp
    3. Tracing paths from tips back to scalp

    Args:
        hair_obj: The Blender hair mesh object
        config: Hair physics configuration
        scalp_reference: Optional tuple of (scalp_center, scalp_radius) from human mesh.
                        If provided, uses this to identify scalp vertices instead of
                        boundary detection.
        verbose: Enable verbose output

    Returns:
        List of HairStrand objects, or None if detection fails
    """
    if not GEODESIC_AVAILABLE:
        if verbose:
            print("  Geodesic detection not available")
            print(f"    Module not loaded")
        return None

    # Check dependencies
    available, msg = check_dependencies()
    if not available:
        if verbose:
            print(f"  {msg}")
        return None

    if config is None:
        config = HairPhysicsConfig()

    if verbose:
        print("  Extracting mesh data...")

    try:
        # Extract mesh data for geodesic computation
        vertices, faces, _ = extract_mesh_for_geodesic(hair_obj)

        if verbose:
            print(f"    {len(vertices)} vertices, {len(faces)} faces")

        # Convert config
        geo_config = config.to_geodesic_config()

        # Determine scalp vertices
        scalp_indices = None
        if scalp_reference is not None:
            scalp_center, scalp_radius = scalp_reference
            if verbose:
                print("  Using external scalp reference from human mesh...")
            scalp_indices = find_scalp_vertices_from_reference(
                vertices, scalp_center, scalp_radius, verbose=verbose
            )

        # Detect strands (pass scalp_indices if we have them)
        strands = detect_strands_geodesic(
            vertices, faces, geo_config, verbose,
            external_scalp_indices=scalp_indices
        )

        if strands is None:
            if verbose:
                print("  Geodesic strand detection failed")
            return None

        if verbose:
            print(f"  Detected {len(strands)} hair strands")

        return strands

    except Exception as e:
        if verbose:
            print(f"  Geodesic detection error: {e}")
            traceback.print_exc()
        return None


# ============================================================================
# Path Resampling Utility
# ============================================================================

def _resample_path(
    path: List[Tuple[float, float, float]],
    num_points: int
) -> List[Tuple[float, float, float]]:
    """Resample a path to have a specific number of evenly-spaced points."""
    import numpy as np

    if len(path) < 2:
        return path

    if num_points <= 2:
        return [path[0], path[-1]]

    # Calculate cumulative distances
    distances = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        dz = path[i][2] - path[i-1][2]
        distances.append(distances[-1] + np.sqrt(dx*dx + dy*dy + dz*dz))

    total_length = distances[-1]
    if total_length < 0.0001:
        return [path[0]] * num_points

    target_distances = np.linspace(0, total_length, num_points)

    resampled = []
    path_idx = 0

    for target_dist in target_distances:
        while path_idx < len(distances) - 1 and distances[path_idx + 1] < target_dist:
            path_idx += 1

        if path_idx >= len(path) - 1:
            resampled.append(path[-1])
            continue

        seg_start = distances[path_idx]
        seg_end = distances[path_idx + 1]
        seg_length = seg_end - seg_start

        if seg_length < 0.0001:
            t = 0.0
        else:
            t = (target_dist - seg_start) / seg_length

        p0 = path[path_idx]
        p1 = path[path_idx + 1]

        interpolated = (
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]),
            p0[2] + t * (p1[2] - p0[2])
        )
        resampled.append(interpolated)

    return resampled


# ============================================================================
# Direct Blender Armature Integration
# ============================================================================

def add_hair_bones_to_armature(
    armature_obj,
    hair_obj,
    human_obj=None,
    parent_bone_name: str = "head",
    config: Optional[HairPhysicsConfig] = None,
    verbose: bool = False
) -> Tuple[List[str], Optional[Dict]]:
    """
    Add hair physics bones directly to an existing Blender armature.

    Analyzes the hair mesh using geodesic strand detection and creates
    bones in the armature for each detected strand.

    Args:
        armature_obj: Blender armature object to add bones to
        hair_obj: Hair mesh object to analyze
        human_obj: Optional MPFB human mesh object for scalp reference.
                  If provided, uses the head bone position to identify scalp region.
        parent_bone_name: Name of the bone to parent hair bones to (default: "head")
        config: Hair physics configuration (uses defaults if None)
        verbose: Enable verbose output

    Returns:
        Tuple of (created_bone_names, weight_info):
        - created_bone_names: List of created bone names
        - weight_info: Dict containing strand/vertex data for weight calculation,
                       or None if analysis failed
    """
    import bpy

    if config is None:
        config = HairPhysicsConfig()

    # Extract scalp reference from human mesh if provided
    scalp_reference = None
    if human_obj is not None and armature_obj is not None:
        if verbose:
            print("  Extracting scalp reference from human mesh...")
        scalp_reference = extract_scalp_reference(
            human_obj, armature_obj, parent_bone_name, verbose
        )

    # Analyze hair mesh to detect strands
    if verbose:
        print("  Analyzing hair mesh for strand detection...")

    strands = analyze_hair_mesh_geodesic(hair_obj, config, scalp_reference, verbose)

    if strands is None or len(strands) == 0:
        if verbose:
            print("  No hair strands detected")
        return [], None

    if verbose:
        print(f"  Detected {len(strands)} hair strands")

    # Extract vertices for weight calculation
    mesh = hair_obj.data
    world_matrix = hair_obj.matrix_world
    vertices = []
    for v in mesh.vertices:
        world_pos = world_matrix @ v.co
        vertices.append((world_pos.x, world_pos.y, world_pos.z))

    # Store edit mode state
    original_mode = bpy.context.object.mode if bpy.context.object else 'OBJECT'
    original_active = bpy.context.view_layer.objects.active

    created_bones = []
    bone_prefix = "hair"

    try:
        # Switch to armature and enter edit mode
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')

        edit_bones = armature_obj.data.edit_bones

        # Find parent bone
        parent_bone = edit_bones.get(parent_bone_name)
        if parent_bone is None and verbose:
            print(f"  Warning: Parent bone '{parent_bone_name}' not found, using no parent")

        # Create bones for each strand
        for strand_idx, strand in enumerate(strands):
            path = strand.path_coords

            # Determine number of bones
            bone_count = int(strand.length * 100 * config.bones_per_10cm / 10.0)
            bone_count = max(config.min_bones_per_strand,
                            min(config.max_bones_per_strand, bone_count))

            # Resample path to get bone positions
            resampled = _resample_path(path, bone_count + 1)

            strand_parent = parent_bone

            for bone_idx in range(bone_count):
                bone_name = f"{bone_prefix}_{strand_idx}_{bone_idx}"

                head_pos = resampled[bone_idx]
                tail_pos = resampled[bone_idx + 1]

                # Create bone
                new_bone = edit_bones.new(bone_name)
                new_bone.head = head_pos
                new_bone.tail = tail_pos

                # Set parent
                if bone_idx == 0:
                    new_bone.parent = strand_parent
                    new_bone.use_connect = False
                else:
                    prev_bone_name = f"{bone_prefix}_{strand_idx}_{bone_idx - 1}"
                    prev_bone = edit_bones.get(prev_bone_name)
                    if prev_bone:
                        new_bone.parent = prev_bone
                        new_bone.use_connect = True

                # Set bone properties
                new_bone.use_deform = True

                created_bones.append(bone_name)

        # Exit edit mode
        bpy.ops.object.mode_set(mode='OBJECT')

        if verbose:
            print(f"  Created {len(created_bones)} hair bones")

        # Prepare weight info for calculate_hair_vertex_weights
        weight_info = {
            "strands": strands,
            "vertices": vertices,
            "config": config,
            "bone_prefix": bone_prefix
        }

        return created_bones, weight_info

    except Exception as e:
        if verbose:
            print(f"  Error creating bones: {e}")
            traceback.print_exc()

        # Try to exit edit mode
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except:
            pass

        return [], None

    finally:
        # Restore original state
        try:
            bpy.context.view_layer.objects.active = original_active
            if original_active and original_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode=original_mode)
        except:
            pass


def calculate_hair_vertex_weights(
    hair_obj,
    weight_info: Dict,
    verbose: bool = False
) -> bool:
    """
    Calculate and assign vertex weights for hair physics bones.

    Uses the strand detection results from add_hair_bones_to_armature
    to assign appropriate weights to each vertex.

    Args:
        hair_obj: Hair mesh object to assign weights to
        weight_info: Dict containing strands, vertices, and config from
                     add_hair_bones_to_armature
        verbose: Enable verbose output

    Returns:
        True if weights were successfully assigned, False otherwise
    """
    import bpy
    import numpy as np

    if weight_info is None:
        if verbose:
            print("  No weight info provided")
        return False

    strands = weight_info.get("strands")
    vertices = weight_info.get("vertices")
    config = weight_info.get("config", HairPhysicsConfig())
    bone_prefix = weight_info.get("bone_prefix", "hair")

    if strands is None or len(strands) == 0:
        if verbose:
            print("  No strands in weight info")
        return False

    if vertices is None or len(vertices) == 0:
        if verbose:
            print("  No vertices in weight info")
        return False

    try:
        vertices_array = np.array(vertices)

        # Assign vertices to strands
        if GEODESIC_AVAILABLE:
            geo_config = config.to_geodesic_config()
            strand_assignments = assign_vertices_to_strands(vertices_array, strands, geo_config)
        else:
            strand_assignments = {}

        # Calculate weights for each strand/bone
        bone_weights = {}  # bone_name -> list of (vert_idx, weight)

        for strand_idx, strand in enumerate(strands):
            path = strand.path_coords

            # Determine number of bones
            bone_count = int(strand.length * 100 * config.bones_per_10cm / 10.0)
            bone_count = max(config.min_bones_per_strand,
                            min(config.max_bones_per_strand, bone_count))

            # Get vertices assigned to this strand
            if strand_idx in strand_assignments:
                assigned_verts = strand_assignments[strand_idx]
            else:
                # Fallback: use path vertices
                assigned_verts = [(idx, 1.0) for idx in strand.path_vertex_indices]

            # Resample path for bone positions
            resampled = _resample_path(path, bone_count + 1)

            for vert_idx, base_weight in assigned_verts:
                if vert_idx >= len(vertices):
                    continue

                vert_pos = vertices_array[vert_idx]

                # Find which bone segment this vertex is closest to
                min_dist = float('inf')
                best_bone_idx = 0
                best_t = 0.0

                for i in range(bone_count):
                    bone_head = np.array(resampled[i])
                    bone_tail = np.array(resampled[i + 1])

                    # Project vertex onto bone segment
                    segment = bone_tail - bone_head
                    seg_len_sq = np.dot(segment, segment)

                    if seg_len_sq < 0.00001:
                        t = 0.0
                        proj = bone_head
                    else:
                        t = max(0.0, min(1.0, np.dot(vert_pos - bone_head, segment) / seg_len_sq))
                        proj = bone_head + t * segment

                    dist = np.linalg.norm(vert_pos - proj)

                    if dist < min_dist:
                        min_dist = dist
                        best_bone_idx = i
                        best_t = t

                # Calculate weight based on position along bone
                primary_weight = base_weight * (1.0 - best_t * 0.3)
                bone_name = f"{bone_prefix}_{strand_idx}_{best_bone_idx}"

                if bone_name not in bone_weights:
                    bone_weights[bone_name] = []
                bone_weights[bone_name].append((vert_idx, primary_weight))

                # Secondary bone weight for smooth blending
                if best_bone_idx + 1 < bone_count and best_t > 0.3:
                    secondary_weight = base_weight * best_t * 0.3
                    next_bone = f"{bone_prefix}_{strand_idx}_{best_bone_idx + 1}"

                    if next_bone not in bone_weights:
                        bone_weights[next_bone] = []
                    bone_weights[next_bone].append((vert_idx, secondary_weight))

        # Create vertex groups and assign weights
        total_weights = 0
        for bone_name, vert_weights in bone_weights.items():
            if not vert_weights:
                continue

            # Get or create vertex group
            vgroup = hair_obj.vertex_groups.get(bone_name)
            if vgroup is None:
                vgroup = hair_obj.vertex_groups.new(name=bone_name)

            # Assign weights
            for vert_idx, weight in vert_weights:
                vgroup.add([vert_idx], weight, 'REPLACE')
                total_weights += 1

        if verbose:
            print(f"  Assigned {total_weights} vertex weights to {len(bone_weights)} vertex groups")

        return total_weights > 0

    except Exception as e:
        if verbose:
            print(f"  Error calculating weights: {e}")
            traceback.print_exc()
        return False
