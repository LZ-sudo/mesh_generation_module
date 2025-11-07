"""
Measurement extraction functions for human models.

This module provides functions to extract body measurements from a rigged
human mesh in Blender using PURE MESH-BASED METHODS (no bone dependency):

Approach:
1. Find anatomical landmarks directly from mesh vertices
2. Use horizontal plane slicing for circumferences
3. Use bounding boxes and vertex extrema for lengths and widths
4. Calculate distances along mesh surface where appropriate

All measurements are returned in centimeters.
"""

import bpy
import math
from typing import Dict, List, Tuple, Optional
import bmesh
from mathutils import Vector

# ==================== VALIDATION RANGES ====================

# Expected ranges for adult human measurements (in cm)
VALIDATION_RANGES = {
    "height": (140, 220),
    "shoulder_width": (30, 60),
    "forearm_length": (20, 40),
    "upper_arm_length": (25, 45),
    "hand_length": (15, 25),
    "neck_length": (8, 20),
    # Width measurements
    "chest_width": (20, 50),      # Front-to-back chest depth
    "head_width": (12, 20)        # Head width (side to side)
}


# ==================== BONE NAME MAPPINGS ====================

# MPFB2 standard rig bone names for different rig types
BONE_NAMES = {
    "default": {
        "head_top": "head",
        "foot_left": "foot.L",
        "foot_right": "foot.R",
        "shoulder_left": "upperarm01.L",       # Start of upper arm (shoulder joint)
        "shoulder_right": "upperarm01.R",
        "hip_left": "upperleg01.L",            # Start of upper leg (hip joint)
        "hip_right": "upperleg01.R",
        "pelvis": "root",                      # Root bone (pelvis center)
        "elbow_left": "lowerarm01.L",          # Start of forearm (elbow joint)
        "elbow_right": "lowerarm01.R",
        "wrist_left": "wrist.L",               # Wrist joint
        "wrist_right": "wrist.R",
        "chest": "spine03",                    # Mid-upper spine (chest level)
        "waist": "spine02",                    # Lower-mid spine (waist level)
        "neck": "neck01"                       # Base of neck
    },
    "default_no_toes": {
        "head_top": "head",
        "foot_left": "foot.L",
        "foot_right": "foot.R",
        "shoulder_left": "upperarm01.L",       # Start of upper arm (shoulder joint)
        "shoulder_right": "upperarm01.R",
        "hip_left": "upperleg01.L",            # Start of upper leg (hip joint)
        "hip_right": "upperleg01.R",
        "pelvis": "root",                      # Root bone (pelvis center)
        "elbow_left": "lowerarm01.L",          # Start of forearm (elbow joint)
        "elbow_right": "lowerarm01.R",
        "wrist_left": "wrist.L",               # Wrist joint
        "wrist_right": "wrist.R",
        "chest": "spine03",                    # Mid-upper spine (chest level)
        "waist": "spine02",                    # Lower-mid spine (waist level)
        "neck": "neck01"                       # Base of neck
    }
}


# ==================== HELPER FUNCTIONS ====================

def get_bone_names(armature) -> Dict[str, str]:
    """
    Detect rig type and return appropriate bone name mapping.
    
    Args:
        armature: Blender armature object
        
    Returns:
        Dictionary mapping anatomical locations to bone names
    """
    # Try to detect rig type from armature name or bone structure
    # For now, default to "default_no_toes" as it's most commonly used
    return BONE_NAMES["default_no_toes"]


def get_bone_world_position(armature, bone_name: str, use_tail: bool = False) -> Tuple[float, float, float]:
    """
    Get world-space position of a bone.
    
    Args:
        armature: Blender armature object
        bone_name: Name of the bone
        use_tail: If True, use tail position; otherwise use head position
        
    Returns:
        Tuple of (x, y, z) coordinates in world space
    """
    if bone_name not in armature.pose.bones:
        raise ValueError(f"Bone '{bone_name}' not found in armature")
    
    bone = armature.pose.bones[bone_name]
    
    if use_tail:
        local_pos = bone.tail
    else:
        local_pos = bone.head
    
    # Convert to world space
    world_pos = armature.matrix_world @ local_pos
    
    return (world_pos.x, world_pos.y, world_pos.z)


def distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """
    Calculate Euclidean distance between two 3D points.
    
    Args:
        p1: First point (x, y, z)
        p2: Second point (x, y, z)
        
    Returns:
        Distance between points
    """
    return math.sqrt(
        (p2[0] - p1[0])**2 +
        (p2[1] - p1[1])**2 +
        (p2[2] - p1[2])**2
    )


def validate_measurement(name: str, value: float) -> bool:
    """
    Validate that a measurement is within expected range.
    
    Args:
        name: Measurement name
        value: Measurement value in cm
        
    Returns:
        True if valid, False otherwise
    """
    if name not in VALIDATION_RANGES:
        print(f"⚠ Warning: No validation range for '{name}'")
        return True
    
    min_val, max_val = VALIDATION_RANGES[name]
    
    if not (min_val <= value <= max_val):
        print(f"⚠ Warning: {name} = {value:.1f}cm is outside expected range [{min_val}, {max_val}]")
        return False
    
    return True


# ==================== MESH ANALYSIS HELPER FUNCTIONS ====================

def get_mesh_vertices_world_space(mesh_obj) -> List[Vector]:
    """
    Get all mesh vertices in world space coordinates.

    Args:
        mesh_obj: Blender mesh object

    Returns:
        List of vertex positions in world space
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)

    vertices = [mesh_obj.matrix_world @ v.co for v in eval_mesh.vertices]

    eval_obj.to_mesh_clear()
    return vertices


def find_anatomical_landmarks(mesh_obj) -> Dict[str, float]:
    """
    Find key anatomical landmark heights directly from mesh geometry.

    This analyzes the mesh to find anatomical features without relying on bones.

    Args:
        mesh_obj: Blender mesh object

    Returns:
        Dictionary with anatomical landmark Z-heights in world space
    """
    vertices = get_mesh_vertices_world_space(mesh_obj)

    # Get all Z coordinates
    z_coords = [v.z for v in vertices]
    x_coords = [v.x for v in vertices]

    # Basic landmarks from extrema
    landmarks = {
        'head_top': max(z_coords),
        'feet_bottom': min(z_coords),
    }

    # Height of the model
    height = landmarks['head_top'] - landmarks['feet_bottom']

    # Estimate anatomical positions as fractions of height
    # These ratios are based on human anatomy proportions
    landmarks['chest'] = landmarks['feet_bottom'] + (height * 0.68)      # ~68% up from feet
    landmarks['waist'] = landmarks['feet_bottom'] + (height * 0.58)      # ~58% up from feet
    landmarks['hips'] = landmarks['feet_bottom'] + (height * 0.52)       # ~52% up from feet
    landmarks['neck_base'] = landmarks['feet_bottom'] + (height * 0.85)  # ~85% up from feet
    landmarks['crotch'] = landmarks['feet_bottom'] + (height * 0.48)     # ~48% up from feet (inseam reference)

    # Find shoulder height by looking for widest point in upper body
    upper_body_verts = [v for v in vertices if v.z > landmarks['chest']]
    if upper_body_verts:
        # Group by height and find widest horizontal span
        max_width = 0
        shoulder_height = landmarks['chest']

        # Sample at different heights
        for sample_z in [landmarks['feet_bottom'] + (height * h) for h in [0.75, 0.78, 0.81]]:
            nearby_verts = [v for v in vertices if abs(v.z - sample_z) < height * 0.03]
            if nearby_verts:
                width = max(v.x for v in nearby_verts) - min(v.x for v in nearby_verts)
                if width > max_width:
                    max_width = width
                    shoulder_height = sample_z

        landmarks['shoulders'] = shoulder_height
    else:
        landmarks['shoulders'] = landmarks['feet_bottom'] + (height * 0.80)

    return landmarks


def measure_circumference_at_height(mesh_obj, target_z: float, tolerance: float = 0.01) -> float:
    """
    Measure circumference at a specific height using edge intersection method.

    This is an improved version that properly slices the mesh.

    Args:
        mesh_obj: Blender mesh object
        target_z: Height (Z coordinate) to measure at in world space
        tolerance: Tolerance for finding intersections

    Returns:
        Circumference in centimeters
    """
    # Create a bmesh copy of the evaluated mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)

    bm = bmesh.new()
    bm.from_mesh(eval_mesh)
    bm.transform(mesh_obj.matrix_world)  # Transform to world space
    bmesh.ops.triangulate(bm, faces=bm.faces)

    intersection_points = []

    # Find edge intersections with the horizontal plane at target_z
    for edge in bm.edges:
        v1, v2 = edge.verts
        z1, z2 = v1.co.z, v2.co.z

        # Check if edge crosses the slicing plane
        if (z1 - target_z) * (z2 - target_z) < 0:
            # Linear interpolation to find intersection point
            t = (target_z - z1) / (z2 - z1)
            intersection = v1.co.lerp(v2.co, t)
            intersection_points.append(Vector((intersection.x, intersection.y, target_z)))

    bm.free()
    eval_obj.to_mesh_clear()

    if len(intersection_points) < 3:
        print(f"⚠ Warning: Only {len(intersection_points)} intersection points at z={target_z:.3f}m")
        return 0.0

    # Remove duplicate points
    unique_points = []
    threshold = 1e-4
    for point in intersection_points:
        if not any((point - existing).length < threshold for existing in unique_points):
            unique_points.append(point)

    if len(unique_points) < 3:
        return 0.0

    # Compute 2D convex hull to get outer contour
    from mathutils import geometry
    coords_2d = [(p.x, p.y) for p in unique_points]

    try:
        hull_indices = geometry.convex_hull_2d(coords_2d)
    except:
        print(f"⚠ Warning: Failed to compute convex hull at z={target_z:.3f}m")
        return 0.0

    if not hull_indices:
        return 0.0

    # Calculate perimeter
    perimeter = 0.0
    for i in range(len(hull_indices)):
        p1 = Vector((coords_2d[hull_indices[i]][0], coords_2d[hull_indices[i]][1], 0))
        p2 = Vector((coords_2d[hull_indices[(i + 1) % len(hull_indices)]][0],
                     coords_2d[hull_indices[(i + 1) % len(hull_indices)]][1], 0))
        perimeter += (p2 - p1).length

    return perimeter * 100.0  # Convert to cm


def measure_width_at_height(mesh_obj, target_z: float, tolerance: float = 0.05) -> float:
    """
    Measure horizontal width (X-axis span) at a specific height.

    Args:
        mesh_obj: Blender mesh object
        target_z: Height (Z coordinate) to measure at
        tolerance: Height tolerance for selecting vertices

    Returns:
        Width in centimeters
    """
    vertices = get_mesh_vertices_world_space(mesh_obj)

    # Find vertices near the target height
    nearby_verts = [v for v in vertices if abs(v.z - target_z) < tolerance]

    if len(nearby_verts) < 2:
        return 0.0

    # Calculate width as max X - min X
    x_coords = [v.x for v in nearby_verts]
    width = (max(x_coords) - min(x_coords)) * 100  # Convert to cm

    return width


def measure_depth_at_height(mesh_obj, target_z: float, tolerance: float = 0.05) -> float:
    """
    Measure front-to-back depth (Y-axis span) at a specific height.

    Args:
        mesh_obj: Blender mesh object
        target_z: Height (Z coordinate) to measure at
        tolerance: Height tolerance for selecting vertices

    Returns:
        Depth in centimeters
    """
    vertices = get_mesh_vertices_world_space(mesh_obj)

    # Find vertices near the target height
    nearby_verts = [v for v in vertices if abs(v.z - target_z) < tolerance]

    if len(nearby_verts) < 2:
        return 0.0

    # Calculate depth as max Y - min Y
    y_coords = [v.y for v in nearby_verts]
    depth = (max(y_coords) - min(y_coords)) * 100  # Convert to cm

    return depth


def measure_vertical_distance(mesh_obj, z_top: float, z_bottom: float) -> float:
    """
    Measure vertical distance between two heights.

    Args:
        mesh_obj: Blender mesh object
        z_top: Top Z coordinate
        z_bottom: Bottom Z coordinate

    Returns:
        Distance in centimeters
    """
    return abs(z_top - z_bottom) * 100  # Convert to cm


# ==================== POSE MANIPULATION ====================

def set_armature_to_tpose(armature):
    """
    Set armature to T-pose by rotating shoulder bones to horizontal.

    This makes arm measurements more accurate by extending arms perpendicular to body.

    Args:
        armature: Blender armature object

    Returns:
        Dictionary with original rotations for restoration
    """
    import bpy
    from mathutils import Euler
    import math

    if armature is None or armature.type != 'ARMATURE':
        print("⚠ Warning: No armature provided for T-pose")
        return {}

    # Store original pose
    original_rotations = {}

    # Enter pose mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    # Common shoulder bone name patterns
    shoulder_patterns = {
        'left': ['upperarm01.L', 'upper_arm.L', 'shoulder.L', 'arm.L'],
        'right': ['upperarm01.R', 'upper_arm.R', 'shoulder.R', 'arm.R']
    }

    # Find and rotate shoulder bones
    for side, patterns in shoulder_patterns.items():
        for pattern in patterns:
            if pattern in armature.pose.bones:
                bone = armature.pose.bones[pattern]

                # Store original rotation
                original_rotations[pattern] = bone.rotation_euler.copy()

                # Set to T-pose (arms horizontal)
                # Left arm: rotate up, Right arm: rotate up
                if side == 'left':
                    bone.rotation_euler = Euler((0, 0, math.radians(90)), 'XYZ')
                else:
                    bone.rotation_euler = Euler((0, 0, math.radians(-90)), 'XYZ')

                print(f"  Set {pattern} to T-pose")
                break  # Found the bone, move to next side

    # Update armature
    bpy.context.view_layer.update()

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    return original_rotations


def restore_armature_pose(armature, original_rotations):
    """
    Restore armature to original pose.

    Args:
        armature: Blender armature object
        original_rotations: Dictionary of original rotations from set_armature_to_tpose
    """
    import bpy

    if not original_rotations or armature is None:
        return

    # Enter pose mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    # Restore rotations
    for bone_name, rotation in original_rotations.items():
        if bone_name in armature.pose.bones:
            armature.pose.bones[bone_name].rotation_euler = rotation

    # Update armature
    bpy.context.view_layer.update()

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')


# ==================== JOINT-BASED SEGMENTATION & BOUNDING BOX MEASUREMENTS ====================

def find_body_joints(mesh_obj) -> Dict[str, Vector]:
    """
    Find major body joint locations by analyzing mesh geometry.

    This finds joints by looking for narrowest points and anatomical proportions.

    Args:
        mesh_obj: Blender mesh object

    Returns:
        Dictionary mapping joint names to 3D positions
    """
    vertices = get_mesh_vertices_world_space(mesh_obj)

    # Get basic dimensions
    z_coords = [v.z for v in vertices]
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]

    min_z, max_z = min(z_coords), max(z_coords)
    height = max_z - min_z

    joints = {}

    # Head and feet (extrema)
    joints['head_top'] = Vector((0, sum(y_coords)/len(y_coords), max_z))
    joints['feet'] = Vector((0, sum(y_coords)/len(y_coords), min_z))

    # Use anatomical proportions for main body joints
    joints['neck'] = Vector((0, 0, min_z + height * 0.87))
    joints['shoulders_center'] = Vector((0, 0, min_z + height * 0.82))
    joints['chest'] = Vector((0, 0, min_z + height * 0.68))
    joints['waist'] = Vector((0, 0, min_z + height * 0.58))
    joints['hips_center'] = Vector((0, 0, min_z + height * 0.52))
    joints['crotch'] = Vector((0, 0, min_z + height * 0.48))

    # Find left/right shoulders by looking for widest points in shoulder region
    shoulder_height = joints['shoulders_center'].z
    shoulder_verts = [v for v in vertices if abs(v.z - shoulder_height) < height * 0.02]
    if shoulder_verts:
        left_shoulder = min(shoulder_verts, key=lambda v: v.x)
        right_shoulder = max(shoulder_verts, key=lambda v: v.x)
        joints['shoulder_left'] = left_shoulder
        joints['shoulder_right'] = right_shoulder

    # Find elbows (narrowest point in middle of arm region)
    # Arms typically extend from 52% to 82% of height
    arm_mid_z = min_z + height * 0.67  # Roughly elbow height

    # Look for arms extending laterally
    arm_verts = [v for v in vertices if arm_mid_z - height*0.1 < v.z < arm_mid_z + height*0.1]

    # Left elbow: find narrowest point on left side
    left_arm_verts = [v for v in arm_verts if v.x < -height*0.05]  # Left side
    if left_arm_verts:
        # Group by height and find narrowest
        elbow_candidates = []
        for test_z in [arm_mid_z + height * offset for offset in [-0.05, 0, 0.05]]:
            nearby = [v for v in left_arm_verts if abs(v.z - test_z) < height * 0.03]
            if nearby:
                # Find the most lateral point at this height (farthest left)
                leftmost = min(nearby, key=lambda v: v.x)
                elbow_candidates.append(leftmost)

        if elbow_candidates:
            joints['elbow_left'] = min(elbow_candidates, key=lambda v: abs(v.x))  # Closest to body

    # Right elbow
    right_arm_verts = [v for v in arm_verts if v.x > height*0.05]  # Right side
    if right_arm_verts:
        elbow_candidates = []
        for test_z in [arm_mid_z + height * offset for offset in [-0.05, 0, 0.05]]:
            nearby = [v for v in right_arm_verts if abs(v.z - test_z) < height * 0.03]
            if nearby:
                rightmost = max(nearby, key=lambda v: v.x)
                elbow_candidates.append(rightmost)

        if elbow_candidates:
            joints['elbow_right'] = max(elbow_candidates, key=lambda v: v.x)

    # Find wrists (lowest point on arms, laterally positioned)
    wrist_z = min_z + height * 0.48  # Wrist height
    wrist_verts = [v for v in vertices if abs(v.z - wrist_z) < height * 0.05]

    left_wrist_verts = [v for v in wrist_verts if v.x < -height*0.02]
    right_wrist_verts = [v for v in wrist_verts if v.x > height*0.02]

    if left_wrist_verts:
        joints['wrist_left'] = min(left_wrist_verts, key=lambda v: v.x)
    if right_wrist_verts:
        joints['wrist_right'] = max(right_wrist_verts, key=lambda v: v.x)

    return joints


def segment_mesh_region(mesh_obj, z_min: float, z_max: float,
                        x_min: float = None, x_max: float = None) -> List[Vector]:
    """
    Extract vertices from a specific region of the mesh.

    Args:
        mesh_obj: Blender mesh object
        z_min: Minimum Z coordinate
        z_max: Maximum Z coordinate
        x_min: Optional minimum X coordinate
        x_max: Optional maximum X coordinate

    Returns:
        List of vertices in the region
    """
    vertices = get_mesh_vertices_world_space(mesh_obj)

    filtered = []
    for v in vertices:
        # Z bounds
        if v.z < z_min or v.z > z_max:
            continue

        # Optional X bounds
        if x_min is not None and v.x < x_min:
            continue
        if x_max is not None and v.x > x_max:
            continue

        filtered.append(v)

    return filtered


def calculate_oriented_bounding_box_length(vertices: List[Vector]) -> float:
    """
    Calculate the length of an oriented bounding box for a set of vertices.

    This finds the longest dimension of the bounding box.

    Args:
        vertices: List of vertex positions

    Returns:
        Length in meters
    """
    if len(vertices) < 2:
        return 0.0

    # Simple axis-aligned bounding box
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    z_coords = [v.z for v in vertices]

    x_span = max(x_coords) - min(x_coords)
    y_span = max(y_coords) - min(y_coords)
    z_span = max(z_coords) - min(z_coords)

    # Return the maximum span
    return max(x_span, y_span, z_span)


def measure_segment_length(mesh_obj, z_bottom: float, z_top: float,
                           x_min: float = None, x_max: float = None) -> float:
    """
    Measure the length of a body segment between two heights.

    Args:
        mesh_obj: Blender mesh object
        z_bottom: Bottom Z coordinate
        z_top: Top Z coordinate
        x_min: Optional X minimum (for left/right isolation)
        x_max: Optional X maximum (for left/right isolation)

    Returns:
        Length in centimeters
    """
    # Get vertices in this segment
    segment_verts = segment_mesh_region(mesh_obj, z_bottom, z_top, x_min, x_max)

    if len(segment_verts) < 2:
        return 0.0

    # Calculate bounding box length
    length = calculate_oriented_bounding_box_length(segment_verts)

    return length * 100  # Convert to cm


def measure_limb_with_joints(mesh_obj, joint_start: Vector, joint_end: Vector,
                             lateral_offset: float = 0.0) -> float:
    """
    Measure a limb segment between two joints using bounding box.

    Args:
        mesh_obj: Blender mesh object
        joint_start: Starting joint position
        joint_end: Ending joint position
        lateral_offset: Offset in X direction (negative for left, positive for right)

    Returns:
        Length in centimeters
    """
    z_min = min(joint_start.z, joint_end.z)
    z_max = max(joint_start.z, joint_end.z)

    # Define X bounds based on lateral offset
    if lateral_offset < 0:  # Left side
        x_min = None
        x_max = lateral_offset * 0.5  # Midline
    elif lateral_offset > 0:  # Right side
        x_min = lateral_offset * 0.5
        x_max = None
    else:  # Center
        x_min = None
        x_max = None

    return measure_segment_length(mesh_obj, z_min, z_max, x_min, x_max)


# ==================== VERTEX GROUP-BASED MEASUREMENTS ====================

def get_vertex_group_names(mesh_obj) -> List[str]:
    """
    Get all vertex group names from a mesh object.

    Args:
        mesh_obj: Blender mesh object

    Returns:
        List of vertex group names
    """
    return [vg.name for vg in mesh_obj.vertex_groups]


def print_vertex_groups(mesh_obj):
    """
    Print all vertex groups for debugging purposes.

    Args:
        mesh_obj: Blender mesh object
    """
    groups = get_vertex_group_names(mesh_obj)
    print(f"\nVertex Groups found ({len(groups)} total):")
    for i, group in enumerate(groups, 1):
        print(f"  {i:3d}. {group}")
    print()


def find_relevant_vertex_groups(mesh_obj) -> Dict[str, str]:
    """
    Find vertex groups relevant for body measurements.

    This function searches for common naming patterns in MakeHuman/MPFB2 models.

    Args:
        mesh_obj: Blender mesh object

    Returns:
        Dictionary mapping anatomical parts to vertex group names
    """
    all_groups = get_vertex_group_names(mesh_obj)
    mapping = {}

    # Common patterns for different body parts
    patterns = {
        'left_arm': ['leftarm', 'left_arm', 'arm.L', 'upperarm.L', 'l_arm'],
        'right_arm': ['rightarm', 'right_arm', 'arm.R', 'upperarm.R', 'r_arm'],
        'left_forearm': ['left_forearm', 'forearm.L', 'lowerarm.L', 'l_forearm'],
        'right_forearm': ['right_forearm', 'forearm.R', 'lowerarm.R', 'r_forearm'],
        'left_hand': ['left_hand', 'hand.L', 'l_hand'],
        'right_hand': ['right_hand', 'hand.R', 'r_hand'],
        'left_shoulder': ['left_shoulder', 'shoulder.L', 'l_shoulder'],
        'right_shoulder': ['right_shoulder', 'shoulder.R', 'r_shoulder'],
    }

    # Search for matches (case-insensitive)
    for part, pattern_list in patterns.items():
        for pattern in pattern_list:
            for group in all_groups:
                if pattern.lower() in group.lower():
                    mapping[part] = group
                    break
            if part in mapping:
                break

    return mapping


def get_vertex_group_vertices(mesh_obj, group_name: str) -> List[Vector]:
    """
    Get all vertices belonging to a vertex group in world space.

    Args:
        mesh_obj: Blender mesh object
        group_name: Name of the vertex group

    Returns:
        List of vertex positions in world space
    """
    if group_name not in mesh_obj.vertex_groups:
        return []

    # Get the vertex group
    vg = mesh_obj.vertex_groups[group_name]
    vg_index = vg.index

    # Get evaluated mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)

    # Collect vertices that belong to this group
    vertices = []
    for vert in eval_mesh.vertices:
        for group in vert.groups:
            if group.group == vg_index and group.weight > 0.1:  # Threshold to avoid negligible weights
                world_pos = mesh_obj.matrix_world @ vert.co
                vertices.append(world_pos)
                break

    eval_obj.to_mesh_clear()
    return vertices


def measure_limb_length_from_vertex_group(mesh_obj, group_name: str) -> float:
    """
    Measure the length of a limb segment using its vertex group.

    This calculates the length as the maximum distance along the limb's primary axis.

    Args:
        mesh_obj: Blender mesh object
        group_name: Name of the vertex group

    Returns:
        Length in centimeters
    """
    vertices = get_vertex_group_vertices(mesh_obj, group_name)

    if len(vertices) < 2:
        return 0.0

    # Get bounding box of the vertex group
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    z_coords = [v.z for v in vertices]

    # Calculate spans in each direction
    x_span = max(x_coords) - min(x_coords)
    y_span = max(y_coords) - min(y_coords)
    z_span = max(z_coords) - min(z_coords)

    # The limb length is the maximum span (limbs are typically aligned along one axis)
    length = max(x_span, y_span, z_span) * 100  # Convert to cm

    return length


def measure_arm_from_vertex_groups(mesh_obj, side: str = "left") -> Dict[str, float]:
    """
    Measure arm dimensions using vertex groups.

    Args:
        mesh_obj: Blender mesh object
        side: "left" or "right"

    Returns:
        Dictionary with arm measurements in centimeters
    """
    measurements = {
        'arm_length': 0.0,
        'upper_arm_length': 0.0,
        'forearm_length': 0.0
    }

    # Find relevant vertex groups
    vg_mapping = find_relevant_vertex_groups(mesh_obj)

    # Determine which groups to use based on side
    arm_key = f'{side}_arm'
    forearm_key = f'{side}_forearm'

    # Measure upper arm (if vertex group exists)
    if arm_key in vg_mapping:
        measurements['upper_arm_length'] = measure_limb_length_from_vertex_group(
            mesh_obj, vg_mapping[arm_key]
        )
        validate_measurement('upper_arm_length', measurements['upper_arm_length'])

    # Measure forearm (if vertex group exists)
    if forearm_key in vg_mapping:
        measurements['forearm_length'] = measure_limb_length_from_vertex_group(
            mesh_obj, vg_mapping[forearm_key]
        )
        validate_measurement('forearm_length', measurements['forearm_length'])

    # Total arm length is sum of segments
    measurements['arm_length'] = measurements['upper_arm_length'] + measurements['forearm_length']
    if measurements['arm_length'] > 0:
        validate_measurement('arm_length', measurements['arm_length'])

    return measurements


# ==================== SKELETAL MEASUREMENTS (BONE-BASED) ====================

def measure_bone_chain_length(armature, bone_names: List[str]) -> float:
    """
    Measure the length of a chain of bones by finding the distance between
    the bottom of the first bone and the top of the last bone.

    This is more accurate than joint-to-joint distance for multi-bone segments.

    Args:
        armature: Blender armature object
        bone_names: List of bone names in order (e.g., ['upperarm01.L', 'upperarm02.L'])

    Returns:
        Length in centimeters
    """
    if not bone_names:
        return 0.0

    # Get the bottom point of the first bone (head position)
    first_bone_pos = get_bone_world_position(armature, bone_names[0], use_tail=False)

    # Get the top point of the last bone (tail position)
    last_bone_pos = get_bone_world_position(armature, bone_names[-1], use_tail=True)

    # Calculate distance
    length = distance_3d(first_bone_pos, last_bone_pos) * 100  # Convert to cm

    return length


def measure_height(armature) -> float:
    """
    Measure total height from head top to feet.
    
    Args:
        armature: Blender armature object
        
    Returns:
        Height in centimeters
    """
    bone_names = get_bone_names(armature)
    
    # Get head top position (use tail of head bone for top)
    head_pos = get_bone_world_position(armature, bone_names["head_top"], use_tail=True)
    
    # Get foot positions (use minimum Z of left and right feet)
    foot_left_pos = get_bone_world_position(armature, bone_names["foot_left"])
    foot_right_pos = get_bone_world_position(armature, bone_names["foot_right"])
    foot_bottom_z = min(foot_left_pos[2], foot_right_pos[2])
    
    # Calculate height
    height = (head_pos[2] - foot_bottom_z) * 100  # Convert to cm
    
    validate_measurement("height", height)
    return height


def measure_shoulder_width(armature) -> float:
    """
    Measure shoulder width (biacromial breadth).
    
    Args:
        armature: Blender armature object
        
    Returns:
        Shoulder width in centimeters
    """
    bone_names = get_bone_names(armature)
    
    # Get shoulder joint positions (head of upper arm bones)
    left_shoulder = get_bone_world_position(armature, bone_names["shoulder_left"])
    right_shoulder = get_bone_world_position(armature, bone_names["shoulder_right"])
    
    # Calculate width (horizontal distance)
    width = abs(right_shoulder[0] - left_shoulder[0]) * 100  # Convert to cm
    
    validate_measurement("shoulder_width", width)
    return width


def measure_hip_width(armature) -> float:
    """
    Measure hip width (bi-iliac breadth).
    
    Args:
        armature: Blender armature object
        
    Returns:
        Hip width in centimeters
    """
    bone_names = get_bone_names(armature)
    
    # Get hip joint positions (head of thigh bones)
    left_hip = get_bone_world_position(armature, bone_names["hip_left"])
    right_hip = get_bone_world_position(armature, bone_names["hip_right"])
    
    # Calculate width (horizontal distance)
    width = abs(right_hip[0] - left_hip[0]) * 100  # Convert to cm
    
    validate_measurement("hip_width", width)
    return width


def measure_inseam(armature) -> float:
    """
    Measure inseam (leg length from crotch to floor).
    
    Args:
        armature: Blender armature object
        
    Returns:
        Inseam in centimeters
    """
    bone_names = get_bone_names(armature)
    
    # Get pelvis position (crotch level)
    pelvis_pos = get_bone_world_position(armature, bone_names["pelvis"])
    
    # Get foot position
    foot_left_pos = get_bone_world_position(armature, bone_names["foot_left"])
    foot_right_pos = get_bone_world_position(armature, bone_names["foot_right"])
    foot_bottom_z = min(foot_left_pos[2], foot_right_pos[2])
    
    # Calculate inseam (vertical distance)
    inseam = (pelvis_pos[2] - foot_bottom_z) * 100  # Convert to cm
    
    validate_measurement("inseam", inseam)
    return inseam


def measure_arm_length(armature, side: str = "left") -> float:
    """
    Measure full arm length from shoulder to wrist.
    
    Args:
        armature: Blender armature object
        side: "left" or "right"
        
    Returns:
        Arm length in centimeters
    """
    bone_names = get_bone_names(armature)
    
    shoulder_key = f"shoulder_{side}"
    wrist_key = f"wrist_{side}"
    
    # Get shoulder and wrist positions
    shoulder_pos = get_bone_world_position(armature, bone_names[shoulder_key])
    wrist_pos = get_bone_world_position(armature, bone_names[wrist_key])
    
    # Calculate length
    length = distance_3d(shoulder_pos, wrist_pos) * 100  # Convert to cm
    
    validate_measurement("arm_length", length)
    return length


def measure_forearm_length(armature, side: str = "left") -> float:
    """
    Measure forearm length using lowerarm01 and lowerarm02 bones.

    Measures the distance between the bottom of lowerarm01 and the top of lowerarm02.

    Args:
        armature: Blender armature object
        side: "left" or "right"

    Returns:
        Forearm length in centimeters
    """
    # Determine bone suffix
    suffix = ".L" if side == "left" else ".R"

    # Build bone chain
    bone_chain = [f"lowerarm01{suffix}", f"lowerarm02{suffix}"]

    # Measure using bone chain
    length = measure_bone_chain_length(armature, bone_chain)

    validate_measurement("forearm_length", length)
    return length


def measure_upper_arm_length(armature, side: str = "left") -> float:
    """
    Measure upper arm length using upperarm01 and upperarm02 bones.

    Measures the distance between the bottom of upperarm01 and the top of upperarm02.

    Args:
        armature: Blender armature object
        side: "left" or "right"

    Returns:
        Upper arm length in centimeters
    """
    # Determine bone suffix
    suffix = ".L" if side == "left" else ".R"

    # Build bone chain
    bone_chain = [f"upperarm01{suffix}", f"upperarm02{suffix}"]

    # Measure using bone chain
    length = measure_bone_chain_length(armature, bone_chain)

    validate_measurement("upper_arm_length", length)
    return length


def measure_neck_length(armature) -> float:
    """
    Measure neck length using neck01 and neck02 bones.

    Measures the distance between the bottom of neck01 and the top of neck02.

    Args:
        armature: Blender armature object

    Returns:
        Neck length in centimeters
    """
    # Build bone chain
    bone_chain = ["neck01", "neck02"]

    # Measure using bone chain
    length = measure_bone_chain_length(armature, bone_chain)

    validate_measurement("neck_length", length)
    return length


def measure_hand_length(armature, side: str = "left") -> float:
    """
    Measure hand length using wrist to finger3-3 bones.

    Measures the distance between the bottom of wrist and the top of finger3-3
    (middle finger tip).

    Args:
        armature: Blender armature object
        side: "left" or "right"

    Returns:
        Hand length in centimeters
    """
    # Determine bone suffix
    suffix = ".L" if side == "left" else ".R"

    # Build bone chain from wrist to middle finger tip
    bone_chain = [f"wrist{suffix}", f"finger3-3{suffix}"]

    # Measure using bone chain
    length = measure_bone_chain_length(armature, bone_chain)

    validate_measurement("hand_length", length)
    return length


# ==================== CIRCUMFERENCE MEASUREMENTS (MESH-BASED) ====================


def measure_circumference(
    mesh,
    armature,
    bone_name: str,
    height_offset: float = 0.0,
    tolerance: float = 0.001
) -> float:
    """
    Measure circumference at a bone-anchored horizontal plane using bmesh slicing.

    Args:
        mesh: Blender mesh object
        armature: Blender armature object
        bone_name: Name of reference bone
        height_offset: Offset (in meters) from bone head position
        tolerance: Small tolerance for slicing plane

    Returns:
        Circumference in centimeters
    """
    # Get slice height in world space
    bone = armature.pose.bones[bone_name]
    bone_world_pos = armature.matrix_world @ bone.head
    target_z = bone_world_pos.z + height_offset

    # Create a bmesh copy of the evaluated mesh (after modifiers)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)

    bm = bmesh.new()
    bm.from_mesh(eval_mesh)
    bm.transform(mesh.matrix_world)  # bring to world space
    bmesh.ops.triangulate(bm, faces=bm.faces)

    plane_co = Vector((0, 0, target_z))
    plane_no = Vector((0, 0, 1))

    intersection_points = []

    # Loop over edges and find intersections with the Z-plane
    for e in bm.edges:
        v1, v2 = e.verts
        z1, z2 = v1.co.z, v2.co.z

        # Check if edge crosses the slicing plane
        if (z1 - target_z) * (z2 - target_z) < 0:
            t = (target_z - z1) / (z2 - z1)
            p = v1.co.lerp(v2.co, t)
            intersection_points.append(Vector((p.x, p.y, target_z)))

    bm.free()
    mesh.to_mesh_clear()

    if len(intersection_points) < 3:
        print(f"⚠ Warning: Only {len(intersection_points)} intersection points at {bone_name} (z={target_z:.3f})")
        return 0.0

    # Group nearby points to reduce duplicates
    merged_points = []
    threshold = 1e-4
    for p in intersection_points:
        if not any((p - q).length < threshold for q in merged_points):
            merged_points.append(p)

    # Compute 2D convex hull (approximate outer contour)
    from mathutils import geometry
    coords_2d = [(p.x, p.y) for p in merged_points]
    hull_indices = geometry.convex_hull_2d(coords_2d)
    if not hull_indices:
        print(f"⚠ Warning: Failed to compute convex hull at {bone_name}")
        return 0.0

    perimeter = 0.0
    for i in range(len(hull_indices)):
        v1 = Vector((coords_2d[hull_indices[i]][0], coords_2d[hull_indices[i]][1], 0))
        v2 = Vector((coords_2d[hull_indices[(i + 1) % len(hull_indices)]][0],
                     coords_2d[hull_indices[(i + 1) % len(hull_indices)]][1], 0))
        perimeter += (v2 - v1).length

    return perimeter * 100.0  # Convert to cm

def measure_chest_circumference(mesh, armature) -> float:
    """
    Measure chest circumference at nipple line.
    
    Args:
        mesh: Blender mesh object
        armature: Blender armature object
        
    Returns:
        Chest circumference in centimeters
    """
    bone_names = get_bone_names(armature)
    circumference = measure_circumference(mesh, armature, bone_names["chest"])
    validate_measurement("chest_circumference", circumference)
    return circumference


def measure_waist_circumference(mesh, armature) -> float:
    """
    Measure waist circumference at narrowest point.
    
    Args:
        mesh: Blender mesh object
        armature: Blender armature object
        
    Returns:
        Waist circumference in centimeters
    """
    bone_names = get_bone_names(armature)
    circumference = measure_circumference(mesh, armature, bone_names["waist"])
    validate_measurement("waist_circumference", circumference)
    return circumference


def measure_hip_circumference(mesh, armature) -> float:
    """
    Measure hip circumference at widest point.
    
    Args:
        mesh: Blender mesh object
        armature: Blender armature object
        
    Returns:
        Hip circumference in centimeters
    """
    bone_names = get_bone_names(armature)
    # Measure slightly below pelvis bone for maximum hip circumference
    circumference = measure_circumference(mesh, armature, bone_names["pelvis"], height_offset=-0.05)
    validate_measurement("hip_circumference", circumference)
    return circumference


def measure_neck_circumference(mesh, armature) -> float:
    """
    Measure neck circumference at base of neck.
    
    Args:
        mesh: Blender mesh object
        armature: Blender armature object
        
    Returns:
        Neck circumference in centimeters
    """
    bone_names = get_bone_names(armature)
    circumference = measure_circumference(mesh, armature, bone_names["neck"])
    validate_measurement("neck_circumference", circumference)
    return circumference


def measure_head_circumference(mesh, armature) -> float:
    """
    Measure head circumference at widest point (forehead level).
    
    Args:
        mesh: Blender mesh object
        armature: Blender armature object
        
    Returns:
        Head circumference in centimeters
    """
    bone_names = get_bone_names(armature)
    # Measure at head bone with slight offset to get forehead level
    circumference = measure_circumference(mesh, armature, bone_names["head_top"], height_offset=-0.05)
    validate_measurement("head_circumference", circumference)
    return circumference


# ==================== MAIN MEASUREMENT FUNCTION ====================

def extract_all_measurements_joint_based(mesh, armature=None) -> Dict[str, float]:
    """
    Extract all body measurements using JOINT-BASED SEGMENTATION approach.

    This method:
    1. Finds joint locations from mesh geometry
    2. Segments the mesh between joints
    3. Creates bounding boxes for each segment
    4. Measures segment lengths
    5. Uses T-pose for accurate arm measurements (if armature provided)

    Args:
        mesh: Blender mesh object (human body)
        armature: Optional armature for T-pose arm measurements

    Returns:
        Dictionary with all measurements in centimeters
    """
    print("\nExtracting measurements (joint-based segmentation method)...")

    measurements = {}

    # Step 1: Find all body joints
    print("  Finding body joints...")
    joints = find_body_joints(mesh)

    # Debug: print found joints
    print(f"  Found {len(joints)} joints")

    # Step 2: Height measurement (simple)
    print("  Measuring height...")
    measurements["height"] = (joints['head_top'].z - joints['feet'].z) * 100
    validate_measurement("height", measurements["height"])

    # Step 3: Torso measurements using horizontal slices
    print("  Measuring torso dimensions...")

    # Shoulder width (distance between shoulder joints)
    if 'shoulder_left' in joints and 'shoulder_right' in joints:
        measurements["shoulder_width"] = (joints['shoulder_right'].x - joints['shoulder_left'].x) * 100
        validate_measurement("shoulder_width", measurements["shoulder_width"])
    else:
        measurements["shoulder_width"] = measure_width_at_height(mesh, joints['shoulders_center'].z)

    # Step 4: Body widths (depth measurements)
    print("  Measuring body widths...")

    # Chest width (front-to-back depth at chest level)
    measurements["chest_width"] = measure_depth_at_height(mesh, joints['chest'].z)
    validate_measurement("chest_width", measurements["chest_width"])

    # Head width (side-to-side at widest part of head)
    head_width_height = joints['head_top'].z - (joints['head_top'].z - joints['neck'].z) * 0.4
    measurements["head_width"] = measure_width_at_height(mesh, head_width_height, tolerance=0.03)
    validate_measurement("head_width", measurements["head_width"])

    # Step 5: Arm, hand, and neck measurements using BONE CHAINS
    print("  Measuring arm, hand, and neck dimensions (bone-based)...")

    if armature is not None:
        # Measure upper arm using upperarm01 and upperarm02
        measurements["upper_arm_length"] = measure_upper_arm_length(armature, side="left")

        # Measure forearm using lowerarm01 and lowerarm02
        measurements["forearm_length"] = measure_forearm_length(armature, side="left")

        # Measure hand using wrist to finger3-3
        measurements["hand_length"] = measure_hand_length(armature, side="left")

        # Measure neck using neck01 and neck02
        measurements["neck_length"] = measure_neck_length(armature)

    else:
        print("  ⚠ Warning: No armature available for bone-based measurements")
        measurements["forearm_length"] = 0.0
        measurements["upper_arm_length"] = 0.0
        measurements["hand_length"] = 0.0
        measurements["neck_length"] = 0.0

    print("✓ All measurements extracted")

    return measurements


def extract_all_measurements_mesh_only(mesh) -> Dict[str, float]:
    """
    Extract all body measurements using PURE MESH-BASED approach (no bones required).

    This function analyzes the mesh geometry directly to find anatomical landmarks
    and calculate measurements, making it more reliable than bone-based methods.

    Args:
        mesh: Blender mesh object (human body)

    Returns:
        Dictionary with all measurements in centimeters
    """
    print("\nExtracting measurements (pure mesh-based method)...")

    measurements = {}

    # Step 1: Find anatomical landmarks from mesh geometry
    print("  Analyzing mesh geometry for anatomical landmarks...")
    landmarks = find_anatomical_landmarks(mesh)

    # Step 2: Height measurement
    print("  Measuring height...")
    measurements["height"] = measure_vertical_distance(mesh, landmarks['head_top'], landmarks['feet_bottom'])
    validate_measurement("height", measurements["height"])

    # Step 3: Width measurements at key heights
    print("  Measuring widths...")
    measurements["shoulder_width"] = measure_width_at_height(mesh, landmarks['shoulders'])
    validate_measurement("shoulder_width", measurements["shoulder_width"])

    measurements["hip_width"] = measure_width_at_height(mesh, landmarks['hips'])
    validate_measurement("hip_width", measurements["hip_width"])

    # Step 4: Circumference measurements
    print("  Measuring circumferences...")
    measurements["chest_circumference"] = measure_circumference_at_height(mesh, landmarks['chest'])
    validate_measurement("chest_circumference", measurements["chest_circumference"])

    measurements["waist_circumference"] = measure_circumference_at_height(mesh, landmarks['waist'])
    validate_measurement("waist_circumference", measurements["waist_circumference"])

    measurements["hip_circumference"] = measure_circumference_at_height(mesh, landmarks['hips'])
    validate_measurement("hip_circumference", measurements["hip_circumference"])

    measurements["neck_circumference"] = measure_circumference_at_height(mesh, landmarks['neck_base'])
    validate_measurement("neck_circumference", measurements["neck_circumference"])

    # Head circumference at a bit below head top
    head_circ_height = landmarks['head_top'] - (landmarks['head_top'] - landmarks['neck_base']) * 0.3
    measurements["head_circumference"] = measure_circumference_at_height(mesh, head_circ_height)
    validate_measurement("head_circumference", measurements["head_circumference"])

    # Step 5: Leg measurements
    print("  Measuring leg dimensions...")
    measurements["inseam"] = measure_vertical_distance(mesh, landmarks['crotch'], landmarks['feet_bottom'])
    validate_measurement("inseam", measurements["inseam"])

    # Step 6: Arm measurements using vertex groups
    print("  Measuring arm dimensions (vertex group-based)...")
    arm_measurements = measure_arm_from_vertex_groups(mesh, side="left")
    measurements.update(arm_measurements)

    # If vertex groups didn't work, set to 0
    if measurements["arm_length"] == 0:
        print("  ⚠ Note: Arm vertex groups not found - arm measurements will be 0")

    print("✓ All measurements extracted")

    return measurements


def extract_all_measurements(mesh, armature=None) -> Dict[str, float]:
    """
    Extract all measurements from a human mesh using JOINT-BASED SEGMENTATION.

    This is the primary measurement function that uses:
    - Joint detection from mesh geometry
    - Mesh segmentation between joints
    - Bounding box measurements for accurate limb lengths
    - T-pose for accurate arm measurements (if armature provided)

    Args:
        mesh: Blender mesh object (human body)
        armature: Optional armature object (used for T-pose arm measurements)

    Returns:
        Dictionary with all measurements in centimeters
    """
    # Use the joint-based segmentation approach with T-pose
    return extract_all_measurements_joint_based(mesh, armature)


def print_measurements(measurements: Dict[str, float]):
    """
    Print measurements in a formatted table.

    Args:
        measurements: Dictionary of measurements
    """
    print("\n" + "="*60)
    print("BODY MEASUREMENTS (Joint-Based Segmentation)")
    print("="*60)

    print("\nVertical Dimensions:")
    print(f"  Height:            {measurements.get('height', 0):6.1f} cm")
    print(f"  Neck Length:       {measurements.get('neck_length', 0):6.1f} cm")

    print("\nHorizontal Widths (side-to-side):")
    print(f"  Shoulder Width:    {measurements.get('shoulder_width', 0):6.1f} cm")
    print(f"  Head Width:        {measurements.get('head_width', 0):6.1f} cm")

    print("\nBody Depths (front-to-back):")
    print(f"  Chest Width:       {measurements.get('chest_width', 0):6.1f} cm")

    print("\nArm & Hand Dimensions (bone-based):")
    upper_arm = measurements.get('upper_arm_length', 0)
    if upper_arm > 0:
        print(f"  Upper Arm Length:  {measurements.get('upper_arm_length', 0):6.1f} cm")
        print(f"  Forearm Length:    {measurements.get('forearm_length', 0):6.1f} cm")
        print(f"  Hand Length:       {measurements.get('hand_length', 0):6.1f} cm")
    else:
        print(f"  (Could not measure arms - no armature)")

    print("="*60 + "\n")
