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
    "height": (130, 220),
    "shoulder_width": (20, 60),
    "forearm_length": (20, 60),
    "upper_arm_length": (20, 45),
    "hand_length": (15, 25),
    "neck_length": (6, 20),
    # Width measurements
    "hip_width": (20, 50),        # Hip width (bi-iliac breadth)
    "head_width": (10, 20)        # Head width (side to side)
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



# # ==================== MESH ANALYSIS HELPER FUNCTIONS ====================

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


def get_bone_extremity_vertices(mesh_obj, bone_name: str, axis: str = 'x') -> Tuple[Optional[Vector], Optional[Vector]]:
    """
    Get the extreme vertices (min and max) of a bone's vertex group along a specified axis.

    Args:
        mesh_obj: Blender mesh object
        bone_name: Name of the bone (vertex group name)
        axis: Axis to measure along ('x', 'y', or 'z')

    Returns:
        Tuple of (min_vertex, max_vertex) or (None, None) if bone not found
    """
    if bone_name not in mesh_obj.vertex_groups:
        print(f"⚠ Warning: Vertex group '{bone_name}' not found")
        return (None, None)

    # Get the vertex group
    vg = mesh_obj.vertex_groups[bone_name]
    vg_index = vg.index

    # Get evaluated mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)

    # Collect vertices that belong to this bone
    vertices = []
    for vert in eval_mesh.vertices:
        for group in vert.groups:
            if group.group == vg_index and group.weight > 0.1:  # Threshold to avoid negligible weights
                world_pos = mesh_obj.matrix_world @ vert.co
                vertices.append(world_pos)
                break

    eval_obj.to_mesh_clear()

    if not vertices:
        return (None, None)

    # Get axis index
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]

    # Find min and max vertices along the specified axis
    min_vert = min(vertices, key=lambda v: v[axis_idx])
    max_vert = max(vertices, key=lambda v: v[axis_idx])

    return (min_vert, max_vert)


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
    return height


def measure_shoulder_width(armature) -> float:
    """
    Measure shoulder width using bone positions.
    First tries shoulder01.L/R bones, then falls back to clavicle or upperarm01 bones.

    Args:
        armature: Blender armature object

    Returns:
        Shoulder width in centimeters
    """
    # Try different bone name possibilities in order of preference
    bone_options = [
        ("shoulder01.L", "shoulder01.R", True)  # Tail of shoulder01 bones (outer shoulder) - BEST
        # ("clavicle.L", "clavicle.R", True),       # Tail of clavicle bones (outer shoulder)
        # ("upperarm01.L", "upperarm01.R", False),  # Head of upper arm bones (shoulder joint)
    ]

    for left_bone, right_bone, use_tail in bone_options:
        try:
            # Get left shoulder bone position
            left_shoulder = get_bone_world_position(armature, left_bone, use_tail=use_tail)

            # Get right shoulder bone position
            right_shoulder = get_bone_world_position(armature, right_bone, use_tail=use_tail)

            # Calculate width (horizontal distance)
            width = abs(right_shoulder[0] - left_shoulder[0]) * 100  # Convert to cm

            print(f"  Using bones: {left_bone}/{right_bone} ({'tail' if use_tail else 'head'} position)")
            return width
        except ValueError:
            continue  # Try next bone option

    print(f"⚠ Warning: Could not find any suitable shoulder bones")
    return 0.0


def measure_hip_width(armature) -> float:
    """
    Measure hip width using bone positions.
    Measures the distance between upperleg01.L and upperleg01.R bones
    (the actual hip joint positions where legs connect to pelvis).

    Args:
        armature: Blender armature object

    Returns:
        Hip width in centimeters
    """
    try:
        # Get left hip joint bone position (head of upperleg01.L)
        left_hip = get_bone_world_position(armature, "upperleg01.L", use_tail=False)

        # Get right hip joint bone position (head of upperleg01.R)
        right_hip = get_bone_world_position(armature, "upperleg01.R", use_tail=False)

        # Calculate width (horizontal distance)
        width = abs(right_hip[0] - left_hip[0]) * 100  # Convert to cm

        return width
    except ValueError as e:
        print(f"⚠ Warning: Could not measure hip width: {e}")
        return 0.0


def measure_head_width(armature) -> float:
    """
    Measure head width using bone positions.
    Measures the distance between temporalis02.L and temporalis02.R bones.

    Args:
        armature: Blender armature object

    Returns:
        Head width in centimeters
    """
    try:
        # Get left temporalis bone position (head of temporalis02.L)
        left_temporalis = get_bone_world_position(armature, "temporalis02.L", use_tail=False)

        # Get right temporalis bone position (head of temporalis02.R)
        right_temporalis = get_bone_world_position(armature, "temporalis02.R", use_tail=False)

        # Calculate width (horizontal distance)
        width = abs(right_temporalis[0] - left_temporalis[0]) * 100  # Convert to cm

        return width
    except ValueError as e:
        print(f"⚠ Warning: Could not measure head width: {e}")
        return 0.0


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

    return length


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

    # Step 3: Width measurements using bone positions
    print("  Measuring body widths...")

    if armature is not None:
        # Shoulder width using shoulder01.L and shoulder01.R bones
        measurements["shoulder_width"] = measure_shoulder_width(armature)

        # Hip width using upperleg01.L and upperleg01.R bones (hip joints)
        measurements["hip_width"] = measure_hip_width(armature)

        # Head width using temporalis02.L and temporalis02.R bones
        measurements["head_width"] = measure_head_width(armature)
    else:
        print("  ⚠ Warning: No armature available for width measurements")
        measurements["shoulder_width"] = 0.0
        measurements["hip_width"] = 0.0
        measurements["head_width"] = 0.0

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
    print(f"  Hip Width:         {measurements.get('hip_width', 0):6.1f} cm")
    print(f"  Head Width:        {measurements.get('head_width', 0):6.1f} cm")

    print("\nArm & Hand Dimensions (bone-based):")
    upper_arm = measurements.get('upper_arm_length', 0)
    if upper_arm > 0:
        print(f"  Upper Arm Length:  {measurements.get('upper_arm_length', 0):6.1f} cm")
        print(f"  Forearm Length:    {measurements.get('forearm_length', 0):6.1f} cm")
        print(f"  Hand Length:       {measurements.get('hand_length', 0):6.1f} cm")
    else:
        print(f"  (Could not measure arms - no armature)")

    print("="*60 + "\n")
