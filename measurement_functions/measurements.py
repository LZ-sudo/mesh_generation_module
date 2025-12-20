"""
Measurement extraction functions for human models.

This module provides functions to extract body measurements from a rigged
human mesh in Blender using armature-based methods.

All measurements are returned in centimeters.
"""

import math
from typing import Dict, List, Tuple
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
    "hip_width": (20, 50),
    "head_width": (12, 20)
}


# ==================== BONE NAME MAPPINGS ====================

# MPFB2 standard rig bone names for different rig types
BONE_NAMES = {
    "default": {
        "head_top": "head",
        "foot_left": "foot.L",
        "foot_right": "foot.R",
        "shoulder_left": "upperarm01.L",
        "shoulder_right": "upperarm01.R",
        "hip_left": "upperleg01.L",
        "hip_right": "upperleg01.R",
        "pelvis": "root",
        "elbow_left": "lowerarm01.L",
        "elbow_right": "lowerarm01.R",
        "wrist_left": "wrist.L",
        "wrist_right": "wrist.R",
        "chest": "spine03",
        "waist": "spine02",
        "neck": "neck01"
    },
    "default_no_toes": {
        "head_top": "head",
        "foot_left": "foot.L",
        "foot_right": "foot.R",
        "shoulder_left": "upperarm01.L",
        "shoulder_right": "upperarm01.R",
        "hip_left": "upperleg01.L",
        "hip_right": "upperleg01.R",
        "pelvis": "root",
        "elbow_left": "lowerarm01.L",
        "elbow_right": "lowerarm01.R",
        "wrist_left": "wrist.L",
        "wrist_right": "wrist.R",
        "chest": "spine03",
        "waist": "spine02",
        "neck": "neck01"
    }
}


# ==================== HELPER FUNCTIONS ====================

def get_bone_names(armature) -> Dict[str, str]:
    """
    Detect rig type and return appropriate bone name mappings.

    Args:
        armature: Blender armature object

    Returns:
        Dictionary mapping anatomical names to bone names for this rig
    """
    return BONE_NAMES["default"]


def get_bone_world_position(armature, bone_name: str, use_tail: bool = False) -> Tuple[float, float, float]:
    """
    Get world-space position of a bone in the armature.

    Args:
        armature: Blender armature object
        bone_name: Name of the bone
        use_tail: If True, return tail position; if False, return head position

    Returns:
        (x, y, z) tuple in world coordinates
    """
    if bone_name not in armature.pose.bones:
        raise ValueError(f"Bone '{bone_name}' not found in armature")

    pose_bone = armature.pose.bones[bone_name]

    if use_tail:
        # Get tail position in world space
        bone_vec = pose_bone.tail
        world_pos = armature.matrix_world @ bone_vec
    else:
        # Get head position in world space
        bone_vec = pose_bone.head
        world_pos = armature.matrix_world @ bone_vec

    return (world_pos.x, world_pos.y, world_pos.z)


def distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """
    Calculate Euclidean distance between two 3D points.

    Args:
        p1: First point (x, y, z)
        p2: Second point (x, y, z)

    Returns:
        Distance in Blender units
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def validate_measurement(name: str, value: float) -> bool:
    """
    Validate a measurement against expected ranges.

    Args:
        name: Measurement name
        value: Measurement value in cm

    Returns:
        True if valid, False otherwise
    """
    if name not in VALIDATION_RANGES:
        return True  # No validation range defined

    min_val, max_val = VALIDATION_RANGES[name]

    if value < min_val or value > max_val:
        print(f"  ⚠ Warning: {name} = {value:.1f} cm is outside expected range [{min_val}, {max_val}]")
        return False

    return True


def get_mesh_vertices_world_space(mesh_obj) -> List[Vector]:
    """
    Get all vertices from a mesh in world coordinates.

    Args:
        mesh_obj: Blender mesh object

    Returns:
        List of vertex positions in world space
    """
    # Get the world matrix to transform from local to world space
    world_matrix = mesh_obj.matrix_world

    # Get all vertices in world space
    vertices = [world_matrix @ v.co for v in mesh_obj.data.vertices]

    return vertices


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


def measure_bone_chain_length(armature, bone_names: List[str]) -> float:
    """
    Measure the combined length of a chain of bones.

    Args:
        armature: Blender armature object
        bone_names: List of bone names in sequence

    Returns:
        Total length in centimeters
    """
    total_length = 0.0

    for bone_name in bone_names:
        if bone_name in armature.pose.bones:
            pose_bone = armature.pose.bones[bone_name]

            # Get bone head and tail in world space
            head_world = armature.matrix_world @ pose_bone.head
            tail_world = armature.matrix_world @ pose_bone.tail

            # Calculate bone length
            bone_length = (tail_world - head_world).length
            total_length += bone_length
        else:
            print(f"  ⚠ Warning: Bone '{bone_name}' not found in armature")

    # Convert to centimeters
    return total_length * 100


# ==================== CORE MEASUREMENT FUNCTIONS ====================

def measure_height(armature) -> float:
    """
    Measure total height from feet to top of head.

    Args:
        armature: Blender armature object

    Returns:
        Height in centimeters
    """
    bone_names = get_bone_names(armature)

    # Get head top position
    head_top_pos = get_bone_world_position(armature, bone_names["head_top"], use_tail=True)

    # Get foot position (use lowest of left or right foot)
    foot_left_pos = get_bone_world_position(armature, bone_names["foot_left"], use_tail=False)
    foot_right_pos = get_bone_world_position(armature, bone_names["foot_right"], use_tail=False)

    foot_pos = foot_left_pos if foot_left_pos[2] < foot_right_pos[2] else foot_right_pos

    # Height is vertical distance (z-axis)
    height = head_top_pos[2] - foot_pos[2]

    # Convert to centimeters
    height_cm = height * 100

    validate_measurement("height", height_cm)

    return height_cm


def measure_shoulder_width(armature) -> float:
    """
    Measure shoulder width (distance between shoulder joints).

    Uses shoulder01.L and shoulder01.R bones.

    Args:
        armature: Blender armature object

    Returns:
        Shoulder width in centimeters
    """
    # Get shoulder bone positions
    left_shoulder = get_bone_world_position(armature, "shoulder01.L", use_tail=False)
    right_shoulder = get_bone_world_position(armature, "shoulder01.R", use_tail=False)

    # Calculate horizontal distance
    width = distance_3d(left_shoulder, right_shoulder)

    # Convert to centimeters
    width_cm = width * 100

    validate_measurement("shoulder_width", width_cm)

    return width_cm


def measure_hip_width(armature) -> float:
    """
    Measure hip width (distance between hip joints).

    Uses upperleg01.L and upperleg01.R bones (hip joints).

    Args:
        armature: Blender armature object

    Returns:
        Hip width in centimeters
    """
    bone_names = get_bone_names(armature)

    # Get hip bone positions (top of upper legs)
    left_hip = get_bone_world_position(armature, bone_names["hip_left"], use_tail=False)
    right_hip = get_bone_world_position(armature, bone_names["hip_right"], use_tail=False)

    # Calculate horizontal distance
    width = distance_3d(left_hip, right_hip)

    # Convert to centimeters
    width_cm = width * 100

    validate_measurement("hip_width", width_cm)

    return width_cm


def measure_head_width(armature) -> float:
    """
    Measure head width (distance between temporal bones).

    Uses temporalis02.L and temporalis02.R bones.

    Args:
        armature: Blender armature object

    Returns:
        Head width in centimeters
    """
    # Get temporal bone positions (widest part of head)
    left_temporal = get_bone_world_position(armature, "temporalis02.L", use_tail=False)
    right_temporal = get_bone_world_position(armature, "temporalis02.R", use_tail=False)

    # Calculate horizontal distance
    width = distance_3d(left_temporal, right_temporal)

    # Convert to centimeters
    width_cm = width * 100

    validate_measurement("head_width", width_cm)

    return width_cm


def measure_forearm_length(armature, side: str = "left") -> float:
    """
    Measure forearm length (elbow to wrist).

    Uses lowerarm01 and lowerarm02 bones.

    Args:
        armature: Blender armature object
        side: "left" or "right"

    Returns:
        Forearm length in centimeters
    """
    suffix = ".L" if side == "left" else ".R"
    bone_chain = [f"lowerarm01{suffix}", f"lowerarm02{suffix}"]

    length_cm = measure_bone_chain_length(armature, bone_chain)

    validate_measurement("forearm_length", length_cm)

    return length_cm


def measure_upper_arm_length(armature, side: str = "left") -> float:
    """
    Measure upper arm length (shoulder to elbow).

    Uses upperarm01 and upperarm02 bones.

    Args:
        armature: Blender armature object
        side: "left" or "right"

    Returns:
        Upper arm length in centimeters
    """
    suffix = ".L" if side == "left" else ".R"
    bone_chain = [f"upperarm01{suffix}", f"upperarm02{suffix}"]

    length_cm = measure_bone_chain_length(armature, bone_chain)

    validate_measurement("upper_arm_length", length_cm)

    return length_cm


def measure_neck_length(armature) -> float:
    """
    Measure neck length.

    Uses neck01 and neck02 bones.

    Args:
        armature: Blender armature object

    Returns:
        Neck length in centimeters
    """
    bone_chain = ["neck01", "neck02"]

    length_cm = measure_bone_chain_length(armature, bone_chain)

    validate_measurement("neck_length", length_cm)

    return length_cm


def measure_hand_length(armature, side: str = "left") -> float:
    """
    Measure hand length (wrist to fingertip).

    Uses wrist and finger3-3 bones (middle finger).

    Args:
        armature: Blender armature object
        side: "left" or "right"

    Returns:
        Hand length in centimeters
    """
    suffix = ".L" if side == "left" else ".R"

    # Get wrist position
    wrist_pos = get_bone_world_position(armature, f"wrist{suffix}", use_tail=False)

    # Get middle fingertip position (finger3-3 tail)
    fingertip_pos = get_bone_world_position(armature, f"finger3-3{suffix}", use_tail=True)

    # Calculate distance
    length = distance_3d(wrist_pos, fingertip_pos)

    # Convert to centimeters
    length_cm = length * 100

    validate_measurement("hand_length", length_cm)

    return length_cm


# ==================== EXTRACTION FUNCTIONS ====================

def extract_all_measurements_joint_based(mesh, armature=None) -> Dict[str, float]:
    """
    Extract all body measurements using armature-based approach.

    This method uses armature bones for accurate measurements.

    Args:
        mesh: Blender mesh object (human body)
        armature: Armature for bone-based measurements

    Returns:
        Dictionary with all measurements in centimeters
    """
    print("\nExtracting measurements (armature-based method)...")

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

    # Step 4: Arm, hand, and neck measurements using BONE CHAINS
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
    Extract all measurements from a human mesh using armature-based methods.

    Args:
        mesh: Blender mesh object (human body)
        armature: Optional armature object

    Returns:
        Dictionary with all measurements in centimeters
    """
    return extract_all_measurements_joint_based(mesh, armature)


def print_measurements(measurements: Dict[str, float]):
    """
    Print measurements in a formatted table.

    Args:
        measurements: Dictionary of measurements
    """
    print("\n" + "="*60)
    print("BODY MEASUREMENTS")
    print("="*60)

    # Group measurements by category
    height_measures = ["height"]
    width_measures = ["shoulder_width", "hip_width", "head_width"]
    length_measures = ["upper_arm_length", "forearm_length", "hand_length", "neck_length"]

    # Print height
    print("\nHeight:")
    for measure in height_measures:
        if measure in measurements:
            print(f"  {measure:25s}: {measurements[measure]:6.1f} cm")

    # Print widths
    print("\nWidths:")
    for measure in width_measures:
        if measure in measurements:
            print(f"  {measure:25s}: {measurements[measure]:6.1f} cm")

    # Print lengths
    print("\nLengths:")
    for measure in length_measures:
        if measure in measurements:
            print(f"  {measure:25s}: {measurements[measure]:6.1f} cm")

    print("="*60)
