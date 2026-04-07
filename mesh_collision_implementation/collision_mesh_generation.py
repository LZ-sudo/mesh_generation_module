"""
Collision Mesh Generation for Unreal Engine Physics Assets

Extracts per-region vertex geometry from the MPFB2 basemesh, runs CoACD
(Approximate Convex Decomposition) via a subprocess in the project myenv,
and creates UCX_ collision mesh objects in the Blender scene ready for
inclusion in the FBX export.

Unreal Engine convention: collision meshes named UCX_<MeshName>_<N> are
automatically imported as the PhysicsAsset for <MeshName>.

Usage (from inside generate_human.py / Blender context):
    from mesh_collision_implementation.collision_mesh_generation import generate_collision_meshes
    ucx_objects = generate_collision_meshes(basemesh, armature, script_dir)
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Region -> bone group mapping
# Bone names match the MPFB2 default rig (see bone_names_reference.txt)
# ============================================================================

COLLISION_REGIONS: Dict[str, List[str]] = {
    # Vertex group names match CMU mocap skinning weights assigned by MPFB2
    "head": [
        "Head", "Neck", "Neck1",
    ],
    "torso": [
        "Hips", "LowerBack", "Spine", "Spine1",
        "LHipJoint", "RHipJoint",
    ],
    "upper_arm_L": ["LeftArm", "LeftShoulder"],
    "upper_arm_R": ["RightArm", "RightShoulder"],
    "lower_arm_L": ["LeftForeArm"],
    "lower_arm_R": ["RightForeArm"],
    "hand_L": ["LeftHand", "LThumb", "LeftFingerBase", "LeftHandFinger1"],
    "hand_R": ["RightHand", "RThumb", "RightFingerBase", "RightHandFinger1"],
    "upper_leg_L": ["LeftUpLeg"],
    "upper_leg_R": ["RightUpLeg"],
    "lower_leg_L": ["LeftLeg"],
    "lower_leg_R": ["RightLeg"],
    "foot_L": ["LeftFoot", "LeftToeBase"],
    "foot_R": ["RightFoot", "RightToeBase"],
}

# Minimum bone weight for a vertex to be included in a region
WEIGHT_THRESHOLD = 0.1


def _extract_region_mesh(
    mesh_obj,
    bone_names: List[str],
    weight_threshold: float = WEIGHT_THRESHOLD,
) -> Optional[Tuple[List, List]]:
    """
    Extract vertices and triangulated faces for a body region.

    Vertices are included if their weight for ANY bone in bone_names is >= weight_threshold.
    Faces (triangles) are included only when all three of their vertices qualify.

    Args:
        mesh_obj: Blender mesh object with vertex groups matching bone names.
        bone_names: Bones that define this body region.
        weight_threshold: Minimum weight to count a vertex as belonging to the region.

    Returns:
        (vertices, faces) as plain Python lists (serialisable to JSON), or None
        if the region has insufficient geometry for CoACD.
    """
    # Collect vertex group indices that correspond to region bones
    vg_indices = {
        vg.index
        for vg in mesh_obj.vertex_groups
        if vg.name in bone_names
    }
    if not vg_indices:
        return None

    # Find vertices meeting the weight threshold for any region bone
    region_verts = set()
    for v in mesh_obj.data.vertices:
        for g in v.groups:
            if g.group in vg_indices and g.weight >= weight_threshold:
                region_verts.add(v.index)
                break

    if len(region_verts) < 4:
        return None

    # Remap original vertex indices to contiguous 0-based indices
    sorted_verts = sorted(region_verts)
    old_to_new = {old: new for new, old in enumerate(sorted_verts)}

    # Export vertex positions in Blender's world space (metres)
    verts_out = [list(mesh_obj.data.vertices[i].co) for i in sorted_verts]

    # Export triangulated faces whose every vertex belongs to the region
    faces_out = []
    for poly in mesh_obj.data.polygons:
        poly_verts = list(poly.vertices)
        if not all(i in region_verts for i in poly_verts):
            continue
        if len(poly_verts) == 3:
            faces_out.append([old_to_new[i] for i in poly_verts])
        elif len(poly_verts) == 4:
            a, b, c, d = [old_to_new[i] for i in poly_verts]
            faces_out.append([a, b, c])
            faces_out.append([a, c, d])

    if len(faces_out) < 4:
        return None

    return verts_out, faces_out


def _create_blender_meshes(
    parts: List[Dict],
    ucx_name: str,
    armature,
) -> List:
    """
    Create Blender mesh objects for each convex part returned by CoACD.

    When CoACD returns multiple parts for a region, each part is a separate
    object with a numeric suffix (_0, _1, ...).

    Args:
        parts: List of {"vertices": [...], "faces": [...]} dicts.
        ucx_name: Base UCX name, e.g. "UCX_Body_00".
        armature: Armature object the collision meshes will be parented to.

    Returns:
        List of created Blender mesh objects.
    """
    import bpy

    created = []
    for i, part in enumerate(parts):
        name = ucx_name if len(parts) == 1 else f"{ucx_name}_{i}"

        mesh_data = bpy.data.meshes.new(name)
        mesh_data.from_pydata(
            [tuple(v) for v in part["vertices"]],
            [],
            [tuple(f) for f in part["faces"]],
        )
        mesh_data.update()

        obj = bpy.data.objects.new(name, mesh_data)
        bpy.context.scene.collection.objects.link(obj)

        if armature:
            obj.parent = armature

        created.append(obj)

    return created


def generate_collision_meshes(
    basemesh,
    armature,
    script_dir: Path,
    threshold: float = 0.3,
    max_vertices: int = 2000,
    verbose: bool = False,
) -> List:
    """
    Generate UCX_ collision meshes for a basemesh using CoACD.

    Steps:
    1. Extract per-region vertex/face data from vertex groups.
    2. Write to a temporary JSON file.
    3. Run run_coacd.py via myenv Python in a subprocess.
    4. Read CoACD output and create Blender mesh objects.

    Args:
        basemesh: MPFB2 Blender mesh object (must have vertex groups).
        armature: Armature to parent collision meshes to.
        script_dir: Root of mesh_generation_module (used to locate myenv and run_coacd.py).
        threshold: CoACD decomposition threshold. Lower = tighter fit, more parts.
                   0.01 (fine) / 0.05 (default) / 0.1 (coarse).
        verbose: Print per-region extraction and decomposition details.

    Returns:
        List of UCX_ Blender objects to pass to export_fbx as additional_objects.
    """
    print("Generating collision meshes with CoACD...")

    mesh_name = basemesh.name

    # ------------------------------------------------------------------
    # Step 1: Extract region geometry from vertex groups
    # ------------------------------------------------------------------
    region_data: Dict = {}
    for region_name, bones in COLLISION_REGIONS.items():
        result = _extract_region_mesh(basemesh, bones)
        if result is None:
            if verbose:
                print(f"  Skipping '{region_name}': insufficient geometry")
            continue
        verts, faces = result
        region_data[region_name] = {"vertices": verts, "faces": faces}
        if verbose:
            print(f"  Extracted '{region_name}': {len(verts)} verts, {len(faces)} faces")

    if not region_data:
        print("  Warning: No usable region data found - skipping collision generation")
        return []

    # ------------------------------------------------------------------
    # Step 2: Write temp input JSON
    # ------------------------------------------------------------------
    tmp_dir = Path(tempfile.gettempdir()) / "mpfb_collision"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_path = tmp_dir / "coacd_input.json"
    output_path = tmp_dir / "coacd_output.json"

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(region_data, f)

    # ------------------------------------------------------------------
    # Step 3: Run CoACD via myenv Python subprocess
    # ------------------------------------------------------------------
    myenv_python = script_dir / "myenv" / "Scripts" / "python.exe"
    if not myenv_python.exists():
        myenv_python = script_dir / "myenv" / "bin" / "python"

    run_coacd_script = (
        script_dir / "mesh_collision_implementation" / "run_coacd.py"
    )

    cmd = [
        str(myenv_python),
        str(run_coacd_script),
        "--input", str(input_path),
        "--output", str(output_path),
        "--threshold", str(threshold),
        "--max-vertices", str(max_vertices),
    ]

    print(
        f"  Running CoACD (threshold={threshold}, max_vertices={max_vertices}) "
        f"on {len(region_data)} regions..."
    )

    # Stream stdout line-by-line so progress prints in real time
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip()
        if line:
            print(f"    {line}", flush=True)
    proc.stdout.close()
    stderr_output = proc.stderr.read()
    proc.stderr.close()
    returncode = proc.wait(timeout=300)

    if returncode != 0:
        print(f"  Warning: CoACD subprocess failed:\n{stderr_output}")
        return []

    if not output_path.exists():
        print("  Warning: CoACD output file not found - skipping collision generation")
        return []

    # ------------------------------------------------------------------
    # Step 4: Create Blender mesh objects from CoACD output
    # ------------------------------------------------------------------
    with open(output_path, "r", encoding="utf-8") as f:
        coacd_output = json.load(f)

    ucx_objects = []
    for region_name, parts in coacd_output.items():
        if not parts:
            continue
        # Name by body region so Unreal's physics asset editor shows meaningful labels
        ucx_name = f"UCX_{mesh_name}_{region_name}"
        created = _create_blender_meshes(parts, ucx_name, armature)
        ucx_objects.extend(created)
        if verbose:
            print(f"  '{region_name}' -> {len(created)} convex part(s) -> {ucx_name}")

    print(f"  Created {len(ucx_objects)} UCX collision mesh object(s)")
    return ucx_objects
