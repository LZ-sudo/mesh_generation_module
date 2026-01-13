#!/usr/bin/env python3
"""
BASIC hair generation - back to basics with proper styling.

Creates individual hair strands that:
1. Start from scalp vertices (using vertex group)
2. Follow surface normals outward
3. Drape naturally with gravity
"""

import sys
import os
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import utils


def create_hair_strands(mesh_obj, scalp_vgroup_name, hair_length=0.25, num_strands=500):
    """
    Create hair as individual curve strands with proper styling.

    Each strand:
    - Starts at a scalp vertex
    - Grows outward along the surface normal
    - Curves downward naturally (gravity)
    """
    import bpy
    import random

    print(f"\nCreating {num_strands} hair strands...")
    print(f"  Length: {hair_length * 100:.1f} cm")

    # Get scalp vertices from vertex group
    vgroup = mesh_obj.vertex_groups[scalp_vgroup_name]
    vgroup_index = vgroup.index
    mesh = mesh_obj.data

    scalp_verts = []
    for vert in mesh.vertices:
        for group in vert.groups:
            if group.group == vgroup_index and group.weight > 0.5:
                scalp_verts.append(vert)
                break

    print(f"  Found {len(scalp_verts)} scalp vertices")

    # Sample vertices
    sampled = random.sample(scalp_verts, min(num_strands, len(scalp_verts)))

    # Create collection for hair
    hair_col = bpy.data.collections.new("Hair_Strands")
    bpy.context.scene.collection.children.link(hair_col)

    # Create each hair strand
    for i, vert in enumerate(sampled):
        if i % 100 == 0:
            print(f"  Creating strand {i}/{len(sampled)}...")

        # Get vertex position and normal in world space
        pos = mesh_obj.matrix_world @ vert.co
        normal = (mesh_obj.matrix_world.to_3x3() @ vert.normal).normalized()

        # Create curve
        curve_data = bpy.data.curves.new(f"Hair_{i}", 'CURVE')
        curve_data.dimensions = '3D'
        curve_data.bevel_depth = 0.0005  # Thin strand

        spline = curve_data.splines.new('POLY')
        spline.points.add(3)  # 4 points total for smooth curve

        # Point 0: Root (at vertex)
        spline.points[0].co = (*pos, 1)

        # Point 1: Grow outward along normal (25% of length)
        p1 = pos + normal * (hair_length * 0.25)
        spline.points[1].co = (*p1, 1)

        # Point 2: Continue outward but start drooping (50% of length)
        p2 = pos + normal * (hair_length * 0.5)
        p2.z -= hair_length * 0.1  # Slight downward bend
        spline.points[2].co = (*p2, 1)

        # Point 3: End with gravity droop (100% of length)
        p3 = pos + normal * (hair_length * 0.7)
        p3.z -= hair_length * 0.4  # More droop at the end
        spline.points[3].co = (*p3, 1)

        # Create object
        curve_obj = bpy.data.objects.new(f"Hair_{i}", curve_data)
        hair_col.objects.link(curve_obj)

    print(f"✓ Created {len(sampled)} hair strands")

    # Convert to mesh and join
    print("  Converting to mesh...")
    bpy.ops.object.select_all(action='DESELECT')
    for obj in hair_col.objects:
        obj.select_set(True)

    if len(hair_col.objects) > 0:
        bpy.context.view_layer.objects.active = hair_col.objects[0]
        bpy.ops.object.convert(target='MESH')

        if len(hair_col.objects) > 1:
            bpy.ops.object.join()

        hair_mesh = bpy.context.active_object
        hair_mesh.name = "Hair"
        print(f"  ✓ Hair converted to mesh: {hair_mesh.name}")
        return hair_mesh

    return None


def main():
    """Main test."""
    import bpy

    print("\n" + "="*70)
    print("BASIC HAIR GENERATION TEST")
    print("="*70 + "\n")

    print("✓ Blender", bpy.app.version_string)

    if not utils.check_mpfb2_installed():
        sys.exit(1)

    # Setup
    utils.setup_blender_scene()

    # Create human
    print("\nCreating human...")
    import importlib
    mpfb_path = utils._get_mpfb_module_path()
    HumanService = importlib.import_module(f'{mpfb_path}.services.humanservice').HumanService

    basemesh = HumanService.create_human()
    simple_macro = {
        'gender': 0.5, 'age': 0.5, 'muscle': 0.5, 'weight': 0.5,
        'height': 0.5, 'proportions': 0.5, 'cupsize': 0.5, 'firmness': 0.5,
        'race': {'asian': 0.33, 'caucasian': 0.34, 'african': 0.33}
    }
    utils.apply_macro_settings_to_human(basemesh, simple_macro, bake=True)
    print(f"✓ Created: {basemesh.name}")

    # Add rig
    print("\nAdding rig...")
    armature, _ = utils.add_standard_rig(basemesh, rig_type='default_no_toes')

    # Create scalp vertex group
    print("\nCreating scalp vertex group...")
    mesh = basemesh.data

    # Simple: top 10% of vertices by height
    max_z = max((basemesh.matrix_world @ v.co).z for v in mesh.vertices)
    min_z = min((basemesh.matrix_world @ v.co).z for v in mesh.vertices)
    threshold = min_z + (max_z - min_z) * 0.90

    vgroup = basemesh.vertex_groups.new(name="Scalp")
    count = 0
    for vert in mesh.vertices:
        if (basemesh.matrix_world @ vert.co).z > threshold:
            vgroup.add([vert.index], 1.0, 'ADD')
            count += 1

    print(f"✓ Created scalp group ({count} vertices)")

    # Generate hair
    print("\nGenerating hair...")
    hair_obj = create_hair_strands(basemesh, "Scalp", hair_length=0.25, num_strands=500)

    if not hair_obj:
        print("✗ Hair generation failed")
        sys.exit(1)

    # Export
    print("\nExporting FBX...")
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_hair_basic.fbx"

    bpy.ops.object.select_all(action='DESELECT')
    for obj in [basemesh, armature, hair_obj]:
        if obj:
            obj.select_set(True)
    bpy.context.view_layer.objects.active = armature

    bpy.ops.export_scene.fbx(
        filepath=str(output_path),
        use_selection=True,
        object_types={'ARMATURE', 'MESH'},
        use_mesh_modifiers=True
    )

    print(f"\n{'='*70}")
    print("✓ SUCCESS!")
    print(f"{'='*70}")
    print(f"\nOutput: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
