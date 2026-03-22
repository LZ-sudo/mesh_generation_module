#!/usr/bin/env python3
"""
MPFB Hair Assets Application Library

This module provides functions to apply MakeHuman hair assets (.mhclo format)
to MPFB2 human meshes with head-bone rigging for Chaos Cloth simulation in
Unreal Engine.

Functions:
    - find_hair_asset(): Locate a hair asset by name in the mpfb_hair_assets folder
    - add_hair_to_human(): Add hair asset to a human mesh
    - setup_hair_for_cloth(): Rig hair mesh to head bone with Chaos Cloth weight hints
    - apply_hair_asset(): Complete workflow to add and rig hair asset

Example usage:
    import mpfb_hair_assets_application as hair_lib

    # Apply hair to existing human and armature
    hair_obj = hair_lib.apply_hair_asset(
        human_obj=basemesh,
        armature_obj=armature,
        hair_asset_name="Short_Hair_B"
    )
"""

from pathlib import Path
import sys
from typing import Optional
import importlib
import traceback

import numpy as np

# Add parent directory to path for utils import
# (this file is in mesh_hair_generation, utils is in parent dir)
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import utils


def find_hair_asset(hair_name: str, assets_dir: Optional[Path] = None) -> dict:
    """
    Find a hair asset by name in the mpfb_hair_assets folder.

    Args:
        hair_name: Name of the hair asset (e.g., "Short_Hair_B")
        assets_dir: Optional custom path to assets directory.
                   If None, uses ./mpfb_hair_assets

    Returns:
        Dictionary with paths:
        {
            'mhclo': Path to .mhclo file,
            'obj': Path to .obj file,
            'mhmat': Path to .mhmat file,
        }

    Raises:
        FileNotFoundError: If hair asset or required files are not found
    """
    if assets_dir is None:
        assets_dir = parent_dir / "mpfb_hair_assets"

    # Look for hair asset in subfolder
    hair_folder = assets_dir / hair_name

    if not hair_folder.exists():
        raise FileNotFoundError(
            f"Hair asset folder not found: {hair_folder}\n"
            f"Available assets: {[d.name for d in assets_dir.iterdir() if d.is_dir()]}"
        )

    # Find files by extension (don't assume matching names)
    mhclo_files = list(hair_folder.glob("*.mhclo"))
    obj_files = list(hair_folder.glob("*.obj"))
    mhmat_files = list(hair_folder.glob("*.mhmat"))

    if not mhclo_files:
        raise FileNotFoundError(f"No .mhclo file found in {hair_folder}")

    if not obj_files:
        raise FileNotFoundError(
            f"No .obj file found in {hair_folder}\n"
            f"Hair asset requires all three files: .mhclo, .obj, .mhmat"
        )

    if not mhmat_files:
        raise FileNotFoundError(
            f"No .mhmat file found in {hair_folder}\n"
            f"Hair asset requires all three files: .mhclo, .obj, .mhmat"
        )

    return {
        'mhclo': mhclo_files[0],
        'obj': obj_files[0],
        'mhmat': mhmat_files[0],
    }


def add_hair_to_human(human_obj, hair_asset_path: Path, verbose: bool = False):
    """
    Add a hair asset to a human mesh using MPFB2 API.

    Args:
        human_obj: The Blender human mesh object
        hair_asset_path: Path to the .mhclo hair asset file
        verbose: Enable verbose output

    Returns:
        The created hair object, or None if failed

    Raises:
        ImportError: If bpy (Blender) is not available
        Exception: If hair asset addition fails
    """
    try:
        import bpy

        # Get MPFB module path
        mpfb_path = utils._get_mpfb_module_path()
        HumanService = importlib.import_module(f'{mpfb_path}.services.humanservice').HumanService

        if verbose:
            print(f"  Adding hair asset: {hair_asset_path.name}")

        # Make sure human is selected and active
        bpy.context.view_layer.objects.active = human_obj
        human_obj.select_set(True)

        # Track objects before import to find the newly created one
        objects_before = set(bpy.data.objects)

        # Add the MHCLO asset
        HumanService.add_mhclo_asset(
            str(hair_asset_path),
            human_obj,
            asset_type="hair",
            subdiv_levels=0
        )

        # Find the newly created hair object by comparing before/after
        objects_after = set(bpy.data.objects)
        new_objects = objects_after - objects_before

        # Find the mesh object (should be only one new mesh)
        hair_obj = None
        for obj in new_objects:
            if obj.type == 'MESH':
                hair_obj = obj
                break

        if hair_obj:
            # Rename to a stable, predictable name so that Unreal Engine can
            # reliably identify the hair mesh section when applying Chaos Cloth.
            hair_obj.name = "Hair_Mesh"
            hair_obj.data.name = "Hair_Mesh"
            if verbose:
                print(f"  Hair mesh created and renamed to: {hair_obj.name}")
                print(f"    Vertices: {len(hair_obj.data.vertices)}")
                print(f"    Faces: {len(hair_obj.data.polygons)}")
        else:
            print(f"  Warning: Could not locate hair object after creation")

        return hair_obj

    except Exception as e:
        print(f"Error adding hair asset: {e}")
        if verbose:
            traceback.print_exc()
        raise


def setup_hair_for_cloth(
    hair_obj,
    human_obj,
    armature_obj,
    verbose: bool = False
):
    """
    Set up hair mesh for Chaos Cloth simulation in Unreal Engine.

    Skins all hair vertices to the head bone with weight 1.0, so the hair
    follows the character's head. Also adds a vertex color layer
    ('cloth_weights') encoding simulation weight hints: 0.0 at the
    scalp/roots (pinned), 1.0 at the hair tips (free to simulate), computed
    from each vertex's distance to the head bone center.

    In Unreal Engine, after FBX import:
    - Add a Chaos Cloth asset to the Skeletal Mesh
    - Reference the 'cloth_weights' vertex color layer to auto-populate
      Max Distance values in the Cloth Paint tool

    Args:
        hair_obj: The hair mesh object
        human_obj: The human mesh object (unused, kept for API compatibility)
        armature_obj: The armature/rig object
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            print("  Setting up hair mesh for Chaos Cloth simulation...")

        # Detect head bone name
        bone_names = [bone.name for bone in armature_obj.data.bones]

        if 'mixamorig:Head' in bone_names:
            head_bone_name = 'mixamorig:Head'
        elif 'head' in bone_names:
            head_bone_name = 'head'
        else:
            head_bones = [name for name in bone_names if 'head' in name.lower()]
            if head_bones:
                head_bone_name = head_bones[0]
            else:
                print("  Error: Cannot find head bone in armature")
                print(f"    Available bones: {bone_names[:10]}...")
                return False

        if verbose:
            print(f"    Head bone: {head_bone_name}")

        mesh = hair_obj.data

        # Clear existing vertex groups and assign all vertices to head bone
        hair_obj.vertex_groups.clear()
        head_vg = hair_obj.vertex_groups.new(name=head_bone_name)
        all_indices = [v.index for v in mesh.vertices]
        head_vg.add(all_indices, 1.0, 'REPLACE')

        if verbose:
            print(f"    Assigned {len(all_indices)} vertices to '{head_bone_name}' with weight 1.0")

        # Compute cloth weight hints from distance to head bone (rest pose)
        # head_local is in armature local space; matrix_world converts to world space
        head_bone = armature_obj.data.bones[head_bone_name]
        head_bone_world_pos = armature_obj.matrix_world @ head_bone.head_local
        world_matrix = hair_obj.matrix_world
        vertices_world = np.array([world_matrix @ v.co for v in mesh.vertices])
        head_pos = np.array(head_bone_world_pos)
        distances = np.linalg.norm(vertices_world - head_pos, axis=1)

        min_d, max_d = distances.min(), distances.max()
        if max_d > min_d:
            cloth_weights = (distances - min_d) / (max_d - min_d)
        else:
            cloth_weights = np.zeros(len(distances))

        # Add vertex color layer for cloth weight hints (0.0=pinned, 1.0=free)
        if "cloth_weights" in mesh.color_attributes:
            mesh.color_attributes.remove(mesh.color_attributes["cloth_weights"])
        color_attr = mesh.color_attributes.new(
            name="cloth_weights",
            type='BYTE_COLOR',
            domain='POINT'
        )
        for i, w in enumerate(cloth_weights):
            w_f = float(w)
            color_attr.data[i].color = (w_f, w_f, w_f, 1.0)

        if verbose:
            print("    Added 'cloth_weights' vertex color layer (0.0=scalp, 1.0=tips)")

        # Parent hair mesh to armature
        hair_obj.parent = armature_obj
        hair_obj.matrix_parent_inverse = armature_obj.matrix_world.inverted()

        # Remove existing armature modifiers
        for mod in list(hair_obj.modifiers):
            if mod.type == 'ARMATURE':
                hair_obj.modifiers.remove(mod)

        arm_mod = hair_obj.modifiers.new(name="Armature", type='ARMATURE')
        arm_mod.object = armature_obj
        arm_mod.use_vertex_groups = True

        if verbose:
            print("    Added armature modifier")
            print("  Hair mesh ready for Chaos Cloth simulation in Unreal Engine")

        return True

    except Exception as e:
        print(f"Error setting up hair for cloth: {e}")
        if verbose:
            traceback.print_exc()
        return False


def apply_hair_asset(
    human_obj,
    armature_obj,
    hair_asset_name: str,
    assets_dir: Optional[Path] = None,
    verbose: bool = False
):
    """
    Complete workflow to add and rig a hair asset to a human.

    This is the main function that orchestrates:
    1. Finding the hair asset files
    2. Adding the hair mesh to the human
    3. Setting up rigging with dynamic bone generation

    Args:
        human_obj: The Blender human mesh object
        armature_obj: The armature/rig object
        hair_asset_name: Name of the hair asset (e.g., "Short_Hair_B")
        assets_dir: Optional custom path to assets directory
        verbose: Enable verbose output

    Returns:
        The created hair object, or None if failed

    Example:
        hair_obj = apply_hair_asset(
            human_obj=basemesh,
            armature_obj=armature,
            hair_asset_name="Short_Hair_B",
            verbose=True
        )
    """
    try:
        if verbose:
            print(f"\nApplying hair asset: {hair_asset_name}")

        # Find the hair asset files
        asset_files = find_hair_asset(hair_asset_name, assets_dir)

        if verbose:
            print(f"  Found hair asset:")
            print(f"    .mhclo: {asset_files['mhclo'].name}")
            print(f"    .obj:   {asset_files['obj'].name}")
            print(f"    .mhmat: {asset_files['mhmat'].name}")

        # Add hair to human
        hair_obj = add_hair_to_human(human_obj, asset_files['mhclo'], verbose=verbose)

        if not hair_obj:
            print(f"Failed to create hair object")
            return None

        success = setup_hair_for_cloth(
            hair_obj,
            human_obj,
            armature_obj,
            verbose=verbose
        )

        if not success:
            print(f"Warning: Hair cloth setup had issues")

        if verbose:
            print(f"Hair asset applied successfully")

        return hair_obj

    except FileNotFoundError as e:
        print(f"Hair asset not found: {e}")
        return None
    except Exception as e:
        print(f"Error applying hair asset: {e}")
        if verbose:
            traceback.print_exc()
        return None


def list_available_hair_assets(assets_dir: Optional[Path] = None) -> list:
    """
    List all available hair assets in the mpfb_hair_assets folder.

    Args:
        assets_dir: Optional custom path to assets directory.
                   If None, uses ./mpfb_hair_assets

    Returns:
        List of hair asset names (folder names)
    """
    if assets_dir is None:
        assets_dir = parent_dir / "mpfb_hair_assets"

    if not assets_dir.exists():
        return []

    # Find all subfolders that contain .mhclo files
    hair_assets = []
    for folder in assets_dir.iterdir():
        if folder.is_dir():
            mhclo_files = list(folder.glob("*.mhclo"))
            if mhclo_files:
                hair_assets.append(folder.name)

    return sorted(hair_assets)


# Example standalone usage
if __name__ == "__main__":
    print("MPFB Hair Assets Application Library")
    print("=" * 50)
    print("\nAvailable hair assets:")

    assets = list_available_hair_assets()
    if assets:
        for asset in assets:
            print(f"  - {asset}")
    else:
        print("  No hair assets found in mpfb_hair_assets folder")

    print("\nThis is a library module. Import it in your scripts:")
    print("  import mpfb_hair_assets_application as hair_lib")
    print("  hair_obj = hair_lib.apply_hair_asset(human, armature, 'Short_Hair_B')")
