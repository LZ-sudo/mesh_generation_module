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


def add_hair_to_human(human_obj, hair_asset_path: Path, hair_asset_name: Optional[str] = None, verbose: bool = False):
    """
    Add a hair asset to a human mesh using MPFB2 API.

    Args:
        human_obj: The Blender human mesh object
        hair_asset_path: Path to the .mhclo hair asset file
        hair_asset_name: Name used to label the resulting mesh object and its
                         mesh data block. Defaults to "Hair_Mesh" when not
                         provided. Passing the hair folder name (e.g.
                         "elvs_ashley_may_hair") ensures the mesh appears under
                         that name in Unreal Engine after FBX import.
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
            mesh_name = f"Human_Hair_{hair_asset_name}" if hair_asset_name else "Human_Hair_Mesh"
            hair_obj.name = mesh_name
            hair_obj.data.name = mesh_name
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


def apply_hair_material(hair_obj, mhmat_path: Path, verbose: bool = False):
    """
    Build a standard Principled BSDF material from a .mhmat file and assign it
    to the hair mesh, replacing whatever MPFB2 created.

    MPFB2 converts MakeHuman's litsphere shader to a custom Blender node group
    that Blender's FBX exporter cannot introspect for texture extraction.  This
    function creates a plain Principled BSDF material whose texture connections
    the FBX exporter understands, ensuring textures are embedded in the exported
    FBX file.

    Supported .mhmat directives:
        diffuseTexture       -> Base Color input (alpha channel -> Alpha input
                                when transparent is True)
        normalmapTexture     -> Normal Map node -> Normal input
        transparencymapTexture -> Alpha input (overrides diffuse alpha)
        diffuseColor         -> Base Color default value (used when no texture)
        transparent          -> enables Alpha Hashed blend mode

    Texture paths that are absolute and do not exist on this system (e.g.
    authored on another machine) are skipped silently.

    Args:
        hair_obj:   The Blender hair mesh object whose material slots will be
                    replaced.
        mhmat_path: Path to the .mhmat file for this hair asset.
        verbose:    Print applied texture names.

    Returns:
        The created bpy.types.Material, or None on failure.
    """
    try:
        import bpy

        mat_dir = mhmat_path.parent
        diffuse_tex_path = None
        normal_tex_path = None
        alpha_tex_path = None
        diffuse_color = (0.5, 0.5, 0.5, 1.0)
        is_transparent = False

        with open(mhmat_path, 'r') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                key = parts[0].lower()
                value = parts[1]

                if key == 'diffusetexture':
                    p = Path(value)
                    resolved = p if p.is_absolute() else mat_dir / p
                    if resolved.exists():
                        diffuse_tex_path = resolved

                elif key == 'normalmaptexture':
                    p = Path(value)
                    resolved = p if p.is_absolute() else mat_dir / p
                    if resolved.exists():
                        normal_tex_path = resolved

                elif key == 'transparencymaptexture':
                    p = Path(value)
                    resolved = p if p.is_absolute() else mat_dir / p
                    if resolved.exists():
                        alpha_tex_path = resolved

                elif key == 'diffusecolor':
                    rgb = value.split()
                    if len(rgb) >= 3:
                        diffuse_color = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)

                elif key == 'transparent' and value.strip().lower() == 'true':
                    is_transparent = True

        # Build Principled BSDF material
        mat = bpy.data.materials.new(name=mhmat_path.stem)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        output_node = nodes.new('ShaderNodeOutputMaterial')
        output_node.location = (400, 0)

        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

        # Base Color
        if diffuse_tex_path:
            img = bpy.data.images.load(str(diffuse_tex_path), check_existing=True)
            diff_node = nodes.new('ShaderNodeTexImage')
            diff_node.image = img
            diff_node.location = (-500, 200)
            links.new(diff_node.outputs['Color'], bsdf.inputs['Base Color'])
            if is_transparent:
                links.new(diff_node.outputs['Alpha'], bsdf.inputs['Alpha'])
        else:
            bsdf.inputs['Base Color'].default_value = diffuse_color

        # Normal map
        if normal_tex_path:
            img = bpy.data.images.load(str(normal_tex_path), check_existing=True)
            img.colorspace_settings.name = 'Non-Color'
            norm_img = nodes.new('ShaderNodeTexImage')
            norm_img.image = img
            norm_img.location = (-500, -100)
            norm_map = nodes.new('ShaderNodeNormalMap')
            norm_map.location = (-200, -100)
            links.new(norm_img.outputs['Color'], norm_map.inputs['Color'])
            links.new(norm_map.outputs['Normal'], bsdf.inputs['Normal'])

        # Separate alpha map (overrides the diffuse alpha channel if present)
        if alpha_tex_path and alpha_tex_path != diffuse_tex_path:
            img = bpy.data.images.load(str(alpha_tex_path), check_existing=True)
            alpha_node = nodes.new('ShaderNodeTexImage')
            alpha_node.image = img
            alpha_node.location = (-500, -350)
            links.new(alpha_node.outputs['Color'], bsdf.inputs['Alpha'])

        # Transparency render mode (API changed in Blender 4.2)
        if is_transparent:
            if hasattr(mat, 'blend_mode'):
                # Blender < 4.2
                mat.blend_mode = 'HASHED'
                if hasattr(mat, 'shadow_method'):
                    mat.shadow_method = 'CLIP'
            elif hasattr(mat, 'surface_render_method'):
                # Blender 4.2+
                mat.surface_render_method = 'DITHERED'

        # Replace all material slots on the hair mesh
        hair_obj.data.materials.clear()
        hair_obj.data.materials.append(mat)

        if verbose:
            print(f"  Applied material '{mat.name}' to {hair_obj.name}")
            if diffuse_tex_path:
                print(f"    Base Color : {diffuse_tex_path.name}")
            if normal_tex_path:
                print(f"    Normal Map : {normal_tex_path.name}")
            if alpha_tex_path:
                print(f"    Alpha      : {alpha_tex_path.name}")
            print(f"    Transparent: {is_transparent}")

        return mat

    except Exception as e:
        print(f"  Warning: Could not apply hair material from {mhmat_path.name}: {e}")
        if verbose:
            traceback.print_exc()
        return None


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
        hair_obj = add_hair_to_human(human_obj, asset_files['mhclo'], hair_asset_name=hair_asset_name, verbose=verbose)

        if not hair_obj:
            print(f"Failed to create hair object")
            return None

        # Replace MPFB2's litsphere-based material with a standard Principled BSDF
        # so that the FBX exporter can introspect and embed the textures correctly.
        apply_hair_material(hair_obj, asset_files['mhmat'], verbose=verbose)

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
