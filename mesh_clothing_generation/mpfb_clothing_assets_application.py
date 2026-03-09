#!/usr/bin/env python3
"""
MPFB Clothing Assets Application Library

This module provides functions to apply MakeHuman clothing assets (.mhclo format)
to MPFB2 human meshes with armature binding.

Functions:
    - find_clothing_asset(): Locate a clothing asset by name in the mpfb_clothing_assets folder
    - add_clothing_to_human(): Add clothing asset to a human mesh via MPFB2 API
    - setup_clothing_for_armature(): Ensure clothing is parented to the armature with a modifier
    - apply_clothing_asset(): Complete workflow to add and bind a clothing asset

Example usage:
    import mpfb_clothing_assets_application as clothing_lib

    # Apply a single clothing item
    clothing_obj = clothing_lib.apply_clothing_asset(
        human_obj=basemesh,
        armature_obj=armature,
        clothing_asset_name="Scrub_Shirt"
    )

    # Apply multiple clothing items
    for name in ["Scrub_Shirt", "Scrub_Pants"]:
        clothing_lib.apply_clothing_asset(basemesh, armature, name)
"""

from pathlib import Path
import sys
from typing import Optional
import importlib
import traceback

# Add parent directory to path for utils import
# (this file is in mesh_clothing_generation, utils is in parent dir)
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import utils


def find_clothing_asset(clothing_name: str, assets_dir: Optional[Path] = None) -> dict:
    """
    Find a clothing asset by name in the mpfb_clothing_assets folder.

    Args:
        clothing_name: Name of the clothing asset (e.g., "Scrub_Shirt")
        assets_dir: Optional custom path to assets directory.
                   If None, uses ./mpfb_clothing_assets

    Returns:
        Dictionary with paths:
        {
            'mhclo': Path to .mhclo file,
            'obj': Path to .obj file,
            'mhmat': Path to .mhmat file,
        }

    Raises:
        FileNotFoundError: If clothing asset or required files are not found
    """
    if assets_dir is None:
        assets_dir = parent_dir / "mpfb_clothing_assets"

    clothing_folder = assets_dir / clothing_name

    if not clothing_folder.exists():
        raise FileNotFoundError(
            f"Clothing asset folder not found: {clothing_folder}\n"
            f"Available assets: {[d.name for d in assets_dir.iterdir() if d.is_dir()]}"
        )

    mhclo_files = list(clothing_folder.glob("*.mhclo"))
    obj_files = list(clothing_folder.glob("*.obj"))
    mhmat_files = list(clothing_folder.glob("*.mhmat"))

    if not mhclo_files:
        raise FileNotFoundError(f"No .mhclo file found in {clothing_folder}")

    if not obj_files:
        raise FileNotFoundError(
            f"No .obj file found in {clothing_folder}\n"
            f"Clothing asset requires all three files: .mhclo, .obj, .mhmat"
        )

    if not mhmat_files:
        raise FileNotFoundError(
            f"No .mhmat file found in {clothing_folder}\n"
            f"Clothing asset requires all three files: .mhclo, .obj, .mhmat"
        )

    return {
        'mhclo': mhclo_files[0],
        'obj': obj_files[0],
        'mhmat': mhmat_files[0],
    }


def add_clothing_to_human(human_obj, clothing_asset_path: Path, verbose: bool = False):
    """
    Add a clothing asset to a human mesh using MPFB2 API.

    Args:
        human_obj: The Blender human mesh object
        clothing_asset_path: Path to the .mhclo clothing asset file
        verbose: Enable verbose output

    Returns:
        The created clothing object, or None if failed

    Raises:
        ImportError: If bpy (Blender) is not available
        Exception: If clothing asset addition fails
    """
    try:
        import bpy

        mpfb_path = utils._get_mpfb_module_path()
        HumanService = importlib.import_module(f'{mpfb_path}.services.humanservice').HumanService

        if verbose:
            print(f"  Adding clothing asset: {clothing_asset_path.name}")

        bpy.context.view_layer.objects.active = human_obj
        human_obj.select_set(True)

        objects_before = set(bpy.data.objects)

        HumanService.add_mhclo_asset(
            str(clothing_asset_path),
            human_obj,
            asset_type="clothes",
            subdiv_levels=0
        )

        objects_after = set(bpy.data.objects)
        new_objects = objects_after - objects_before

        clothing_obj = None
        for obj in new_objects:
            if obj.type == 'MESH':
                clothing_obj = obj
                break

        if clothing_obj:
            if verbose:
                print(f"  Clothing mesh created: {clothing_obj.name}")
                print(f"    Vertices: {len(clothing_obj.data.vertices)}")
                print(f"    Faces: {len(clothing_obj.data.polygons)}")
        else:
            print("  Warning: Could not locate clothing object after creation")

        return clothing_obj

    except Exception as e:
        print(f"Error adding clothing asset: {e}")
        if verbose:
            traceback.print_exc()
        raise


def setup_clothing_for_armature(clothing_obj, armature_obj, verbose: bool = False):
    """
    Ensure clothing mesh is properly parented to the armature with an armature modifier.

    MPFB2 assigns vertex groups to the clothing mesh when applying proxy meshes,
    so this step only handles parenting and modifier linkage.

    Args:
        clothing_obj: The clothing mesh object
        armature_obj: The armature/rig object
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            print("  Setting up clothing mesh armature binding...")

        clothing_obj.parent = armature_obj
        clothing_obj.matrix_parent_inverse = armature_obj.matrix_world.inverted()

        has_armature_mod = any(mod.type == 'ARMATURE' for mod in clothing_obj.modifiers)
        if not has_armature_mod:
            arm_mod = clothing_obj.modifiers.new(name="Armature", type='ARMATURE')
            arm_mod.object = armature_obj
            arm_mod.use_vertex_groups = True
            if verbose:
                print("    Added armature modifier")
        elif verbose:
            print("    Armature modifier already present")

        return True

    except Exception as e:
        print(f"Error setting up clothing for armature: {e}")
        if verbose:
            traceback.print_exc()
        return False


def apply_clothing_asset(
    human_obj,
    armature_obj,
    clothing_asset_name: str,
    assets_dir: Optional[Path] = None,
    verbose: bool = False
):
    """
    Complete workflow to add and bind a clothing asset to a human.

    This is the main function that orchestrates:
    1. Finding the clothing asset files
    2. Adding the clothing mesh to the human via MPFB2
    3. Binding the mesh to the armature

    Args:
        human_obj: The Blender human mesh object
        armature_obj: The armature/rig object
        clothing_asset_name: Name of the clothing asset (e.g., "Scrub_Shirt")
        assets_dir: Optional custom path to assets directory
        verbose: Enable verbose output

    Returns:
        The created clothing object, or None if failed

    Example:
        clothing_obj = apply_clothing_asset(
            human_obj=basemesh,
            armature_obj=armature,
            clothing_asset_name="Scrub_Shirt",
            verbose=True
        )
    """
    try:
        if verbose:
            print(f"\nApplying clothing asset: {clothing_asset_name}")

        asset_files = find_clothing_asset(clothing_asset_name, assets_dir)

        if verbose:
            print(f"  Found clothing asset:")
            print(f"    .mhclo: {asset_files['mhclo'].name}")
            print(f"    .obj:   {asset_files['obj'].name}")
            print(f"    .mhmat: {asset_files['mhmat'].name}")

        clothing_obj = add_clothing_to_human(human_obj, asset_files['mhclo'], verbose=verbose)

        if not clothing_obj:
            print(f"Failed to create clothing object for '{clothing_asset_name}'")
            return None

        success = setup_clothing_for_armature(clothing_obj, armature_obj, verbose=verbose)

        if not success:
            print(f"Warning: Armature binding for '{clothing_asset_name}' had issues")

        if verbose:
            print(f"Clothing asset '{clothing_asset_name}' applied successfully")

        return clothing_obj

    except FileNotFoundError as e:
        print(f"Clothing asset not found: {e}")
        return None
    except Exception as e:
        print(f"Error applying clothing asset: {e}")
        if verbose:
            traceback.print_exc()
        return None


def list_available_clothing_assets(assets_dir: Optional[Path] = None) -> list:
    """
    List all available clothing assets in the mpfb_clothing_assets folder.

    Args:
        assets_dir: Optional custom path to assets directory.
                   If None, uses ./mpfb_clothing_assets

    Returns:
        List of clothing asset names (folder names)
    """
    if assets_dir is None:
        assets_dir = parent_dir / "mpfb_clothing_assets"

    if not assets_dir.exists():
        return []

    clothing_assets = []
    for folder in assets_dir.iterdir():
        if folder.is_dir():
            mhclo_files = list(folder.glob("*.mhclo"))
            if mhclo_files:
                clothing_assets.append(folder.name)

    return sorted(clothing_assets)


if __name__ == "__main__":
    print("MPFB Clothing Assets Application Library")
    print("=" * 50)
    print("\nAvailable clothing assets:")

    assets = list_available_clothing_assets()
    if assets:
        for asset in assets:
            print(f"  - {asset}")
    else:
        print("  No clothing assets found in mpfb_clothing_assets folder")

    print("\nThis is a library module. Import it in your scripts:")
    print("  import mpfb_clothing_assets_application as clothing_lib")
    print("  clothing_obj = clothing_lib.apply_clothing_asset(human, armature, 'Scrub_Shirt')")
