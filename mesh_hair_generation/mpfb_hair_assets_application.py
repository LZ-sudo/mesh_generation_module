#!/usr/bin/env python3
"""
MPFB Hair Assets Application Library

This module provides functions to apply MakeHuman hair assets (.mhclo format)
to MPFB2 human meshes with proper rigging for game engines like Unreal.

Functions:
    - find_hair_asset(): Locate a hair asset by name in the mpfb_hair_assets folder
    - add_hair_to_human(): Add hair asset to a human mesh
    - setup_hair_rigging(): Set up rigging for hair with weight transfer
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
from typing import Optional, Tuple

# Add parent directory to path for utils import
# (this file is in mesh_hair_generation, utils is in parent dir)
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import utils


def find_hair_asset(hair_name: str, assets_dir: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """
    Find a hair asset by name in the mpfb_hair_assets folder.

    Args:
        hair_name: Name of the hair asset (e.g., "Short_Hair_B")
        assets_dir: Optional custom path to assets directory.
                   If None, uses ./mpfb_hair_assets

    Returns:
        Tuple of (mhclo_path, obj_path, mhmat_path)

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

    # Find the .mhclo file
    mhclo_files = list(hair_folder.glob("*.mhclo"))
    if not mhclo_files:
        raise FileNotFoundError(f"No .mhclo file found in {hair_folder}")

    mhclo_path = mhclo_files[0]

    # Check for required .obj and .mhmat files
    obj_path = mhclo_path.with_suffix('.obj')
    mhmat_path = mhclo_path.with_suffix('.mhmat')

    if not obj_path.exists():
        raise FileNotFoundError(
            f"Required .obj file not found: {obj_path}\n"
            f"Hair asset requires all three files: .mhclo, .obj, .mhmat"
        )

    if not mhmat_path.exists():
        raise FileNotFoundError(
            f"Required .mhmat file not found: {mhmat_path}\n"
            f"Hair asset requires all three files: .mhclo, .obj, .mhmat"
        )

    return mhclo_path, obj_path, mhmat_path


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
        import importlib

        # Get MPFB module path
        mpfb_path = utils._get_mpfb_module_path()
        HumanService = importlib.import_module(f'{mpfb_path}.services.humanservice').HumanService

        if verbose:
            print(f"  Adding hair asset: {hair_asset_path.name}")

        # Make sure human is selected and active
        bpy.context.view_layer.objects.active = human_obj
        human_obj.select_set(True)

        # Add the MHCLO asset
        result = HumanService.add_mhclo_asset(
            str(hair_asset_path),
            human_obj,
            asset_type="hair",
            subdiv_levels=0
        )

        # Find the hair object in the scene
        hair_obj = None
        for obj in bpy.data.objects:
            if obj != human_obj and obj.type == 'MESH' and 'hair' in obj.name.lower():
                # Check if this is a newly created object (not in our previous object list)
                hair_obj = obj
                break

        if hair_obj:
            if verbose:
                print(f"  ✓ Hair mesh created: {hair_obj.name}")
                print(f"    Vertices: {len(hair_obj.data.vertices)}")
                print(f"    Faces: {len(hair_obj.data.polygons)}")
        else:
            print(f"  ⚠ Warning: Could not locate hair object after creation")

        return hair_obj

    except Exception as e:
        print(f"✗ Error adding hair asset: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise


def setup_hair_rigging(hair_obj, human_obj, armature_obj, verbose: bool = False):
    """
    Set up rigging for hair mesh with automatic weight transfer from human.

    This function:
    1. Adds an armature modifier to the hair
    2. Parents the hair to the armature
    3. Transfers bone weights from the human to the hair

    Args:
        hair_obj: The hair mesh object
        human_obj: The human mesh object (for weight transfer)
        armature_obj: The armature/rig object
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    try:
        import bpy

        if verbose:
            print("  Setting up hair rigging...")

        # Check if hair already has armature modifier
        has_armature = any(m.type == 'ARMATURE' for m in hair_obj.modifiers)

        if has_armature:
            if verbose:
                print("  ✓ Hair already has armature modifier")
            return True

        # Add armature modifier to hair
        arm_mod = hair_obj.modifiers.new(name="Armature", type='ARMATURE')
        arm_mod.object = armature_obj

        if verbose:
            print("  ✓ Added armature modifier")

        # Parent hair to armature (without changing position)
        hair_obj.parent = armature_obj
        hair_obj.matrix_parent_inverse = armature_obj.matrix_world.inverted()

        if verbose:
            print("  ✓ Parented hair to armature")

        # Transfer weights from human to hair using data transfer modifier
        try:
            # Add data transfer modifier
            dt_mod = hair_obj.modifiers.new(name="DataTransfer", type='DATA_TRANSFER')
            dt_mod.object = human_obj
            dt_mod.use_vert_data = True
            dt_mod.data_types_verts = {'VGROUP_WEIGHTS'}
            dt_mod.vert_mapping = 'NEAREST'

            # Apply the modifier to bake the weights
            bpy.context.view_layer.objects.active = hair_obj
            bpy.ops.object.modifier_apply(modifier=dt_mod.name)

            if verbose:
                print("  ✓ Transferred bone weights from human to hair")

        except Exception as e:
            print(f"  ⚠ Warning: Weight transfer failed: {e}")
            if verbose:
                print("  Trying automatic weights as fallback...")

            # Fallback to automatic weights
            try:
                bpy.context.view_layer.objects.active = armature_obj
                hair_obj.select_set(True)
                bpy.ops.object.parent_set(type='ARMATURE_AUTO')

                if verbose:
                    print("  ✓ Applied automatic weights")
            except Exception as e2:
                print(f"  ⚠ Automatic weights also failed: {e2}")
                return False

        return True

    except Exception as e:
        print(f"✗ Error setting up hair rigging: {e}")
        if verbose:
            import traceback
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
    3. Setting up rigging with weight transfer

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
        mhclo_path, obj_path, mhmat_path = find_hair_asset(hair_asset_name, assets_dir)

        if verbose:
            print(f"  Found hair asset:")
            print(f"    .mhclo: {mhclo_path.name}")
            print(f"    .obj:   {obj_path.name}")
            print(f"    .mhmat: {mhmat_path.name}")

        # Add hair to human
        hair_obj = add_hair_to_human(human_obj, mhclo_path, verbose=verbose)

        if not hair_obj:
            print(f"✗ Failed to create hair object")
            return None

        # Set up rigging
        success = setup_hair_rigging(hair_obj, human_obj, armature_obj, verbose=verbose)

        if not success:
            print(f"⚠ Warning: Hair rigging setup had issues")

        if verbose:
            print(f"✓ Hair asset applied successfully")

        return hair_obj

    except FileNotFoundError as e:
        print(f"✗ Hair asset not found: {e}")
        return None
    except Exception as e:
        print(f"✗ Error applying hair asset: {e}")
        if verbose:
            import traceback
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
