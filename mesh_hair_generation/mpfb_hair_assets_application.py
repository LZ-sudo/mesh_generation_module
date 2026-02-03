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
from typing import Optional
import importlib
import traceback

# Add parent directory to path for utils import
# (this file is in mesh_hair_generation, utils is in parent dir)
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import utils

# Import hair rigging functions from generate_hair_rigging
from generate_hair_rigging import (
    add_hair_bones_to_armature,
    calculate_hair_vertex_weights,
)


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
            'mpfbskel': Path to .mpfbskel file (or None if not found),
            'mhw': Path to .mhw file (or None if not found)
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
    mpfbskel_files = list(hair_folder.glob("*.mpfbskel"))
    mhw_files = list(hair_folder.glob("*.mhw"))

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
        'mpfbskel': mpfbskel_files[0] if mpfbskel_files else None,
        'mhw': mhw_files[0] if mhw_files else None
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
            traceback.print_exc()
        raise


def load_hair_weights(
    hair_obj,
    mhw_path: Path,
    armature_obj,
    verbose: bool = False
):
    """
    Load vertex weights from .mhw file to hair mesh.

    Args:
        hair_obj: The hair mesh object
        mhw_path: Path to the .mhw weight file
        armature_obj: The armature object (for bone name validation)
        verbose: Enable verbose output

    Returns:
        True if weights were loaded successfully
    """
    import bpy
    import json

    try:
        with open(mhw_path, 'r') as f:
            weight_data = json.load(f)

        weights = weight_data.get('weights', {})
        if not weights:
            if verbose:
                print("  Warning: No weights defined in .mhw file")
            return False

        bone_names = set(b.name for b in armature_obj.data.bones)
        loaded_count = 0

        for group_name, vertex_weights in weights.items():
            # Only create groups for bones that exist in the armature
            if group_name not in bone_names:
                if verbose:
                    print(f"    Skipping {group_name} - not in armature")
                continue

            # Create or get vertex group
            if group_name in hair_obj.vertex_groups:
                vg = hair_obj.vertex_groups[group_name]
            else:
                vg = hair_obj.vertex_groups.new(name=group_name)

            # Add weights
            for vertex_idx, weight in vertex_weights:
                try:
                    vg.add([vertex_idx], weight, 'REPLACE')
                except (IndexError, RuntimeError):
                    # Vertex index out of range - skip
                    pass

            loaded_count += 1

        if verbose:
            print(f"  Loaded weights for {loaded_count} bone groups")

        return loaded_count > 0

    except Exception as e:
        print(f"  Error loading weights: {e}")
        if verbose:
            traceback.print_exc()
        return False


def setup_hair_subrig(
    hair_obj,
    human_obj,
    armature_obj,
    _mpfbskel_path: Optional[Path] = None,
    _mhw_path: Optional[Path] = None,
    verbose: bool = False
):
    """
    Set up hair rigging by adding bones directly to the main armature.

    This approach (different from MPFB2's separate subrig) integrates
    hair bones into the main rig for proper FBX export compatibility.

    The function uses dynamic mesh analysis to position bones correctly
    within the actual hair mesh geometry (not pre-computed positions).

    Steps:
    1. Analyze hair mesh geometry to determine bone positions
    2. Add bones to main armature (parented to head bone)
    3. Calculate and assign vertex weights
    4. Set up armature modifier on hair mesh

    Args:
        hair_obj: The hair mesh object
        human_obj: The human mesh object (basemesh, used for weight transfer fallback)
        armature_obj: The main armature/rig object
        _mpfbskel_path: Unused, kept for API compatibility
        _mhw_path: Unused, kept for API compatibility
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    try:
        import bpy

        if verbose:
            print("  Setting up hair bones in main armature...")

        # Step 1: Add bones to main armature using mesh analysis
        # Uses dynamic bone positioning from generate_hair_rigging
        # Pass human_obj for accurate scalp reference from MPFB human mesh
        created_bones, weight_info = add_hair_bones_to_armature(
            armature_obj,
            hair_obj,
            human_obj=human_obj,
            parent_bone_name="head",
            verbose=verbose
        )

        if not created_bones:
            print("  Failed to create hair bones")
            return False

        if verbose:
            print(f"  Hair bones added: {len(created_bones)}")

        # Step 2: Parent hair mesh to armature
        hair_obj.parent = armature_obj
        hair_obj.matrix_parent_inverse = armature_obj.matrix_world.inverted()

        # Step 3: Add armature modifier to hair
        # Remove existing armature modifiers first
        for mod in list(hair_obj.modifiers):
            if mod.type == 'ARMATURE':
                hair_obj.modifiers.remove(mod)

        arm_mod = hair_obj.modifiers.new(name="Armature", type='ARMATURE')
        arm_mod.object = armature_obj
        arm_mod.use_vertex_groups = True

        if verbose:
            print("  Added armature modifier to hair")

        # Step 4: Calculate and assign vertex weights
        weights_assigned = False

        if weight_info:
            # Use dynamically calculated weights (preferred)
            if verbose:
                print("  Calculating vertex weights from mesh analysis...")
            weights_assigned = calculate_hair_vertex_weights(
                hair_obj,
                weight_info,
                verbose=verbose
            )

        # Step 5: Fallback - transfer weights from human if dynamic weights failed
        if not weights_assigned:
            if verbose:
                print("  Transferring weights from human mesh as fallback...")
            try:
                # Add data transfer modifier to copy weights from human
                dt_mod = hair_obj.modifiers.new(name="DataTransfer", type='DATA_TRANSFER')
                dt_mod.object = human_obj
                dt_mod.use_vert_data = True
                dt_mod.data_types_verts = {'VGROUP_WEIGHTS'}
                dt_mod.vert_mapping = 'NEAREST'

                # Apply the modifier to bake the weights
                bpy.context.view_layer.objects.active = hair_obj
                bpy.ops.object.modifier_apply(modifier=dt_mod.name)

                if verbose:
                    print("  Transferred weights from human mesh")
            except Exception as e:
                print(f"  Warning: Weight transfer failed: {e}")

        if verbose:
            print("  Hair rigging setup complete")

        return True

    except Exception as e:
        print(f"Error setting up hair rigging: {e}")
        if verbose:
            traceback.print_exc()
        return False


def setup_hair_rigging(
    hair_obj,
    human_obj,
    armature_obj,
    mpfbskel_path: Optional[Path] = None,
    mhw_path: Optional[Path] = None,
    verbose: bool = False
):
    """
    Set up rigging for hair mesh using dynamic bone generation.

    Analyzes the hair mesh geometry and generates physics bones dynamically,
    using the human mesh's scalp as reference for accurate bone placement.

    Args:
        hair_obj: The hair mesh object
        human_obj: The human mesh object (used for scalp reference)
        armature_obj: The armature/rig object
        mpfbskel_path: Unused (kept for API compatibility)
        mhw_path: Unused (kept for API compatibility)
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            print("  Setting up hair rigging with dynamic bone generation...")

        # Always use dynamic bone generation with human mesh scalp reference
        success = setup_hair_subrig(
            hair_obj,
            human_obj,
            armature_obj,
            mpfbskel_path,
            mhw_path,
            verbose
        )
        return success

    except Exception as e:
        print(f"Error setting up hair rigging: {e}")
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
    3. Setting up rigging (using subrig if .mpfbskel exists, otherwise weight transfer)

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
            print(f"    .mhclo:    {asset_files['mhclo'].name}")
            print(f"    .obj:      {asset_files['obj'].name}")
            print(f"    .mhmat:    {asset_files['mhmat'].name}")
            if asset_files['mpfbskel']:
                print(f"    .mpfbskel: {asset_files['mpfbskel'].name} (hair subrig)")
            if asset_files['mhw']:
                print(f"    .mhw:      {asset_files['mhw'].name} (custom weights)")

        # Add hair to human
        hair_obj = add_hair_to_human(human_obj, asset_files['mhclo'], verbose=verbose)

        if not hair_obj:
            print(f"Failed to create hair object")
            return None

        # Set up rigging (with subrig if available)
        success = setup_hair_rigging(
            hair_obj,
            human_obj,
            armature_obj,
            mpfbskel_path=asset_files['mpfbskel'],
            mhw_path=asset_files['mhw'],
            verbose=verbose
        )

        if not success:
            print(f"Warning: Hair rigging setup had issues")

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
