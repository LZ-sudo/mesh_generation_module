#!/usr/bin/env python3
"""
Animation utilities for importing and baking Mixamo animations.

This module provides functions to:
- Import Mixamo animation FBX files
- Transfer animation data to MPFB2 Mixamo-rigged characters
- Bake animations for export

Functions:
    - import_mixamo_animation(): Import Mixamo animation FBX and extract animation data
    - apply_animation_to_armature(): Apply animation action to target armature
    - bake_animation(): Bake animation data into keyframes for export
    - import_and_bake_mixamo_animation(): Complete workflow (import → apply → bake)

Example usage:
    import animation_utils

    # Import and apply Mixamo animation to character
    success = animation_utils.import_and_bake_mixamo_animation(
        armature_obj=armature,
        animation_fbx_path="path/to/mixamo_walk.fbx",
        frame_start=1,
        frame_end=120
    )
"""

from pathlib import Path
from typing import Optional, Tuple
import sys


def import_mixamo_animation(animation_fbx_path: str, verbose: bool = False) -> Optional[any]:
    """
    Import a Mixamo animation FBX file and extract the animation action.

    Args:
        animation_fbx_path: Path to Mixamo animation FBX file (downloaded "without skin")
        verbose: Enable verbose output

    Returns:
        The imported armature object with animation data, or None if failed

    Note:
        - Mixamo animations should be downloaded with "Without Skin" option
        - The FBX will contain an armature with animation data
        - Bone names will have "mixamorig:" prefix
    """
    try:
        import bpy

        if verbose:
            print(f"  Importing Mixamo animation: {Path(animation_fbx_path).name}")

        # Track objects before import
        objects_before = set(bpy.data.objects)

        # Import Mixamo FBX
        # Note: Mixamo FBX uses specific settings for best compatibility
        bpy.ops.import_scene.fbx(
            filepath=animation_fbx_path,
            use_anim=True,  # Import animation data
            anim_offset=0,  # Start at frame 0
            automatic_bone_orientation=True,  # Let Blender figure out bone orientation
            ignore_leaf_bones=False,  # Keep all bones
            force_connect_children=False,  # Preserve Mixamo hierarchy
            use_prepost_rot=True  # Use pre/post rotation for compatibility
        )

        # Find the newly imported armature
        objects_after = set(bpy.data.objects)
        new_objects = objects_after - objects_before

        # Find armature in new objects
        imported_armature = None
        for obj in new_objects:
            if obj.type == 'ARMATURE':
                imported_armature = obj
                break

        if not imported_armature:
            print("  ✗ No armature found in imported FBX")
            return None

        if verbose:
            print(f"  ✓ Imported armature: {imported_armature.name}")

            # Check if it has animation data
            if imported_armature.animation_data and imported_armature.animation_data.action:
                action = imported_armature.animation_data.action
                frame_range = action.frame_range
                print(f"    Animation: {action.name}")
                print(f"    Frame range: {int(frame_range[0])} - {int(frame_range[1])}")
                print(f"    Duration: {int(frame_range[1] - frame_range[0])} frames")
            else:
                print("    ⚠ Warning: No animation data found in armature")

        return imported_armature

    except Exception as e:
        print(f"  ✗ Error importing animation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def transfer_animation_blender50(
    source_armature,
    target_armature,
    remove_root_motion: bool = True,
    verbose: bool = False
) -> bool:
    """
    Transfer animation from source to target armature using Blender 5.0 API.

    Uses Copy Transforms constraints with Blender's built-in bake_action() function,
    which correctly handles the Blender 5.0 layered action system (slots, layers,
    channelbags).

    Args:
        source_armature: Armature with animation (imported from Mixamo FBX)
        target_armature: Target armature (MPFB2 Mixamo rig)
        remove_root_motion: If True, keeps armature at origin (no object-level animation)
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    try:
        import bpy
        from bpy_extras.anim_utils import bake_action, BakeOptions

        if verbose:
            print(f"  Transferring animation from {source_armature.name} to {target_armature.name}")

        # Check if source has animation
        if not source_armature.animation_data or not source_armature.animation_data.action:
            print("  ✗ Source armature has no animation data")
            return False

        source_action = source_armature.animation_data.action
        frame_start = int(source_action.frame_range[0])
        frame_end = int(source_action.frame_range[1])

        if verbose:
            print(f"    Source animation: {source_action.name}")
            print(f"    Frame range: {frame_start} - {frame_end}")

        # Get matching bones between source and target
        source_bone_names = set(bone.name for bone in source_armature.pose.bones)
        target_bone_names = set(bone.name for bone in target_armature.pose.bones)
        matching_bones = list(source_bone_names & target_bone_names)

        if not matching_bones:
            print("  ✗ No matching bones found between source and target")
            return False

        if verbose:
            print(f"    Matching bones: {len(matching_bones)}/{len(target_bone_names)}")

        # Step 1: Add Copy Transforms constraints to each matching target bone
        # Using LOCAL space so bone orientations are transferred correctly
        if verbose:
            print(f"    Adding Copy Transforms constraints...")

        for bone_name in matching_bones:
            target_bone = target_armature.pose.bones[bone_name]
            constraint = target_bone.constraints.new('COPY_TRANSFORMS')
            constraint.name = "TEMP_AnimTransfer"
            constraint.target = source_armature
            constraint.subtarget = bone_name
            constraint.target_space = 'LOCAL'
            constraint.owner_space = 'LOCAL'

        if verbose:
            print(f"    Added {len(matching_bones)} constraints")

        # Step 2: Set up scene frame range
        bpy.context.scene.frame_start = frame_start
        bpy.context.scene.frame_end = frame_end

        # Step 3: Select and activate target armature
        bpy.ops.object.select_all(action='DESELECT')
        target_armature.select_set(True)
        bpy.context.view_layer.objects.active = target_armature

        # Step 4: Bake using Blender's built-in bake_action()
        # This correctly handles Blender 5.0 layered actions, slots, channelbags
        if verbose:
            print(f"    Baking animation with bpy_extras.anim_utils.bake_action()...")

        bake_options = BakeOptions(
            only_selected=False,
            do_pose=True,
            do_object=False,
            do_visual_keying=True,
            do_constraint_clear=True,
            do_parents_clear=False,
            do_clean=False,
            do_location=True,
            do_rotation=True,
            do_scale=True,
            do_bbone=False,
            do_custom_props=False,
        )

        baked_action = bake_action(
            target_armature,
            action=None,
            frames=range(frame_start, frame_end + 1),
            bake_options=bake_options,
        )

        if not baked_action:
            print("  ✗ bake_action() returned None")
            return False

        if verbose:
            print(f"    Baked action: {baked_action.name}")

        # Step 5: Handle root motion
        if remove_root_motion:
            if verbose:
                print(f"    Removing root motion (keeping armature at origin)")

            # Identify the root bone (usually named "Hips" in Mixamo rigs)
            root_bone_candidates = ['Hips', 'mixamorig:Hips', 'pelvis', 'root']
            root_bone_name = None
            for candidate in root_bone_candidates:
                if candidate in target_armature.pose.bones:
                    root_bone_name = candidate
                    break

            if root_bone_name and verbose:
                print(f"    Found root motion bone: {root_bone_name}")

            # Remove location animation from the root bone (Hips)
            # This prevents the character from translating through space
            if target_armature.animation_data and target_armature.animation_data.action:
                action = target_armature.animation_data.action
                try:
                    atd = target_armature.animation_data
                    if hasattr(atd, 'action_slot') and atd.action_slot:
                        # Blender 5.0: Use layered action API
                        for layer in action.layers:
                            for strip in layer.strips:
                                bag = strip.channelbag(atd.action_slot)
                                if bag:
                                    # Remove object-level fcurves (armature translation)
                                    object_paths = ['location', 'rotation_euler',
                                                    'rotation_quaternion', 'scale']
                                    object_fcurves = [fc for fc in bag.fcurves
                                                     if fc.data_path in object_paths]
                                    for fc in object_fcurves:
                                        bag.fcurves.remove(fc)
                                    if verbose and object_fcurves:
                                        print(f"    Removed {len(object_fcurves)} object-level fcurves")

                                    # Remove Hips location fcurves (root motion)
                                    if root_bone_name:
                                        hips_loc_paths = [
                                            f'pose.bones["{root_bone_name}"].location',
                                        ]
                                        hips_fcurves = [fc for fc in bag.fcurves
                                                       if fc.data_path in hips_loc_paths]
                                        for fc in hips_fcurves:
                                            bag.fcurves.remove(fc)
                                        if verbose and hips_fcurves:
                                            print(f"    Removed {len(hips_fcurves)} root bone location fcurves")
                    else:
                        # Fallback for older Blender versions
                        if hasattr(action, 'fcurves'):
                            # Remove object-level fcurves
                            object_paths = ['location', 'rotation_euler',
                                            'rotation_quaternion', 'scale']
                            for path in object_paths:
                                for i in range(3):
                                    fc = action.fcurves.find(path, index=i)
                                    if fc:
                                        action.fcurves.remove(fc)

                            # Remove Hips location fcurves
                            if root_bone_name:
                                hips_loc_path = f'pose.bones["{root_bone_name}"].location'
                                for i in range(3):
                                    fc = action.fcurves.find(hips_loc_path, index=i)
                                    if fc:
                                        action.fcurves.remove(fc)
                except Exception as e:
                    if verbose:
                        print(f"    Note: Could not remove root motion fcurves: {e}")

            # Force armature and root bone to origin
            target_armature.location = (0, 0, 0)
            target_armature.rotation_euler = (0, 0, 0)
            target_armature.rotation_quaternion = (1, 0, 0, 0)

            # Also reset the root bone (Hips) to its rest position
            if root_bone_name and root_bone_name in target_armature.pose.bones:
                hips_bone = target_armature.pose.bones[root_bone_name]
                hips_bone.location = (0, 0, 0)
                if verbose:
                    print(f"    Reset {root_bone_name} bone to origin")

        if verbose:
            total_frames = frame_end - frame_start + 1
            print(f"  ✓ Animation transferred successfully")
            print(f"    Action: {baked_action.name}")
            print(f"    Frames: {total_frames}")
            print(f"    Bones: {len(matching_bones)}")

        return True

    except Exception as e:
        print(f"  ✗ Error transferring animation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def bake_animation(
    armature,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    verbose: bool = False
) -> bool:
    """
    Bake animation data into keyframes for export.

    This ensures all bone transforms are keyframed and ready for FBX export.

    Args:
        armature: Armature with animation to bake
        frame_start: Start frame (if None, uses action's start frame)
        frame_end: End frame (if None, uses action's end frame)
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise

    Note:
        Baking converts all constraints and drivers into simple keyframes.
        This is required for proper FBX animation export.
    """
    try:
        import bpy

        if verbose:
            print(f"  Baking animation on {armature.name}...")

        # Check if armature has animation
        if not armature.animation_data or not armature.animation_data.action:
            print("  ✗ No animation to bake")
            return False

        action = armature.animation_data.action

        # Determine frame range
        if frame_start is None or frame_end is None:
            frame_range = action.frame_range
            frame_start = int(frame_range[0]) if frame_start is None else frame_start
            frame_end = int(frame_range[1]) if frame_end is None else frame_end

        if verbose:
            print(f"    Frame range: {frame_start} - {frame_end}")

        # Set frame range in scene
        bpy.context.scene.frame_start = frame_start
        bpy.context.scene.frame_end = frame_end

        # Select armature and set as active
        bpy.ops.object.select_all(action='DESELECT')
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature

        # Bake animation using NLA
        # This converts all animation into keyframes
        bpy.ops.nla.bake(
            frame_start=frame_start,
            frame_end=frame_end,
            step=1,  # Bake every frame
            only_selected=False,  # Bake all bones
            visual_keying=True,  # Use visual transforms (includes constraints)
            clear_constraints=False,  # Keep constraints (we don't have any for Mixamo)
            clear_parents=False,  # Keep parent relationships
            use_current_action=True,  # Bake into current action
            bake_types={'POSE'}  # Only bake pose (not object transform)
        )

        if verbose:
            print(f"  ✓ Animation baked successfully")

        return True

    except Exception as e:
        print(f"  ✗ Error baking animation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def import_and_apply_mixamo_animation(
    mesh_obj,
    current_armature,
    animation_fbx_path: str,
    remove_root_motion: bool = True,
    verbose: bool = False
) -> tuple:
    """
    Complete workflow: Import Mixamo animation and transfer to MPFB2 rig.

    Uses manual keyframe sampling to transfer animation from imported Mixamo FBX
    to the MPFB2 Mixamo rig. Works with Blender 5.0's new layered action system.

    Workflow:
    1. Import Mixamo animation FBX (contains rig + animation)
    2. Transfer animation using frame-by-frame sampling
    3. Delete imported rig
    4. Return current_armature with animation applied

    Args:
        mesh_obj: Your MPFB2 character mesh
        current_armature: Current MPFB2 Mixamo armature (will receive animation)
        animation_fbx_path: Path to Mixamo animation FBX file
        remove_root_motion: If True, keeps armature at origin (prevents "flying")
        verbose: Enable verbose output

    Returns:
        Tuple of (armature, success)
        - armature: The current_armature with animation applied (or None if failed)
        - success: True if successful, False otherwise

    Example:
        armature, success = import_and_apply_mixamo_animation(
            mesh_obj=basemesh,
            current_armature=armature,
            animation_fbx_path="mixamo_walk.fbx",
            verbose=True
        )

    Note:
        - Your character must be rigged with Mixamo rig first
        - Mixamo animation should be downloaded "Without Skin"
        - Uses manual keyframe sampling (Blender 5.0 compatible)
        - Root motion is removed by default to prevent "flying" behavior
    """
    try:
        import bpy

        if verbose:
            print("\n" + "-"*70)
            print("IMPORTING AND TRANSFERRING MIXAMO ANIMATION")
            print("-"*70)
            print(f"Mesh: {mesh_obj.name}")
            print(f"Target armature: {current_armature.name}")
            print(f"Animation file: {Path(animation_fbx_path).name}")

        # Step 1: Import Mixamo animation FBX
        if verbose:
            print("\nStep 1: Importing Mixamo animation FBX...")

        imported_armature = import_mixamo_animation(animation_fbx_path, verbose=verbose)

        if not imported_armature:
            return None, False

        # Verify animation exists
        if not imported_armature.animation_data or not imported_armature.animation_data.action:
            print("  ✗ Imported armature has no animation")
            bpy.data.objects.remove(imported_armature, do_unlink=True)
            return None, False

        # Step 2: Transfer animation using Blender 5.0 compatible method
        if verbose:
            print("\nStep 2: Transferring animation to target armature...")

        success = transfer_animation_blender50(
            imported_armature,
            current_armature,
            remove_root_motion=remove_root_motion,
            verbose=verbose
        )

        if not success:
            # Clean up imported armature
            bpy.data.objects.remove(imported_armature, do_unlink=True)
            return None, False

        # Step 3: Clean up imported armature (no longer needed)
        if verbose:
            print(f"\nStep 3: Cleaning up imported armature...")

        bpy.data.objects.remove(imported_armature, do_unlink=True)

        if verbose:
            print("\n" + "-"*70)
            print("✓ MIXAMO ANIMATION TRANSFERRED")
            print("-"*70)
            print(f"  Mesh: {mesh_obj.name}")
            print(f"  Armature: {current_armature.name}")
            if current_armature.animation_data and current_armature.animation_data.action:
                action = current_armature.animation_data.action
                print(f"  Animation: {action.name}")
                print(f"  Frame range: {int(action.frame_range[0])} - {int(action.frame_range[1])}")

        return current_armature, True

    except Exception as e:
        print(f"\n✗ Error in animation transfer workflow: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None, False


# Legacy alias for backward compatibility
def import_and_bake_mixamo_animation(
    armature_obj,
    animation_fbx_path: str,
    cleanup_imported: bool = True,
    verbose: bool = False
) -> bool:
    """
    DEPRECATED: Use import_and_apply_mixamo_animation instead.

    This function is kept for backward compatibility but will fail with
    a helpful error message directing users to the new approach.
    """
    print("\n" + "="*70)
    print("ERROR: Old animation baking method no longer supported")
    print("="*70)
    print("\nThe animation baking API changed in Blender 5.0.")
    print("Please update your code to use the new approach:")
    print("")
    print("OLD (deprecated):")
    print("  import_and_bake_mixamo_animation(armature, fbx_path)")
    print("")
    print("NEW (Blender 5.0 compatible):")
    print("  new_armature, success = import_and_apply_mixamo_animation(")
    print("      mesh_obj=basemesh,")
    print("      current_armature=armature,")
    print("      animation_fbx_path=fbx_path")
    print("  )")
    print("")
    print("See animation_utils.py documentation for details.")
    print("="*70)
    return False


# Example standalone usage
if __name__ == "__main__":
    print("Animation Utils Library")
    print("=" * 70)
    print("\nThis is a library module. Import it in your scripts:")
    print("  import animation_utils")
    print("  animation_utils.import_and_bake_mixamo_animation(armature, 'walk.fbx')")
