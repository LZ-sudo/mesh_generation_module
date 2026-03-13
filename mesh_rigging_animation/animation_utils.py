#!/usr/bin/env python3
"""
Animation utilities for applying animations to MPFB2 rigs.

Functions:
    - apply_cmu_mb_animation(): Retarget BVH/FBX animation onto a CMU MB rig
                                via the retarget_bvh addon (mcp.load_and_retarget)
"""

import sys
import traceback
from pathlib import Path


def _enable_retarget_bvh(verbose: bool = False) -> bool:
    """
    Ensure the retarget_bvh addon operators are registered. Returns True if
    the operators are available after the attempt.

    bpy.ops.mcp is a dynamic namespace that always exists in Blender regardless
    of whether any operators are registered, so hasattr() on bpy.ops is
    unreliable. Calling poll() on the specific operator is the correct check:
    it raises RuntimeError when the operator is not registered, and returns
    a bool when it is.
    """
    import bpy

    try:
        bpy.ops.mcp.load_and_retarget.poll()
        return True
    except Exception:
        pass

    if verbose:
        print("  retarget_bvh operators not registered, enabling addon...")

    # Prefer the exact module name from the user's installed addons
    candidates = []
    for addon in bpy.context.preferences.addons:
        name = addon.module.lower()
        if 'retarget_bvh' in name or ('retarget' in name and 'bvh' in name):
            candidates.insert(0, addon.module)

    for fallback in ('retarget_bvh', 'bl_ext.user_default.retarget_bvh'):
        if fallback not in candidates:
            candidates.append(fallback)

    for module_name in candidates:
        try:
            bpy.ops.preferences.addon_enable(module=module_name)
            bpy.ops.mcp.load_and_retarget.poll()
            if verbose:
                print(f"  Enabled retarget_bvh as: {module_name}")
            return True
        except Exception:
            pass

    return False


def apply_cmu_mb_animation(
    armature,
    animation_path: str,
    verbose: bool = False
) -> tuple:
    """
    Apply BVH or FBX animation to a CMU MB rig via the retarget_bvh addon.

    Requires the retarget_bvh (MakeWalk) addon to be installed and enabled in
    Blender. The armature must use the MPFB2 'cmu_mb' rig type.

    The retarget_bvh operator handles FBX-to-BVH conversion, bone renaming
    from the cmu-mb source rig JSON, T-pose compensated retargeting, and
    source rig cleanup.

    Args:
        armature: The MPFB2 CMU MB armature object
        animation_path: Path to BVH or FBX animation file
        verbose: Enable verbose output

    Returns:
        Tuple of (armature, success)
    """
    import bpy

    anim_path = Path(animation_path)
    if not anim_path.exists():
        print(f"  Error: Animation file not found: {animation_path}")
        return None, False

    if verbose:
        print(f"  Animation: {anim_path.name}")
        print(f"  Armature : {armature.name}")

    if not _enable_retarget_bvh(verbose):
        print("  Error: retarget_bvh addon could not be enabled.")
        print("  Install the Diffeomorphic retarget_bvh addon and enable it in Blender.")
        return None, False

    # BD.ensureInited loads the known_rigs JSON tables (sourceInfos, targetInfos).
    # In the GUI this is called from invoke(), but scripts call execute() directly,
    # bypassing invoke(). Must be called after _enable_retarget_bvh() so that the
    # bvh_retargeter PropertyGroup is registered on the scene.
    for mod in sys.modules.values():
        if hasattr(mod, 'BD') and hasattr(mod.BD, 'ensureInited'):
            try:
                mod.BD.ensureInited(bpy.context.scene)
                if verbose:
                    print("  BD initialized: source/target rig tables loaded")
            except Exception as bd_err:
                print(f"  Warning: BD.ensureInited failed: {bd_err}")
                print("  Retarget will use Automatic rig detection")
            break

    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)

    try:
        # useAllFrames=True hard-codes the range to (-9999, 9999), capping any
        # BVH at exactly 10 000 frames regardless of actual length.  Passing
        # useAllFrames=False with a large endFrame delegates the upper bound to
        # the BVH header's own Frames count (the loader also checks frame < nFrames),
        # so no frames beyond the file are ever read.
        result = bpy.ops.mcp.load_and_retarget(
            filepath=str(anim_path),
            useAllFrames=False,
            startFrame=0,
            endFrame=1_000_000,
        )
        if verbose:
            print(f"  Operator result: {result}")
        if 'FINISHED' not in result:
            print(f"  Error: mcp.load_and_retarget returned {result}")
            return None, False
    except Exception as e:
        print(f"  Error: mcp.load_and_retarget failed: {e}")
        if verbose:
            traceback.print_exc()
        return None, False

    if not (armature.animation_data and armature.animation_data.action):
        print("  Error: No animation data found after retargeting")
        return None, False

    action = armature.animation_data.action
    frame_range = action.frame_range
    bpy.context.scene.frame_start = int(frame_range[0])
    bpy.context.scene.frame_end = int(frame_range[1])

    print(f"  Action: {action.name}, Frames: {int(frame_range[0])} - {int(frame_range[1])}")

    # Remove orphaned actions left by retarget_bvh. With bake_anim_use_all_actions=True
    # the FBX exporter iterates bpy.data.actions and exports every action whose slot
    # validates against the armature - including the source BVH action that
    # retarget_bvh loaded then orphaned when it deleted the source rig. If that
    # orphaned action shares bone names with the target armature (which it does for
    # CMU MB rigs), it passes find_validate_action_slot() and is exported as a second
    # take. The FBX importer activates the first take on import, which may be the
    # wrong one and appear as T-pose.
    removed = []
    for orphan in list(bpy.data.actions):
        if orphan is action:
            continue
        if orphan.users == 0:
            removed.append(orphan.name)
            bpy.data.actions.remove(orphan)
    if removed:
        print(f"  Removed orphaned actions: {removed}")

    return armature, True


if __name__ == "__main__":
    print("Animation Utils Library")
    print("=" * 70)
    print("\nThis is a library module. Import it in your scripts:")
    print("  import animation_utils")
    print("  armature, ok = animation_utils.apply_cmu_mb_animation(")
    print("      armature=armature,")
    print("      animation_path='walk.bvh'")
    print("  )")
