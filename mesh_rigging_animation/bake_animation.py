"""
Animation Baker - Blender headless script.

Imports an FBX avatar with a CMU MB rig, retargets a BVH animation onto
the rig using the retarget_bvh addon, and exports the result as a new FBX
file with the animation baked in.

Usage (via run_blender.py):
    python run_blender.py --script mesh_rigging_animation/bake_animation.py \
        -- --fbx avatar.fbx --bvh walk.bvh --output avatar_animated.fbx
"""

import argparse
import builtins
import sys
from pathlib import Path

# Patch builtins.print so every call inside Blender flushes immediately.
# Without this, Python output is buffered and only appears after the script
# exits, making the GUI log widget appear frozen during long operations.
_orig_print = builtins.print

def _print_flush(*args, **kwargs):
    """Wrap builtins.print to force flush=True for real-time GUI streaming."""
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

builtins.print = _print_flush

script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent.absolute()

if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def parse_arguments():
    """
    Parse command line arguments passed after '--' to Blender.

    Returns:
        Parsed argparse.Namespace with fields: fbx, bvh, output.
    """
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Bake BVH animation onto an FBX avatar CMU MB rig"
    )
    parser.add_argument("--fbx", required=True, help="Path to input FBX avatar file")
    parser.add_argument("--bvh", required=True, help="Path to BVH animation file")
    parser.add_argument("--output", required=True, help="Path for output FBX file")
    return parser.parse_args(argv)


def main():
    """Entry point for headless BVH-to-FBX animation baking in Blender."""
    args = parse_arguments()

    import bpy
    from animation_utils import apply_cmu_mb_animation
    from utils import export_fbx

    fbx_path = Path(args.fbx)
    bvh_path = Path(args.bvh)
    output_path = Path(args.output)

    if not fbx_path.exists():
        print(f"ERROR: FBX file not found: {fbx_path}")
        sys.exit(1)
    if not bvh_path.exists():
        print(f"ERROR: BVH file not found: {bvh_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear the default scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    print(f"Importing FBX: {fbx_path.name}")
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))

    # Identify armature, primary mesh, and any additional meshes (clothing, hair)
    armature = None
    basemesh = None
    additional_objects = []

    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and armature is None:
            armature = obj
        elif obj.type == 'MESH':
            if basemesh is None:
                basemesh = obj
            else:
                additional_objects.append(obj)

    if armature is None:
        print("ERROR: No armature found in the FBX file")
        sys.exit(1)
    if basemesh is None:
        print("ERROR: No mesh found in the FBX file")
        sys.exit(1)

    print(f"Armature : {armature.name}")
    print(f"Mesh     : {basemesh.name}")
    if additional_objects:
        print(f"Additional meshes: {[o.name for o in additional_objects]}")

    print(f"\nApplying animation: {bvh_path.name}")
    armature, success = apply_cmu_mb_animation(armature, str(bvh_path), verbose=True)

    if not success:
        print("ERROR: Animation baking failed")
        sys.exit(1)

    print(f"\nExporting animated FBX: {output_path.name}")
    export_fbx(
        basemesh=basemesh,
        armature=armature,
        output_path=str(output_path),
        export_settings={
            "bake_anim": True,
            # Export only the currently active action so the importer
            # activates the retargeted animation, not an arbitrary take.
            "bake_anim_use_all_actions": False,
            "bake_anim_use_nla_strips": False,
            # Preserve every keyframe — mocap data must not be simplified.
            "bake_anim_simplify_factor": 0.0,
        },
        additional_objects=additional_objects if additional_objects else None,
    )

    print(f"SUCCESS: Saved to {output_path}")


if __name__ == "__main__":
    main()
