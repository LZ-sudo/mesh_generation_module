#!/usr/bin/env python3
"""
Main script for generating human meshes with MPFB2 in headless Blender.

This script can be run from the command line without opening the Blender GUI:
    

The script will:
1. Load configuration from JSON file
2. Create a human mesh with specified parameters
3. Add standard rigging
4. Export as FBX file
"""

import sys
import os
import builtins
import argparse
from pathlib import Path
import traceback
import importlib

# Add the script directory to Python path to import utils
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import utils as utils

# Force all print() calls in this process to flush immediately.
# Blender replaces sys.stdout with its own buffered object, so PYTHONUNBUFFERED
# has no effect once Blender is running. Patching builtins.print ensures that
# every print() call -- including those from MPFB and retarget_bvh -- flushes
# to the pipe on each line, enabling real-time output in the GUI log widget.
_orig_print = builtins.print

def _print_flush(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

builtins.print = _print_flush


def parse_arguments():
    """
    Parse command line arguments.
    
    When running with Blender, arguments after '--' are passed to the script.
    Example: python run_blender.py --script generation_scripts/generate_human.py -- --config human_female.json
    
    Returns:
        Parsed arguments
    """
    # Find where Blender arguments end and script arguments begin
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(
        description='Generate human mesh with MPFB2 in headless Blender',
        epilog="""
Example usage:
    python run_blender.py --script generate_human.py --config human_female.json

    python run_blender.py --script generate_human.py --config human_female.json --rig-type default_no_toes

    python run_blender.py --script generate_human.py --config human_female.json --fk-ik-hybrid --instrumented-arm right

    python run_blender.py --script generate_human.py --config human_female.json --fk-ik-hybrid --instrumented-arm right --t-pose
    """
    
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--rig-type',
        type=str,
        default='default_no_toes',
        choices=['default_no_toes', 'cmu_mb'],
        help='Type of rig to add (default_no_toes for measurements, cmu_mb for CMU mocap BVH/FBX animation retargeting via retarget_bvh)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for generated mesh (default: ./output). Filename will match the input config name.'
    )

    parser.add_argument(
        '--hair',
        type=str,
        default=None,
        help='Hair asset name to apply from mpfb_hair_assets folder (e.g., "Short_Hair_B"). Optional.'
    )

    parser.add_argument(
        '--clothing',
        type=str,
        nargs='+',
        default=None,
        help='One or more clothing asset names from mpfb_clothing_assets folder (e.g., "Scrub_Shirt" "Scrub_Pants"). Optional.'
    )

    parser.add_argument(
        '--t-pose',
        action='store_true',
        help='Export the model in T-pose instead of the default A-pose'
    )

    parser.add_argument(
        '--animation',
        type=str,
        default=None,
        help='Path to animation file (BVH or FBX). Requires --rig-type cmu_mb. Retargeted via the retarget_bvh addon (mcp.load_and_retarget). Optional.'
    )

    parser.add_argument(
        '--collision',
        action='store_true',
        help='Generate UCX_ collision meshes for Unreal Engine physics asset import using CoACD convex decomposition.'
    )

    parser.add_argument(
        '--collision-threshold',
        type=float,
        default=0.05,
        help='CoACD concavity threshold for collision mesh generation (default: 0.05). Normalized ratio of each region bounding box. Lower = tighter fit, more convex pieces.'
    )

    parser.add_argument(
        '--collision-max-vertices',
        type=int,
        default=5000,
        help='Vertex cap per body region before CoACD runs (default: 2000). Lower = faster, coarser decimation.'
    )

    return parser.parse_args(argv)


def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("MPFB2 HUMAN GENERATOR - HEADLESS MODE")
    print("="*70 + "\n")
    
    # Parse arguments
    try:
        args = parse_arguments()
    except SystemExit as e:
        # Handle argparse errors gracefully
        if e.code != 0:
            print("\n✗ Error parsing arguments")
        sys.exit(e.code)
    
    if args.verbose:
        print(f"Arguments: {vars(args)}\n")
    
    # Check if we're running in Blender
    try:
        import bpy
        print("✓ Running in Blender environment")
        print(f"  Blender version: {bpy.app.version_string}")
    except ImportError:
        print("✗ ERROR: This script must be run with Blender!")
        sys.exit(1)
    
    # Check if MPFB2 is installed
    if not utils.check_mpfb2_installed():
        sys.exit(1)
    
    # Load and validate configuration
    print("\n" + "-"*70)
    print("STEP 1: Loading Configuration")
    print("-"*70)
    
    try:
        config = utils.load_json_config(args.config)

        # Add output section from command-line args if not in config
        if 'output' not in config:
            # Derive output filename from input config filename
            config_path = Path(args.config)
            output_filename = config_path.stem + '.fbx'  # e.g., 'example_subject_asian_female.json' -> 'example_subject_asian_female.fbx'

            config['output'] = {
                'directory': args.output_dir,
                'filename': output_filename
            }
            print(f"ℹ Using output directory from arguments: {args.output_dir}")
            print(f"ℹ Output filename derived from config: {output_filename}")

        # Add export_settings if not in config (use defaults)
        if 'export_settings' not in config:
            config['export_settings'] = {
                'use_mesh_modifiers': True,
                'add_leaf_bones': True
            }

        utils.validate_json_structure(config)
        validated_macro = utils.validate_macro_settings(config)
        output_path = utils.get_output_path(config)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n✗ Configuration Error: {e}")
        sys.exit(1)
    
    # Print configuration summary
    utils.print_configuration_summary(validated_macro, output_path)
    
    # Set up Blender scene
    print("-"*70)
    print("STEP 2: Setting Up Blender Scene")
    print("-"*70)
    utils.setup_blender_scene()
    
    # Create human mesh
    print("\n" + "-"*70)
    print("STEP 3: Creating Human Mesh")
    print("-"*70)

    try:

        # Get correct MPFB module path (supports both Blender 4.2+ and legacy)
        mpfb_path = utils._get_mpfb_module_path()
        HumanService = importlib.import_module(f'{mpfb_path}.services.humanservice').HumanService

        print("Creating base human mesh...")
        basemesh = HumanService.create_human()
        print(f"✓ Base mesh created: {basemesh.name}")

    except Exception as e:
        print(f"\n✗ Error creating human mesh: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if microparameters will be applied
    has_micros = 'micro_settings' in config and config['micro_settings']

    # Apply macro settings
    print("\n" + "-"*70)
    print("STEP 4: Applying Macroparameters")
    print("-"*70)

    try:
        # Don't bake macros if microparameters will be applied next
        utils.apply_macro_settings_to_human(basemesh, validated_macro, bake=not has_micros)
    except Exception as e:
        print(f"\n✗ Error applying macro settings: {e}")
        
        traceback.print_exc()
        sys.exit(1)

    # Apply microparameters if provided
    if has_micros:
        print("\n" + "-"*70)
        print("STEP 4.5: Applying Microparameters (Fine-Tuning)")
        print("-"*70)

        try:
            utils.apply_microparameters_to_human(basemesh, config['micro_settings'], bake=True, verbose=args.verbose)
            print(f"✓ Applied {len(config['micro_settings'])} microparameters")
        except Exception as e:
            print(f"\n⚠ Warning: Failed to apply microparameters: {e}")
            
            if args.verbose:
                traceback.print_exc()

    # Add rigging
    print("\n" + "-"*70)
    print("STEP 5: Adding Rig")
    print("-"*70)

    try:
        armature, _ = utils.add_standard_rig(basemesh, args.rig_type)
    except Exception as e:
        print(f"\n✗ Error adding rig: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Apply hair asset if requested
    hair_obj = None
    clothing_objs = []
    if args.hair:
        print("\n" + "-"*70)
        print("STEP 5.5: Applying Hair Asset")
        print("-"*70)

        if not armature:
            print("⚠ Warning: Cannot apply hair without rig. Use --rig-type to add a rig.")
            print("Skipping hair application...")
        else:
            try:
                # Import hair application library
                sys.path.insert(0, str(script_dir / "mesh_hair_generation"))
                import mpfb_hair_assets_application as hair_lib

                # Apply hair asset
                hair_obj = hair_lib.apply_hair_asset(
                    human_obj=basemesh,
                    armature_obj=armature,
                    hair_asset_name=args.hair,
                    verbose=args.verbose
                )

                if hair_obj:
                    print(f"✓ Hair asset '{args.hair}' applied successfully")
                else:
                    print(f"⚠ Warning: Failed to apply hair asset '{args.hair}'")

            except Exception as e:
                print(f"⚠ Warning: Error applying hair asset: {e}")
                if args.verbose:
                    
                    traceback.print_exc()

    # Apply clothing assets if requested
    if args.clothing:
        print("\n" + "-"*70)
        print("STEP 5.6: Applying Clothing Assets")
        print("-"*70)

        if not armature:
            print("⚠ Warning: Cannot apply clothing without rig. Use --rig-type to add a rig.")
            print("Skipping clothing application...")
        else:
            try:
                sys.path.insert(0, str(script_dir / "mesh_clothing_generation"))
                import mpfb_clothing_assets_application as clothing_lib

                for clothing_name in args.clothing:
                    clothing_obj = clothing_lib.apply_clothing_asset(
                        human_obj=basemesh,
                        armature_obj=armature,
                        clothing_asset_name=clothing_name,
                        verbose=args.verbose
                    )
                    if clothing_obj:
                        clothing_objs.append(clothing_obj)
                        print(f"✓ Clothing asset '{clothing_name}' applied successfully")
                    else:
                        print(f"⚠ Warning: Failed to apply clothing asset '{clothing_name}'")

            except Exception as e:
                print(f"⚠ Warning: Error applying clothing assets: {e}")
                if args.verbose:
                    traceback.print_exc()

    # Apply T-pose if requested
    if args.t_pose and armature:
        print("\n" + "-"*70)
        print("STEP 5.7: Setting T-Pose")
        print("-"*70)

        try:
            success = utils.set_tpose(armature, args.rig_type)
            if success:
                print("T-pose applied successfully")
            else:
                print("Warning: T-pose application may have issues")
        except Exception as e:
            print(f"Warning: Failed to apply T-pose: {e}")
            if args.verbose:
                traceback.print_exc()

    # Apply animation if provided
    if args.animation and armature:
        print("\n" + "-"*70)
        print("STEP 5.8: Applying Animation")
        print("-"*70)

        if args.rig_type != 'cmu_mb':
            print("Warning: Animation import requires --rig-type cmu_mb")
            print(f"  Current rig type: {args.rig_type}")
            print("  Skipping animation import...")
        else:
            try:
                sys.path.insert(0, str(script_dir / "mesh_rigging_animation"))
                import animation_utils

                new_armature, success = animation_utils.apply_cmu_mb_animation(
                    armature=armature,
                    animation_path=args.animation,
                    verbose=args.verbose
                )

                if success and new_armature:
                    print("Animation applied successfully")
                    armature = new_armature
                else:
                    print("Warning: Failed to apply animation")
                    print("  Continuing with export without animation...")

            except Exception as e:
                print(f"Warning: Error applying animation: {e}")
                if args.verbose:
                    traceback.print_exc()
                print("  Continuing with export...")

    # Generate collision meshes if requested
    ucx_objects = []
    if args.collision:
        print("\n" + "-"*70)
        print("STEP 5.9: Generating Collision Meshes (CoACD)")
        print("-"*70)

        try:
            sys.path.insert(0, str(script_dir / "mesh_collision_implementation"))
            import collision_mesh_generation as collision_lib

            ucx_objects = collision_lib.generate_collision_meshes(
                basemesh=basemesh,
                script_dir=script_dir,
                threshold=args.collision_threshold,
                max_vertices=args.collision_max_vertices,
                verbose=args.verbose,
            )

            if ucx_objects:
                print(f"  {len(ucx_objects)} UCX collision mesh(es) created")
            else:
                print("  Warning: No collision meshes generated - continuing without them")

        except Exception as e:
            print(f"  Warning: Collision mesh generation failed: {e}")
            if args.verbose:
                traceback.print_exc()
            print("  Continuing without collision meshes...")

    # Export FBX
    print("\n" + "-"*70)
    print("STEP 6: Exporting FBX")
    print("-"*70)

    try:
        # Get export settings from config if provided
        export_settings = config.get("export_settings", {})

        # Enable animation baking if animation was applied
        has_animation = (armature and
                        armature.animation_data and
                        armature.animation_data.action is not None)

        if has_animation:
            export_settings['bake_anim'] = True
            # bake_anim_use_all_actions=True is required for Blender 5.0 slot-based
            # actions. It is the ONLY export path that explicitly reassigns
            # ob.animation_data.action_slot before sampling frames, which triggers
            # the depsgraph to evaluate slot-based animation correctly.
            # The single-action path (use_all_actions=False) skips the slot
            # reassignment and samples rest poses at every frame (static export).
            export_settings['bake_anim_use_all_actions'] = True
            export_settings['bake_anim_use_nla_strips'] = False
            # Disable simplification: factor > 0 can collapse animation curves.
            export_settings['bake_anim_simplify_factor'] = 0.0
            if args.verbose:
                print(f"  Animation detected - enabling bake_anim with use_all_actions=True (Blender 5.0 slot path)")

        # Prepare list of additional objects (hair, clothes, collision meshes)
        additional_objects = []
        if hair_obj:
            additional_objects.append(hair_obj)
        additional_objects.extend(clothing_objs)
        additional_objects.extend(ucx_objects)

        utils.export_fbx(basemesh, armature, output_path, export_settings, additional_objects)
        
    except Exception as e:
        print(f"\n✗ Error exporting FBX: {e}")
        
        traceback.print_exc()
        sys.exit(1)
    
    # Success!
    print("\n" + "="*70)
    print("✓ HUMAN GENERATION COMPLETE!")
    print("="*70)
    print(f"\nOutput file: {output_path}")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
    print(f"File size: {file_size:.2f} MB")
    
    print("\n" + "="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        
        traceback.print_exc()
        sys.exit(1)
