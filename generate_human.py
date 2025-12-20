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
import argparse
from pathlib import Path

# Add the script directory to Python path to import utils
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import utils


def parse_arguments():
    """
    Parse command line arguments.
    
    When running with Blender, arguments after '--' are passed to the script.
    Example: python run_blender.py --script generate_human.py -- --config human_female.json
    
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
    python run_blender.py --script generate_human.py -- --config human_female.json

    python run_blender.py --script generate_human.py -- --config human_female.json --rig-type default_no_toes

    python run_blender.py --script generate_human.py -- --config human_female.json --fk-ik-hybrid --instrumented-arm right
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
        default='default',
        choices=['default', 'default_no_toes', 'game_engine'],
        help='Type of rig to add (default: default)'
    )
    
    parser.add_argument(
        '--no-rig',
        action='store_true',
        help='Skip adding rig (export mesh only)'
    )

    parser.add_argument(
        '--fk-ik-hybrid',
        action='store_true',
        help='Configure FK/IK hybrid rig for IMU sensor-based motion capture'
    )

    parser.add_argument(
        '--instrumented-arm',
        type=str,
        default='left',
        choices=['left', 'right'],
        help='Which arm has IMU sensors attached (default: left)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
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
        import importlib

        # Get correct MPFB module path (supports both Blender 4.2+ and legacy)
        mpfb_path = utils._get_mpfb_module_path()
        HumanService = importlib.import_module(f'{mpfb_path}.services.humanservice').HumanService

        print("Creating base human mesh...")
        basemesh = HumanService.create_human()
        print(f"✓ Base mesh created: {basemesh.name}")

    except Exception as e:
        print(f"\n✗ Error creating human mesh: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Apply macro settings
    print("\n" + "-"*70)
    print("STEP 4: Applying Body Parameters")
    print("-"*70)
    
    try:
        # Don't bake macros if microparameters will be applied next
        utils.apply_macro_settings_to_human(basemesh, validated_macro, bake=not has_micros)
    except Exception as e:
        print(f"\n✗ Error applying macro settings: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Add rigging
    armature = None
    if not args.no_rig:
        print("\n" + "-"*70)
        print("STEP 5: Adding Rig")
        print("-"*70)

        try:
            armature, _ = utils.add_standard_rig(basemesh, args.rig_type)

            # Configure FK/IK hybrid rig if requested
            if args.fk_ik_hybrid:
                print("\n" + "-"*70)
                print("STEP 5.5: Configuring FK/IK Hybrid Rig")
                print("-"*70)
                try:
                    utils.configure_fk_ik_hybrid_rig(armature, args.instrumented_arm)
                except Exception as e:
                    print(f"\n⚠ Warning: Failed to configure FK/IK hybrid rig: {e}")
                    import traceback
                    if args.verbose:
                        traceback.print_exc()
        except Exception as e:
            print(f"\n⚠ Warning: Failed to add rig: {e}")
            print("Continuing without rig...")
            import traceback
            if args.verbose:
                traceback.print_exc()
    else:
        print("\n" + "-"*70)
        print("STEP 5: Skipping Rig (--no-rig specified)")
        print("-"*70)
    
    # Export FBX
    print("\n" + "-"*70)
    print("STEP 6: Exporting FBX")
    print("-"*70)
    
    try:
        # Get export settings from config if provided
        export_settings = config.get("export_settings", {})
        
        utils.export_fbx(basemesh, armature, output_path, export_settings)
        
    except Exception as e:
        print(f"\n✗ Error exporting FBX: {e}")
        import traceback
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
        import traceback
        traceback.print_exc()
        sys.exit(1)
