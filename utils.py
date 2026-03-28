"""
Utility functions for MPFB2 human generation.

This module provides helper functions for:
- Loading and validating JSON configuration files
- Validating macro parameters
- Setting up Blender scene
- Applying settings to human mesh
"""

import json
import numpy as np
import os
import sys
import importlib
import platform
from typing import Dict, Any, Tuple

# NOTE: bpy and mpfb imports are done inside functions to avoid import errors
# when this module is imported outside of Blender environment


def _ensure_extensions_loaded():
    """
    Ensure extensions are loaded in headless mode (Blender 5.0+).

    In Blender 5.0, extensions are not automatically loaded when running
    in --background mode. This function adds the extension path to sys.path
    so that extensions can be imported.

    Returns:
        bool: True if extensions were successfully enabled, False otherwise
    """

    # Check if MPFB extension is already loaded
    if 'bl_ext.user_default.mpfb' in sys.modules:
        return True  # Already loaded

    # Check if legacy addon is loaded
    if 'mpfb' in sys.modules:
        return True  # Already loaded (legacy format)

    # Try to enable extensions in headless mode
    try:
        import bpy

        # Check Blender version - extensions introduced in 4.2
        if bpy.app.version < (4, 2, 0):
            return False  # Old Blender version, doesn't use extensions

        # In Blender 5.0+, extensions are stored in user extensions directory
        # We need to manually add this path to sys.path in headless mode

        # Get the extensions directory path
        # Extensions are typically in: %APPDATA%\Blender Foundation\Blender\5.0\extensions\user_default

        if platform.system() == "Windows":
            appdata = os.environ.get('APPDATA')
            if appdata:
                version_str = f"{bpy.app.version[0]}.{bpy.app.version[1]}"
                extensions_path = os.path.join(appdata, "Blender Foundation", "Blender", version_str, "extensions", "user_default")

                if os.path.exists(extensions_path) and extensions_path not in sys.path:
                    sys.path.insert(0, extensions_path)

        elif platform.system() == "Darwin":  # macOS
            home = os.path.expanduser("~")
            version_str = f"{bpy.app.version[0]}.{bpy.app.version[1]}"
            extensions_path = os.path.join(home, "Library", "Application Support", "Blender", version_str, "extensions", "user_default")

            if os.path.exists(extensions_path) and extensions_path not in sys.path:
                sys.path.insert(0, extensions_path)

        else:  # Linux
            home = os.path.expanduser("~")
            version_str = f"{bpy.app.version[0]}.{bpy.app.version[1]}"
            extensions_path = os.path.join(home, ".config", "blender", version_str, "extensions", "user_default")

            if os.path.exists(extensions_path) and extensions_path not in sys.path:
                sys.path.insert(0, extensions_path)

        # Try importing the extension now that the path is added
        try:
            import bl_ext.user_default.mpfb
            return True
        except ImportError:
            return False

    except Exception:
        return False


def _get_mpfb_module_path():
    """
    Determine the correct MPFB module path for the current Blender version.

    In Blender 5.0+, extensions can be installed from different repositories:
    - bl_ext.blender_org.mpfb (official Blender Extensions repository)

    """


    # Ensure extensions are loaded in headless mode (Blender 5.0+)
    _ensure_extensions_loaded()

    # Priority order for checking:
    # 1. Official Blender repository (most common in Blender 5.0+)
    if 'bl_ext.blender_org.mpfb' in sys.modules:
        return 'bl_ext.blender_org.mpfb'

    # 2. Importing
    try:
        import bl_ext.blender_org.mpfb
        return 'bl_ext.blender_org.mpfb'
    except ImportError:
        pass


def load_json_config(json_path: str) -> Dict[str, Any]:
    """
    Load and parse JSON configuration file.
    
    Args:
        json_path: Path to JSON configuration file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Configuration file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    print(f"✓ Loaded configuration from: {json_path}")
    return config


def validate_macro_value(value: float, param_name: str) -> float:
    """
    Validate that a macro parameter value is within valid range [0.0, 1.0].
    
    Args:
        value: Parameter value to validate
        param_name: Name of parameter (for error messages)
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value is outside valid range
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{param_name} must be a number, got {type(value)}")
    
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{param_name} must be between 0.0 and 1.0, got {value}")
    
    return float(value)


def validate_race_values(race_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Validate race parameter dictionary.
    
    Args:
        race_dict: Dictionary with race values
        
    Returns:
        Validated race dictionary
        
    Raises:
        ValueError: If race values are invalid
    """
    required_keys = {"asian", "caucasian", "african"}
    
    if not isinstance(race_dict, dict):
        raise ValueError("Race must be a dictionary")
    
    missing_keys = required_keys - set(race_dict.keys())
    if missing_keys:
        raise ValueError(f"Race dictionary missing keys: {missing_keys}")
    
    # Validate each race value
    validated = {}
    for key in required_keys:
        validated[key] = validate_macro_value(race_dict[key], f"race.{key}")
    
    # Check that values sum to approximately 1.0 (allow some tolerance)
    total = sum(validated.values())
    if abs(total - 1.0) > 0.01:
        print(f"⚠ Warning: Race values sum to {total:.3f} (expected 1.0)")
    
    return validated


def validate_macro_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate macro settings from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated macro settings dictionary
        
    Raises:
        ValueError: If required settings are missing or invalid
    """
    if "macro_settings" not in config:
        raise ValueError("Configuration must contain 'macro_settings' key")
    
    macro = config["macro_settings"]
    
    # Define expected parameters and their defaults
    expected_params = {
        "gender": 0.5,
        "age": 0.5,
        "muscle": 0.5,
        "weight": 0.5,
        "proportions": 0.5,
        "height": 0.5,
        "cupsize": 0.5,
        "firmness": 0.5
    }
    
    validated = {}
    
    # Validate each parameter
    for param, default_value in expected_params.items():
        if param in macro:
            validated[param] = validate_macro_value(macro[param], param)
        else:
            validated[param] = default_value
            print(f"⚠ Using default value for {param}: {default_value}")
    
    # Validate race separately
    if "race" in macro:
        validated["race"] = validate_race_values(macro["race"])
    else:
        validated["race"] = {"asian": 0.33, "caucasian": 0.33, "african": 0.34}
        print("⚠ Using default race values")
    
    return validated


def get_output_path(config: Dict[str, Any], default_name: str = "human.fbx") -> str:
    """
    Get output file path from configuration.
    
    Args:
        config: Configuration dictionary
        default_name: Default filename if not specified
        
    Returns:
        Absolute path for output file
    """
    if "output" not in config:
        raise ValueError("Configuration must contain 'output' key")
    
    output_config = config["output"]
    
    # Get directory
    if "directory" not in output_config:
        raise ValueError("Output configuration must contain 'directory' key")
    
    output_dir = output_config["directory"]
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename
    filename = output_config.get("filename", default_name)
    
    # Ensure .fbx extension
    if not filename.lower().endswith(".fbx"):
        filename += ".fbx"
    
    output_path = os.path.abspath(os.path.join(output_dir, filename))
    
    print(f"✓ Output will be saved to: {output_path}")
    return output_path


def print_configuration_summary(validated_macro: Dict[str, Any], output_path: str):
    """
    Print a summary of the configuration to be used.
    
    Args:
        validated_macro: Validated macro settings
        output_path: Output file path
    """
    print("\n" + "="*70)
    print("HUMAN GENERATION CONFIGURATION")
    print("="*70)
    print("\nBody Parameters:")
    print(f"  Gender:      {validated_macro['gender']:.3f} ({'female' if validated_macro['gender'] < 0.5 else 'male'})")
    print(f"  Age:         {validated_macro['age']:.3f}")
    print(f"  Muscle:      {validated_macro['muscle']:.3f}")
    print(f"  Weight:      {validated_macro['weight']:.3f}")
    print(f"  Height:      {validated_macro['height']:.3f}")
    print(f"  Proportions: {validated_macro['proportions']:.3f}")
    print(f"  Cup Size:    {validated_macro['cupsize']:.3f}")
    print(f"  Firmness:    {validated_macro['firmness']:.3f}")
    
    print("\nRace:")
    for race_type, value in validated_macro['race'].items():
        print(f"  {race_type.capitalize():12} {value:.3f}")
    
    print(f"\nOutput: {output_path}")
    print("="*70 + "\n")


def setup_blender_scene():
    """
    Set up Blender scene for human generation.
    Removes default objects and prepares clean scene.
    """
    import bpy
    
    # Delete default objects (cube, light, camera)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clear any existing collections
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)
    
    print("✓ Blender scene prepared")


def check_mpfb2_installed() -> bool:
    """
    Check if MPFB2 addon is installed and enabled in Blender.

    Returns:
        True if MPFB2 is available, False otherwise
    """
    try:
    

        # Ensure extensions are loaded in headless mode (Blender 5.0+)
        _ensure_extensions_loaded()

        # Check for MPFB2 in different possible locations
        # 1. Official Blender repository (most common in Blender 5.0+)
        if 'bl_ext.blender_org.mpfb' in sys.modules:
            print("✓ MPFB2 extension detected (Official Blender repository)")
            return True

        # Try importing in order of preference
        # Try official repository first
        try:
            import bl_ext.blender_org.mpfb
            print("✓ MPFB2 extension imported successfully (Official repository)")
            return True
        except ImportError:
            pass

        # Nothing worked
        raise ImportError("MPFB2 not found in any location")

    except ImportError as e:
        print(f"✗ ERROR: MPFB2 addon not found! ({e})")
        print("\nPlease install MPFB2:")
        print("1. Open Blender normally")
        print("2. Go to Edit → Preferences → Get Extensions")
        print("3. Search for 'MPFB' and click Install")
        print("4. Restart Blender")
        return False


def apply_macro_settings_to_human(basemesh, macro_settings: Dict[str, Any], bake: bool = True):
    """
    Apply macro settings to the human basemesh.

    Args:
        basemesh: Blender object representing the human basemesh
        macro_settings: Dictionary of validated macro settings
        bake: Whether to bake targets after applying (default True).
              Set to False if microparameters will be applied afterward.
    """
    import bpy

    mpfb_path = _get_mpfb_module_path()
    HumanObjectProperties = importlib.import_module(f'{mpfb_path}.entities.objectproperties').HumanObjectProperties
    TargetService = importlib.import_module(f'{mpfb_path}.services.targetservice').TargetService

    print("\nApplying macro settings to human mesh...")

    # Apply each macro parameter
    for param, value in macro_settings.items():
        if param == "race":
            # Handle race separately
            for race_type, race_value in value.items():
                HumanObjectProperties.set_value(race_type, race_value, entity_reference=basemesh)
                print(f"  {race_type}: {race_value:.3f}")
        else:
            HumanObjectProperties.set_value(param, value, entity_reference=basemesh)
            print(f"  {param}: {value:.3f}")

    # CRITICAL: Force scene update to ensure macro properties are recognized
    print("\nUpdating scene...")
    bpy.context.view_layer.update()

    # Apply macro details first (this loads the target shape keys)
    print("Loading macro detail targets...")
    try:
        TargetService.reapply_macro_details(basemesh)
    except Exception as e:
        print(f"  Warning: Could not reapply macro details: {e}")

    # Optionally bake targets (skip if microparameters will be applied next)
    if bake:
        print("Baking all targets to mesh...")
        TargetService.bake_targets(basemesh)

        # Final scene update
        bpy.context.view_layer.update()
        basemesh.data.update()

    print("✓ Macro settings applied successfully")


def apply_microparameters_to_human(basemesh, micro_settings: Dict[str, float], bake: bool = True, verbose: bool = False):
    """
    Apply microparameters (MPFB2 targets) to the human basemesh for fine-tuning.

    This should be called AFTER apply_macro_settings_to_human with bake=False.

    Args:
        basemesh: Blender object representing the human basemesh
        micro_settings: Dictionary mapping microparameter names to values (0.0-1.0)
        bake: Whether to bake all targets (macros + micros) after applying (default True)
        verbose: Whether to print detailed progress information

    Example:
        >>> micro_settings = {
        ...     'measure-shoulder-dist-incr': 0.8,
        ...     'measure-upperarm-length-incr': 0.6
        ... }
        >>> apply_microparameters_to_human(basemesh, micro_settings)
    """
    if not micro_settings:
        if verbose:
            print("No microparameters to apply")
        return

    import bpy

    mpfb_path = _get_mpfb_module_path()
    TargetService = importlib.import_module(f'{mpfb_path}.services.targetservice').TargetService

    if verbose:
        print(f"\nApplying {len(micro_settings)} microparameters...")

    # Load each microparameter target
    for micro_name, micro_value in micro_settings.items():
        try:
            # Get full path for the target
            full_path = TargetService.target_full_path(micro_name)

            if full_path is None:
                print(f"  ✗ Warning: Could not find target '{micro_name}'")
                continue

            # Load target with specified value
            TargetService.load_target(basemesh, full_path, weight=micro_value, name=micro_name)

            if verbose:
                print(f"  ✓ {micro_name}: {micro_value:.3f}")

        except Exception as e:
            print(f"  ✗ Error loading '{micro_name}': {e}")

    # Bake all targets (macros + micros) to apply them permanently
    if bake:
        print("Baking all targets (macros + microparameters) to mesh...")
        TargetService.bake_targets(basemesh)

        # Final scene update
        bpy.context.view_layer.update()
        basemesh.data.update()

        if verbose:
            print("  ✓ All targets baked to mesh")


def add_standard_rig(basemesh, rig_type: str = "default") -> Tuple[Any, Any]:
    """
    Add standard rig to the human mesh using the new MPFB API.

    Args:
        basemesh: Human basemesh object
        rig_type: Type of rig to add ("default", "default_no_toes", "game_engine")

    Returns:
        Tuple of (armature_object, basemesh)
    """

    mpfb_path = _get_mpfb_module_path()
    HumanService = importlib.import_module(f'{mpfb_path}.services.humanservice').HumanService

    print(f"\nAdding {rig_type} rig to human...")

    # Use the new add_builtin_rig method
    # rig_name should be just the name like "default", "default_no_toes", "game_engine"
    armature = HumanService.add_builtin_rig(
        basemesh,
        rig_name=rig_type,
        import_weights=True
    )

    if armature:
        print(f"✓ {rig_type} rig added successfully")
        print(f"  Armature name: {armature.name}")
        return armature, basemesh
    else:
        raise RuntimeError(f"Failed to add {rig_type} rig")


def set_tpose(armature, rig_type: str = "default") -> bool:
    """
    Set the armature to T-pose for accurate measurements.

    Uses MPFB2's built-in T-pose presets when available, with a fallback
    to clearing pose transforms (rest pose).

    Args:
        armature: Blender armature object
        rig_type: Type of rig ("default", "default_no_toes", "game_engine")

    Returns:
        True if T-pose was successfully applied, False otherwise
    """
    import bpy
    from pathlib import Path

    mpfb_path = _get_mpfb_module_path()
    RigService = importlib.import_module(f'{mpfb_path}.services.rigservice').RigService
    LocationService = importlib.import_module(f'{mpfb_path}.services.locationservice').LocationService

    print(f"\nSetting T-pose for {rig_type} rig...")

    # Ensure global poses are available (copies from source if needed)
    try:
        RigService.ensure_global_poses_are_available()
    except Exception as e:
        print(f"  Warning: Could not ensure poses are available: {e}")

    # Try to find T-pose preset file
    # MPFB2 stores poses in subdirectories by rig type
    tpose_found = False
    pose_dict = None

    # Check both MPFB data directory and user data directory for T-pose
    pose_directories = []
    try:
        mpfb_poses_dir = LocationService.get_mpfb_data("poses")
        if mpfb_poses_dir:
            pose_directories.append(Path(mpfb_poses_dir))
    except Exception:
        pass

    try:
        user_poses_dir = LocationService.get_user_data("poses")
        if user_poses_dir:
            pose_directories.append(Path(user_poses_dir))
    except Exception:
        pass

    # T-pose file naming patterns to search for
    # MPFB2 stores poses in subdirectories named {rig_type}_fk/
    # e.g., default_fk/t-pose.json, game_engine_fk/t-pose.json
    tpose_patterns = [
        # Primary pattern: {rig_type}_fk/t-pose.json
        f"{rig_type}_fk/t-pose.json",
        # Fallback for "default_no_toes" -> use "default_fk"
        "default_fk/t-pose.json",
        # Additional patterns for compatibility
        f"{rig_type}/t-pose.json",
        "t-pose.json",
    ]

    for poses_dir in pose_directories:
        if not poses_dir.exists():
            continue

        for pattern in tpose_patterns:
            tpose_path = poses_dir / pattern
            if tpose_path.exists():
                try:
                    with open(tpose_path, 'r') as f:
                        pose_dict = json.load(f)
                    print(f"  Found T-pose preset: {tpose_path}")
                    tpose_found = True
                    break
                except Exception as e:
                    print(f"  Warning: Could not load {tpose_path}: {e}")

        if tpose_found:
            break

    # Apply T-pose if found
    if pose_dict is not None:
        try:
            # RigService.set_pose_from_dict requires correct context (pose mode)
            original_active = bpy.context.view_layer.objects.active
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')

            RigService.set_pose_from_dict(armature, pose_dict, from_rest_pose=True)

            # Return to object mode
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active = original_active

            print("  T-pose applied successfully using MPFB2 preset")

            # Update scene to reflect pose changes
            bpy.context.view_layer.update()
            return True
        except Exception as e:
            print(f"  Warning: Could not apply T-pose preset: {e}")
            # Try to return to object mode if we failed mid-way
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except:
                pass

    # Fallback: Clear all pose transforms (returns to rest pose)
    # This may be A-pose depending on the rig, but it's better than nothing
    print("  T-pose preset not found, falling back to rest pose...")

    try:
        # Store current active object
        original_active = bpy.context.view_layer.objects.active

        # Set armature as active and switch to pose mode
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')

        # Select all bones and clear transforms
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()

        # Return to original mode
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = original_active

        # Update scene
        bpy.context.view_layer.update()

        print("  Rest pose applied (transforms cleared)")
        return True

    except Exception as e:
        print(f"  Error setting rest pose: {e}")
        return False


def configure_fk_ik_hybrid_rig(armature, instrumented_arm: str = "left"):
    """
    Configure FK/IK hybrid rigging system for IMU sensor-based motion capture.

    This function tags bones with custom properties to identify their control type
    for use in game engines (Unity, Unreal, etc.). No Blender constraints are added,
    ensuring clean export without pose deformation.

    Bone tagging scheme:
    - FK (Forward Kinematics): Bones controlled directly by IMU sensors
      * spine01, spine02 (chest sensor)
      * instrumented arm: upperarm01, lowerarm01, wrist (arm sensors)

    - IK (Inverse Kinematics): Bones with positional constraints
      * foot.L, foot.R (ground contact)
      * root (pelvis anchor)

    - Copy/Mirror: Bones that derive motion from other bones
      * non-instrumented arm (mirrors instrumented arm)
      * head, neck01 (follows chest)
      * spine03 (interpolates between root and chest)

    - Anchored: Bones fixed relative to root/pelvis
      * spine04, spine05

    Args:
        armature: Blender armature object with standard MPFB2 rig
        instrumented_arm: Which arm has sensors ("left" or "right")

    Raises:
        ValueError: If instrumented_arm is not "left" or "right"
        RuntimeError: If required bones are not found in the armature
    """
    import bpy

    if instrumented_arm not in ["left", "right"]:
        raise ValueError(f"instrumented_arm must be 'left' or 'right', got '{instrumented_arm}'")

    print(f"\n" + "="*70)
    print("CONFIGURING FK/IK HYBRID RIG (TAGGING ONLY)")
    print("="*70)
    print(f"\nInstrumented arm: {instrumented_arm.upper()}")
    print("Mode: Bone tagging without constraints (clean export)")

    # Switch to pose mode to access bones
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    pose_bones = armature.pose.bones

    # Helper function to check if bone exists
    def get_bone(bone_name: str):
        if bone_name not in pose_bones:
            raise RuntimeError(f"Required bone '{bone_name}' not found in armature")
        return pose_bones[bone_name]

    print("\n" + "-"*70)
    print("STEP 1: Tagging FK Bones (Direct Sensor Control)")
    print("-"*70)

    # Configure FK for chest (spine1 and spine2)
    print("\nConfiguring chest FK:")
    try:
        spine1 = get_bone("spine01")
        spine2 = get_bone("spine02")

        # Tag these bones for FK control (no constraints needed for FK, just marking)
        spine1["fk_controlled"] = True
        spine1["sensor_target"] = "chest"
        spine2["fk_controlled"] = True
        spine2["sensor_target"] = "chest"

        print(f"  ✓ spine01: Tagged for chest sensor FK control")
        print(f"  ✓ spine02: Tagged for chest sensor FK control")
    except RuntimeError as e:
        print(f"  ⚠ Warning: {e}")

    # Configure FK for instrumented arm
    print(f"\nConfiguring {instrumented_arm} arm FK:")
    arm_suffix = ".L" if instrumented_arm == "left" else ".R"

    try:
        upperarm = get_bone(f"upperarm01{arm_suffix}")
        lowerarm = get_bone(f"lowerarm01{arm_suffix}")
        wrist = get_bone(f"wrist{arm_suffix}")

        # Tag arm bones for FK control
        upperarm["fk_controlled"] = True
        upperarm["sensor_target"] = "upper_arm"
        lowerarm["fk_controlled"] = True
        lowerarm["sensor_target"] = "forearm"
        wrist["fk_controlled"] = True
        wrist["sensor_target"] = "hand"

        print(f"  ✓ upperarm01{arm_suffix}: Tagged for upper arm sensor FK control")
        print(f"  ✓ lowerarm01{arm_suffix}: Tagged for forearm sensor FK control")
        print(f"  ✓ wrist{arm_suffix}: Tagged for hand sensor FK control")
    except RuntimeError as e:
        print(f"  ⚠ Warning: {e}")

    print("\n" + "-"*70)
    print("STEP 2: Tagging IK Bones (Positional Constraints)")
    print("-"*70)

    # Tag feet for IK ground contact
    print("\nTagging foot bones for IK:")
    for side, suffix in [("left", ".L"), ("right", ".R")]:
        try:
            foot = get_bone(f"foot{suffix}")

            # Tag foot as IK controlled
            foot["ik_controlled"] = True
            foot["ik_purpose"] = "ground_constraint"
            foot["ik_pole_target"] = f"toe3-1{suffix}"  # Suggested pole target for game engine

            print(f"  ✓ {side} foot: Tagged for IK ground constraint")
        except RuntimeError as e:
            print(f"  ⚠ Warning: Could not tag {side} foot - {e}")

    # Tag root (pelvis) as anchor
    print("\nTagging root/pelvis anchor:")
    try:
        root = get_bone("root")

        root["ik_controlled"] = True
        root["ik_purpose"] = "anchor"

        print("  ✓ root: Tagged as pelvis anchor")
    except RuntimeError as e:
        print(f"  ⚠ Warning: {e}")

    print("\n" + "-"*70)
    print("STEP 3: Tagging Copy/Mirror Bones (Derived Movement)")
    print("-"*70)

    # Tag non-instrumented arm for mirroring
    print(f"\nTagging {'right' if instrumented_arm == 'left' else 'left'} arm for mirroring:")
    mirror_suffix = ".R" if instrumented_arm == "left" else ".L"
    source_suffix = ".L" if instrumented_arm == "left" else ".R"

    for bone_type in ["upperarm01", "lowerarm01", "wrist"]:
        try:
            mirror_bone = get_bone(f"{bone_type}{mirror_suffix}")
            source_bone_name = f"{bone_type}{source_suffix}"

            mirror_bone["copy_controlled"] = True
            mirror_bone["copy_source"] = source_bone_name
            mirror_bone["copy_type"] = "mirror"

            print(f"  ✓ {bone_type}{mirror_suffix}: Tagged to mirror {source_bone_name}")
        except RuntimeError as e:
            print(f"  ⚠ Warning: {e}")

    # Tag head/neck to follow chest
    print("\nTagging head/neck to follow chest:")
    try:
        neck01 = get_bone("neck01")
        head = get_bone("head")

        for bone, bone_name in [(neck01, "neck01"), (head, "head")]:
            bone["copy_controlled"] = True
            bone["copy_source"] = "spine02"
            bone["copy_type"] = "follow"
            bone["copy_influence"] = 0.5  # Suggested influence for natural movement

            print(f"  ✓ {bone_name}: Tagged to follow chest (spine02)")
    except RuntimeError as e:
        print(f"  ⚠ Warning: {e}")

    # Tag spine03 for interpolation
    print("\nTagging spine for interpolation:")
    try:
        spine03 = get_bone("spine03")

        # Determine lower spine source
        lower_spine_source = "root"
        try:
            get_bone("spine04")
            lower_spine_source = "spine04"
        except RuntimeError:
            try:
                get_bone("spine05")
                lower_spine_source = "spine05"
            except RuntimeError:
                pass

        spine03["interpolated"] = True
        spine03["interpolation_source_lower"] = lower_spine_source
        spine03["interpolation_source_upper"] = "spine02"
        spine03["interpolation_influence"] = 0.5  # 50/50 blend

        print(f"  ✓ spine03: Tagged to interpolate between {lower_spine_source} and spine02")
    except RuntimeError as e:
        print(f"  ⚠ Warning: {e}")

    # Tag spine04 and spine05 as anchored to root/pelvis
    print("\nTagging lower spine anchoring:")
    for spine_num in ["04", "05"]:
        try:
            spine = get_bone(f"spine{spine_num}")
            spine["anchored_to"] = "root"
            print(f"  ✓ spine{spine_num}: Tagged as anchored to root/pelvis")
        except RuntimeError:
            print(f"  ⚠ Warning: spine{spine_num} not found")

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    print("\n" + "="*70)
    print("✓ FK/IK HYBRID RIG TAGGING COMPLETE")
    print("="*70)
    print("\nBone Tagging Summary:")
    print("  FK Controlled: chest (spine01, spine02), instrumented arm")
    print("  IK Controlled: both feet (ground), root (anchor)")
    print("  Mirrored: non-instrumented arm")
    print("  Tracked: head, neck01 (follows chest)")
    print("  Interpolated: spine03 (between root/pelvis and chest)")
    print("  Anchored: spine04, spine05 (to root/pelvis)")
    print("\nNOTE: No Blender constraints added - only custom property tags")
    print("      Use tags in game engine to apply IMU data and IK")
    print("="*70 + "\n")


def export_fbx(basemesh, armature, output_path: str, export_settings: Dict[str, Any] = None, additional_objects: list = None):
    """
    Export the human mesh and rig to FBX format.

    Args:
        basemesh: Human mesh object
        armature: Armature/rig object
        output_path: Path to save FBX file
        export_settings: Optional dictionary of export settings
        additional_objects: Optional list of additional objects to export (e.g., hair, clothes)
    """
    import bpy

    print("\nPreparing for FBX export...")

    # Apply all transforms to ensure consistent export
    # This ensures the armature and mesh have identity transforms
    print("Applying transforms...")
    bpy.ops.object.select_all(action='DESELECT')

    if armature:
        arm_has_anim = (
            armature.animation_data is not None
            and armature.animation_data.action is not None
        )
        if arm_has_anim:
            # Skip transform_apply on animated armatures. Calling it after
            # animation is assigned disrupts the depsgraph evaluation in
            # Blender 5.0's slot-based action system, causing the FBX bake to
            # read the rest pose at every frame (completely static export).
            # The MPFB2 Mixamo rig already has identity transforms so this is safe.
            print("  Skipping transform_apply on armature (has animation data)")
        else:
            armature.select_set(True)
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    basemesh.select_set(True)
    bpy.context.view_layer.objects.active = basemesh
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Apply transforms to additional objects (hair, clothes, etc.)
    if additional_objects:
        for obj in additional_objects:
            if obj:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Select all objects for export
    bpy.ops.object.select_all(action='DESELECT')
    basemesh.select_set(True)
    if armature:
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature  # Set armature as active

    # Select additional objects
    if additional_objects:
        for obj in additional_objects:
            if obj:
                obj.select_set(True)
                print(f"  Including additional object: {obj.name}")

    # Default export settings optimized for rigged characters
    default_settings = {
        'use_selection': True,
        'use_active_collection': False,
        'global_scale': 1.0,
        'apply_unit_scale': True,
        'apply_scale_options': 'FBX_SCALE_NONE',
        'use_space_transform': True,
        'bake_space_transform': True,  # Changed to True to bake transforms into mesh
        'object_types': {'ARMATURE', 'MESH'},
        'use_mesh_modifiers': True,
        'use_mesh_modifiers_render': True,
        'mesh_smooth_type': 'OFF',
        'use_subsurf': False,
        'use_mesh_edges': False,
        'use_tspace': False,
        'use_custom_props': True,  # Changed to True to preserve FK/IK tags
        'add_leaf_bones': False,
        'primary_bone_axis': 'Y',
        'secondary_bone_axis': 'X',
        'armature_nodetype': 'NULL',
        'bake_anim': False,
        'bake_anim_use_all_bones': True,
        'bake_anim_use_nla_strips': True,
        'bake_anim_use_all_actions': True,
        'bake_anim_force_startend_keying': True,
        'bake_anim_step': 1.0,
        'bake_anim_simplify_factor': 1.0,
        'path_mode': 'COPY',
        'embed_textures': True,  # Embed textures as binary data in the FBX file
        'batch_mode': 'OFF',
        'use_batch_own_dir': True,
        'axis_forward': '-Z',
        'axis_up': 'Y'
    }

    # Merge with user settings if provided
    if export_settings:
        default_settings.update(export_settings)

    print(f"Exporting to: {output_path}")
    print(f"Export settings:")
    print(f"  - Scale: {default_settings['global_scale']}")
    print(f"  - Axis: Forward={default_settings['axis_forward']}, Up={default_settings['axis_up']}")
    print(f"  - Bone axis: Primary={default_settings['primary_bone_axis']}, Secondary={default_settings['secondary_bone_axis']}")
    print(f"  - Animation baking: {'ENABLED' if default_settings['bake_anim'] else 'DISABLED'}")

    # Confirm animation is still assigned immediately before export.
    # transform_apply or any other intervening op could silently clear it.
    if default_settings['bake_anim'] and armature:
        action = (armature.animation_data.action
                  if armature.animation_data else None)
        if action:
            fr = action.frame_range
            print(f"  - Action before export  : {action.name}  "
                  f"frames {int(fr[0])}-{int(fr[1])}")
            slot = (armature.animation_data.action_slot
                    if hasattr(armature.animation_data, 'action_slot') else None)
            print(f"  - Slot before export    : "
                  f"{slot.name_display if slot else 'none'}")
        else:
            print("  WARNING: bake_anim is True but armature has no action assigned!")

    # Export FBX
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        **default_settings
    )

    print(f"✓ Successfully exported to: {output_path}")

    # Check if custom properties were exported
    if armature and default_settings.get('use_custom_props', False):
        print("\nNote: FK/IK bone tags exported as custom properties")
        print("  Read these properties in your game engine to identify bone roles")


def validate_json_structure(config: Dict[str, Any]) -> bool:
    """
    Validate that JSON has required structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If structure is invalid
    """
    required_keys = {"macro_settings", "output"}
    
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Configuration missing required keys: {missing_keys}")
    
    return True


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object that may contain numpy types

    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj