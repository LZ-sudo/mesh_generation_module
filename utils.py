"""
Utility functions for MPFB2 human generation.

This module provides helper functions for:
- Loading and validating JSON configuration files
- Validating macro parameters
- Setting up Blender scene
- Applying settings to human mesh
"""

import json
import os
import sys
from typing import Dict, Any, Tuple

# NOTE: bpy and mpfb imports are done inside functions to avoid import errors
# when this module is imported outside of Blender environment


def _get_mpfb_module_path():
    """
    Determine the correct MPFB module path for the current Blender version.

    Returns:
        str: Either 'bl_ext.user_default.mpfb' for Blender 4.2+ or 'mpfb' for older versions
    """
    import sys

    # Check if using new extension system (Blender 4.2+)
    if 'bl_ext.user_default.mpfb' in sys.modules:
        return 'bl_ext.user_default.mpfb'

    # Check if using legacy addon system
    if 'mpfb' in sys.modules:
        return 'mpfb'

    # Try importing to detect which format is available
    try:
        import bl_ext.user_default.mpfb
        return 'bl_ext.user_default.mpfb'
    except ImportError:
        # Fall back to legacy format
        return 'mpfb'


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
        import sys

        # In Blender 4.2+, extensions are loaded as bl_ext.user_default.mpfb
        # In older versions, it might be just mpfb
        if 'bl_ext.user_default.mpfb' in sys.modules:
            print("✓ MPFB2 extension detected (Blender 4.2+ format)")
            return True
        elif 'mpfb' in sys.modules:
            print("✓ MPFB2 addon detected (legacy format)")
            return True

        # Try importing the extension format (Blender 4.2+)
        try:
            import bl_ext.user_default.mpfb
            print("✓ MPFB2 extension imported successfully")
            return True
        except ImportError:
            pass

        # Try legacy format
        import mpfb
        print("✓ MPFB2 addon imported successfully")
        return True

    except ImportError as e:
        print(f"✗ ERROR: MPFB2 addon not found! ({e})")
        print("\nPlease install MPFB2:")
        print("1. Open Blender normally")
        print("2. Go to Edit → Preferences → Extensions")
        print("3. Search for 'MPFB' and click Install")
        print("4. Restart Blender")
        return False


def apply_macro_settings_to_human(basemesh, macro_settings: Dict[str, Any]):
    """
    Apply macro settings to the human basemesh.

    Args:
        basemesh: Blender object representing the human basemesh
        macro_settings: Dictionary of validated macro settings
    """
    import importlib
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

    # Now bake all targets to permanently apply them to the mesh
    print("Baking all targets to mesh...")
    TargetService.bake_targets(basemesh)

    # Final scene update
    bpy.context.view_layer.update()
    basemesh.data.update()

    print("✓ Macro settings applied successfully")


def add_standard_rig(basemesh, rig_type: str = "default") -> Tuple[Any, Any]:
    """
    Add standard rig to the human mesh using the new MPFB API.

    Args:
        basemesh: Human basemesh object
        rig_type: Type of rig to add ("default", "default_no_toes", "game_engine")

    Returns:
        Tuple of (armature_object, basemesh)
    """
    import importlib

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


def export_fbx(basemesh, armature, output_path: str, export_settings: Dict[str, Any] = None):
    """
    Export the human mesh and rig to FBX format.

    Args:
        basemesh: Human mesh object
        armature: Armature/rig object
        output_path: Path to save FBX file
        export_settings: Optional dictionary of export settings
    """
    import bpy

    print("\nPreparing for FBX export...")

    # Apply all transforms to ensure consistent export
    # This ensures the armature and mesh have identity transforms
    print("Applying transforms...")
    bpy.ops.object.select_all(action='DESELECT')

    if armature:
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    basemesh.select_set(True)
    bpy.context.view_layer.objects.active = basemesh
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Select both mesh and armature for export
    bpy.ops.object.select_all(action='DESELECT')
    basemesh.select_set(True)
    if armature:
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature  # Set armature as active

    # Default export settings optimized for rigged characters
    default_settings = {
        'use_selection': True,
        'use_active_collection': False,
        'global_scale': 1.0,
        'apply_unit_scale': True,
        'apply_scale_options': 'FBX_SCALE_NONE',
        'use_space_transform': True,
        'bake_space_transform': False,
        'object_types': {'ARMATURE', 'MESH'},
        'use_mesh_modifiers': True,
        'use_mesh_modifiers_render': True,
        'mesh_smooth_type': 'OFF',
        'use_subsurf': False,
        'use_mesh_edges': False,
        'use_tspace': False,
        'use_custom_props': False,
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
        'path_mode': 'AUTO',
        'embed_textures': False,
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

    # Export FBX
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        **default_settings
    )

    print(f"✓ Successfully exported to: {output_path}")


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
