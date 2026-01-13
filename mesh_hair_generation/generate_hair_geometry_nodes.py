# #!/usr/bin/env python3
# """
# Hair Generation using Geometry Nodes in Blender 5.0+

# This script generates hair on a mesh using available Geometry Nodes:
# 1. Distribute points on scalp surface
# 2. Create curves at each point
# 3. Style with radius, direction, and deformation
# 4. Convert to mesh for FBX export

# Run with:
#     python run_blender.py --script mesh_hair_generation/generate_hair_geometry_nodes.py
# """

# import sys
# from pathlib import Path

# # Add parent directory to path
# script_dir = Path(__file__).parent.absolute()
# parent_dir = script_dir.parent.absolute()
# if str(parent_dir) not in sys.path:
#     sys.path.insert(0, str(parent_dir))

# from utils import _get_mpfb_module_path
# import importlib


# def create_scalp_vertex_group(mesh_obj, threshold_percentage=0.40):
#     """
#     Create a vertex group for the scalp (top portion of head).

#     Args:
#         mesh_obj: The mesh object (human head)
#         threshold_percentage: Percentage of head height to use (0.40 = top 40%)

#     Returns:
#         str: Name of the created vertex group
#     """
#     import bpy

#     mesh = mesh_obj.data

#     # Get armature
#     armature = None
#     for modifier in mesh_obj.modifiers:
#         if modifier.type == 'ARMATURE':
#             armature = modifier.object
#             break

#     if not armature:
#         print("⚠ Warning: No armature found, using mesh bounds for scalp detection")
#         # Fallback to mesh bounds
#         vertices = [mesh_obj.matrix_world @ v.co for v in mesh.vertices]
#         max_z = max(v.z for v in vertices)
#         head_bone_z = max_z * 0.85  # Estimate
#     else:
#         # Get head bone position
#         head_bone = armature.data.bones.get("head")
#         if head_bone:
#             head_bone_z = (armature.matrix_world @ head_bone.head_local).z
#         else:
#             print("⚠ Warning: No 'head' bone found")
#             vertices = [mesh_obj.matrix_world @ v.co for v in mesh.vertices]
#             max_z = max(v.z for v in vertices)
#             head_bone_z = max_z * 0.85

#     # Calculate scalp region
#     vertices = [mesh_obj.matrix_world @ v.co for v in mesh.vertices]
#     max_z = max(v.z for v in vertices)
#     scalp_range = max_z - head_bone_z
#     threshold = head_bone_z + (scalp_range * threshold_percentage)

#     # Create vertex group
#     vgroup_name = "Scalp"
#     if vgroup_name in mesh_obj.vertex_groups:
#         vgroup = mesh_obj.vertex_groups[vgroup_name]
#     else:
#         vgroup = mesh_obj.vertex_groups.new(name=vgroup_name)

#     # Assign vertices
#     for vert in mesh.vertices:
#         world_pos = mesh_obj.matrix_world @ vert.co
#         if world_pos.z >= threshold:
#             vgroup.add([vert.index], 1.0, 'REPLACE')

#     print(f"✓ Created scalp vertex group: {vgroup_name}")
#     print(f"  Threshold height: {threshold:.3f} m")

#     return vgroup_name


# def create_hair_geometry_nodes(mesh_obj, scalp_vgroup_name, hair_params):
#     """
#     Create hair using Geometry Nodes.

#     Args:
#         mesh_obj: The mesh object
#         scalp_vgroup_name: Name of the scalp vertex group
#         hair_params: Dictionary with hair parameters:
#             - density: Number of hair strands
#             - length: Hair length in meters
#             - radius: Hair strand thickness
#             - hair_type: 'straight', 'wavy', or 'curly'
#     """
#     import bpy

#     print(f"\nCreating hair with Geometry Nodes...")
#     print(f"  Hair type: {hair_params.get('hair_type', 'straight')}")
#     print(f"  Density: {hair_params.get('density', 1000)} strands")
#     print(f"  Length: {hair_params.get('length', 0.15)} m")

#     # Add Geometry Nodes modifier
#     modifier = mesh_obj.modifiers.new("HairGeneration", 'NODES')

#     # Create node tree
#     node_tree = bpy.data.node_groups.new("HairGeneration", 'GeometryNodeTree')
#     modifier.node_group = node_tree

#     # CRITICAL: In Blender 5.0+, add interface sockets BEFORE creating nodes
#     node_tree.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
#     node_tree.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

#     nodes = node_tree.nodes
#     links = node_tree.links
#     nodes.clear()

#     # Create nodes
#     x_offset = 0
#     y_pos = 0

#     # Input node
#     input_node = nodes.new('NodeGroupInput')
#     input_node.location = (x_offset, y_pos)
#     x_offset += 250

#     # Named attribute node to get scalp group
#     scalp_attr = nodes.new('GeometryNodeInputNamedAttribute')
#     scalp_attr.location = (x_offset, y_pos - 200)
#     scalp_attr.data_type = 'FLOAT'
#     scalp_attr.inputs['Name'].default_value = scalp_vgroup_name

#     # Distribute points on faces (scalp only)
#     distribute_points = nodes.new('GeometryNodeDistributePointsOnFaces')
#     distribute_points.location = (x_offset, y_pos)
#     distribute_points.distribute_method = 'POISSON'
#     # For POISSON distribution, use 'Density Max' instead of 'Density'
#     distribute_points.inputs['Density Max'].default_value = hair_params.get('density', 1000)
#     # Set minimum distance between hair strands (prevents clumping)
#     distribute_points.inputs['Distance Min'].default_value = 0.001
#     x_offset += 250

#     # Link scalp selection
#     # Use index-based socket access for Blender 5.0+ compatibility
#     links.new(input_node.outputs[0], distribute_points.inputs['Mesh'])  # outputs[0] = Geometry
#     links.new(scalp_attr.outputs['Attribute'], distribute_points.inputs['Selection'])

#     # Get normal at each point
#     normal_node = nodes.new('GeometryNodeInputNormal')
#     normal_node.location = (x_offset, y_pos - 300)

#     # Create curves from points
#     # We'll use instance on points with a line curve

#     # Create a simple line curve primitive
#     curve_line = nodes.new('GeometryNodeCurvePrimitiveLine')
#     curve_line.location = (x_offset, y_pos + 200)
#     curve_line.inputs['End'].default_value = (0, 0, hair_params.get('length', 0.15))
#     x_offset += 250

#     # Instance curves at each point
#     instance_on_points = nodes.new('GeometryNodeInstanceOnPoints')
#     instance_on_points.location = (x_offset, y_pos)
#     links.new(distribute_points.outputs['Points'], instance_on_points.inputs['Points'])
#     links.new(curve_line.outputs['Curve'], instance_on_points.inputs['Instance'])

#     # Align to normal (Blender 5.0 uses FunctionNodeAlignRotationToVector)
#     align_node = nodes.new('FunctionNodeAlignRotationToVector')
#     align_node.location = (x_offset - 50, y_pos - 200)
#     align_node.axis = 'Z'  # Align Z-axis of hair to normal
#     links.new(normal_node.outputs['Normal'], align_node.inputs['Vector'])
#     links.new(align_node.outputs['Rotation'], instance_on_points.inputs['Rotation'])

#     x_offset += 250

#     # Realize instances (convert to actual curves)
#     realize_instances = nodes.new('GeometryNodeRealizeInstances')
#     realize_instances.location = (x_offset, y_pos)
#     links.new(instance_on_points.outputs['Instances'], realize_instances.inputs['Geometry'])
#     x_offset += 250

#     # Add hair styling based on type
#     hair_type = hair_params.get('hair_type', 'straight')

#     if hair_type in ['wavy', 'curly']:
#         # Subdivide for smoother curves
#         subdivide = nodes.new('GeometryNodeSubdivideCurve')
#         subdivide.location = (x_offset, y_pos)
#         subdivide.inputs['Cuts'].default_value = 4 if hair_type == 'wavy' else 6
#         links.new(realize_instances.outputs['Geometry'], subdivide.inputs['Curve'])
#         x_offset += 250

#         # Set curve radius
#         set_radius = nodes.new('GeometryNodeSetCurveRadius')
#         set_radius.location = (x_offset, y_pos)
#         set_radius.inputs['Radius'].default_value = hair_params.get('radius', 0.001)
#         links.new(subdivide.outputs['Curve'], set_radius.inputs['Curve'])

#         last_node = set_radius
#     else:
#         # Straight hair - just set radius
#         set_radius = nodes.new('GeometryNodeSetCurveRadius')
#         set_radius.location = (x_offset, y_pos)
#         set_radius.inputs['Radius'].default_value = hair_params.get('radius', 0.001)
#         links.new(realize_instances.outputs['Geometry'], set_radius.inputs['Curve'])

#         last_node = set_radius

#     x_offset += 250

#     # Convert curves to mesh for FBX export
#     curve_to_mesh = nodes.new('GeometryNodeCurveToMesh')
#     curve_to_mesh.location = (x_offset, y_pos)

#     # Create profile curve (circle) for hair thickness
#     profile_circle = nodes.new('GeometryNodeCurvePrimitiveCircle')
#     profile_circle.location = (x_offset, y_pos + 200)
#     profile_circle.inputs['Resolution'].default_value = 4  # Low poly for efficiency
#     profile_circle.inputs['Radius'].default_value = 1.0  # Radius controlled by curve radius

#     links.new(last_node.outputs['Curve'], curve_to_mesh.inputs['Curve'])
#     links.new(profile_circle.outputs['Curve'], curve_to_mesh.inputs['Profile Curve'])
#     x_offset += 250

#     # Join with original mesh
#     join_geometry = nodes.new('GeometryNodeJoinGeometry')
#     join_geometry.location = (x_offset, y_pos)
#     links.new(input_node.outputs[0], join_geometry.inputs['Geometry'])  # outputs[0] = Geometry
#     links.new(curve_to_mesh.outputs['Mesh'], join_geometry.inputs['Geometry'])
#     x_offset += 250

#     # Output node
#     output_node = nodes.new('NodeGroupOutput')
#     output_node.location = (x_offset, y_pos)
#     links.new(join_geometry.outputs['Geometry'], output_node.inputs[0])  # inputs[0] = Geometry

#     print("✓ Hair geometry nodes created successfully")

#     return modifier, node_tree


# def main():
#     import bpy

#     print("\n" + "="*80)
#     print("HAIR GENERATION WITH GEOMETRY NODES - BLENDER 5.0")
#     print("="*80)

#     # Check for MPFB2
#     mpfb_path = _get_mpfb_module_path()
#     HumanService = importlib.import_module(f'{mpfb_path}.services.humanservice').HumanService

#     # Clear scene
#     bpy.ops.object.select_all(action='SELECT')
#     bpy.ops.object.delete()

#     # Create test human
#     print("\nCreating test human mesh...")
#     basemesh = HumanService.create_human()
#     print(f"✓ Created human: {basemesh.name}")

#     # Create scalp vertex group
#     scalp_vgroup = create_scalp_vertex_group(basemesh, threshold_percentage=0.40)

#     # Hair parameters
#     hair_params = {
#         'hair_type': 'straight',  # 'straight', 'wavy', or 'curly'
#         'density': 500,           # Number of hair strands
#         'length': 0.20,           # 20 cm
#         'radius': 0.0008,         # Hair thickness
#     }

#     # Generate hair
#     modifier, node_tree = create_hair_geometry_nodes(basemesh, scalp_vgroup, hair_params)

#     # Save blend file for inspection
#     output_path = Path(__file__).parent / "test_hair_output.blend"
#     bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

#     print("\n" + "="*80)
#     print("HAIR GENERATION COMPLETE!")
#     print("="*80)
#     print(f"\n✓ Hair generated with Geometry Nodes")
#     print(f"  Modifier: {modifier.name}")
#     print(f"  Node tree: {node_tree.name}")
#     print(f"  Saved to: {output_path}")
#     print("\nOpen in Blender GUI to inspect the result!")
#     print("="*80 + "\n")


# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print(f"\n✗ Error: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)
