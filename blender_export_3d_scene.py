import os
import os.path as osp
import argparse
import bpy
from typing import Optional
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.blender import (
    add_light,
    reset_scene,
    build_world,
    flip_normals,
)


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def build_scene(
    import_mesh_path: str,
    scene_name: str = 'Scene',
    material_name: str = "material0",
    use_backface_culling: bool = True,
    flip_normal: bool = True,
    use_sphere: bool = False,
    sphere_radius: float = 1.0,
    scene_scale: float = 5.0,
):
    assert import_mesh_path.endswith('.obj'), "Only .obj mesh format is supported."
    albedo_path = import_mesh_path.replace('.obj', '_albedo.png')
    normal_path = import_mesh_path.replace('.obj', '_normal.png')
    metallic_path = import_mesh_path.replace('.obj', '_metallic.png')
    roughness_path = import_mesh_path.replace('.obj', '_roughness.png')
    alpha_path = import_mesh_path.replace('.obj', '_alpha.png')
    depth_path = import_mesh_path.replace('.obj', '_depth.png')

    if not osp.isfile(albedo_path):
        albedo_path = None
    if not osp.isfile(normal_path):
        normal_path = None
    if not osp.isfile(metallic_path):
        metallic_path = None
    if not osp.isfile(roughness_path):
        roughness_path = None
    alpha_path = None
    depth_path = None


    # 清空场景
    reset_scene()

    # 导入网格
    if use_sphere:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=sphere_radius)
        obj = bpy.context.active_object
        obj.name = scene_name
    else:
        bpy.ops.wm.obj_import(filepath=import_mesh_path)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                obj.select_set(True)
        obj.name = scene_name
        obj = bpy.context.active_object
        # 缩放场景
        obj.scale = (scene_scale, scene_scale, scene_scale)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    obj.data.materials.clear()
    
    # 创建材质
    mat = bpy.data.materials.new(material_name)
    obj.data.materials.append(mat)
    mat.use_nodes = True
    mat.blend_method = 'HASHED'  # [OPAQUE, CLIP, HASHED, BLEND]
    mat.use_backface_culling = use_backface_culling
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # 获取Principled BSDF节点
    bsdf = nodes.get('Principled BSDF')
    assert bsdf, "Principled BSDF node not found in material."
    bsdf.location = (300, 0)

    # 获取材质输出节点
    material_output = nodes.get('Material Output')
    assert material_output, "Material Output node not found in material."
    material_output.location = (600, 0)

    # Albedo
    if albedo_path is not None:
        albedo_tex = nodes.new(type='ShaderNodeTexImage')
        albedo_tex.location = (-300, 0)
        albedo_tex.image = bpy.data.images.load(albedo_path)
        albedo_tex.label = 'Albedo'
        albedo_tex.image.colorspace_settings.name = 'sRGB'
        albedo_tex.image.alpha_mode = 'STRAIGHT'  # [STRAIGHT, PREMUL, CHANNEL_PACKED, NONE]
        links.new(albedo_tex.outputs['Color'], bsdf.inputs['Base Color'])
        # links.new(albedo_tex.outputs['Alpha'], bsdf.inputs['Alpha'])

    # # Alpha
    if alpha_path is not None:
        alpha_tex = nodes.new(type='ShaderNodeTexImage')
        alpha_tex.location = (-300, 300)
        alpha_tex.image = bpy.data.images.load(alpha_path)
        alpha_tex.label = 'Alpha'
        alpha_tex.image.colorspace_settings.name = 'Non-Color'
        links.new(alpha_tex.outputs['Color'], bsdf.inputs['Alpha'])
    
    # Normal
    if normal_path is not None:
        normal_tex = nodes.new(type='ShaderNodeTexImage')
        normal_tex.location = (-300, -300)
        normal_tex.image = bpy.data.images.load(normal_path)
        normal_tex.label = 'Normal'
        normal_tex.image.colorspace_settings.name = 'Non-Color'
        normal_map = nodes.new(type='ShaderNodeNormalMap')
        normal_map.location = (0, -300)
        normal_map.name = 'Normal Map'
        normal_map.space = 'WORLD'
        links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])

    # Metallic
    if metallic_path is not None:
        metallic_tex = nodes.new(type='ShaderNodeTexImage')
        metallic_tex.location = (-300, -600)
        metallic_tex.image = bpy.data.images.load(metallic_path)
        metallic_tex.label = 'Metallic'
        metallic_tex.image.colorspace_settings.name = 'Non-Color'
        links.new(metallic_tex.outputs['Color'], bsdf.inputs['Metallic'])

    # Roughness
    if roughness_path is not None:
        roughness_tex = nodes.new(type='ShaderNodeTexImage')
        roughness_tex.location = (0, -600)
        roughness_tex.image = bpy.data.images.load(roughness_path)
        roughness_tex.label = 'Roughness'
        roughness_tex.image.colorspace_settings.name = 'Non-Color'
        links.new(roughness_tex.outputs['Color'], bsdf.inputs['Roughness'])

    # Depth
    if depth_path is not None:
        depth_tex = nodes.new(type='ShaderNodeTexImage')
        depth_tex.location = (-300, -900)
        depth_tex.image = bpy.data.images.load(depth_path)
        depth_tex.label = 'Distance'
        depth_tex.image.colorspace_settings.name = 'Non-Color'
        bump = nodes.new(type='ShaderNodeBump')
        bump.name = "Bump"
        bump.inputs['Strength'].default_value = 1.0
        links.new(depth_tex.outputs['Color'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    
    # Mapping
    if use_sphere:
        # Texture Coordinate
        tex_coord = nodes.new(type='ShaderNodeTexCoord')

        # Flip U Axis
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.inputs['Scale'].default_value[0] = -1  # 反转U轴（水平翻转）
        links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
        if albedo_path is not None:
            links.new(mapping.outputs['Vector'], albedo_tex.inputs['Vector'])
        if normal_path is not None:
            links.new(mapping.outputs['Vector'], normal_tex.inputs['Vector'])
        if depth_path is not None:
            links.new(mapping.outputs['Vector'], depth_tex.inputs['Vector'])
        if metallic_path is not None:
            links.new(mapping.outputs['Vector'], metallic_tex.inputs['Vector'])
        if roughness_path is not None:
            links.new(mapping.outputs['Vector'], roughness_tex.inputs['Vector'])

    # Flip Normals
    if flip_normal:
        flip_normals()


if __name__ == '__main__':
    parser = ArgumentParserForBlender()
    parser.add_argument('--import_mesh_path', type=str, required=True, help='Path to the textured mesh (.obj).')

    parser.add_argument('--export_blend_path', type=str, default='', help='Export .blend path.')
    parser.add_argument('--export_fbx_path', type=str, default='', help='Export .fbx path.')
    parser.add_argument('--export_gltf_path', type=str, default='', help='Export .gltf path.')

    parser.add_argument('--light_type', type=str, default='POINT', help='light type')
    parser.add_argument('--light_energy', type=float, default=0.0, help='energy of light')
    parser.add_argument('--environment_path', type=str, default='', help='Environment path.')

    parser.add_argument('--use_backface_culling', action='store_true', help='use_backface_culling')
    parser.add_argument('--use_sphere', action='store_true', help='Use sphere.')
    parser.add_argument('--flip_normal', action='store_true', help='Whether to flip normals.')

    args = parser.parse_args()

    build_scene(
        import_mesh_path=args.import_mesh_path,
        use_backface_culling=args.use_backface_culling,
        flip_normal=args.flip_normal,
        use_sphere=args.use_sphere,
    )

    if args.environment_path != '':
        build_world(args.environment_path, strength=1.0)

    if args.light_energy > 0.0:
        light = add_light(
            light_type=args.light_type,
            location=(0, 0, 0),
            energy=args.light_energy,
        )

    if args.export_gltf_path != '':
        os.makedirs(osp.dirname(args.export_gltf_path), exist_ok=True)
        bpy.ops.export_scene.gltf(filepath=args.export_gltf_path)

    if args.export_fbx_path != '':
        os.makedirs(osp.dirname(args.export_fbx_path), exist_ok=True)
        bpy.ops.export_scene.fbx(filepath=args.export_fbx_path, path_mode='COPY')

    if args.export_blend_path != '':
        if osp.isfile(args.export_blend_path):
            os.remove(args.export_blend_path)
        os.makedirs(osp.dirname(args.export_blend_path), exist_ok=True)
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=args.export_blend_path)
