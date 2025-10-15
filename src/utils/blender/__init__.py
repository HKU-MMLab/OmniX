import os
import bpy
import math
from typing import Optional


def add_camera(
    type: str = 'PERSP',
    location: tuple = (0, 0, 0),
    rotation: tuple = (math.radians(90), 0, 0),
    fov: float = 90.0,
):
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.active_object
    cam.data.type = type
    cam.rotation_euler = rotation
    if type == 'PERSP':
        cam.data.angle = math.radians(fov)
    elif type == 'PANO':
        cam.data.panorama_type = 'EQUIRECTANGULAR'
    bpy.context.scene.camera = cam
    return cam


def add_light(
    light_type: str = 'POINT',
    location: tuple = (0, 0, 0),
    rotation_euler: tuple = (0, 0, 0),
    energy: float = 0.0,
):
    if light_type == 'POINT':
        bpy.ops.object.light_add(type=light_type, location=location)
    elif light_type == 'SUN':
        rotation_euler = (
            math.radians(rotation_euler[0]),
            math.radians(rotation_euler[1]),
            math.radians(rotation_euler[2]),
        )
        bpy.ops.object.light_add(type=light_type, rotation=rotation_euler)
    else:
        raise ValueError("Unsupported light type. Please use 'POINT' or 'SUN'.")
    light = bpy.context.active_object
    light.data.energy = energy
    return light


def reset_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    world = bpy.context.scene.world
    for n in world.node_tree.nodes:
        world.node_tree.nodes.remove(n)


def import_scene(mesh_path: str):
    if mesh_path.endswith('.obj'):
        bpy.ops.wm.obj_import(filepath=mesh_path)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                obj.select_set(True)
    elif mesh_path.endswith('.gltf') or mesh_path.endswith('.glb'):
        bpy.ops.import_scene.gltf(filepath=mesh_path)


def export_scene(mesh_path: str):
    if mesh_path.lower().endswith('.obj'):
        os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
        bpy.ops.wm.obj_export(
            filepath=mesh_path,
            path_mode='COPY',
        )
    elif mesh_path.lower().endswith('.glb'):
        os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
        bpy.ops.export_scene.gltf(filepath=mesh_path, export_format='GLB')
    else:
        raise ValueError("Unsupported mesh format. Please use .obj or .glb/.gltf files.")


def convert_obj_to_glb(
    obj_path: str,
    glb_path: Optional[str] = None,
    shade_type: str = "SMOOTH",
    auto_smooth_angle: float = 60,
    merge_vertices: bool = False,
) -> bool:
    """Convert OBJ file to GLB format using Blender."""
    try:
        if glb_path is None:
            glb_path = obj_path.lower().replace('.obj', '.glb')
        
        if "convert" not in bpy.data.scenes:
            bpy.data.scenes.new("convert")
        bpy.context.window.scene = bpy.data.scenes["convert"]

        for obj in bpy.context.scene.objects:
            obj.select_set(True)
            bpy.data.objects.remove(obj, do_unlink=True)

        # Import OBJ file
        bpy.ops.wm.obj_import(filepath=obj_path)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                obj.select_set(True)

        # Process meshes
        if merge_vertices:
            for obj in bpy.context.selected_objects:
                if obj.type == "MESH":
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.remove_doubles()
                    bpy.ops.object.mode_set(mode="OBJECT")

        def _apply_auto_smooth(auto_smooth_angle: float):
            """Apply auto smooth based on Blender version."""
            angle_rad = math.radians(auto_smooth_angle)

            if bpy.app.version < (4, 1, 0):
                bpy.ops.object.shade_smooth(use_auto_smooth=True, auto_smooth_angle=angle_rad)
            elif bpy.app.version < (4, 2, 0):
                bpy.ops.object.shade_smooth_by_angle(angle=angle_rad)
            else:
                bpy.ops.object.shade_auto_smooth(angle=angle_rad)

        shading_ops = {
            "SMOOTH": lambda: bpy.ops.object.shade_smooth(),
            "FLAT": lambda: bpy.ops.object.shade_flat(),
            "AUTO_SMOOTH": lambda: _apply_auto_smooth(auto_smooth_angle),
        }

        if shade_type in shading_ops:
            shading_ops[shade_type]()

        # Export to GLB
        bpy.ops.export_scene.gltf(filepath=glb_path, use_active_scene=True)
        return True
    
    except Exception:
        return False


def build_world(
    environment_path: str,
    strength: float = 1.0,
    rotation: tuple = (0, 0, 0),
):
    world = bpy.context.scene.world
    world_nodes = world.node_tree.nodes
    world_links = world.node_tree.links

    env_tex = world_nodes.new("ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(environment_path)
    bg = world_nodes.new("ShaderNodeBackground")
    output = world_nodes.new("ShaderNodeOutputWorld")
    mapping = world_nodes.new("ShaderNodeMapping")
    tex_coord = world_nodes.new("ShaderNodeTexCoord")

    world_links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    world_links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    world_links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    world_links.new(bg.outputs["Background"], output.inputs["Surface"])

    mapping.inputs['Rotation'].default_value = rotation
    bg.inputs['Strength'].default_value = strength

    return mapping


def setup_render(
    camera_type: str,
    resolution_x: int,
    resolution_y: int,
    engine: str = 'CYCLES',  # 'CYCLES', 'BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'
    filepath: Optional[str] = None,
    samples: Optional[int] = None,
):
    if engine == "EEVEE":
        engine = "BLENDER_EEVEE_NEXT"
    
    elif engine == "WORKBENCH":
        engine = "BLENDER_WORKBENCH"
    
    if camera_type == 'PANO':
        assert engine == 'CYCLES', "Panoramic camera is only supported in Cycles engine."
    
    scene = bpy.context.scene
    render = bpy.context.scene.render
    
    if filepath is not None:
        render.filepath = filepath
    
    render.engine = engine

    if engine == 'CYCLES':
        scene.cycles.device = 'GPU'
        if samples is not None:
            scene.cycles.samples = samples
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # or "OPENCL"
    elif engine == 'BLENDER_EEVEE_NEXT':
        if samples is not None:
            scene.eevee.taa_render_samples = samples
            scene.eevee.taa_samples = samples
    
    render.resolution_x = resolution_x
    render.resolution_y = resolution_y
    return render


def flip_normals():
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # 设为活动对象
            bpy.context.view_layer.objects.active = obj
            # 进入编辑模式
            bpy.ops.object.mode_set(mode='EDIT')
            # 选择全部面
            bpy.ops.mesh.select_all(action='SELECT')
            # 翻转法线
            bpy.ops.mesh.flip_normals()
            # 返回对象模式
            bpy.ops.object.mode_set(mode='OBJECT')
