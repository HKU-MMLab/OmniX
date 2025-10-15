import bpy
import mathutils
from typing import Iterable, Tuple, Optional

DEFAULT_RESTITUTION = 1.0
DEFAULT_FRICTION = 0.5


def to_vec(v):
    return mathutils.Vector(v)


def create_sphere(
    name: str = "Ball",
    radius: float = 0.05,
    location: tuple = (0.0, 0.0, 0.0),
    material_type: Optional[str] = None,  # "MIRROR" or "PLASTIC"
):
    # 创建球对象
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=location,
    )
    ball = bpy.context.active_object
    ball.name = name

    # 创建新材质
    mat = bpy.data.materials.new(name="BallMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    if material_type is not None:
        # 清除默认节点
        nodes.clear()

        # 添加 Principled BSDF 节点
        principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

        # 设置金属度和粗糙度以实现镜面效果
        if material_type == "MIRROR":
            principled_bsdf.inputs['Metallic'].default_value = 1.0
            principled_bsdf.inputs['Roughness'].default_value = 0.0
        elif material_type == "PLASTIC":
            principled_bsdf.inputs['Metallic'].default_value = 0.0
            principled_bsdf.inputs['Roughness'].default_value = 1.0

        # 添加输出节点
        material_output = nodes.new(type='ShaderNodeOutputMaterial')

        # 连接BSDF到输出
        links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

        # 将材质赋给球体
        if ball.data.materials:
            ball.data.materials[0] = mat
        else:
            ball.data.materials.append(mat)
        
    return ball


def create_rigid_ball(
    radius: float,
    name: str = "Ball",
    location: tuple = (0.0, 0.0, 0.0),
    mass: float = 1.0,
    restitution: float = DEFAULT_RESTITUTION,
    friction: float = DEFAULT_FRICTION,
    collision_shape: str = 'SPHERE',
    rigid_body_type: str = 'ACTIVE',
):
    # 创建球对象
    ball = create_sphere(
        name=name,
        radius=radius,
        location=location,
    )

    # 为球添加 ACTIVE 刚体，形状为 SPHERE，设置初始属性
    if ball.rigid_body is None:
        bpy.context.view_layer.objects.active = ball
        bpy.ops.rigidbody.object_add()
    ball.rigid_body.type = rigid_body_type
    ball.rigid_body.mass = mass
    ball.rigid_body.collision_shape = collision_shape
    ball.rigid_body.restitution = restitution
    ball.rigid_body.friction = friction
    ball.rigid_body.use_margin = True
    ball.rigid_body.enabled = True

    # 返回创建的球对象
    return ball


def setup_rigid_scene(
    exclude_objs: Iterable[bpy.types.Object] = (),
    rigid_body_type: str = 'PASSIVE',
    collision_shape: str = 'MESH',  # 'BOX', 'SPHERE', 'CAPSULE', 'CYLINDER', 'CONE', 'CONVEX_HULL', 'MESH'
    restitution: float = DEFAULT_RESTITUTION,
    friction: float = DEFAULT_FRICTION,
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81),
):
    # 为场景中的其他网格对象自动添加被动刚体
    for obj in bpy.context.scene.objects:
        if obj in exclude_objs:
            continue
        if obj.type != 'MESH':
            continue
        if obj.name.startswith("Camera") or obj.name.startswith("Light"):
            continue
        if obj.rigid_body is None:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.rigidbody.object_add()
        
        obj.rigid_body.type = rigid_body_type
        obj.rigid_body.collision_shape = collision_shape
        obj.rigid_body.restitution = restitution
        obj.rigid_body.friction = friction


def setup_force_field(
    obj: bpy.types.Object,
    acceleration: Optional[Tuple[float, float, float]] = None,
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81),
    frame_start: int = 1,
    frame_duration: int = 5,
) -> None:
    assert obj.rigid_body is not None and obj.rigid_body.type == 'ACTIVE', "对象需为 ACTIVE 刚体"

    scene = bpy.context.scene

    if acceleration is not None:
        scene.frame_set(frame_start)
        scene.gravity = to_vec(acceleration)
        scene.keyframe_insert(data_path="gravity", frame=frame_start)

        frame_end = frame_start + frame_duration
        scene.frame_set(frame_end)
        scene.gravity = to_vec(gravity)
        scene.keyframe_insert(data_path="gravity", frame=frame_end)
    else:
        scene.gravity = to_vec(gravity)


def bake_physics_simulation(
    frame_start: int,
    frame_end: int,
):
    # 确保有一个Rigidbody World
    scene = bpy.context.scene
    if not hasattr(scene, "rigidbody_world") or scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    
    # 启用刚体世界
    rw = bpy.context.scene.rigidbody_world
    rw.enabled = True
    rw.solver_iterations = 10

    # 设置步数
    rw.time_scale = 1.0
    
    # 物理缓存范围
    if rw.point_cache:
        rw.point_cache.frame_start = frame_start
        rw.point_cache.frame_end = frame_end
    
    # 烘焙所有物理缓存
    bpy.ops.ptcache.bake_all(bake=True)
