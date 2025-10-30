import os
import os.path as osp
import numpy as np
from PIL import Image
from einops import einsum, rearrange
import torch
import torch.nn.functional as F
from typing import Optional, Literal, Union
from shutil import copyfile
import open3d as o3d
import pymeshlab
import trimesh
import xatlas


# Create ray directions (right-handed coordinate system, X: left, Y: up, Z: forward)
def get_equi_raymap_numpy(height: int, width: int) -> np.ndarray:
    """
    计算ERP全景图像的raymap (遵循Structured3D的坐标系定义: Front: +X, Up: +Y, Left: +Z).
    
    Args:
        height (int): 图像的高度（像素数）
        width (int): 图像的宽度（像素数）
        
    Returns:
        np.ndarray: 形状为 (height, width, 3) 的数组，存储每个像素对应的射线方向向量。
    """
    # 创建像素坐标网格
    i = np.arange(height)
    j = np.arange(width)
    # 转换为网格
    v, u = np.meshgrid(i, j, indexing='ij')
    
    # 将像素索引转换为归一化的角度
    lon = (0.5 - (u + 0.5) / width) * 2 * np.pi  # 从左到右的横向角度: π 到 -π
    lat = (0.5 - (v + 0.5) / height) * np.pi     # 从上到下的纵向角度: π/2 到 -π/2

    # 计算空间中的射线方向向量
    # 使用球面坐标到笛卡尔坐标的转换
    x = np.cos(lat) * np.cos(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.sin(lon)
    
    # 拼接成一个三维数组
    raymap = np.stack((x, y, z), axis=-1)

    return raymap

def get_equi_raymap_torch(h: int, w: int, device, up_axis: str = 'y') -> torch.Tensor:
    u = (torch.arange(w, device=device).float() + 0.5) / w
    v = (torch.arange(h, device=device).float() + 0.5) / h
    vv, uu = torch.meshgrid(v, u, indexing="ij")   # (H,W) order here

    phi   = uu * 2 * torch.pi - torch.pi  # [-pi, +pi]
    theta = torch.pi / 2 - vv * torch.pi  # [pi/2, -pi/2]

    if up_axis == 'y':
        x = - torch.cos(theta) * torch.sin(phi)
        y = torch.sin(theta)
        z = torch.cos(theta) * torch.cos(phi)
    elif up_axis == 'z':
        x = - torch.cos(theta) * torch.sin(phi)
        y = torch.cos(theta) * torch.cos(phi)
        z = torch.sin(theta)
    else:
        raise ValueError(f"Unsupported up_axis: {up_axis}")
    return torch.stack((x, y, z), dim=-1)  # (h, w, 3)

def get_cube_raymap_torch(h: int, w: int, device):
    assert h == w, "Cube rays require equal height and width"
    u = (torch.arange(w, device=device).float() + 0.5) / w
    v = (torch.arange(h, device=device).float() + 0.5) / h
    vv, uu = torch.meshgrid(v, u, indexing="ij")   # (H, W) order here
    rays = []
    for i in range(6):
        # +Z
        if i == 0:
            x = 1 - uu * 2
            y = 1 - vv * 2
            z = torch.ones_like(vv)
        # -X
        elif i == 1:
            x = -torch.ones_like(vv)
            y = 1 - vv * 2
            z = 1 - uu * 2
        # -Z
        elif i == 2:
            x = uu * 2 - 1
            y = 1 - vv * 2
            z = -torch.ones_like(vv)
        # +X
        elif i == 3:
            x = torch.ones_like(vv)
            y = 1 - vv * 2
            z = uu * 2 - 1
        # +Y
        elif i == 4:
            x = 1 - uu * 2
            y = torch.ones_like(vv)
            z = vv * 2 - 1
        # -Y
        elif i == 5:
            x = 1 - uu * 2
            y = -torch.ones_like(vv)
            z = - vv * 2 + 1

        rays.append(torch.stack((x, y, z), dim=-1))
    
    rays = torch.stack(rays, dim=0)  # (6, h, w, 3)
    rays = rays / torch.linalg.norm(rays, dim=-1, keepdim=True)  # Normalize rays
    
    return rays


# Resize distance and mask if needed
def resize_if_needed(
    distance: torch.Tensor,                  # (H, W)
    mask: Optional[torch.Tensor] = None,     # (H, W) Optional boolean mask
    max_size: int = 2048,
) -> Union[torch.Tensor, Optional[torch.Tensor]]:
    H, W = distance.shape
    if max_size is not None and max(H, W) > max_size:
        scale = max_size / max(H, W)

        distance = F.interpolate(
            distance.unsqueeze(0).unsqueeze(0),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False
        ).squeeze(0).squeeze(0)

        if mask is not None:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(), # Needs float for interpolation
                scale_factor=scale,
                mode="bilinear", # Or 'nearest' if sharp boundaries are critical
                align_corners=False,
                recompute_scale_factor=False
            ).squeeze(0).squeeze(0)
            mask = mask > 0.5 # Convert back to boolean

    return distance, mask


# Convert 3D points to UV coordinates
def xyzs_to_uvs(xyz: torch.Tensor, flip_v: bool = False, up_axis: str = 'y') -> torch.Tensor:
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    if up_axis == 'y':
        phi = torch.atan2(-x, z)  # [-pi, +pi]
        theta = torch.asin(y)     # [-pi/2, +pi/2]
    elif up_axis == 'z':
        phi = torch.atan2(-x, y)
        theta = torch.asin(z)     # [-pi/2, +pi/2]
    else:
        raise ValueError(f"Unsupported up_axis: {up_axis}")
    u = (phi + torch.pi) / (2 * torch.pi)
    v = (torch.pi / 2 - theta) / torch.pi
    if flip_v:
        v = 1.0 - v  # Flip v to match graphics definition
    return torch.stack((u, v), dim=-1)  # (..., 2)


# Rotate points around a specified axis
def rotate_points(points: np.ndarray, axis: str, angle_degrees: float):
    """
    绕指定轴旋转点云。

    Parameters:
    - points: numpy.ndarray of shape (N, 3)，点集
    - axis: str，旋转轴，'x'、'y' 或 'z'
    - angle_degrees: float，旋转角度（度）

    Returns:
    - rotated_points: numpy.ndarray of shape (N, 3)，旋转后的点
    """
    angle_radians = np.deg2rad(angle_degrees)
    c = np.cos(angle_radians)
    s = np.sin(angle_radians)

    if axis == 'x':
        R = np.array([[1, 0, 0],
                    [0, c, -s],
                    [0, s, c]])
    elif axis == 'y':
        R = np.array([[c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c]])
    elif axis == 'z':
        R = np.array([[c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]])
    else:
        raise ValueError("轴参数必须是 'x', 'y' 或 'z'.")

    # 对每个点应用旋转矩阵
    rotated_points = points @ R.T
    return rotated_points


# Custom equi UV mapping to handle seam crossing
def equi_parametrize(vertices, faces):
    """
    在检测跨缝后，为左边顶点复制新顶点，保证UV连续。
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)

    # 转成torch tensor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vertices_pt = torch.tensor(vertices, dtype=torch.float32, device=device)
    v_norm = torch.linalg.norm(vertices_pt, dim=1, keepdim=True)
    v_unit = vertices_pt / (v_norm + 1e-8)
    uv_all = xyzs_to_uvs(v_unit, flip_v=True).cpu().numpy().astype(np.float32)

    # 初始化新顶点列表
    uvs = []
    vmapping = []
    vertex_uv_map = {}  # (orig_idx, seam_side) -> new_vertex_idx
    indices = []

    def crosses_seam(uvs_tri):
        u = uvs_tri[:, 0]
        return np.ptp(u) > 0.5

    for face in faces:
        tri_idx = []
        uv_tri = uv_all[face]
        if crosses_seam(uv_tri):
            # 找到右边顶点（U > 0.5）
            right_idx = None
            for i, u in enumerate(uv_tri[:, 0]):
                if u > 0.5:
                    right_idx = i
                    break
            assert right_idx is not None, "Seam crossing but no vertex on right side?"
            # 跨缝处理
            for i, vi in enumerate(face):
                # 判断是否在左边还是右边
                if uv_tri[i, 0] < 0.5:  # 在左边，复制顶点
                    key = (vi, 'left')
                    if key not in vertex_uv_map:  # 检查是否已复制
                        vertex_uv_map[key] = len(uvs)
                        uvs.append(uv_tri[right_idx].copy())  # UV设置为右边顶点的UV
                        vmapping.append(vi)
                    tri_idx.append(vertex_uv_map[key])
                else:  # 右边顶点，直接使用原始索引
                    key = (vi, 'right')
                    if key not in vertex_uv_map:
                        vertex_uv_map[key] = len(uvs)
                        uvs.append(uv_all[vi])
                        vmapping.append(vi)
                    tri_idx.append(vertex_uv_map[key])
        else:
            # 不跨缝，正常处理
            tri_idx = []
            for i, vi in enumerate(face):
                key = (vi, 'main')
                if key not in vertex_uv_map:
                    vertex_uv_map[key] = len(uvs)
                    uvs.append(uv_all[vi])
                    vmapping.append(vi)
                tri_idx.append(vertex_uv_map[key])
        indices.append(tri_idx)

    # 转为numpy数组
    uvs = np.array(uvs, dtype=np.float32)
    vmapping = np.array(vmapping, dtype=np.uint32)
    indices = np.array(indices, dtype=np.uint32)

    return vmapping, indices, uvs


# Calculate barycentric coordinates for 2D points
def barycentric_coords_2d(p, a, b, c):
    # p: (N, 2), a/b/c: (2,)
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0) if v2.ndim == 1 else np.dot(v2, v0)
    d21 = np.dot(v2, v1) if v2.ndim == 1 else np.dot(v2, v1)
    denom = (d00 * d11 - d01 * d01) + 1e-8
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.stack([u, v, w], axis=-1)


# Map ERP to UV texture
def map_equi_to_uv_texture(mesh: trimesh.Trimesh, panorama: np.ndarray, texture_size: int = 4096) -> np.ndarray:
    uv = mesh.visual.uv
    faces = mesh.faces
    vertices = mesh.vertices
    
    if panorama.ndim == 3:
        pano_height, pano_width, pano_channel = panorama.shape
        texture = np.zeros((texture_size, texture_size, pano_channel), dtype=panorama.dtype)
    else:
        pano_height, pano_width = panorama.shape
        texture = np.zeros((texture_size, texture_size), dtype=panorama.dtype)

    for face in faces:
        tri_uv = uv[face]
        tri_xyz = vertices[face]
        tri_pixels = tri_uv * texture_size
        tri_pixels[:, 1] = texture_size - tri_pixels[:, 1]  # Y轴翻转
        tri_pixels = tri_pixels.astype(np.int32)
        min_x = max(0, np.min(tri_pixels[:, 0]))
        max_x = min(texture_size - 1, np.max(tri_pixels[:, 0]))
        min_y = max(0, np.min(tri_pixels[:, 1]))
        max_y = min(texture_size - 1, np.max(tri_pixels[:, 1]))

        if min_x > max_x or min_y > max_y:
            continue

        # 批量生成边界框内所有像素坐标
        grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

        # 批量计算重心坐标
        w = barycentric_coords_2d(points, *tri_pixels)
        mask = (w >= -1e-5).all(axis=1) & (np.abs(w.sum(axis=1) - 1) < 1e-5)
        if not np.any(mask):
            continue

        points_in = points[mask]
        ws = w[mask]

        # 用重心插值，得到每个像素对应的3D点
        xyz_points = ws @ tri_xyz  # (N,3) x (3,3) -> (N,3)

        # 方向归一化（假设相机在原点）
        dirs = xyz_points / np.linalg.norm(xyz_points, axis=1, keepdims=True)
        
        # 计算经纬度
        lon = np.arctan2(- dirs[:, 0], dirs[:, 2])           # [-π, π]
        lat = - np.arcsin(dirs[:, 1])                        # [-π/2, π/2]

        x_pano = ((lon + np.pi) / (2 * np.pi) * pano_width).astype(np.int32)
        y_pano = ((lat + np.pi / 2) / np.pi * pano_height).astype(np.int32)
        x_pano = np.clip(x_pano, 0, pano_width - 1)
        y_pano = np.clip(y_pano, 0, pano_height - 1)

        # 写入纹理贴图
        tx, ty = points_in[:, 0], points_in[:, 1]
        texture[ty, tx] = panorama[y_pano, x_pano]

    return texture


# Albedo Post-processing
def copyfile_with_albedo_postprocess(albedo_path, save_albedo_path, mask_path=None):
    albedo = np.array(Image.open(albedo_path))
    height, width, n_channels = albedo.shape
    if n_channels == 3:
        if mask_path is not None:
            raise NotImplementedError("Mask path handling not implemented yet.")
        else:
            mask = np.full((height, width, 1), fill_value=255, dtype=np.uint8)
        albedo = np.concatenate([albedo, mask], axis=-1)
    else:
        assert n_channels == 4, "Albedo image must have 3 or 4 channels."
    Image.fromarray(albedo).save(save_albedo_path)


# Normal Post-processing
def copyfile_with_normal_postprocess(normal_path, save_normal_path, from_panox=False, eps=1e-8):
    normal = np.array(Image.open(normal_path)).astype(np.float32) / 127.5 - 1.0
    
    if from_panox:
        normal[:, :, [1, 2]] = normal[:, :, [2, 1]]
        normal[:, :, [2]] *= -1
    
    h, w, _ = normal.shape
    rays = get_equi_raymap_numpy(h, w)  # (H, W, 3)
    flip_mask = (einsum(normal, rays, 'h w c, h w c -> h w') > 0.)
    normal[flip_mask] *= -1

    normal[:, :, [1, 2]] = normal[:, :, [2, 1]]
    normal *= -1

    normal_dist = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / (normal_dist + eps)

    normal = (normal + 1.0) * 127.5
    normal = np.clip(normal, 0, 255).astype(np.uint8)
    Image.fromarray(normal).save(save_normal_path)


def write_mtl(
    mtl_path,
    albedo_path=None,
    alpha_path=None,
    metallic_path=None,
    roughness_path=None,
    normal_path=None,
    material_name="material0",
):
    """
    创建一个简单的MTL文件, 指定漫反射贴图。
    Args:
        mtl_path: MTL文件名
        texture_path: 纹理图片文件名
        material_name: 材质名称
    """
    lines = [
        "# material file",
        f"newmtl {material_name}",
        "Kd 1.000000 1.000000 1.000000",
        "Ka 1.000000 1.000000 1.000000",
        "Ke 0.000000 0.000000 0.000000",
        "Ks 0.000000 0.000000 0.000000",
        "Ni 1.450000",
        "d 1.0",
        "illum 2"
    ]
    if albedo_path is not None:
        lines.append(f"map_Kd {osp.split(albedo_path)[-1]}")
    if alpha_path is not None:
        lines.append(f"map_d {osp.split(alpha_path)[-1]}")
    if metallic_path is not None:
        lines.append(f"map_Pm {osp.split(metallic_path)[-1]}")
    if roughness_path is not None:
        lines.append(f"map_Pr {osp.split(roughness_path)[-1]}")
    if normal_path is not None:
        lines.append(f"map_Bump -bm 1.000000 {osp.split(normal_path)[-1]}")
    
    with open(mtl_path, "w") as f:
        f.write("\n".join(lines))


def write_obj_with_material(obj_path: str, mtl_path: str, save_obj_path: str, material_name="material0"):
    def insert_line_at_top(file_path, line_content):
        # 读取原始文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 在首行插入内容
        lines.insert(0, line_content)
        # 写回文件（覆盖原文件）
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    copyfile(obj_path, save_obj_path)
    
    line_content = f'mtllib {osp.split(mtl_path)[-1]}\n' + f'usemtl {material_name}\n'
    insert_line_at_top(save_obj_path, line_content)


class MeshBuilder(object):
    def __init__(self):
        super(MeshBuilder, self).__init__()
    
    def _create_vertices(
        self,
        distance: torch.Tensor,  # (..., H, W)
        ray: torch.Tensor       # (..., H, W, 3)
    ) -> torch.Tensor:
        """
        Create 3D vertices from distance and ray directions.

        Args:
            distance: (H, W) Distance map.
            ray: (H, W, 3) Ray directions.

        Returns:
            vertices: (H*W, 3) 3D vertices.
        """
        distance_flat = distance.reshape(-1, 1)    # (H*W, 1)
        ray_flat = ray.reshape(-1, 3)              # (H*W, 3)
        vertices = distance_flat * ray_flat        # (H*W, 3)
        return vertices

    def _create_triangles_from_equirectangular(
        self,
        distance: torch.Tensor,  # (H, W)
        mask: torch.Tensor,      # (H, W)
        closed_boundary: bool = True,
    ) -> torch.Tensor:
        device = distance.device

        # --- Generate Mesh Faces ---
        H_new, W_new = distance.shape # Get new dimensions
        if not closed_boundary:
            row_indices = torch.arange(0, H_new - 1, device=device)
            col_indices = torch.arange(0, W_new - 1, device=device)
            row = row_indices.repeat(W_new - 1)
            col = col_indices.repeat_interleave(H_new - 1)

            tl = row * W_new + col      # Top-left
            tr = tl + 1                 # Top-right
            bl = tl + W_new             # Bottom-left
            br = bl + 1                 # Bottom-right
        else:
            row_indices = torch.arange(0, H_new - 1, device=device)
            col_indices = torch.arange(0, W_new, device=device)
            row = row_indices.repeat(W_new)
            col = col_indices.repeat_interleave(H_new - 1)

            tl = row * W_new + col                      # Top-left
            tr = row * W_new + (col + 1) % W_new        # Top-right
            bl = tl + W_new                             # Bottom-left
            br = (row + 1) * W_new + (col + 1) % W_new  # Bottom-right
            
        # Apply mask if provided
        if mask is not None:
            mask_tl = mask[row, col]
            mask_tr = mask[row, (col + 1) % W_new]
            mask_bl = mask[row + 1, col]
            mask_br = mask[row + 1, (col + 1) % W_new]

            quad_keep_mask = ~(mask_tl | mask_tr | mask_bl | mask_br)

            keep_indices = quad_keep_mask.nonzero(as_tuple=False).squeeze(-1)
            tl = tl[keep_indices]
            tr = tr[keep_indices]
            bl = bl[keep_indices]
            br = br[keep_indices]

        # --- Create Triangles ---
        tri1 = torch.stack([tl, tr, bl], dim=1)
        tri2 = torch.stack([tr, br, bl], dim=1)
        faces = torch.cat([tri1, tri2], dim=0)

        return faces

    def _create_triangles_from_cubemap(self, H: int, W: int) -> torch.Tensor:
        """
        Generate triangle indices for a closed cube mesh from a CubeMap of shape [6, H, W].
        Returns: [N, 3] triangle index tensor
        """
        M = 6
        all_triangles = []

        # --- Vectorized per-face triangles ---
        grid_y, grid_x = torch.meshgrid(torch.arange(H - 1), torch.arange(W - 1), indexing='ij')
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)

        v0 = grid_y * W + grid_x
        v1 = grid_y * W + (grid_x + 1)
        v2 = (grid_y + 1) * W + grid_x
        v3 = (grid_y + 1) * W + (grid_x + 1)

        tris1 = torch.stack([v0, v1, v2], dim=1)
        tris2 = torch.stack([v2, v1, v3], dim=1)
        face_tris = torch.cat([tris1, tris2], dim=0)

        for f in range(M):
            all_triangles.append(face_tris + f * H * W)

        # --- Face edge connections ---
        def get_edge(face, edge):
            if edge == 'top':
                return torch.arange(W) + face * H * W
            if edge == 'bottom':
                return torch.arange(W) + face * H * W + (H - 1) * W
            if edge == 'left':
                return torch.arange(H) * W + face * H * W
            if edge == 'right':
                return torch.arange(H) * W + face * H * W + (W - 1)

        edge_pairs = [
            (0, 'right', 1, 'left', False),
            (1, 'right', 2, 'left', False),
            (2, 'right', 3, 'left', False),
            (3, 'right', 0, 'left', False),

            (0, 'top', 4, 'bottom', False),
            (1, 'top', 4, 'right', True),
            (2, 'top', 4, 'top', True),
            (3, 'top', 4, 'left', False),

            (0, 'bottom', 5, 'top', False),
            (1, 'bottom', 5, 'right', False),
            (2, 'bottom', 5, 'bottom', True),
            (3, 'bottom', 5, 'left', True),
        ]

        for fA, eA, fB, eB, rev in edge_pairs:
            edgeA = get_edge(fA, eA)
            edgeB = get_edge(fB, eB)
            if rev:
                edgeB = edgeB.flip(0)

            v0 = edgeA[:-1]
            v1 = edgeA[1:]
            v2 = edgeB[:-1]
            v3 = edgeB[1:]

            tris1 = torch.stack([v0, v1, v2], dim=1)
            tris2 = torch.stack([v2, v1, v3], dim=1)
            all_triangles.append(tris1)
            all_triangles.append(tris2)

        # --- Add corner sealing triangles ---
        def vid(f, y, x):
            return f * H * W + y * W + x

        corners = [
            # (left, up, front)
            [vid(3, 0, W - 1), vid(4, H - 1, 0), vid(0, 0, 0)],
            # (front, up, right)
            [vid(0, 0, W - 1), vid(4, H - 1, W - 1), vid(1, 0, 0)],
            # (right, up, back)
            [vid(1, 0, W - 1), vid(4, 0, W - 1), vid(2, 0, 0)],
            # (back, up, left)
            [vid(2, 0, W - 1), vid(4, 0, 0), vid(3, 0, 0)],

            # (left, down, back)
            [vid(3, H - 1, 0), vid(5, H - 1, 0), vid(2, H - 1, W - 1)],
            # (back, down, right)
            [vid(2, H - 1, 0), vid(5, H - 1, W - 1), vid(1, H - 1, W - 1)],
            # (right, down, front)
            [vid(1, H - 1, 0), vid(5, 0, W - 1), vid(0, H - 1, W - 1)],
            # (front, down, left)
            [vid(0, H - 1, 0), vid(5, 0, 0), vid(3, H - 1, W - 1)],
        ]

        all_triangles.extend([torch.tensor(corner, dtype=torch.long).unsqueeze(0) for corner in corners])

        return torch.cat(all_triangles, dim=0).long()

    def create_mesh_from_equi_distance(
        self,
        distance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        max_size: int = 2048,
        device: Optional[Literal['cuda', 'cpu', 'npu']] = None,
        closed_boundary: bool = True,
    ) -> o3d.geometry.TriangleMesh:
        """
        Converts panoramic distance data (distance, rays) into an Open3D mesh.

        Args:
            distance: Input distance map tensor (H, W).
            mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
            max_size: Maximum size (height or width) to resize inputs to.
            save_path: Optional path to save the resulting mesh.
            device: Device to perform computations on. Defaults to self.device.
            closed_boundary: Whether to treat the panorama as having a closed horizontal boundary.

        Returns:
            An Open3D TriangleMesh object.
        """
        if device is None:
            device = distance.device

        assert distance.ndim == 2, "Distance must be HxW"
        assert torch.all(distance >= 0), "Distance values must be non-negative"
        
        if mask is not None:
            assert mask.dtype == torch.bool, "Mask must be a boolean tensor"
            assert mask.shape == distance.shape, "Mask must be the same shape as distance"

        distance = distance.to(device)
        if mask is not None:
            mask = mask.to(device)

        distance, mask = resize_if_needed(distance, mask, max_size)

        ray = get_equi_raymap_torch(distance.shape[0], distance.shape[1], device)
        
        vertices = self._create_vertices(distance, ray)  # (H*W, 3)
        triangles = self._create_triangles_from_equirectangular(distance, mask, closed_boundary)  # (F, 3)

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
        mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles.cpu().numpy())

        mesh_o3d.remove_unreferenced_vertices()
        mesh_o3d.remove_degenerate_triangles()

        if save_path is not None:
            o3d.io.write_triangle_mesh(save_path, mesh_o3d)

        return mesh_o3d

    def create_mesh_from_cube_distance(
        self,
        distance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        max_size: int = 2048,
        device: Optional[Literal['cuda', 'cpu', 'npu']] = None,
        closed_boundary: bool = True,
    ) -> o3d.geometry.TriangleMesh:
        """
        Converts cubemap distance into an Open3D mesh.

        Args:
            distance: Input distance map tensor (M, H, W), meters.
            mask: Optional boolean mask tensor (M, H, W). True values indicate regions to potentially exclude.
            max_size: Maximum size (height or width) to resize inputs to.
            device: The torch device ('cuda' or 'cpu') to use for computations.

        Returns:
            An Open3D TriangleMesh object.
        """
        assert distance.ndim == 3, "Distance must be MxHxW"
        if mask is not None:
            assert mask.ndim == 3 and mask.shape[:3] == distance.shape[:3], "Mask shape must match"
            assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

        if device is None:
            device = distance.device
        
        distance = distance.to(device)
        if mask is not None:
            mask = mask.to(device)

        _, H, W = distance.shape
        if max_size is not None and max(H, W) > max_size:
            scale = max_size / max(H, W)
        else:
            scale = None

        if scale is not None:
            distance = F.interpolate(
                distance.unsqueeze(1),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False
            ).squeeze(1)

            if mask is not None:
                mask = F.interpolate(
                    mask.unsqueeze(1).float(), # Needs float for interpolation
                    scale_factor=scale,
                    mode="bilinear", # Or 'nearest' if sharp boundaries are critical
                    align_corners=False,
                    recompute_scale_factor=False
                ).squeeze(1)
                mask = mask > 0.5 # Convert back to boolean

        rays = get_cube_raymap_torch(distance.shape[1], distance.shape[2], device)
        
        _, H_new, W_new = distance.shape # Get new dimensions

        # --- Calculate 3D Vertices ---
        vertices = self._create_vertices(distance, rays)  # (M*H*W, 3)

        # --- Generate Mesh Faces (Triangles from Quads for Cubemap) ---
        faces = []
        for face_idx in range(6):  # Iterate over the 6 faces of the cubemap
            if not closed_boundary or face_idx >= 4:
                row_indices = torch.arange(0, H_new - 1, device=device)
                col_indices = torch.arange(0, W_new - 1, device=device)
                row = row_indices.repeat(W_new - 1)
                col = col_indices.repeat_interleave(H_new - 1)

                offsets = face_idx * (H_new * W_new)  # Offset for each face

                tl = offsets + row * W_new + col      # Top-left
                tr = tl + 1                           # Top-right
                bl = tl + W_new                       # Bottom-left
                br = bl + 1                           # Bottom-right
            else:
                row_indices = torch.arange(0, H_new - 1, device=device)
                col_indices = torch.arange(0, W_new, device=device)
                row = row_indices.repeat(W_new)
                col = col_indices.repeat_interleave(H_new - 1)

                offset = H_new * W_new
                start = face_idx * (H_new * W_new)  # Offset for each face

                if face_idx in (0, 1, 2):
                    tl = start + row * W_new + col                               # Top-left
                    tr = start + row * W_new + (col + 1) % W_new + (col + 1) // W_new * offset # Top-right
                    bl = tl + W_new                                                 # Bottom-left
                    br = start + (row + 1) * W_new + (col + 1) % W_new + (col + 1) // W_new * offset  # Bottom-right
                else:
                    tl = start + row * W_new + col                               # Top-left
                    tr = start + row * W_new + (col + 1) % W_new + (col + 1) // W_new * (-3 * offset) # Top-right
                    bl = tl + W_new                                                 # Bottom-left
                    br = start + (row + 1) * W_new + (col + 1) % W_new + (col + 1) // W_new * (-3 * offset)  # Bottom-right

            # Apply mask if provided
            if mask is not None:
                mask_face = mask[face_idx]

                mask_tl = mask_face[row, col]
                mask_tr = mask_face[row, (col + 1) % W_new]
                mask_bl = mask_face[row + 1, col]
                mask_br = mask_face[row + 1, (col + 1) % W_new]

                quad_keep_mask = ~(mask_tl | mask_tr | mask_bl | mask_br)

                keep_indices = quad_keep_mask.nonzero(as_tuple=False).squeeze(-1)
                tl = tl[keep_indices]
                tr = tr[keep_indices]
                bl = bl[keep_indices]
                br = br[keep_indices]

            # --- Create Triangles ---
            tri1 = torch.stack([tl, tr, bl], dim=1)
            tri2 = torch.stack([tr, br, bl], dim=1)
            faces.append(torch.cat([tri1, tri2], dim=0))

        faces = torch.cat(faces, dim=0)  # Combine faces from all cubemap sides
        faces = self._create_triangles_from_cubemap(H_new, W_new)

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())

        mesh_o3d.remove_unreferenced_vertices()
        mesh_o3d.remove_degenerate_triangles()

        if save_path is not None:
            o3d.io.write_triangle_mesh(save_path, mesh_o3d)

        return mesh_o3d

    def simplify_mesh(
        self,
        input_path: str,
        output_path: str,
        target_count: int = 40000,
    ):
        # 先去除离散面
        ms = pymeshlab.MeshSet()
        if input_path.endswith(".glb"):
            ms.load_new_mesh(input_path, load_in_a_single_layer=True)
        else:
            ms.load_new_mesh(input_path)
        ms.save_current_mesh(output_path.replace(".glb", ".obj"), save_textures=False)
        
        # 调用减面函数
        courent = trimesh.load(output_path.replace(".glb", ".obj"), force="mesh")
        face_num = courent.faces.shape[0]
        if face_num > target_count:
            courent = courent.simplify_quadric_decimation(target_count)
        courent.export(output_path)

    def unwrap_mesh(
        self,
        input_path: str,
        output_path: str,
        method: str = 'equi',  # 'xatlas' or 'equi'
    ):
        mesh = trimesh.load_mesh(input_path)
        
        if method == 'xatlas':
            vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
            """
                # The parametrization potentially duplicates vertices.
                # `vmapping` contains the original vertex index for each new vertex (shape N, type uint32).
                # `indices` contains the vertex indices of the new triangles (shape Fx3, type uint32)
                # `uvs` contains texture coordinates of the new vertices (shape Nx2, type float32)
            """
            xatlas.export(output_path, mesh.vertices[vmapping], indices, uvs)
        
        elif method == 'equi':
            vmapping, indices, uvs = equi_parametrize(mesh.vertices, mesh.faces)
            vertices = mesh.vertices[vmapping]  # np.ndarray, [N, 3]
            vertices = rotate_points(vertices, axis='y', angle_degrees=90)
            xatlas.export(output_path, vertices, indices, uvs)
        
        else:
            raise ValueError(f"Unsupported UV mapping method: {method}")

    def texture_mesh(
        self,
        input_path: str,
        output_path: str,
        input_albedo_path: Optional[str] = None,
        input_alpha_path: Optional[str] = None,
        input_normal_path: Optional[str] = None,
        input_roughness_path: Optional[str] = None,
        input_metallic_path: Optional[str] = None,
        method: str = 'equi',  # 'xatlas' or 'equi'
    ):
        output_file = osp.splitext(output_path)[0]
        output_mtl_path = output_file + '.mtl'

        output_albedo_path = output_file + '_albedo.png'
        output_alpha_path = output_file + '_alpha.png'
        output_normal_path = output_file + '_normal.png'
        output_metallic_path = output_file + '_metallic.png'
        output_roughness_path = output_file + '_roughness.png'

        if method == 'xatlas':
            mesh = trimesh.load(input_path, process=False)
            
            if input_albedo_path is not None:
                albedo_texture = map_equi_to_uv_texture(mesh, np.array(Image.open(input_albedo_path)))
                Image.fromarray(albedo_texture).save(output_albedo_path)
            
            if input_alpha_path is not None:
                alpha_texture = map_equi_to_uv_texture(mesh, np.array(Image.open(input_alpha_path)))
                Image.fromarray(alpha_texture).save(output_alpha_path)
            
            if input_normal_path is not None:
                normal_texture = map_equi_to_uv_texture(mesh, np.array(Image.open(input_normal_path)))
                Image.fromarray(normal_texture).save(output_normal_path)

            if input_roughness_path is not None:
                roughness_texture = map_equi_to_uv_texture(mesh, np.array(Image.open(input_roughness_path)))
                Image.fromarray(roughness_texture).save(output_roughness_path)
                
            if input_metallic_path is not None:
                metallic_texture = map_equi_to_uv_texture(mesh, np.array(Image.open(input_metallic_path)))
                Image.fromarray(metallic_texture).save(output_metallic_path)
            
        else:
            if input_albedo_path is not None:
                copyfile(input_albedo_path, output_albedo_path)
                # copyfile_with_albedo_postprocess(input_albedo_path, output_albedo_path)
            
            if input_alpha_path is not None:
                copyfile(input_alpha_path, output_alpha_path)
    
            if input_normal_path is not None:
                copyfile_with_normal_postprocess(input_normal_path, output_normal_path)
    
            if input_roughness_path is not None:
                copyfile(input_roughness_path, output_roughness_path)
    
            if input_metallic_path is not None:
                copyfile(input_metallic_path, output_metallic_path)
        
        material_name = "material0"
        write_mtl(
            mtl_path=output_mtl_path,
            albedo_path=output_albedo_path if osp.isfile(output_albedo_path) else None,
            alpha_path=output_alpha_path if osp.isfile(output_alpha_path) else None,
            normal_path=output_normal_path if osp.isfile(output_normal_path) else None,
            roughness_path=output_roughness_path if osp.isfile(output_roughness_path) else None,
            metallic_path=output_metallic_path if osp.isfile(output_metallic_path) else None,
            material_name=material_name,
        )
        write_obj_with_material(
            obj_path=input_path,
            mtl_path=output_mtl_path,
            save_obj_path=output_path,
            material_name=material_name,
        )
