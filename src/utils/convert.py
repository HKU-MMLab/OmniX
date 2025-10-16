import sys
sys.path.append('external')
import numpy as np
from PIL import Image
import torch
from einops import rearrange
from typing import Any, Union, List, Optional

from external.equilib.equi2cube.base import equi2cube
from external.equilib.cube2equi.base import cube2equi

from external.equilib.equi2cube.numpy import cube_h2dice as cube_h2dice_numpy
from external.equilib.equi2cube.numpy import cube_h2dict as cube_h2dict_numpy
from external.equilib.equi2cube.numpy import cube_h2list as cube_h2list_numpy

from external.equilib.equi2cube.torch import cube_h2dice as cube_h2dice_torch
from external.equilib.equi2cube.torch import cube_h2dict as cube_h2dict_torch
from external.equilib.equi2cube.torch import cube_h2list as cube_h2list_torch

from external.equilib.cube2equi.base import convert2horizon_numpy, convert2horizon_torch

ArrayLike = Union[np.ndarray, torch.Tensor]

__all__ = [
    "equi_to_cube",
    "cube_to_equi",
    "cube_to_cube",
]


def equi_to_cube(equi: ArrayLike, w_face: int, cube_format: str, clip_output: bool = False) -> Any:
    """
    Convert equirectangular panorama to cubemap.
    Args:
        equi (ArrayLike): Equirectangular panorama of shape (B, C, H, W).
        w_face (int): Width of each cube face.
        cube_format (str): Format of the output cubemap. Options are "dice", "horizon", "dict", "list".
    """
    bs, _, h, w = equi.shape
    rots = [{'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0} for _ in range(bs)]
    cube = equi2cube(equi, rots=rots, w_face=w_face, cube_format=cube_format, clip_output=clip_output)
    # For cube_format == 'list': cube[0][0].shape = [3, H, W]
    return cube


def cube_to_equi(cube: Any, height: int, width: int, cube_format: str, clip_output: bool = False) -> ArrayLike:
    """
    Convert cubemap to equirectangular panorama.
    Args:
        cube (Any): Cubemap in specified format.
        height (int): Height of the output equirectangular panorama.
        width (int): Width of the output equirectangular panorama.
        cube_format (str): Format of the input cubemap. Options are "dice", "horizon", "dict", "list".
    Returns:
        ArrayLike: Equirectangular panorama of shape (B, C, H, W).
    """
    return cube2equi(cube, height=height, width=width, cube_format=cube_format, clip_output=clip_output)


def cube_to_cube(cube: Any, src_format: str, dst_format: str) -> Any:
    assert src_format in ['array', 'dice', 'horizon', 'dict', 'list'], f"src_format {src_format} not supported"
    assert dst_format in ['array', 'dice', 'horizon', 'dict', 'list'], f"dst_format {dst_format} not supported"

    if src_format == 'array':
        cube = rearrange(cube, 'b m c h w -> b c h (m w)', m=6)
        src_format = 'horizon'

    if isinstance(cube, np.ndarray):
        horizon = convert2horizon_numpy(cube, cube_format=src_format)
        if dst_format == 'horizon':
            return horizon
        elif dst_format == 'dice':
            return cube_h2dice_numpy(horizon)
        elif dst_format == 'dict':
            return cube_h2dict_numpy(horizon)
        elif dst_format == 'list':
            return cube_h2list_numpy(horizon)
        elif dst_format == 'array':
            return rearrange(horizon, 'b c h (m w) -> b m c h w', m=6)

    elif isinstance(cube, torch.Tensor):
        horizon = convert2horizon_torch(cube, cube_format=src_format)
        if dst_format == 'horizon':
            return horizon
        elif dst_format == 'dice':
            return cube_h2dice_torch(horizon)
        elif dst_format == 'dict':
            return cube_h2dict_torch(horizon)
        elif dst_format == 'list':
            return cube_h2list_torch(horizon)
        elif dst_format == 'array':
            return rearrange(horizon, 'b c h (m w) -> b m c h w', m=6)

    else:
        raise ValueError("Input cube must be either numpy array or torch tensor.")
