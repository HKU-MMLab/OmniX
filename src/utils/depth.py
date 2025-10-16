import torch
import torch.nn.functional as F
import cv2
import numpy as np
import PIL
from PIL import Image
from einops import rearrange, repeat
from typing import Any, Callable, Dict, List, Tuple, Optional, Union

from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor


PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]


def compute_max_depth_with_dynamic_multipliers(
    valid_depth,
    quantiles,
    acceptable_ratio=2.0,
    # acceptable_ratio=None,
    epsilon=1e-6,
    mode='max',
):
    """
    根据分位数动态计算multipliers，计算深度值，排序筛选。

    参数：
    - valid_depth：Tensor，深度数据
    - quantiles：list，分位数，默认为[0.02, 0.25, 0.5, 0.75, 0.98]
    - acceptable_ratio：float，满足条件的最大比例

    返回：
    - max_depth：float，最终选定的最大深度
    - depth_values：list，所有计算得到的深度值
    """
    # 计算multipliers
    multipliers = [1.0 / (q + epsilon) for q in quantiles]
    # 一次性计算所有分位点的值
    quantiles = torch.tensor(quantiles, dtype=valid_depth.dtype, device=valid_depth.device)
    quantile_values = torch.quantile(valid_depth, quantiles)
    # 乘以对应的系数
    depth_values = (quantile_values.detach().cpu().numpy() * multipliers).tolist()

    # 逐个筛选满足比例条件的最小值
    if acceptable_ratio is not None:
        sorted_depths = sorted(depth_values)
        max_depth = sorted_depths[0]
        candidates = [max_depth]

        for d in sorted_depths[1:]:
            if d < max_depth * acceptable_ratio:
                max_depth = d
                candidates.append(max_depth)
    else:
        candidates = depth_values
    
    if mode == 'mean':
        max_depth = np.mean(candidates).item()
    elif mode == 'max':
        max_depth = max(candidates)

    return max_depth, depth_values


class DepthVisualizer:
    def __init__(
        self,
        do_normalize: bool = True,
        do_range_check: bool = True,
    ):
        super().__init__()

    @staticmethod
    def expand_tensor_or_array(images: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Expand a tensor or array to a specified number of images.
        """
        if isinstance(images, np.ndarray):
            if images.ndim == 2:  # [H,W] -> [1,H,W,1]
                images = images[None, ..., None]
            if images.ndim == 3:  # [H,W,C] -> [1,H,W,C]
                images = images[None]
        elif isinstance(images, torch.Tensor):
            if images.ndim == 2:  # [H,W] -> [1,1,H,W]
                images = images[None, None]
            elif images.ndim == 3:  # [1,H,W] -> [1,1,H,W]
                images = images[None]
        else:
            raise ValueError(f"Unexpected input type: {type(images)}")
        return images

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
        """
        Convert a NumPy image to a PyTorch tensor.
        """
        if np.issubdtype(images.dtype, np.integer) and not np.issubdtype(images.dtype, np.unsignedinteger):
            raise ValueError(f"Input image dtype={images.dtype} cannot be a signed integer.")
        if np.issubdtype(images.dtype, np.complexfloating):
            raise ValueError(f"Input image dtype={images.dtype} cannot be complex.")
        if np.issubdtype(images.dtype, bool):
            raise ValueError(f"Input image dtype={images.dtype} cannot be boolean.")

        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images

    @staticmethod
    def resize_antialias(
        image: torch.Tensor, size: Tuple[int, int], mode: str, is_aa: Optional[bool] = None
    ) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        antialias = is_aa and mode in ("bilinear", "bicubic")
        image = F.interpolate(image, size, mode=mode, antialias=antialias)

        return image

    @staticmethod
    def resize_to_max_edge(image: torch.Tensor, max_edge_sz: int, mode: str) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        h, w = image.shape[-2:]
        max_orig = max(h, w)
        new_h = h * max_edge_sz // max_orig
        new_w = w * max_edge_sz // max_orig

        if new_h == 0 or new_w == 0:
            raise ValueError(f"Extreme aspect ratio of the input image: [{w} x {h}]")

        image = MarigoldImageProcessor.resize_antialias(image, (new_h, new_w), mode, is_aa=True)

        return image

    @staticmethod
    def pad_image(image: torch.Tensor, align: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        h, w = image.shape[-2:]
        ph, pw = -h % align, -w % align

        image = F.pad(image, (0, pw, 0, ph), mode="replicate")

        return image, (ph, pw)

    @staticmethod
    def unpad_image(image: torch.Tensor, padding: Tuple[int, int]) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        ph, pw = padding
        uh = None if ph == 0 else -ph
        uw = None if pw == 0 else -pw

        image = image[:, :, :uh, :uw]

        return image

    @staticmethod
    def load_image_canonical(
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, int]:
        if isinstance(image, Image.Image):
            image = np.array(image)

        image_dtype_max = None
        if isinstance(image, (np.ndarray, torch.Tensor)):
            image = MarigoldImageProcessor.expand_tensor_or_array(image)
            if image.ndim != 4:
                raise ValueError("Input image is not 2-, 3-, or 4-dimensional.")
        if isinstance(image, np.ndarray):
            if np.issubdtype(image.dtype, np.integer) and not np.issubdtype(image.dtype, np.unsignedinteger):
                raise ValueError(f"Input image dtype={image.dtype} cannot be a signed integer.")
            if np.issubdtype(image.dtype, np.complexfloating):
                raise ValueError(f"Input image dtype={image.dtype} cannot be complex.")
            if np.issubdtype(image.dtype, bool):
                raise ValueError(f"Input image dtype={image.dtype} cannot be boolean.")
            if np.issubdtype(image.dtype, np.unsignedinteger):
                image_dtype_max = np.iinfo(image.dtype).max
                image = image.astype(np.float32)  # because torch does not have unsigned dtypes beyond torch.uint8
            image = MarigoldImageProcessor.numpy_to_pt(image)

        if torch.is_tensor(image) and not torch.is_floating_point(image) and image_dtype_max is None:
            if image.dtype != torch.uint8:
                raise ValueError(f"Image dtype={image.dtype} is not supported.")
            image_dtype_max = 255

        if not torch.is_tensor(image):
            raise ValueError(f"Input type unsupported: {type(image)}.")

        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)  # [N,1,H,W] -> [N,3,H,W]
        if image.shape[1] != 3:
            raise ValueError(f"Input image is not 1- or 3-channel: {image.shape}.")

        image = image.to(device=device, dtype=dtype)

        if image_dtype_max is not None:
            image = image / image_dtype_max

        return image

    @staticmethod
    def check_image_values_range(image: torch.Tensor) -> None:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.min().item() < 0.0 or image.max().item() > 1.0:
            raise ValueError("Input image data is partially outside of the [0,1] range.")

    def preprocess(
        self,
        image: PipelineImageInput,
        processing_resolution: Optional[int] = None,
        resample_method_input: str = "bilinear",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        if isinstance(image, list):
            images = None
            for i, img in enumerate(image):
                img = self.load_image_canonical(img, device, dtype)  # [N,3,H,W]
                if images is None:
                    images = img
                else:
                    if images.shape[2:] != img.shape[2:]:
                        raise ValueError(
                            f"Input image[{i}] has incompatible dimensions {img.shape[2:]} with the previous images "
                            f"{images.shape[2:]}"
                        )
                    images = torch.cat((images, img), dim=0)
            image = images
            del images
        else:
            image = self.load_image_canonical(image, device, dtype)  # [N,3,H,W]

        original_resolution = image.shape[2:]

        if self.config.do_range_check:
            self.check_image_values_range(image)

        if self.config.do_normalize:
            image = image * 2.0 - 1.0

        if processing_resolution is not None and processing_resolution > 0:
            image = self.resize_to_max_edge(image, processing_resolution, resample_method_input)  # [N,3,PH,PW]

        image, padding = self.pad_image(image, self.config.vae_scale_factor)  # [N,3,PPH,PPW]

        return image, padding, original_resolution

    @staticmethod
    def colormap(
        image: Union[np.ndarray, torch.Tensor],
        cmap: str = "Spectral",
        bytes: bool = False,
        _force_method: Optional[str] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Converts a monochrome image into an RGB image by applying the specified colormap. This function mimics the
        behavior of matplotlib.colormaps, but allows the user to use the most discriminative color maps ("Spectral",
        "binary") without having to install or import matplotlib. For all other cases, the function will attempt to use
        the native implementation.

        Args:
            image: 2D tensor of values between 0 and 1, either as np.ndarray or torch.Tensor.
            cmap: Colormap name.
            bytes: Whether to return the output as uint8 or floating point image.
            _force_method:
                Can be used to specify whether to use the native implementation (`"matplotlib"`), the efficient custom
                implementation of the select color maps (`"custom"`), or rely on autodetection (`None`, default).

        Returns:
            An RGB-colorized tensor corresponding to the input image.
        """
        if not (torch.is_tensor(image) or isinstance(image, np.ndarray)):
            raise ValueError("Argument must be a numpy array or torch tensor.")
        if _force_method not in (None, "matplotlib", "custom"):
            raise ValueError("_force_method must be either `None`, `'matplotlib'` or `'custom'`.")

        supported_cmaps = {
            "binary": [
                (1.0, 1.0, 1.0),
                (0.0, 0.0, 0.0),
            ],
            "Spectral": [  # Taken from matplotlib/_cm.py
                (0.61960784313725492, 0.003921568627450980, 0.25882352941176473),  # 0.0 -> [0]
                (0.83529411764705885, 0.24313725490196078, 0.30980392156862746),
                (0.95686274509803926, 0.42745098039215684, 0.2627450980392157),
                (0.99215686274509807, 0.68235294117647061, 0.38039215686274508),
                (0.99607843137254903, 0.8784313725490196, 0.54509803921568623),
                (1.0, 1.0, 0.74901960784313726),
                (0.90196078431372551, 0.96078431372549022, 0.59607843137254901),
                (0.6705882352941176, 0.8666666666666667, 0.64313725490196083),
                (0.4, 0.76078431372549016, 0.6470588235294118),
                (0.19607843137254902, 0.53333333333333333, 0.74117647058823533),
                (0.36862745098039218, 0.30980392156862746, 0.63529411764705879),  # 1.0 -> [K-1]
            ],
        }

        def method_matplotlib(image, cmap, bytes=False):
            import matplotlib

            arg_is_pt, device = torch.is_tensor(image), None
            if arg_is_pt:
                image, device = image.cpu().numpy(), image.device

            if cmap not in matplotlib.colormaps:
                raise ValueError(
                    f"Unexpected color map {cmap}; available options are: {', '.join(list(matplotlib.colormaps.keys()))}"
                )

            cmap = matplotlib.colormaps[cmap]
            out = cmap(image, bytes=bytes)  # [?,4]
            out = out[..., :3]  # [?,3]

            if arg_is_pt:
                out = torch.tensor(out, device=device)

            return out

        def method_custom(image, cmap, bytes=False):
            arg_is_np = isinstance(image, np.ndarray)
            if arg_is_np:
                image = torch.tensor(image)
            if image.dtype == torch.uint8:
                image = image.float() / 255
            else:
                image = image.float()

            is_cmap_reversed = cmap.endswith("_r")
            if is_cmap_reversed:
                cmap = cmap[:-2]

            if cmap not in supported_cmaps:
                raise ValueError(
                    f"Only {list(supported_cmaps.keys())} color maps are available without installing matplotlib."
                )

            cmap = supported_cmaps[cmap]
            if is_cmap_reversed:
                cmap = cmap[::-1]
            cmap = torch.tensor(cmap, dtype=torch.float, device=image.device)  # [K,3]
            K = cmap.shape[0]

            pos = image.clamp(min=0, max=1) * (K - 1)
            left = pos.long()
            right = (left + 1).clamp(max=K - 1)

            d = (pos - left.float()).unsqueeze(-1)
            left_colors = cmap[left]
            right_colors = cmap[right]

            out = (1 - d) * left_colors + d * right_colors

            if bytes:
                out = (out * 255).to(torch.uint8)

            if arg_is_np:
                out = out.numpy()

            return out

        if _force_method is None and torch.is_tensor(image) and cmap == "Spectral":
            return method_custom(image, cmap, bytes)

        out = None
        if _force_method != "custom":
            out = method_matplotlib(image, cmap, bytes)

        if _force_method == "matplotlib" and out is None:
            raise ImportError("Make sure to install matplotlib if you want to use a color map other than 'Spectral'.")

        if out is None:
            out = method_custom(image, cmap, bytes)

        return out

    @staticmethod
    def visualize_depth(
        depth: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.Tensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.Tensor],
        ],
        val_min: float = 0.0,
        val_max: float = 1.0,
        color_map: str = "Spectral",
    ) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        """
        Visualizes depth maps, such as predictions of the `MarigoldDepthPipeline`.

        Args:
            depth (`Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image], List[np.ndarray],
                List[torch.Tensor]]`): Depth maps.
            val_min (`float`, *optional*, defaults to `0.0`): Minimum value of the visualized depth range.
            val_max (`float`, *optional*, defaults to `1.0`): Maximum value of the visualized depth range.
            color_map (`str`, *optional*, defaults to `"Spectral"`): Color map used to convert a single-channel
                      depth prediction into colored representation.

        Returns: `PIL.Image.Image` or `List[PIL.Image.Image]` with depth maps visualization.
        """
        if val_max <= val_min:
            raise ValueError(f"Invalid values range: [{val_min}, {val_max}].")

        def visualize_depth_one(img, idx=None):
            prefix = "Depth" + (f"[{idx}]" if idx else "")
            if isinstance(img, PIL.Image.Image):
                if img.mode != "I;16":
                    raise ValueError(f"{prefix}: invalid PIL mode={img.mode}.")
                img = np.array(img).astype(np.float32) / (2**16 - 1)
            if isinstance(img, np.ndarray) or torch.is_tensor(img):
                if img.ndim != 2:
                    raise ValueError(f"{prefix}: unexpected shape={img.shape}.")
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                if not torch.is_floating_point(img):
                    raise ValueError(f"{prefix}: unexected dtype={img.dtype}.")
            else:
                raise ValueError(f"{prefix}: unexpected type={type(img)}.")
            if val_min != 0.0 or val_max != 1.0:
                img = (img - val_min) / (val_max - val_min)
            img = MarigoldImageProcessor.colormap(img, cmap=color_map, bytes=True)  # [H,W,3]
            img = PIL.Image.fromarray(img.cpu().numpy())
            return img

        if depth is None or isinstance(depth, list) and any(o is None for o in depth):
            raise ValueError("Input depth is `None`")
        if isinstance(depth, (np.ndarray, torch.Tensor)):
            depth = MarigoldImageProcessor.expand_tensor_or_array(depth)
            if isinstance(depth, np.ndarray):
                depth = MarigoldImageProcessor.numpy_to_pt(depth)  # [N,H,W,1] -> [N,1,H,W]
            if not (depth.ndim == 4 and depth.shape[1] == 1):  # [N,1,H,W]
                raise ValueError(f"Unexpected input shape={depth.shape}, expecting [N,1,H,W].")
            return [visualize_depth_one(img[0], idx) for idx, img in enumerate(depth)]
        elif isinstance(depth, list):
            return [visualize_depth_one(img, idx) for idx, img in enumerate(depth)]
        else:
            raise ValueError(f"Unexpected input type: {type(depth)}")

    @staticmethod
    def export_depth_to_16bit_png(
        depth: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        val_min: float = 0.0,
        val_max: float = 1.0,
    ) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        def export_depth_to_16bit_png_one(img, idx=None):
            prefix = "Depth" + (f"[{idx}]" if idx else "")
            if not isinstance(img, np.ndarray) and not torch.is_tensor(img):
                raise ValueError(f"{prefix}: unexpected type={type(img)}.")
            if img.ndim != 2:
                raise ValueError(f"{prefix}: unexpected shape={img.shape}.")
            if torch.is_tensor(img):
                img = img.cpu().numpy()
            if not np.issubdtype(img.dtype, np.floating):
                raise ValueError(f"{prefix}: unexected dtype={img.dtype}.")
            if val_min != 0.0 or val_max != 1.0:
                img = (img - val_min) / (val_max - val_min)
            img = (img * (2**16 - 1)).astype(np.uint16)
            img = PIL.Image.fromarray(img, mode="I;16")
            return img

        if depth is None or isinstance(depth, list) and any(o is None for o in depth):
            raise ValueError("Input depth is `None`")
        if isinstance(depth, (np.ndarray, torch.Tensor)):
            depth = MarigoldImageProcessor.expand_tensor_or_array(depth)
            if isinstance(depth, np.ndarray):
                depth = MarigoldImageProcessor.numpy_to_pt(depth)  # [N,H,W,1] -> [N,1,H,W]
            if not (depth.ndim == 4 and depth.shape[1] == 1):
                raise ValueError(f"Unexpected input shape={depth.shape}, expecting [N,1,H,W].")
            return [export_depth_to_16bit_png_one(img[0], idx) for idx, img in enumerate(depth)]
        elif isinstance(depth, list):
            return [export_depth_to_16bit_png_one(img, idx) for idx, img in enumerate(depth)]
        else:
            raise ValueError(f"Unexpected input type: {type(depth)}")

    @staticmethod
    def visualize_normals(
        normals: Union[
            np.ndarray,
            torch.Tensor,
            List[np.ndarray],
            List[torch.Tensor],
        ],
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
    ) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        """
        Visualizes surface normals, such as predictions of the `MarigoldNormalsPipeline`.

        Args:
            normals (`Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]`):
                Surface normals.
            flip_x (`bool`, *optional*, defaults to `False`): Flips the X axis of the normals frame of reference.
                      Default direction is right.
            flip_y (`bool`, *optional*, defaults to `False`):  Flips the Y axis of the normals frame of reference.
                      Default direction is top.
            flip_z (`bool`, *optional*, defaults to `False`): Flips the Z axis of the normals frame of reference.
                      Default direction is facing the observer.

        Returns: `PIL.Image.Image` or `List[PIL.Image.Image]` with surface normals visualization.
        """
        flip_vec = None
        if any((flip_x, flip_y, flip_z)):
            flip_vec = torch.tensor(
                [
                    (-1) ** flip_x,
                    (-1) ** flip_y,
                    (-1) ** flip_z,
                ],
                dtype=torch.float32,
            )

        def visualize_normals_one(img, idx=None):
            img = img.permute(1, 2, 0)
            if flip_vec is not None:
                img *= flip_vec.to(img.device)
            img = (img + 1.0) * 0.5
            img = (img * 255).to(dtype=torch.uint8, device="cpu").numpy()
            img = PIL.Image.fromarray(img)
            return img

        if normals is None or isinstance(normals, list) and any(o is None for o in normals):
            raise ValueError("Input normals is `None`")
        if isinstance(normals, (np.ndarray, torch.Tensor)):
            normals = MarigoldImageProcessor.expand_tensor_or_array(normals)
            if isinstance(normals, np.ndarray):
                normals = MarigoldImageProcessor.numpy_to_pt(normals)  # [N,3,H,W]
            if not (normals.ndim == 4 and normals.shape[1] == 3):
                raise ValueError(f"Unexpected input shape={normals.shape}, expecting [N,3,H,W].")
            return [visualize_normals_one(img, idx) for idx, img in enumerate(normals)]
        elif isinstance(normals, list):
            return [visualize_normals_one(img, idx) for idx, img in enumerate(normals)]
        else:
            raise ValueError(f"Unexpected input type: {type(normals)}")

    @staticmethod
    def visualize_uncertainty(
        uncertainty: Union[
            np.ndarray,
            torch.Tensor,
            List[np.ndarray],
            List[torch.Tensor],
        ],
        saturation_percentile=95,
    ) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        """
        Visualizes dense uncertainties, such as produced by `MarigoldDepthPipeline` or `MarigoldNormalsPipeline`.

        Args:
            uncertainty (`Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]`):
                Uncertainty maps.
            saturation_percentile (`int`, *optional*, defaults to `95`):
                Specifies the percentile uncertainty value visualized with maximum intensity.

        Returns: `PIL.Image.Image` or `List[PIL.Image.Image]` with uncertainty visualization.
        """

        def visualize_uncertainty_one(img, idx=None):
            prefix = "Uncertainty" + (f"[{idx}]" if idx else "")
            if img.min() < 0:
                raise ValueError(f"{prefix}: unexected data range, min={img.min()}.")
            img = img.squeeze(0).cpu().numpy()
            saturation_value = np.percentile(img, saturation_percentile)
            img = np.clip(img * 255 / saturation_value, 0, 255)
            img = img.astype(np.uint8)
            img = PIL.Image.fromarray(img)
            return img

        if uncertainty is None or isinstance(uncertainty, list) and any(o is None for o in uncertainty):
            raise ValueError("Input uncertainty is `None`")
        if isinstance(uncertainty, (np.ndarray, torch.Tensor)):
            uncertainty = MarigoldImageProcessor.expand_tensor_or_array(uncertainty)
            if isinstance(uncertainty, np.ndarray):
                uncertainty = MarigoldImageProcessor.numpy_to_pt(uncertainty)  # [N,1,H,W]
            if not (uncertainty.ndim == 4 and uncertainty.shape[1] == 1):
                raise ValueError(f"Unexpected input shape={uncertainty.shape}, expecting [N,1,H,W].")
            return [visualize_uncertainty_one(img, idx) for idx, img in enumerate(uncertainty)]
        elif isinstance(uncertainty, list):
            return [visualize_uncertainty_one(img, idx) for idx, img in enumerate(uncertainty)]
        else:
            raise ValueError(f"Unexpected input type: {type(uncertainty)}")


class DepthScaler:
    def __init__(self,
        normalization: str = 'multi_quantiles',
        expansion: str = 'repeat',
        min_scale: float = 1.0,
        max_scale: float = 1000.0,
        min_depth_percent: Optional[float] = None,
        max_depth_percent: float = 0.98,
        eps: float = 1e-6,
        use_log_scale: bool = False,
        **kwargs,
    ) -> None:
        self.normalization = normalization
        self.expansion = expansion
        self.use_log_scale = use_log_scale

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_log_scale = np.log(min_scale)
        self.max_log_scale = np.log(max_scale)
        assert min_scale > 0.0, "min_scale should be greater than 0.0, otherwise it will cause division by zero."

        self.min_depth_percent = min_depth_percent
        self.max_depth_percent = max_depth_percent
        self.eps = eps
    
    @staticmethod
    def get_max_depth(valid_depth: torch.Tensor, normalization: str):
        """
        Args:
            valid_depth: torch.FloatTensor, [N], [0.0, +inf], float32
        Returns:
            max_depth: float, [0.0, +inf]
        """
        valid_depth = valid_depth.float()

        if normalization == 'multi_quantiles':
            max_depth, max_depth_list = compute_max_depth_with_dynamic_multipliers(
                valid_depth=valid_depth,
                quantiles=[0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98],
                # quantiles=[0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98],
                # quantiles=[0.02, 0.25, 0.50, 0.75, 0.98],
            )
            max_depth = min(max_depth, valid_depth.max().item())
        
        elif normalization == 'jump_point':
            valid_depth, _ = valid_depth.sort()
            diffs = torch.abs(valid_depth[1:] - valid_depth[:-1])
            max_diff_idx = torch.argmax(diffs)
            max_depth = valid_depth[max_diff_idx]
        
        elif normalization == 'median':
            max_depth = torch.median(valid_depth).item() * 4
            max_depth = min(max_depth, valid_depth.max().item())
        
        elif normalization == 'marigold':
            max_depth = torch.quantile(valid_depth, 0.98).item()
        
        else:
            raise NotImplementedError(f"Invalid depth normalization mode: {normalization}")

        return max_depth
        
    def scale(self, depths: torch.Tensor, infinite_masks: Optional[torch.BoolTensor] = None, **kwargs):
        """
        Args:
            depths: torch.FloatTensor, [..., 1, H, W], [0.0, +inf], meters
            infinite_masks: torch.BoolTensor, [..., 1, H, W], bool
        Returns:
            depths: torch.FloatTensor, [..., C, H, W], C=3, [0.0, 1.0]
        """
        assert depths.shape[0] == 1, "Currently only support batch size of 1."

        if infinite_masks is None:
            infinite_masks = torch.isposinf(depths) | (depths < 0.0)
        finite_masks = torch.logical_not(infinite_masks)

        # Normalization
        valid_depth = depths[finite_masks]
        if valid_depth.nelement() <= 1:
            depths = valid_depth = torch.zeros_like(depths)
        valid_depth = valid_depth.float()  # ensure float32 for quantile calculation

        # Logarithmic Scaling
        if self.use_log_scale:
            valid_depth = torch.log(valid_depth + 1.0)

        # Normalization
        min_depth = 0.0
        max_depth = self.get_max_depth(valid_depth, normalization=self.normalization)

        max_depth = max(max_depth, self.min_scale)
        normalized_depths = depths.clip(min=min_depth, max=max_depth)
        normalized_depths = (normalized_depths - min_depth) / (max_depth - min_depth + self.eps)
        
        # Expansion
        if self.expansion == 'repeat':
            packed_depths = repeat(normalized_depths, '... 1 h w -> ... 3 h w')
        
        elif self.expansion == 'scale_and_mask':
            if self.use_log_scale:
                scale = np.exp(max_depth) - np.exp(min_depth)
            else:
                scale = max_depth - min_depth
            log_scale = np.log(scale + 1.0)
            normalized_log_scale = (log_scale - self.min_log_scale) / (self.max_log_scale - self.min_log_scale)
            scales = torch.full_like(normalized_depths, normalized_log_scale)
            masks = infinite_masks.to(dtype=normalized_depths.dtype)
            packed_depths = torch.cat([normalized_depths, scales, masks], dim=-3)
        
        elif self.expansion == 'orthogonal':
            placeholder = torch.full_like(normalized_depths[..., [0], :, :], fill_value=0.5)  # [..., 1, H, W]
            packed_depths = torch.cat([normalized_depths, placeholder], dim=-3)
    
        else:
            raise NotImplementedError(f"Invalid depth expansion mode: {self.expansion}")
        
        unscale_kwargs = {
            'min_depth': min_depth,
            'max_depth': max_depth,
            'infinite_masks': infinite_masks.detach(),
            'use_log_scale': self.use_log_scale,
        }
        return packed_depths, unscale_kwargs

    def unscale(self, depths: torch.Tensor, **kwargs):
        """
        Args:
            depths: torch.FloatTensor, [..., C, H, W], C=3, [0.0, 1.0]
        Returns:
            depths: torch.FloatTensor, [..., 1, H, W], millimeters (mm)
            inf_masks: torch.BoolTensor, [..., 1, H, W]
        """
        extra_outputs = {}
        extra_outputs['use_log_scale'] = self.use_log_scale
        
        assert depths.shape[0] == 1, "Currently only support batch size of 1."

        # Expansion
        if self.expansion == 'repeat':
            depths = depths.mean(dim=-3, keepdim=True)
            if self.use_log_scale:
                depths = torch.exp(depths) - 1.0

            min_depth = kwargs.get('min_depth', 0.0)
            max_depth = kwargs.get('max_depth', 1.0)
            unscaled_depths = depths * (max_depth - min_depth) + min_depth
            
            extra_outputs['min_depth'] = min_depth
            extra_outputs['max_depth'] = max_depth
            extra_outputs['normalized_depths'] = depths.detach()
        
        elif self.expansion == 'scale_and_mask':
            depths, scales, masks = depths.split([1, 1, 1], dim=-3)

            if self.use_log_scale:
                depths = torch.exp(depths) - 1.0

            log_scale = scales.mean() * (self.max_log_scale - self.min_log_scale) + self.min_log_scale
            scale = torch.exp(log_scale).item() - 1.0

            infinite_masks = masks > 0.5

            min_depth, max_depth = 0.0, scale

            unscaled_depths = depths * (max_depth - min_depth) + min_depth
            unscaled_depths[infinite_masks] = float('inf')
            
            extra_outputs['min_depth'] = min_depth
            extra_outputs['max_depth'] = max_depth
            extra_outputs['normalized_depths'] = depths.detach()
            extra_outputs['scales'] = scales.detach()
            extra_outputs['infinite_masks'] = infinite_masks.detach()
        
        elif self.expansion == 'orthogonal':
            horizontal_dists, vertical_dists, _ = depths.split([1, 1, 1], dim=-3)
            depths = torch.sqrt(horizontal_dists**2 + vertical_dists**2)

            min_depth = kwargs.get('min_depth', 0.0)
            max_depth = kwargs.get('max_depth', 1.0)
            unscaled_depths = depths * (max_depth - min_depth) + min_depth
            
            extra_outputs['min_depth'] = min_depth
            extra_outputs['max_depth'] = max_depth
            extra_outputs['normalized_depths'] = depths.detach()
        
        else:
            raise NotImplementedError(f"Invalid depth expansion mode: {self.expansion}")
        
        return unscaled_depths, extra_outputs

    def clip(self, depths: torch.Tensor, infinite_masks: Optional[torch.BoolTensor] = None, min_depth: float = 0.0, max_depth: Optional[float] = None):
        """
        Args:
            depths: torch.FloatTensor, [..., 1, H, W], [0.0, +inf]
            infinite_masks: torch.BoolTensor, [..., 1, H, W], bool
        Returns:
            clipped_depths: torch.FloatTensor, [..., 1, H, W], [0.0, max_clip]
            valid_masks: torch.BoolTensor, [..., 1, H, W], bool
        """
        if max_depth is None:
            if infinite_masks is None:
                infinite_masks = torch.isposinf(depths) | (depths < 0.0)
            finite_masks = torch.logical_not(infinite_masks)

            # Normalization
            valid_depth = depths[finite_masks]
            if valid_depth.nelement() <= 1:
                depths = valid_depth = torch.zeros_like(depths)
            valid_depth = valid_depth.float()  # ensure float32 for quantile calculation
            
            max_depth = self.get_max_depth(valid_depth, normalization=self.normalization)

        valid_masks = torch.logical_and(depths >= min_depth, depths <= max_depth)

        clipped_depths = depths.clip(min=min_depth, max=max_depth)
        
        return clipped_depths, valid_masks

    def normalize(self, depths: torch.Tensor, infinite_masks: Optional[torch.BoolTensor] = None, min_depth: float = 0.0, max_depth: Optional[float] = None):
        """
        Args:
            depths: torch.FloatTensor, [..., 1, H, W], [0.0, +inf]
            infinite_masks: torch.BoolTensor, [..., 1, H, W], bool
        Returns:
            clipped_depths: torch.FloatTensor, [..., 1, H, W], [0.0, max_clip]
            valid_masks: torch.BoolTensor, [..., 1, H, W], bool
        """
        if max_depth is None:
            if infinite_masks is None:
                infinite_masks = torch.isposinf(depths) | (depths < 0.0)
            finite_masks = torch.logical_not(infinite_masks)

            # Normalization
            valid_depth = depths[finite_masks]
            if valid_depth.nelement() <= 1:
                depths = valid_depth = torch.zeros_like(depths)
            valid_depth = valid_depth.float()  # ensure float32 for quantile calculation
            
            max_depth = self.get_max_depth(valid_depth, normalization=self.normalization)

        valid_masks = torch.logical_and(depths >= min_depth, depths <= max_depth)

        clipped_depths = depths.clip(min=min_depth, max=max_depth)
        normalized_depths = (clipped_depths - min_depth) / (max_depth - min_depth + self.eps)
        
        return normalized_depths, valid_masks


def depth_to_orthogonal_distance(distance: torch.Tensor, camray: torch.Tensor) -> torch.Tensor:
    """
    计算每个像素到相机中心点的水平和垂直距离。
    
    参数：
        distance (torch.Tensor): 深度值，形状为 [..., 1, H, W]，表示每个像素到相机的距离。
        camray (torch.Tensor): 相机射线方向，形状为 [..., 3, H, W]，应为单位向量。
        eps (float): 小的数值，用于避免除零错误。
    
    返回：
        torch.Tensor: 每个像素的水平和垂直距离，形状与输入的距离相同。
    """
    # 无穷远区域
    inf_masks = torch.isposinf(distance)  # [..., 1, H, W]

    # 计算水平分量的模长
    horizontal_component = camray[..., [0, 2], :, :]  # [..., 2, H, W]
    horizontal_norm = torch.norm(horizontal_component, dim=-3, keepdim=True)  # [..., 1, H, W]
    
    # 计算垂直分量的模长
    vertical_norm = torch.abs(camray[..., [1], :, :])  # [..., 1, H, W]

    # 计算水平距离：深度乘以水平分量的模长
    horizontal_distance = distance * horizontal_norm
    vertical_distance = distance * vertical_norm

    # 无穷远区域设为无穷大
    horizontal_distance[inf_masks] = float('inf')
    vertical_distance[inf_masks] = float('inf')
    
    orthogonal_distance = torch.cat([horizontal_distance, vertical_distance], dim=-3)  # [..., 2, H, W]
    return orthogonal_distance


def orthogonal_distance_to_depth(orthogonal_distance: torch.Tensor) -> torch.Tensor:
    """
    根据水平距离、垂直距离和相机射线方向，计算对应的深度（沿射线方向的距离）。
    
    参数：
    - orthogonal_distance (torch.Tensor): 水平和垂直距离，形状为 [..., 2, H, W]。
    
    返回：
    - torch.Tensor: 深度（沿射线方向的距离），与输入形状相同。
    """
    horizontal_distance, vertical_distance = orthogonal_distance.split([1, 1], dim=-3)  # [..., 1, H, W], [..., 1, H, W]
    depth = torch.sqrt(horizontal_distance**2 + vertical_distance**2)  # [..., 1, H, W]
    return depth


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)


def detect_depth_jumps(depth_map, gradient_threshold=0.3):
    """
    检测深度图中的跳变像素点。

    Parameters:
        depth_map (ndarray): [H, W] 浮点型深度图
        gradient_threshold (float): 阈值，用于判断跳变的敏感度（根据具体数据调节）

    Returns:
        mask (ndarray): [H, W] 布尔类型的掩码，跳变点为True
    """
    # 计算深度的梯度
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    magnitude = np.abs(grad_x) + np.abs(grad_y)

    # 根据阈值判断跳变点
    mask = magnitude > gradient_threshold

    return mask


def smooth_depth(depth_map, d=9, sigma_color=75, sigma_space=75):
    """
    对深度图进行平滑处理，同时保持边缘结构。

    Parameters:
        depth_map (ndarray): [H, W] 浮点型深度图
        d (int): 双边滤波中邻域的直径（越大越平滑）
        sigma_color (float): 颜色空间的标准差（影响颜色差异的权重）
        sigma_space (float): 空间空间的标准差（影响空间距离的权重）

    Returns:
        smoothed_depth (ndarray): 平滑后的深度图
    """
    # 将深度图转换为8位图（可选，根据深度范围调整）
    # 这里我们假设深度在某个范围内，先归一化到[0,255]
    depth_min = np.nanmin(depth_map)
    depth_max = np.nanmax(depth_map)
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    # 使用双边滤波
    smoothed_uint8 = cv2.bilateralFilter(depth_uint8, d, sigma_color, sigma_space)

    # 转回浮点深度值
    smoothed_norm = smoothed_uint8.astype(np.float32) / 255.0
    smoothed_depth = smoothed_norm * (depth_max - depth_min) + depth_min

    # 处理NaN值（保持原样或用插值填充）
    smoothed_depth[np.isnan(depth_map)] = np.nan

    return smoothed_depth


def fill_invalid_depth(depth_map, mask, method='linear'):
    """
    使用插值填充深度图中无效像素。

    Args:
        depth_map (ndarray): [H, W]，浮点型深度图。
        mask (ndarray): [H, W]，布尔类型，True表示像素无效（需要填充）。
        method (str): 插值方法，可选'linear', 'nearest', 'cubic'。

    Returns:
        ndarray: 填充后的深度图，类型与输入相同。
    """
    from scipy.interpolate import griddata

    H, W = depth_map.shape

    # 获取有效像素的坐标和值
    valid_mask = ~mask  # True为有效
    valid_y, valid_x = np.where(valid_mask)
    valid_depth = depth_map[valid_mask]

    # 获取无效像素的坐标
    invalid_y, invalid_x = np.where(mask)

    # 如果全部无效或全部有效，直接返回原图
    if len(valid_depth) == 0:
        # print("没有有效像素，无法插值。")
        return depth_map
    if len(valid_depth) == H*W:
        # print("所有像素都是有效的，无需填充。")
        return depth_map

    # 进行插值
    points = np.stack((valid_x, valid_y), axis=-1)
    points_invalid = np.stack((invalid_x, invalid_y), axis=-1)

    # 使用griddata进行插值
    filled_values = griddata(points, valid_depth, points_invalid, method=method)

    # 创建新深度图
    new_depth = depth_map.copy()

    # 填充无效像素
    new_depth[invalid_y, invalid_x] = filled_values

    return new_depth