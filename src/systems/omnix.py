import sys
sys.path.append('external')
import torch
from typing import Optional, Any
import torch.nn.functional as F
from einops import rearrange
from diffusers.utils.import_utils import is_torch_npu_available
from PIL import Image
import numpy as np
from external import equilib

from src.utils.depth import DepthVisualizer
from src.pipelines.pipeline_omnix import OmniXPipeline


def get_device():
    if is_torch_npu_available():
        return 'npu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


class OmniX():
    def __init__(self, hf_repo: str = 'KevinHuang/OmniX', device=None, dtype=torch.bfloat16, enable_model_cpu_offload=False) -> None:
        self.hf_repo = hf_repo
        self.device = get_device() if device is None else device
        self.dtype = dtype

        pipe: OmniXPipeline = OmniXPipeline.from_pretrained(
            'black-forest-labs/FLUX.1-dev',
            torch_dtype=self.dtype,
        )
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        else:
            pipe.to(self.device)

        self.pipe = pipe
        self.current_task = None

    def _perceive_panoramic_X(self,
        panorama: Image.Image,
        X: str,
        max_height: int = 720,
        max_width: int = 1440,
        num_inference_steps: int = 50,
        output_type: str = 'pil',
    ) -> Any:
        """
        Returns:
            PIL Image, RGB, [0, 255]                   if output_type is 'pil'
            numpy array, [H, W, C], [0.0, 1.0]         if output_type is 'np'
            torch tensor, [C, H, W], [0.0, 1.0]        if output_type is 'pt'
        """
        height, width = panorama.size[1], panorama.size[0]
        assert height * 2 == width, "Width must be twice the height for equirectangular panorama."
        assert output_type in ['pil', 'np', 'pt'], "output_type must be one of ['pil', 'np', 'pt']"

        self.set_task(f'rgb_to_{X}')
        
        if height > max_height or width > max_width:
            panorama = panorama.resize((max_width, max_height), Image.LANCZOS)
        
        conditions = {'rgb': self.get_condition(panorama)}
        augmented_prompt = self.pipe.get_camera_specific_prompt()

        return self.pipe(
            prompt=augmented_prompt,
            conditions=conditions,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=len(self.pipe.lora_names),
            height=height,
            width=width,
            output_type=output_type,
        ).images[self.pipe.lora_target_index]

    def get_masked_panorama(self,
        image: Image.Image,
        height: int,
        width: int,
        fov_x: float = 90.0,
        output_type: str = 'pt',
    ) -> torch.Tensor:
        pers = np.array(image.convert('RGB')).astype(np.float32)
        pers = rearrange(pers, 'h w c -> 1 c h w')
        rots = [{'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}]
        equi = equilib.pers2equi(pers, rots=rots, height=height, width=width, fov_x=fov_x, clip_output=False)
        mask = equilib.pers2equi(np.ones_like(pers), rots=rots, height=height, width=width, fov_x=fov_x, clip_output=False)
        equi[mask < 0.5] = 127.5  # set to gray
        if output_type == 'pil':
            equi = (np.clip(equi[0].transpose(1, 2, 0), 0, 255).astype(np.uint8))
            return Image.fromarray(equi)
        elif output_type == 'np':
            return (equi / 255.0)[0].transpose(1, 2, 0)
        elif output_type == 'pt':
            equi = torch.tensor((equi / 127.5) - 1.0, device=self.device, dtype=self.dtype)
            return equi

    def get_condition(self, image: Image.Image):
        image = image.convert('RGB')
        image = torch.tensor(np.array(image).astype(np.float32) / 127.5 - 1.0, device=self.device, dtype=self.dtype)
        image = rearrange(image, 'h w c -> 1 c h w')
        return image

    def set_task(self, task: str):
        if self.current_task is not None and self.current_task == task:
            return
        self.pipe.load_task_specific_adapters(task, hf_repo=self.hf_repo)
        self.current_task = task

    def generate_image(self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        output_type: str = 'pil',
    ) -> Any:
        self.set_task('text_to_image')
        return self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            output_type=output_type,
        ).images[0]

    def generate_panorama(self,
        prompt: Optional[str] = None,
        image: Optional[Image.Image] = None,
        height: int = 512,
        width: int = 1024,
        image_fov_x: float = 90.0,
        num_inference_steps: int = 50,
        output_type: str = 'pil',
    ) -> Any:
        """
        Returns:
            PIL Image, RGB, [0, 255]                   if output_type is 'pil'
            numpy array, [H, W, C], [0.0, 1.0]         if output_type is 'np'
            torch tensor, [C, H, W], [0.0, 1.0]        if output_type is 'pt'
        """
        assert height * 2 == width, "Width must be twice the height for equirectangular panorama."
        assert output_type in ['pil', 'np', 'pt'], "output_type must be one of ['pil', 'np', 'pt']"

        if image is None:
            assert prompt is not None, "Either image or prompt must be provided."
            image_hw = height
            image = self.generate_image(prompt=prompt, height=image_hw, width=image_hw, num_inference_steps=num_inference_steps, output_type='pil')
        
        self.set_task('image_to_pano')
        conditions = {'masked_rgb': self.get_masked_panorama(image, height=height, width=width, fov_x=image_fov_x)}
        augmented_prompt = self.pipe.get_camera_specific_prompt(prompt)

        return self.pipe(
            prompt=augmented_prompt,
            conditions=conditions,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=len(self.pipe.lora_names),
            height=height,
            width=width,
            output_type=output_type,
        ).images[self.pipe.lora_target_index]

    def perceive_panoramic_albedo(self,
        panorama: Image.Image,
        max_height: int = 720,
        max_width: int = 1440,
        num_inference_steps: int = 50,
        output_type: str = 'pil',
    ) -> Any:
        """
        Returns:
            PIL Image, RGB, [0, 255]                   if output_type is 'pil'
            numpy array, [H, W, 3], [0.0, 1.0]         if output_type is 'np'
            torch tensor, [3, H, W], [0.0, 1.0]        if output_type is 'pt'
        """
        return self._perceive_panoramic_X(
            panorama=panorama,
            X='albedo',
            max_height=max_height,
            max_width=max_width,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
        )

    def perceive_panoramic_depth(self,
        panorama: Image.Image,
        max_height: int = 720,
        max_width: int = 1440,
        num_inference_steps: int = 50,
        output_type: str = 'pil',
        colormap: Optional[str] = 'Spectral',
    ) -> Any:
        """
        Returns:
            PIL Image, RGB [0, 255] | L [0, 65535]    if output_type is 'pil'
            numpy array, [H, W], [0.0, 1.0]           if output_type is 'np'
            torch tensor, [H, W], [0.0, 1.0]          if output_type is 'pt'
        """
        depth = self._perceive_panoramic_X(
            panorama=panorama,
            X='depth',
            max_height=max_height,
            max_width=max_width,
            num_inference_steps=num_inference_steps,
            output_type='pt',
        ).mean(dim=0)  # [H, W]
        if output_type == 'pt':
            return depth
        elif output_type == 'np':
            return depth.float().cpu().numpy()
        elif output_type == 'pil':
            depth = depth.float().cpu().numpy()
            if colormap is None:
                return Image.fromarray((np.clip(depth * 65535, 0, 65535).astype(np.uint16)))
            else:
                return DepthVisualizer.visualize_depth(depth)[0]

    def perceive_panoramic_normal(self,
        panorama: Image.Image,
        max_height: int = 720,
        max_width: int = 1440,
        num_inference_steps: int = 50,
        output_type: str = 'pil',
        postprocess: bool = False,
    ) -> Any:
        """
        Returns:
            PIL Image, RGB, [0, 255]                   if output_type is 'pil'
            numpy array, [H, W, 3], [0.0, 1.0]         if output_type is 'np'
            torch tensor, [3, H, W], [0.0, 1.0]        if output_type is 'pt'
        """
        normal = self._perceive_panoramic_X(
            panorama=panorama,
            X='normal',
            max_height=max_height,
            max_width=max_width,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
        )
        if postprocess:
            raise NotImplementedError("Normal postprocessing is not implemented yet.")
        return normal

    def perceive_panoramic_pbr(self,
        panorama: Image.Image,
        max_height: int = 720,
        max_width: int = 1440,
        num_inference_steps: int = 50,
        output_type: str = 'pil',
    ) -> Any:
        """
        Returns:
            - roughness: PIL Image, L, [0, 255]                   if output_type is 'pil'
                         numpy array, [H, W], [0.0, 1.0]         if output_type is 'np'
                         torch tensor, [H, W], [0.0, 1.0]          if output_type is 'pt'
            
            - metallic: PIL Image, L, [0, 255]                    if output_type is 'pil'
                        numpy array, [H, W], [0.0, 1.0]         if output_type is 'np'
                        torch tensor, [H, W], [0.0, 1.0]          if output_type is 'pt'
        """
        pbr_material = self._perceive_panoramic_X(
            panorama=panorama,
            X='pbr',
            max_height=max_height,
            max_width=max_width,
            num_inference_steps=num_inference_steps,
            output_type='pt',
        )
        roughness, metallic = pbr_material[0], pbr_material[1]  # [H, W]
        if output_type == 'pt':
            return roughness, metallic
        elif output_type == 'np':
            roughness = roughness.float().cpu().numpy()
            metallic = metallic.float().cpu().numpy()
            return roughness, metallic
        elif output_type == 'pil':
            roughness = roughness.float().cpu().numpy()
            metallic = metallic.float().cpu().numpy()
            roughness_pil = Image.fromarray((np.clip(roughness * 255, 0, 255).astype(np.uint8)), mode='L')
            metallic_pil = Image.fromarray((np.clip(metallic * 255, 0, 255).astype(np.uint8)), mode='L')
            return roughness_pil, metallic_pil

    def perceive_panoramic_semantic(self,
        panorama: Image.Image,
        max_height: int = 720,
        max_width: int = 1440,
        num_inference_steps: int = 50,
        output_type: str = 'pil',
    ) -> Any:
        """
        Returns:
            PIL Image, RGB, [0, 255]                   if output_type is 'pil'
            numpy array, [H, W, 3], [0.0, 1.0]         if output_type is 'np'
            torch tensor, [3, H, W], [0.0, 1.0]        if output_type is 'pt'
        """
        return self._perceive_panoramic_X(
            panorama=panorama,
            X='semantic',
            max_height=max_height,
            max_width=max_width,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
        )
