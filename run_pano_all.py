import os
import os.path as osp
import argparse
from PIL import Image
from typing import Optional
import torch
from diffusers.utils.import_utils import is_torch_npu_available
if is_torch_npu_available():
    import torch_npu

from src.systems.omnix import OmniX
from src.utils.image import stitch_images


def run_generation(image: Optional[str] = None, prompt: Optional[str] = None):
    # Image Generation
    if image is None:
        print('Generating image from prompt...')
        image = omnix.generate_image(prompt=prompt, height=height, width=height)
    else:
        print('Loading input image...')
        image = Image.open(image).convert('RGB')
    
    image.save(osp.join(output_dir, 'input_image.png'))
    print(f'Input image saved to {output_dir}')

    # Save Masked Input for Visualization Only
    masked_input = omnix.get_masked_panorama(image, height=height, width=width, output_type='pil')  # For visualization only
    masked_input.save(osp.join(output_dir, 'input_masked_panorama.png'))
    print(f'Input masked panorama saved to {output_dir}')

    # Panorama Generation
    print('Generating panorama from image...')
    panorama = omnix.generate_panorama(image=image, prompt=prompt, height=height, width=width)
    
    # Save Output Panorama
    panorama.save(osp.join(output_dir, 'output_panorama.png'))
    print(f'Output panorama saved to {output_dir}')

    return panorama, masked_input


def run_perception(panorama: Image.Image):
    # Panorama Perception - Albedo
    print('Perceiving panoramic albedo...')
    albedo = omnix.perceive_panoramic_albedo(panorama, num_inference_steps=num_inference_steps)
    albedo.save(osp.join(output_dir, 'output_albedo.png'))
    print(f'Panoramic albedo saved to {output_dir}')

    # Panorama Perception - Depth
    print('Perceiving panoramic depth...')
    depth = omnix.perceive_panoramic_depth(panorama, num_inference_steps=num_inference_steps)
    depth.save(osp.join(output_dir, 'output_depth.png'))
    print(f'Panoramic depth saved to {output_dir}')
    
    # Panorama Perception - Normal
    print('Perceiving panoramic normal...')
    normal = omnix.perceive_panoramic_normal(panorama, num_inference_steps=num_inference_steps)
    normal.save(osp.join(output_dir, 'output_normal.png'))
    print(f'Panoramic normal saved to {output_dir}')
    
    # Panorama Perception - PBR Material (Roughness, Metallic)
    print('Perceiving panoramic PBR material...')
    roughness, metallic = omnix.perceive_panoramic_pbr(panorama, num_inference_steps=num_inference_steps)
    roughness.save(osp.join(output_dir, 'output_roughness.png'))
    metallic.save(osp.join(output_dir, 'output_metallic.png'))
    print(f'Panoramic PBR material saved to {output_dir}')
    
    # Panorama Perception - Semantic
    print('Perceiving panoramic semantic...')
    semantic = omnix.perceive_panoramic_semantic(panorama, num_inference_steps=num_inference_steps)
    semantic.save(osp.join(output_dir, 'output_semantic.png'))
    print(f'Panoramic semantic saved to {output_dir}')
    
    return albedo, depth, normal, roughness, metallic, semantic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--panorama', type=str, default=None, help='Input panorama path')
    parser.add_argument('--image', type=str, default=None, help='Input image path')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt for image and panorama generation')
    parser.add_argument('--device', type=str, default=None, help='Device to use: "cpu", "cuda", or "npu"')
    parser.add_argument('--height', type=int, default=512, help='Height of the image')
    parser.add_argument('--width', type=int, default=1024, help='Width of the image')
    parser.add_argument('--num_inference_steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--output_dir', type=str, default='outputs/generation_and_perception', help='Output directory')
    args = parser.parse_args()

    panorama, image, prompt = args.panorama, args.image, args.prompt
    assert panorama is not None or image is not None or prompt is not None, "Please provide at least one of --panorama, --image, or --prompt."

    omnix = OmniX(
        hf_repo='KevinHuang/OmniX',
        device=args.device,
        dtype=torch.bfloat16,
        enable_model_cpu_offload=False,
    )

    height, width = args.height, args.width
    num_inference_steps = args.num_inference_steps
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Panorama Preparation
    if panorama is None:
        assert image is not None or prompt is not None, "Please provide at least one of --image or --prompt."
        panorama, masked_input = run_generation(image=image, prompt=prompt)
    else:
        print('Loading input panorama...')
        panorama = Image.open(panorama).convert('RGB')
        masked_input = None

    # Save Input Panorama
    panorama.save(osp.join(output_dir, 'input_panorama.png'))
    print(f'Input panorama saved to {output_dir}')

    # Panorama Perception
    albedo, depth, normal, roughness, metallic, semantic = run_perception(panorama)

    # Stitch and Save Results
    print('Saving stitched results...')
    if masked_input is not None:
        stitched = stitch_images(
            images=[masked_input, panorama, depth, normal, albedo, roughness, metallic, semantic],
            texts=['masked input', 'panorama', 'depth', 'normal', 'albedo', 'roughness', 'metallic', 'semantic'],
            n_cols=4,
        )
    else:
        stitched = stitch_images(
            images=[panorama, depth, normal, albedo, roughness, metallic, semantic],
            texts=['panorama input', 'depth', 'normal', 'albedo', 'roughness', 'metallic', 'semantic'],
            n_cols=4,
        )
    stitched.save(osp.join(output_dir, 'output_stitched.png'))
    print(f'Stitched results saved to {output_dir}')
