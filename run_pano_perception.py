import os
import os.path as osp
import argparse
from PIL import Image
import torch
from diffusers.utils.import_utils import is_torch_npu_available
if is_torch_npu_available():
    import torch_npu

from src.systems.omnix import OmniX
from src.utils.image import stitch_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--panorama', type=str, required=True, help='Input panorama path')
    parser.add_argument('--device', type=str, default=None, help='Device to use: "cpu", "cuda", or "npu"')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--output_dir', type=str, default='outputs/perception', help='Output directory')
    args = parser.parse_args()

    omnix = OmniX(
        hf_repo='KevinHuang/OmniX',
        device=args.device,
        dtype=torch.bfloat16,
        enable_model_cpu_offload=False,
    )

    num_inference_steps = args.num_inference_steps
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Loading Input Panorama
    print('Loading input panorama...')
    panorama = Image.open(args.panorama).convert('RGB')

    # Save Input Panorama
    panorama.save(osp.join(output_dir, 'input_panorama.png'))
    print(f'Input panorama saved to {output_dir}')

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
    
    # Stitch and Save Results
    print('Saving stitched results...')
    stitched = stitch_images(
        images=[panorama, depth, normal, albedo, roughness, metallic, semantic],
        texts=['panorama input', 'depth', 'normal', 'albedo', 'roughness', 'metallic', 'semantic'],
        n_cols=4,
    )
    stitched.save(osp.join(output_dir, 'output_stitched.png'))
    print(f'Stitched results saved to {output_dir}')
