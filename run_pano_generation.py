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
    parser.add_argument('--image', type=str, default=None, help='Input image path')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt for image and panorama generation')
    parser.add_argument('--height', type=int, default=512, help='Height of the generated panorama')
    parser.add_argument('--width', type=int, default=1024, help='Width of the generated panorama')
    parser.add_argument('--device', type=str, default=None, help='Device to use: "cpu", "cuda", or "npu"')
    parser.add_argument('--num_inference_steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--output_dir', type=str, default='outputs/generation', help='Output directory')
    args = parser.parse_args()

    image, prompt = args.image, args.prompt
    assert image is not None or prompt is not None, "Please provide at least one of --image or --prompt."

    height, width = args.height, args.width
    assert height * 2 == width, "The width should be twice the height for panorama."

    num_inference_steps = args.num_inference_steps
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Build OmniX
    omnix = OmniX(
        hf_repo='KevinHuang/OmniX',
        device=args.device,
        dtype=torch.bfloat16,
        enable_model_cpu_offload=False,
    )

    # Image Generation
    if image is None:
        print('Generating image from prompt...')
        size = height  # Use square image as condition
        image = omnix.generate_image(
            prompt=prompt,
            height=size,
            width=size,
            num_inference_steps=num_inference_steps,
        )
    else:
        print('Loading input image...')
        image = Image.open(image).convert('RGB')
    
    # Save Input Image
    image.save(osp.join(output_dir, 'input_image.png'))
    print(f'Input image saved to {output_dir}')

    # Save Masked Input for Visualization Only
    masked_input = omnix.get_masked_panorama(image, height=height, width=width, output_type='pil')
    masked_input.save(osp.join(output_dir, 'input_masked_panorama.png'))
    print(f'Input masked panorama saved to {output_dir}')

    # Panorama Generation
    print('Generating panorama from image...')
    panorama = omnix.generate_panorama(
        image=image,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
    )

    # Save Output Panorama
    panorama.save(osp.join(output_dir, 'output_panorama.png'))
    print(f'Output panorama saved to {output_dir}')

    # Stitch and Save Results
    print('Saving stitched results...')
    stitched = stitch_images(
        images=[image, masked_input, panorama],
        texts=['image input', 'masked input', 'panorama'],
        n_cols=3,
    )
    stitched.save(osp.join(output_dir, 'output_stitched.png'))
    print(f'Stitched results saved to {output_dir}')
