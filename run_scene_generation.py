import os
import os.path as osp
import argparse
import numpy as np
from PIL import Image
from typing import Optional
import torch
from diffusers.utils.import_utils import is_torch_npu_available
if is_torch_npu_available():
    import torch_npu

from src.systems.omnix import OmniX
from src.utils.image import stitch_images
from src.systems.builder import MeshBuilder
from src.utils.depth import detect_depth_jumps, smooth_depth


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
    depth_np, depth = omnix.perceive_panoramic_depth(panorama, output_type='np_and_pil', num_inference_steps=num_inference_steps)
    np.savez_compressed(osp.join(output_dir, 'output_depth.npz'), array=depth_np)
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
    # print('Perceiving panoramic semantic...')
    # semantic = omnix.perceive_panoramic_semantic(panorama, num_inference_steps=num_inference_steps)
    # semantic.save(osp.join(output_dir, 'output_semantic.png'))
    # print(f'Panoramic semantic saved to {output_dir}')
    semantic = None  # Not used in construction for now
    
    return albedo, depth, normal, roughness, metallic, semantic


def export_3d_scene(
    import_mesh_path: str,
    export_gltf_path: str = '',
    export_fbx_path: str = '',
    export_blend_path: str = '',
    light_energy: float = 100.0,
    light_type: str = 'POINT',
    use_sphere: bool = False,
    flip_normal: bool = True,
    use_backface_culling: bool = True,
):
    base_command = \
        f'blender -b -P export_3d_scene_bpy.py -- ' \
        f'--import_mesh_path "{import_mesh_path}" ' \
        f'--export_gltf_path "{export_gltf_path}" ' \
        f'--export_fbx_path "{export_fbx_path}" ' \
        f'--export_blend_path "{export_blend_path}" ' \
        f'--light_type {light_type} ' \
        f'--light_energy {light_energy}'
    
    if use_sphere:
        base_command += ' --use_sphere'
    
    if use_backface_culling:
        base_command += ' --use_backface_culling'
    
    if flip_normal:
        base_command += ' --flip_normal'

    os.system(base_command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--panorama', type=str, default=None, help='Input panorama path')
    parser.add_argument('--image', type=str, default=None, help='Input image path')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt for image and panorama generation')
    parser.add_argument('--which_stage', type=str, default='123', help='Which stage to run: 1 (generation), 2 (perception), 3 (construction), or 123 (all)')
    parser.add_argument('--device', type=str, default=None, help='Device to use: "cpu", "cuda", or "npu"')
    parser.add_argument('--height', type=int, default=512, help='Height of the generated panorama')
    parser.add_argument('--width', type=int, default=1024, help='Width of the generated panorama')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--output_dir', type=str, default='outputs/construction', help='Output directory')
    args = parser.parse_args()

    panorama, image, prompt = args.panorama, args.image, args.prompt
    assert panorama is not None or image is not None or prompt is not None, "Please provide at least one of --panorama, --image, or --prompt."

    STAGE_I, STAGE_II, STAGE_III = '1' in args.which_stage, '2' in args.which_stage, '3' in args.which_stage
    
    # Build OmniX
    if STAGE_I or STAGE_II:
        omnix = OmniX(
            hf_repo='KevinHuang/OmniX',
            device=args.device,
            dtype=torch.bfloat16,
            enable_model_cpu_offload=False,
        )

        height, width = args.height, args.width
        num_inference_steps = args.num_inference_steps

    # Stage I: Panorama Generation (if no input panorama)
    if STAGE_I:
        ## Init
        output_dir = osp.join(args.output_dir, 'panorama')
        os.makedirs(output_dir, exist_ok=True)
        
        ## Panorama Preparation
        if panorama is None:
            assert image is not None or prompt is not None, "Please provide at least one of --image or --prompt."
            panorama, masked_input = run_generation(image=image, prompt=prompt)
        else:
            print('Loading input panorama...')
            panorama = Image.open(panorama).convert('RGB')
            masked_input = None

        ## Save Input Panorama
        panorama.save(osp.join(output_dir, 'input_panorama.png'))
        print(f'Input panorama saved to {output_dir}')

    # Stage II: Panorama Perception
    if STAGE_II:
        ## Init
        output_dir = osp.join(args.output_dir, 'panorama')
        os.makedirs(output_dir, exist_ok=True)

        ## Load Panorama and Run Perception
        panorama = Image.open(osp.join(output_dir, 'input_panorama.png')).convert('RGB')
        albedo, depth, normal, roughness, metallic, semantic = run_perception(panorama)

        ## Stitch and Save Results
        print('Saving stitched results...')
        if masked_input is not None:
            stitched = stitch_images(
                images=[masked_input, panorama, depth, normal, albedo, roughness, metallic],
                texts=['masked input', 'panorama', 'depth', 'normal', 'albedo', 'roughness', 'metallic'],
                n_cols=4,
            )
        else:
            stitched = stitch_images(
                images=[panorama, depth, normal, albedo, roughness, metallic],
                texts=['panorama input', 'depth', 'normal', 'albedo', 'roughness', 'metallic'],
                n_cols=3,
            )
        stitched.save(osp.join(output_dir, 'output_stitched.png'))
        print(f'Stitched results saved to {output_dir}')

    # Stage III: 3D Scene Construction
    if STAGE_III:
        ## Init
        panorama_dir = osp.join(args.output_dir, 'panorama')
        untextured_output_dir = osp.join(args.output_dir, 'untextured_mesh')
        textured_output_dir = osp.join(args.output_dir, 'textured_mesh')
        
        mesh_builder = MeshBuilder()
        
        ## Depth Post-processing
        depth = np.load(osp.join(panorama_dir, 'output_depth.npz'))['array']  # [H, W]
        mask = detect_depth_jumps(depth, gradient_threshold=0.3)
        depth = smooth_depth(depth).clip(0.0, 1.0)

        ## Build Untextured Mesh
        output_dir = untextured_output_dir
        os.makedirs(output_dir, exist_ok=True)
        mesh_builder.create_mesh_from_equi_distance(
            distance=torch.tensor(depth).float(),
            mask=torch.tensor(mask),
            save_path=osp.join(output_dir, 'raw_mesh.obj'),
        )
        mesh_builder.simplify_mesh(
            input_path=osp.join(output_dir, 'raw_mesh.obj'),
            output_path=osp.join(output_dir, 'simplified_mesh.obj'),
            target_count=40000,
        )
        mesh_builder.unwrap_mesh(
            input_path=osp.join(output_dir, 'simplified_mesh.obj'),
            output_path=osp.join(output_dir, 'unwrapped_mesh.obj'),
            method='equi',
        )

        ## Build Textured Mesh
        output_dir = textured_output_dir
        os.makedirs(output_dir, exist_ok=True)
        mesh_builder.texture_mesh(
            input_path=osp.join(untextured_output_dir, 'unwrapped_mesh.obj'),
            output_path=osp.join(output_dir, 'textured_mesh.obj'),
            input_albedo_path=osp.join(panorama_dir, 'output_albedo.png'),
            input_normal_path=osp.join(panorama_dir, 'output_normal.png'),
            input_roughness_path=osp.join(panorama_dir, 'output_roughness.png'),
            input_metallic_path=osp.join(panorama_dir, 'output_metallic.png'),
        )

        ## Export 3D Scene
        export_3d_scene(
            import_mesh_path=osp.join(textured_output_dir, 'textured_mesh.obj'),
            export_gltf_path=osp.join(args.output_dir, 'export_gltf', 'scene.glb'),
            export_fbx_path=osp.join(args.output_dir, 'export_fbx', 'scene.fbx'),
            export_blend_path=osp.join(args.output_dir, 'export_blend', 'scene.blend'),
            light_energy=100.0,
            light_type='POINT',
            use_sphere=False,
            flip_normal=True,
            use_backface_culling=True,
        )
