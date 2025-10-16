import os
import os.path as osp
import argparse
import numpy as np
from PIL import Image
from typing import Optional
from einops import rearrange
import torch
from diffusers.utils.import_utils import is_torch_npu_available
if is_torch_npu_available():
    import torch_npu

from src.systems.omnix import OmniX
from src.utils.image import stitch_images
from src.systems.builder import MeshBuilder
from src.utils.sr import SRModel
from src.utils.depth import detect_depth_jumps, smooth_depth, fill_invalid_depth


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


def run_perception(
    panorama: Image.Image,
    rgb_as_albedo: bool = False,
    disable_normal: bool = False,
    disable_pbr: bool = False,
    use_default_pbr: bool = False,
):
    results = {}

    # Panorama Perception - Albedo
    print('Perceiving panoramic albedo...')
    albedo = omnix.perceive_panoramic_albedo(panorama, num_inference_steps=num_inference_steps) \
        if not rgb_as_albedo else panorama
    albedo.save(osp.join(output_dir, 'output_albedo.png'))
    results['albedo'] = albedo
    print(f'Panoramic albedo saved to {output_dir}')

    # Panorama Perception - Depth
    print('Perceiving panoramic depth...')
    depth_np, depth = omnix.perceive_panoramic_depth(panorama, output_type='np_and_pil', num_inference_steps=num_inference_steps)
    np.savez_compressed(osp.join(output_dir, 'output_depth.npz'), array=depth_np)
    depth.save(osp.join(output_dir, 'output_depth.png'))
    results['depth'] = depth
    print(f'Panoramic depth saved to {output_dir}')
    
    # Panorama Perception - Normal
    if not disable_normal:
        print('Perceiving panoramic normal...')
        normal = omnix.perceive_panoramic_normal(panorama, num_inference_steps=num_inference_steps)
        normal.save(osp.join(output_dir, 'output_normal.png'))
        print(f'Panoramic normal saved to {output_dir}')
        results['normal'] = normal
    else:
        normal = None
    
    # Panorama Perception - PBR Material (Roughness, Metallic)
    if not disable_pbr:
        print('Perceiving panoramic PBR material...')
        if not use_default_pbr:
            roughness, metallic = omnix.perceive_panoramic_pbr(panorama, num_inference_steps=num_inference_steps)
        else:
            roughness = Image.fromarray(np.full_like(np.array(albedo)[:, :, 0], 255))
            metallic = Image.fromarray(np.full_like(np.array(albedo)[:, :, 0], 0))
        roughness.save(osp.join(output_dir, 'output_roughness.png'))
        metallic.save(osp.join(output_dir, 'output_metallic.png'))
        results['roughness'] = roughness
        results['metallic'] = metallic
        print(f'Panoramic PBR material saved to {output_dir}')
    else:
        roughness, metallic = None, None
    
    # Panorama Perception - Semantic
    # print('Perceiving panoramic semantic...')
    # semantic = omnix.perceive_panoramic_semantic(panorama, num_inference_steps=num_inference_steps)
    # semantic.save(osp.join(output_dir, 'output_semantic.png'))
    # results['semantic'] = semantic
    # print(f'Panoramic semantic saved to {output_dir}')
    
    # Panorama Pereption - Alpha
    # print('Perceiving panoramic alpha...')
    # alpha = omnix.perceive_panoramic_alpha(panorama, num_inference_steps=num_inference_steps)
    # alpha.save(osp.join(output_dir, 'output_alpha.png'))
    # results['alpha'] = alpha
    # print(f'Panoramic alpha saved to {output_dir}')
    
    return results


def export_3d_scene(
    import_mesh_path: str,
    export_gltf_path: str = '',
    export_fbx_path: str = '',
    export_blend_path: str = '',
    light_energy: float = 100.0,
    light_type: str = 'POINT',
    use_sphere: bool = False,
    flip_normal: bool = True,
    use_backface_culling: bool = False,
):
    try:
        import bpy
        interpreter = 'python'
    except ImportError:
        interpreter = 'blender -b -P'
    
    base_command = \
        f'{interpreter} export_3d_scene_blender.py -- ' \
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
    parser.add_argument('--which_stage', type=str, default='12345', help='Which stage to run: 1 (generation), 2 (perception), 3 (geometry), 4 (texture), 5 (export), or 12345 (all)')
    parser.add_argument('--device', type=str, default=None, help='Device to use: "cpu", "cuda", or "npu"')
    parser.add_argument('--height', type=int, default=512, help='Height of the generated panorama')
    parser.add_argument('--width', type=int, default=1024, help='Width of the generated panorama')
    parser.add_argument('--sr', type=int, default=4, help='Super-resolution scale factor (1: no SR, 2: 2x SR, 4: 4x SR)')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--output_dir', type=str, default='outputs/construction', help='Output directory')
    parser.add_argument('--lifting_method', type=str, default='cube', help='Lifting method for mesh creation: "equi" or "cube"')
    parser.add_argument('--simplify_mesh', action='store_true', help='Whether to simplify the mesh')
    parser.add_argument('--fill_invalid_depth', action='store_true', help='Whether to fill invalid depth values')
    parser.add_argument('--rgb_as_albedo', action='store_true', help='Use RGB as albedo')
    parser.add_argument('--disable_normal', action='store_true', help='Disable normal texture')
    parser.add_argument('--disable_pbr', action='store_true', help='Disable PBR material')
    parser.add_argument('--use_default_pbr', action='store_true', help='Use default PBR material (roughness=1, metallic=0)')
    args = parser.parse_args()

    panorama, image, prompt = args.panorama, args.image, args.prompt
    assert panorama is not None or image is not None or prompt is not None, "Please provide at least one of --panorama, --image, or --prompt."

    STAGE_I, STAGE_II, STAGE_III, STAGE_IV, STAGE_V = \
        '1' in args.which_stage, '2' in args.which_stage, '3' in args.which_stage, '4' in args.which_stage, '5' in args.which_stage
    
    # Build Models
    if STAGE_I or STAGE_II:
        omnix = OmniX(
            hf_repo='KevinHuang/OmniX',
            device=args.device,
            dtype=torch.bfloat16,
            enable_model_cpu_offload=False,
        )

        height, width = args.height, args.width
        num_inference_steps = args.num_inference_steps

    if STAGE_III or STAGE_IV:
        mesh_builder = MeshBuilder()
        if STAGE_IV and args.sr > 1:
            sr_model = SRModel(scale=args.sr)
    
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
        properties = run_perception(
            panorama,
            rgb_as_albedo=args.rgb_as_albedo,
            disable_normal=args.disable_normal,
            disable_pbr=args.disable_pbr,
            use_default_pbr=args.use_default_pbr,
        )

        ## Stitch and Save Results
        print('Saving stitched results...')
        texts = list(properties.keys())
        images = [properties[k] for k in texts]
        if masked_input is not None:
            stitched = stitch_images(
                images=[masked_input, panorama] + images,
                texts=['masked input', 'panorama'] + texts,
                n_cols=(len(texts) + 2 + 1) // 2,
            )
        else:
            stitched = stitch_images(
                images=[panorama] + images,
                texts=['panorama input'] + texts,
                n_cols=(len(texts) + 1 + 1) // 2,
            )
        stitched.save(osp.join(output_dir, 'output_stitched.png'))
        print(f'Stitched results saved to {output_dir}')

    # Stage III: 3D Scene Construction
    if STAGE_III:
        ## Init
        panorama_dir = osp.join(args.output_dir, 'panorama')
        untextured_output_dir = osp.join(args.output_dir, 'untextured_mesh')
        
        ## Depth Post-processing
        depth = np.load(osp.join(panorama_dir, 'output_depth.npz'))['array']  # [H, W]
        mask = detect_depth_jumps(depth, gradient_threshold=0.3)
        if args.fill_invalid_depth:
            depth = fill_invalid_depth(depth, mask)
            mask = None
        depth = smooth_depth(depth).clip(0.0, 1.0)

        ## Build Untextured Mesh
        output_dir = untextured_output_dir
        os.makedirs(output_dir, exist_ok=True)
        raw_mesh_path = osp.join(output_dir, 'raw_mesh.obj')
        simplified_mesh_path = raw_mesh_path
        unwrapped_mesh_path = osp.join(output_dir, 'unwrapped_mesh.obj')

        if args.lifting_method == 'equi':
            mesh_builder.create_mesh_from_equi_distance(
                distance=torch.tensor(depth).float(),
                mask=torch.tensor(mask) if mask is not None else None,
                save_path=raw_mesh_path,
            )
        else:
            from src.utils.convert import equi_to_cube
            distance = torch.tensor(depth).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            w_face = distance.shape[-2] // 2
            distance = equi_to_cube(distance, w_face=w_face, cube_format='horizon')  # [B, C, H, M*W]
            distance = rearrange(distance, '1 1 h (m w) -> m h w', m=6)  # [6, H, W]
            if mask is not None:
                mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                mask = equi_to_cube(mask, w_face=w_face, cube_format='horizon')  # [B, C, H, M*W]
                mask = rearrange(mask, '1 1 h (m w) -> m h w', m=6)  # [6, H, W]
                mask = mask > 0.5
            mesh_builder.create_mesh_from_cube_distance(distance=distance, mask=mask, save_path=raw_mesh_path)
        
        if args.simplify_mesh:
            simplified_mesh_path = osp.join(output_dir, 'simplified_mesh.obj')
            mesh_builder.simplify_mesh(
                input_path=raw_mesh_path,
                output_path=simplified_mesh_path,
                target_count=40000,
            )
        
        mesh_builder.unwrap_mesh(
            input_path=simplified_mesh_path,
            output_path=unwrapped_mesh_path,
            method='equi',
        )

    # Stage IV: 3D Scene Construction
    if STAGE_IV:
        panorama_dir = osp.join(args.output_dir, 'panorama')
        untextured_output_dir = osp.join(args.output_dir, 'untextured_mesh')
        textured_output_dir = osp.join(args.output_dir, 'textured_mesh')
        
        os.makedirs(textured_output_dir, exist_ok=True)

        input_albedo_path = osp.join(panorama_dir, 'output_albedo.png')
        input_alpha_path = osp.join(panorama_dir, 'output_alpha.png')
        input_normal_path = osp.join(panorama_dir, 'output_normal.png')
        input_roughness_path = osp.join(panorama_dir, 'output_roughness.png')
        input_metallic_path = osp.join(panorama_dir, 'output_metallic.png')
        
        if args.sr > 1:
            input_albedo_path = sr_model(input_path=input_albedo_path)
        
        mesh_builder.texture_mesh(
            input_path=osp.join(untextured_output_dir, 'unwrapped_mesh.obj'),
            output_path=osp.join(textured_output_dir, 'textured_mesh.obj'),
            input_albedo_path=input_albedo_path if osp.exists(input_albedo_path) else None,
            input_alpha_path=input_alpha_path if osp.exists(input_alpha_path) else None,
            input_normal_path=input_normal_path if osp.exists(input_normal_path) else None,
            input_roughness_path=input_roughness_path if osp.exists(input_roughness_path) else None,
            input_metallic_path=input_metallic_path if osp.exists(input_metallic_path) else None,
        )

    # Stage V: Export 3D Scene
    if STAGE_V:
        textured_output_dir = osp.join(args.output_dir, 'textured_mesh')

        export_3d_scene(
            import_mesh_path=osp.join(textured_output_dir, 'textured_mesh.obj'),
            export_gltf_path=osp.join(args.output_dir, 'export_gltf', 'scene.glb'),  # Some textures (e.g., world normal) are incompatible
            # export_fbx_path=osp.join(args.output_dir, 'export_fbx', 'scene.fbx'),    # Untested
            export_blend_path=osp.join(args.output_dir, 'export_blend', 'scene.blend'),
            light_energy=100.0,
            light_type='POINT',
            use_sphere=False,
            flip_normal=True,
            use_backface_culling=False,
        )
