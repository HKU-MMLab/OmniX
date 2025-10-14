from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipeline,
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
    XLA_AVAILABLE,
)

from src.modules.vae import FluxVAEMixin
from src.modules.blend_h import FluxBlendMixin, tiled_decode, _decode
from src.modules.cross_lora import apply_cross_lora_processors, disable_cross_lora_processors

TASK_ADAPTERS = {
    'image_to_pano': {'lora_names': ('masked_rgb', 'filled_rgb'), 'lora_condition_indices': (0,), 'lora_target_index': 1},
    'rgb_to_albedo': {'lora_names': ('rgb', 'albedo'), 'lora_condition_indices': (0,), 'lora_target_index': 1},
    'rgb_to_depth': {'lora_names': ('rgb', 'depth'), 'lora_condition_indices': (0,), 'lora_target_index': 1},
    'rgb_to_normal': {'lora_names': ('rgb', 'normal'), 'lora_condition_indices': (0,), 'lora_target_index': 1},
    'rgb_to_pbr': {'lora_names': ('rgb', 'pbr'), 'lora_condition_indices': (0,), 'lora_target_index': 1},
    'rgb_to_semantic': {'lora_names': ('rgb', 'semantic'), 'lora_condition_indices': (0,), 'lora_target_index': 1},
}


class OmniXPipeline(FluxPipeline, FluxVAEMixin, FluxBlendMixin):

    def load_task_specific_adapters(self, task_name: str, hf_repo: str):
        # Reset to text_to_image (no LoRA)
        if task_name == 'text_to_image':
            self.camera_type = None
            self.lora_names = None
            self.lora_condition_indices = None
            self.lora_target_index = None
            self.unload_lora_weights()
            disable_cross_lora_processors(self.transformer)
            return

        # Set panoramic task-specific LoRA adapters
        self.camera_type = 'equi'
        self.lora_names = TASK_ADAPTERS[task_name]['lora_names']
        self.lora_condition_indices = TASK_ADAPTERS[task_name]['lora_condition_indices']
        self.lora_target_index = TASK_ADAPTERS[task_name]['lora_target_index']

        self.unload_lora_weights()
        for adapter_name in self.lora_names:
            weight_name = f'{task_name}/{adapter_name}/pytorch_lora_weights.safetensors'
            self.load_lora_weights(hf_repo, weight_name=weight_name, adapter_name=adapter_name)

        apply_cross_lora_processors(
            self.transformer,
            lora_names=self.lora_names,
        )

    def prepare_condition_latents(self, conditions: Dict[str, torch.Tensor]):
        condition_latents = []
        for i in self.lora_condition_indices:
            images = conditions[self.lora_names[i]]
            latents = self.encode_images_into_latents(images)
            condition_latents.append(latents)
        return torch.cat(condition_latents, dim=0)

    def get_camera_specific_prompt(self, prompt: Optional[str] = None):
        if self.camera_type is None:
            return prompt
        CAMERA_SPECIFIC_PROMPTS = {
            'pers': '[PERSPECTIVE]',
            'orth': '[ORTHOGRAPHIC]',
            'equi': '[EQUIRECTANGULAR]',
            'fisheye': '[FISHEYE]',
            'cube': '[CUBEMAP]',
            'multiview': '[MULTIVIEW]',
        }
        if prompt is None:
            return CAMERA_SPECIFIC_PROMPTS[self.camera_type]
        else:
            return f'{CAMERA_SPECIFIC_PROMPTS[self.camera_type]} {prompt}'

    @torch.no_grad()
    def __call__(
        self,
        #################################################################
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        camera_type: Optional[str] = None,
        blend_extend: int = 6,
        #################################################################
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        ###########################################
        if camera_type is None:
            camera_type = self.camera_type

        if camera_type == 'equi':
            use_blend_extend = True
            self.vae.enable_tiling()

            if not hasattr(self.vae, 'tiled_decode_orig'):
                self.vae.tiled_decode_orig = self.vae.tiled_decode
            if not hasattr(self.vae, '_decode_orig'):
                self.vae._decode_orig = self.vae._decode
            
            self.vae.tiled_decode = tiled_decode.__get__(self.vae, AutoencoderKL)
            self.vae._decode = _decode.__get__(self.vae, AutoencoderKL)

        else:
            use_blend_extend = False
            if hasattr(self.vae, 'tiled_decode_orig'):
                self.vae.tiled_decode = self.vae.tiled_decode_orig
            if hasattr(self.vae, '_decode_orig'):
                self.vae._decode = self.vae._decode_orig
        ###########################################

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        ###########################################
        latents_unpack = self._unpack_latents(latents, height, width, self.vae_scale_factor)

        if conditions is not None:
            condition_indices = self.lora_condition_indices
            latents_unpack[condition_indices] = self.prepare_condition_latents(conditions)

        if use_blend_extend:
            latents_unpack = torch.cat([latents_unpack, latents_unpack[:, :, :, :blend_extend]], dim=-1)
        
        if conditions is not None:
            conditions_unpack = latents_unpack[condition_indices].clone()

        width_new_blended = latents_unpack.shape[-1] * 8
        latent_image_ids = self._prepare_latent_image_ids(
            batch_size * num_images_per_prompt, 
            height // 16, width_new_blended // 16, 
            latents.device, 
            latents.dtype,
        )
        latents = self._pack_latents(latents_unpack, batch_size * num_images_per_prompt, num_channels_latents, height // 8, width_new_blended // 8)
        ###########################################

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                ##########################################
                latents_unpack = self._unpack_latents(latents, height, width_new_blended, self.vae_scale_factor)
                if use_blend_extend:
                    latents_unpack = self.blend_h(latents_unpack, latents_unpack, blend_extend)
                if conditions is not None:
                    latents_unpack[condition_indices] = conditions_unpack
                latents = self._pack_latents(latents_unpack, batch_size * num_images_per_prompt, num_channels_latents, height // 8, width_new_blended // 8)
                ##########################################
                
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()
            
            if use_blend_extend:
                latents_unpack = self._unpack_latents(latents, height, width_new_blended, self.vae_scale_factor)
                latents_unpack = self.blend_h(latents_unpack, latents_unpack, blend_extend)
                latents_unpack = latents_unpack[:, :, :, :width // self.vae_scale_factor]
                latents = self._pack_latents(latents_unpack, batch_size * num_images_per_prompt, num_channels_latents, height // 8, width // 8)

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
