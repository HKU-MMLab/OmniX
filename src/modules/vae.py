import torch


class FluxVAEMixin():
    def prepare_inputs(self, inputs: torch.Tensor):
        shape = inputs.shape
        dtype = inputs.dtype
        if len(shape) != 4:
            inputs = inputs.reshape(-1, *shape[-3:])
        return inputs.to(self.vae.dtype), {'shape': shape, 'dtype': dtype}
    
    def prepare_outputs(self, outputs: torch.Tensor, shape: torch.Size, dtype: torch.dtype):
        if len(shape) != 4:
            outputs = outputs.reshape(*shape[:-3], *outputs.shape[-3:])
        return outputs.to(dtype)

    @torch.no_grad()
    def encode_images_into_latents(self, images: torch.Tensor):
        """
        Encode images to latents
        Args:
            images (torch.Tensor): Input images, [..., C, H, W]
        Returns:
            torch.Tensor: Latents, [B, 16, H // 8, W // 8]
        """
        images, inputs_info = self.prepare_inputs(images)
        latents = self.vae.encode(images).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latents = self.prepare_outputs(latents, **inputs_info)
        return latents

    @torch.no_grad()
    def decode_latents_into_images(self, latents: torch.Tensor):
        """
        Decode latents to images
        Args:
            latents (torch.Tensor): Latents, [..., 16, H // 8, W // 8]
        Returns:
            torch.Tensor: Output images, [..., C, H, W]
        """
        latents, inputs_info = self.prepare_inputs(latents)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        images = self.vae.decode(latents, return_dict=True).sample
        images = self.prepare_outputs(images, **inputs_info)
        return images
