import torch
from typing import Union
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput


def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
    if self.use_tiling:
        return self.tiled_decode(z, return_dict=return_dict)
    if self.post_quant_conv is not None:
        z = self.post_quant_conv(z)
    dec = self.decoder(z)
    if not return_dict:
        return (dec,)
    return DecoderOutput(sample=dec)


def tiled_decode(
    self,
    z: torch.FloatTensor,
    return_dict: bool = True
) -> Union[DecoderOutput, torch.FloatTensor]:
    r"""Decode a batch of images using a tiled decoder.

    Args:
    When this option is enabled, the VAE will split the input tensor into tiles to compute decoding in several
    steps. This is useful to keep memory use constant regardless of image size.
    The end result of tiled decoding is: different from non-tiled decoding due to each tile using a different
    decoder. To avoid tiling artifacts, the tiles overlap and are blended together to form a smooth output.
    You may still see tile-sized changes in the look of the output, but they should be much less noticeable.
        z (`torch.FloatTensor`): Input batch of latent vectors. return_dict (`bool`, *optional*, defaults to
        `True`):
            Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
    """
    overlap_size = int(self.tile_latent_min_size *
                        (1 - self.tile_overlap_factor))
    blend_extent = int(self.tile_sample_min_size *
                        self.tile_overlap_factor)
    row_limit = self.tile_sample_min_size - blend_extent

    w = z.shape[3]

    z = torch.cat([z, z[:, :, :, :2]], dim=-1)  # [1, 16, 64, 160]

    # Split z into overlapping 64x64 tiles and decode them separately.
    # The tiles have an overlap to avoid seams between tiles.
    rows = []
    for i in range(0, z.shape[2], overlap_size):
        row = []
        tile = z[:, :, i:i + self.tile_latent_min_size, :]
        if self.config.use_post_quant_conv:
            tile = self.post_quant_conv(tile)
        decoded = self.decoder(tile)
        vae_scale_factor = decoded.shape[-1] // tile.shape[-1]
        row.append(decoded)
        rows.append(row)
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            # blend the above tile and the left tile
            # to the current tile and add the current tile to the result row
            if i > 0:
                tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
            if j > 0:
                tile = self.blend_h(row[j - 1], tile, blend_extent)
            result_row.append(
                self.blend_h(
                    tile[:, :, :row_limit, w * vae_scale_factor:],
                    tile[:, :, :row_limit, :w * vae_scale_factor],
                    tile.shape[-1] - w * vae_scale_factor))
        result_rows.append(torch.cat(result_row, dim=3))

    dec = torch.cat(result_rows, dim=2)
    if not return_dict:
        return (dec, )
    return DecoderOutput(sample=dec)


class FluxBlendMixin:
    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b
