import math
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union
from einops import rearrange, repeat
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel, Attention,
)
from diffusers.models.attention_processor import (
    FluxAttnProcessor2_0,
    FluxAttnProcessor2_0_NPU,
)
from peft.tuners.lora.layer import LoraLayer, Linear as Linear_LoRA
from diffusers.utils.import_utils import is_torch_npu_available

if is_torch_npu_available():
    import torch_npu


class FluxAttnProcessor2_0_CrossLoRA:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, active_adapters: Union[str, List[str]]):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.set_adapter(active_adapters)

    def set_adapter(self, active_adapters: Union[str, List[str]]):
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]
        self.active_adapters = active_adapters

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        active_adapters = self.active_adapters
        assert hidden_states.shape[0] == len(active_adapters), f"Number of active adapters {(len(active_adapters))} should equal to hidden_states.shape[0] ({hidden_states.shape[0]})!"

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        key = repeat(key, 'b n s d -> bs n (b s) d', bs=query.shape[0])
        value = repeat(value, 'b n s d -> bs n (b s) d', bs=query.shape[0])

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxAttnProcessor2_0_NPU_CrossLoRA:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, active_adapters: Union[str, List[str]]):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessor2_0_NPU requires PyTorch 2.0 and torch NPU, to use it, please upgrade PyTorch to 2.0 and install torch NPU"
            )
        self.set_adapter(active_adapters)

    def set_adapter(self, active_adapters: Union[str, List[str]]):
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]
        self.active_adapters = active_adapters

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        active_adapters = self.active_adapters
        assert hidden_states.shape[0] == len(active_adapters), f"Number of active adapters {(len(active_adapters))} should equal to hidden_states.shape[0] ({hidden_states.shape[0]})!"

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # [Modify] Cross-LoRA Attention
        # query: [B N C]
        # key: [B N C]
        # value: [B N C]
        # key = repeat(key, 'b n c -> bs (b n) c', bs=query.shape[0])
        # value = repeat(value, 'b n c -> bs (b n) c', bs=query.shape[0])

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

            # if 0:
            #     key = apply_rotary_emb(key, image_rotary_emb)
            # else:
            #     bs = query.shape[0]
            #     image_rotary_emb_cos, image_rotary_emb_sin = image_rotary_emb
            #     image_rotary_emb_cos_new = torch.cat(
            #         [image_rotary_emb_cos[:512],] + [image_rotary_emb_cos[512:] for _ in range(bs)], dim=0)
            #     image_rotary_emb_sin_new = torch.cat(
            #         [image_rotary_emb_sin[:512],] + [image_rotary_emb_sin[512:] for _ in range(bs)], dim=0)
            #     key = apply_rotary_emb(key, (image_rotary_emb_cos_new, image_rotary_emb_sin_new))
        
        key = repeat(key, 'b n s d -> bs n (b s) d', bs=query.shape[0])
        value = repeat(value, 'b n s d -> bs n (b s) d', bs=query.shape[0])

        #----------------------- Modify! -----------------------#
        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # if query.dtype in (torch.float16, torch.bfloat16):
        if query.dtype in (torch.float16, torch.bfloat16) or \
          key.dtype in (torch.float16, torch.bfloat16) or \
          value.dtype in (torch.float16, torch.bfloat16):
            if query.dtype in (torch.float32, torch.float16): query = query.to(torch.bfloat16)
            if key.dtype in (torch.float32, torch.float16): key = key.to(torch.bfloat16)
            if value.dtype in (torch.float32, torch.float16): value = value.to(torch.bfloat16)
            hidden_states = torch_npu.npu_fusion_attention(
                query,
                key,
                value,
                attn.heads,
                input_layout="BNSD",
                pse=None,
                scale=1.0 / math.sqrt(query.shape[-1]),
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1.0,
                sync=False,
                inner_precise=0,
            )[0]
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        
        else:
            return hidden_states


class FluxAttnProcessor2_0_NPU_AsymmetricCrossLoRA:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, active_adapters: Union[str, List[str]], independent_adapters: Union[str, List[str]]):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessor2_0_NPU requires PyTorch 2.0 and torch NPU, to use it, please upgrade PyTorch to 2.0 and install torch NPU"
            )
        self.set_adapter(active_adapters, independent_adapters)

    def set_adapter(self, active_adapters: Union[str, List[str]], independent_adapters: Union[str, List[str]]):
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]
        self.active_adapters = active_adapters

        if isinstance(independent_adapters, str):
            independent_adapters = [independent_adapters]
        self.independent_adapters = independent_adapters

        self.independent_indices = [active_adapters.index(item) for item in independent_adapters]
        self.dependent_indices = [i for i in range(len(self.active_adapters)) if i not in self.independent_indices]

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        active_adapters = self.active_adapters

        assert hidden_states.shape[0] == len(active_adapters), f"Number of active adapters {(len(active_adapters))} should equal to hidden_states.shape[0] ({hidden_states.shape[0]})!"

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # [Modify] Cross-LoRA Attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        bs = query.shape[0]
        s_query = query.shape[-2]
        s_key = key.shape[-2]
        key = repeat(key, 'b n s d -> bs n (b s) d', bs=bs)
        value = repeat(value, 'b n s d -> bs n (b s) d', bs=bs)
        atten_mask = torch.zeros(bs, attn.heads, s_query, bs, s_key, dtype=torch.bool).to(query.device)
        for i in self.independent_indices:
            atten_mask[i, :, :, self.dependent_indices, :] = True
        atten_mask = repeat(atten_mask, 'bs n sq b sk -> bs n sq (b sk)')

        #----------------------- Modify! -----------------------#
        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # if query.dtype in (torch.float16, torch.bfloat16):
        if query.dtype in (torch.float16, torch.bfloat16) or \
          key.dtype in (torch.float16, torch.bfloat16) or \
          value.dtype in (torch.float16, torch.bfloat16):
            if query.dtype in (torch.float32, torch.float16): query = query.to(torch.bfloat16)
            if key.dtype in (torch.float32, torch.float16): key = key.to(torch.bfloat16)
            if value.dtype in (torch.float32, torch.float16): value = value.to(torch.bfloat16)
            hidden_states = torch_npu.npu_fusion_attention(
                query,
                key,
                value,
                attn.heads,
                input_layout="BNSD",
                pse=None,
                scale=1.0 / math.sqrt(query.shape[-1]),
                atten_mask=atten_mask,
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1.0,
                sync=False,
                inner_precise=0,
            )[0]
        else:
            raise NotImplementedError('LoRA masks not implemented!')
            # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        
        else:
            return hidden_states


class FluxAttnProcessor2_0_AsymmetricCrossLoRA:
    pass


def get_linear_forward(self, lora_names: Optional[List[str]] = None):
    def forward(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        active_adapters = self.active_adapter if lora_names is None else lora_names
        outs = []
        for i in range(x.shape[0]):
            self.set_adapter(active_adapters[i])
            outs.append(self._original_forward(x[i:i+1], *args, **kwargs))
        outs = torch.cat(outs, dim=0)
        self.set_adapter(active_adapters)
        return outs
    
    return forward


def get_flux_attention_processor(
    lora_names: Optional[List[str]] = None,
    independent_lora_names: Optional[List[str]] = None,
):
    if is_torch_npu_available():
        if lora_names is None:
            return FluxAttnProcessor2_0_NPU()
        elif independent_lora_names is None:
            return FluxAttnProcessor2_0_NPU_CrossLoRA(lora_names)
        else:
            return FluxAttnProcessor2_0_NPU_AsymmetricCrossLoRA(lora_names, independent_lora_names)
    else:
        if lora_names is None:
            return FluxAttnProcessor2_0()
        elif independent_lora_names is None:
            return FluxAttnProcessor2_0_CrossLoRA(lora_names)
        else:
            return FluxAttnProcessor2_0_AsymmetricCrossLoRA(lora_names, independent_lora_names)


def apply_cross_lora_processors(
    transformer: FluxTransformer2DModel,
    lora_names: List[str],
    independent_lora_names: Optional[List[str]] = None,
):
    transformer.set_adapters(lora_names)
    for n, module in transformer.named_modules():
        if isinstance(module, LoraLayer):
            if isinstance(module, Linear_LoRA):
                if not hasattr(module, '_original_forward'):
                    module._original_forward = module.forward
                module.forward = get_linear_forward(module, lora_names=lora_names)
            else:
                raise NotImplementedError(f"Not supported LoRA Module: {n}")
        
        if isinstance(module, Attention):
            module.set_processor(get_flux_attention_processor(lora_names, independent_lora_names))


def disable_cross_lora_processors(transformer: FluxTransformer2DModel):
    transformer.set_adapters([])
    for n, module in transformer.named_modules():
        if isinstance(module, LoraLayer):
            if isinstance(module, Linear_LoRA):
                if hasattr(module, '_original_forward'):
                    module.forward = module._original_forward
            else:
                raise NotImplementedError(f"Not supported LoRA Module: {n}")
        
        if isinstance(module, Attention):
            module.set_processor(get_flux_attention_processor(None, None))
    