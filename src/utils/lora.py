import os.path as osp
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.flux.pipeline_flux import FluxTransformer2DModel
from peft import LoraConfig


def add_lora_adapter(self,
    transformer: FluxTransformer2DModel,
    adapter_name: str = 'default',
    lora_rank: int = 64,
    target_modules: Optional[List[str]] = None,
):
    print(f"[LoRA] Adding {adapter_name} LoRA to Transformer.")
    if target_modules is None:
        target_modules = [
            # Input layer
            # "x_embedder",
            # Attention
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            # Feedforward
            "ff.net.0.proj",
            "ff.net.2",
        ]
    transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config, adapter_name)


def add_multiple_lora_adapters(
    transformer: FluxTransformer2DModel,
    lora_names: List[str],
    lora_paths: Optional[Union[List[str], str]] = None,
    lora_rank: int = 64,
    weight_name: str = 'pytorch_lora_weights.safetensors',
    target_modules: Optional[List[str]] = None,
):
    print(f"[LoRA] Adding Multiple LoRA(s): {lora_names} to Transformer.")

    if lora_paths is not None:
        if type(lora_paths) is str:
            lora_paths = [osp.join(lora_paths, lora_name) for lora_name in lora_names]
        for lora_path, lora_name in zip(lora_paths, lora_names):
            print(f"[LoRA] Loading LoRA: {lora_name} from {lora_path}.")
            transformer.load_lora_adapter(
                pretrained_model_name_or_path_or_dict=lora_path,
                weight_name=weight_name,
                adapter_name=lora_name,
                local_files_only=True,
            )
        if not transformer._hf_peft_config_loaded:
            transformer._hf_peft_config_loaded = True
    else:
        for lora_name in lora_names:
            add_lora_adapter(
                transformer,
                adapter_name=lora_name,
                lora_rank=lora_rank,
                target_modules=target_modules,
            )
    transformer.set_adapter(lora_names)  # Required! Ensure that all LoRA weights are captured!


def get_lora_params(
    transformer: FluxTransformer2DModel,
    adapter_name: str,
):
    lora_params = []
    for n, p in transformer.named_parameters():
        if n.endswith(f'lora_A.{adapter_name}.weight') or n.endswith(f'lora_B.{adapter_name}.weight'):
            lora_params.append(p)
    return lora_params


def check_no_contains(lora_names: List[str]) -> bool:
    for i in range(len(lora_names)):
        for j in range(len(lora_names)):
            if i != j:
                if lora_names[i] in lora_names[j]:
                    assert False, f"lora_names should not contain each other (because 'get_peft_model_state_dict' cannot handle it), but got {lora_names}."
    return True
