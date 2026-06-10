# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader

from typing import Any, Dict

import torch.nn as nn
import torch
from vllm.model_executor.model_loader.utils import set_default_torch_dtype


def update_hf_weight_loader():
    print("no hf weight loader need to be updated")
    return


def load_hf_weights(actor_weights: Dict, vllm_model: nn.Module):
    assert isinstance(actor_weights, Dict)

    # Zero out all bias parameters BEFORE loading.
    # vLLM's Qwen2 hardcodes bias=True for qkv_proj, but many actor models have
    # attention_bias=False and provide no bias tensors. After offload_model_weights(),
    # these bias tensors are uninitialized (torch.empty_like). Without zeroing them,
    # they contain garbage that corrupts every attention layer.
    with torch.no_grad():
        for name, param in vllm_model.named_parameters():
            if name.endswith('.bias'):
                param.data.zero_()
                # print(f'[load_hf_weights] zeroed bias: {name}, shape={param.shape}', flush=True)

    # --- DEBUG: inspect incoming weights ---
    aw_keys = list[Any](actor_weights.keys())
    print(f'[load_hf_weights] actor_weights has {len(aw_keys)} keys; first 5: {aw_keys[:5]}', flush=True)
    for chk_key in ['model.embed_tokens.weight', 'model.layers.0.self_attn.q_proj.weight',
                    'model.layers.0.self_attn.k_proj.weight', 'lm_head.weight']:
        if chk_key in actor_weights:
            w = actor_weights[chk_key]
            print(f'[load_hf_weights] actor {chk_key}: shape={w.shape}, dtype={w.dtype}, '
                  f'norm={w.float().norm().item():.4f}', flush=True)
        else:
            print(f'[load_hf_weights] WARNING: {chk_key} NOT in actor_weights!', flush=True)

    # --- DEBUG: inspect vllm model params before loading ---
    first_p = next(vllm_model.parameters())
    print(f'[load_hf_weights] vllm first param: device={first_p.device}, dtype={first_p.dtype}, '
          f'norm={first_p.data.float().norm().item():.4f}', flush=True)

    with set_default_torch_dtype(next(vllm_model.parameters()).dtype):  # TODO
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in actor_weights.keys():
            del actor_weights["lm_head.weight"]
        vllm_model.load_weights(actor_weights.items())

    # --- DEBUG: inspect vllm embed_tokens after load_weights but before cuda() ---
    for name, param in vllm_model.named_parameters():
        if 'embed_tokens' in name:
            print(f'[load_hf_weights] AFTER load_weights | {name}: device={param.device}, '
                  f'norm={param.data.float().norm().item():.4f}', flush=True)
            break

    for _, module in vllm_model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
        # FIXME: Remove this after Mixtral is updated
        # to use quant_method.
        if hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading()
    vllm_model = vllm_model.cuda()

    # --- DEBUG: inspect vllm embed_tokens after cuda() ---
    for name, param in vllm_model.named_parameters():
        if 'embed_tokens' in name:
            print(f'[load_hf_weights] AFTER cuda()      | {name}: device={param.device}, dtype={param.dtype}, '
                  f'norm={param.data.float().norm().item():.4f}', flush=True)
            break

    # --- DEBUG: check qkv_proj and lm_head norms ---
    for name, param in vllm_model.named_parameters():
        if 'qkv_proj' in name and 'weight' in name and 'layers.0' in name:
            print(f'[load_hf_weights] AFTER cuda() | {name}: device={param.device}, dtype={param.dtype}, '
                  f'shape={param.shape}, norm={param.data.float().norm().item():.4f}', flush=True)
            break
    for name, param in vllm_model.named_parameters():
        if 'lm_head' in name and 'weight' in name:
            print(f'[load_hf_weights] AFTER cuda() | {name}: device={param.device}, dtype={param.dtype}, '
                  f'shape={param.shape}, norm={param.data.float().norm().item():.4f}', flush=True)
            break
