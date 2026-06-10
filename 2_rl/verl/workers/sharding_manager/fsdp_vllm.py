# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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

import os
import re
import logging
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fsdp_utils import offload_fsdp_param_and_grad, load_fsdp_param_and_grad
from verl.third_party.vllm import vllm_version

from .base import BaseShardingManager

logger = logging.getLogger(__file__)


def _merge_lora_state_dict(state_dict: dict, peft_module) -> dict:
    """Convert a PEFT LoRA state_dict to standard HuggingFace format with merged weights.

    PEFT wraps keys with 'base_model.model.' and splits LoRA target layers into
    base_layer / lora_A / lora_B sub-keys.  This function:
      1. Strips the 'base_model.model.' prefix.
      2. For LoRA layers: computes W_merged = W_base + (lora_B @ lora_A) * scale
         and emits it under the standard '.weight' key.
      3. Drops all remaining LoRA-specific keys (lora_A, lora_B, etc.).
    """
    PREFIX = 'base_model.model.'

    # Collect scaling factors from the PEFT model's LoRA layers (scalars, not params).
    lora_scalings = {}
    for name, module in peft_module.named_modules():
        if hasattr(module, 'scaling') and isinstance(module.scaling, dict):
            hf_name = name[len(PREFIX):] if name.startswith(PREFIX) else name
            lora_scalings[hf_name] = next(iter(module.scaling.values()))

    base_weights = {}   # hf_module_key -> tensor  (from .base_layer.weight)
    lora_a = {}         # hf_module_key -> tensor
    lora_b = {}         # hf_module_key -> tensor
    merged = {}         # output dict

    for key, value in state_dict.items():
        hf_key = key[len(PREFIX):] if key.startswith(PREFIX) else key

        m_base = re.match(r'^(.*?)\.base_layer\.weight$', hf_key)
        m_base_bias = re.match(r'^(.*?)\.base_layer\.bias$', hf_key)
        m_a = re.match(r'^(.*?)\.lora_A\.\w+\.weight$', hf_key)
        m_b = re.match(r'^(.*?)\.lora_B\.\w+\.weight$', hf_key)

        if m_base:
            base_weights[m_base.group(1)] = value
        elif m_base_bias:
            # Bias of a LoRA target layer — pass through under the standard HF key.
            merged[m_base_bias.group(1) + '.bias'] = value
        elif m_a:
            lora_a[m_a.group(1)] = value
        elif m_b:
            lora_b[m_b.group(1)] = value
        elif re.search(r'\.(lora_|base_layer\.)', hf_key):
            pass  # skip other LoRA-specific sub-keys (dropout, embedding, etc.)
        else:
            merged[hf_key] = value  # regular (non-LoRA) parameter

    # Merge LoRA target weights one layer at a time to minimise peak memory.
    # Stay in the original dtype (bfloat16) — upcasting to float32 doubles memory
    # and is unnecessary for the small lora_B @ lora_A product.
    for mod_key, w_base in base_weights.items():
        if mod_key in lora_a and mod_key in lora_b:
            scale = lora_scalings.get(mod_key, 1.0)
            dtype = w_base.dtype
            # Compute delta in original dtype, add in-place, discard intermediates
            delta = (lora_b[mod_key].to(dtype) @ lora_a[mod_key].to(dtype)) * scale
            merged[mod_key + '.weight'] = w_base.add_(delta)
            del delta
            del lora_a[mod_key]
            del lora_b[mod_key]
        else:
            merged[mod_key + '.weight'] = w_base

    torch.cuda.empty_cache()
    return merged
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig())
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)

        unwrapped = self.module._fsdp_wrapped_module
        is_lora = hasattr(unwrapped, 'base_model')

        # For LoRA + sharded state dict: gather the full model on rank 0 only, merge
        # LoRA there, then broadcast to all ranks.  Using rank0_only=True halves the
        # peak CPU memory vs rank0_only=False (which copied the full model to every
        # rank simultaneously — fatal when two jobs share a node).
        # For non-LoRA sharded case we still want DTensors (dtensor weight loader).
        if is_lora and not self.full_params:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            )
            params = self.module.state_dict()  # non-empty only on rank 0
            # Restore sharded type for the rest of training
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
            # Merge LoRA on rank 0, then broadcast merged weights to all ranks
            # so each can load them into its own vLLM instance.
            rank = torch.distributed.get_rank()
            if rank == 0:
                params = _merge_lora_state_dict(params, unwrapped)
            else:
                params = {}
            # Broadcast shape metadata first, then tensors one-at-a-time via GPU
            # to avoid pickling the whole dict (which would double peak memory).
            keys = list(params.keys()) if rank == 0 else None
            keys_container = [keys]
            torch.distributed.broadcast_object_list(keys_container, src=0)
            keys = keys_container[0]
            shape_container = [{k: list(v.shape) for k, v in params.items()}] if rank == 0 else [None]
            torch.distributed.broadcast_object_list(shape_container, src=0)
            shapes = shape_container[0]
            for key in keys:
                if rank == 0:
                    gpu_t = params[key].to(torch.cuda.current_device())
                else:
                    gpu_t = torch.empty(shapes[key], dtype=torch.float32, device=torch.cuda.current_device())
                torch.distributed.broadcast(gpu_t, src=0)
                params[key] = gpu_t.cpu()
                del gpu_t
            torch.cuda.empty_cache()
        else:
            params = self.module.state_dict()

        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)

        print(f'[fsdp_vllm] is_lora={is_lora}, full_params={self.full_params}', flush=True)
        print(f'[fsdp_vllm] state_dict sample keys (first 5): {list(params.keys())[:5]}', flush=True)
        print(f'[fsdp_vllm] first param type: {type(list(params.values())[0])}', flush=True)

        if is_lora and self.full_params:
            # 1-GPU path: full_params=True, all state dict already on this rank
            params = _merge_lora_state_dict(params, unwrapped)
            print(f'[fsdp_vllm] merged keys sample (first 5): {list(params.keys())[:5]}', flush=True)
            embed_key = 'model.embed_tokens.weight'
            if embed_key in params:
                ew = params[embed_key]
                print(f'[fsdp_vllm] merged embed_tokens.weight: shape={ew.shape}, '
                      f'device={ew.device}, dtype={ew.dtype}, norm={ew.float().norm().item():.4f}', flush=True)
            print(f'[fsdp_vllm] merged dict total keys: {len(params)}', flush=True)
        elif is_lora and not self.full_params:
            # multi-GPU path: merge+broadcast already done above; just log from rank 0
            if torch.distributed.get_rank() == 0:
                print(f'[fsdp_vllm] merged keys sample (first 5): {list(params.keys())[:5]}', flush=True)
                embed_key = 'model.embed_tokens.weight'
                if embed_key in params:
                    ew = params[embed_key]
                    print(f'[fsdp_vllm] merged embed_tokens.weight: shape={ew.shape}, '
                          f'device={ew.device}, dtype={ew.dtype}, norm={ew.float().norm().item():.4f}', flush=True)
                print(f'[fsdp_vllm] merged dict total keys: {len(params)}', flush=True)

        # After LoRA merge, keys are in HF format regardless of FSDP sharding strategy.
        # Without LoRA, use dtensor format (matching the FSDP state dict type).
        load_format = 'hf' if (self.full_params or is_lora) else 'dtensor'

        # Move params to CPU and offload FSDP module before vLLM takes GPU memory.
        # (LoRA path already has CPU tensors; this is a no-op for those.)
        params = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in params.items()}
        # Use FSDP-aware offload so that param._local_shard is updated alongside
        # param.data.  Plain module.cpu() only swaps param.data's storage via set_(),
        # leaving _local_shard pointing at the old (now-CPU) storage, which breaks
        # FlatParamHandle._check_on_compute_device() on the next state_dict() call.
        offload_fsdp_param_and_grad(self.module)
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After FSDP offload to CPU before vLLM sync', logger=logger)

        print(f'[fsdp_vllm] calling sync_model_weights with load_format={load_format!r}', flush=True)
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
        else:
            self.inference_engine.wake_up()
            # TODO(ZSL): deal with 'hf' format
            if load_format == 'dtensor':
                from verl.third_party.vllm import load_dtensor_weights
                load_dtensor_weights(
                    params, self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model)
            else:
                raise NotImplementedError(f'load_format {load_format} not implemented')
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del params
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        # TODO(ZSL): check this
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # Restore FSDP params to GPU using the FSDP-aware loader so that both
        # param.data and param._local_shard end up on the compute device.
        load_fsdp_param_and_grad(self.module, device_id=torch.cuda.current_device())
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After FSDP reload to GPU in sharding manager', logger=logger)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                                size=vllm_ps.get_tensor_model_parallel_world_size(),
                                                group=vllm_ps.get_tensor_model_parallel_group(),
                                                dim=0)
        else:
            data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                                size=vllm_ps.get_tensor_model_parallel_world_size(),
                                                group=vllm_ps.get_tensor_model_parallel_group().device_group,
                                                dim=0)

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        local_world_size = vllm_ps.get_tensor_model_parallel_world_size()
        src_rank = (torch.distributed.get_rank() // local_world_size) * local_world_size
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            broadcast_dict_tensor(data.batch, src=src_rank, group=vllm_ps.get_tensor_model_parallel_group())
        else:
            broadcast_dict_tensor(data.batch,
                                  src=src_rank,
                                  group=vllm_ps.get_tensor_model_parallel_group().device_group)
        dp_rank = torch.distributed.get_rank()
        dp_size = torch.distributed.get_world_size()  # not consider torch micro-dp
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]
        return data
