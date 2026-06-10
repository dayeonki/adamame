"""
Merge a verl FSDP LoRA checkpoint into a full HuggingFace model.

Usage:
    python scripts/merge_lora_checkpoint.py \
        --actor_dir checkpoints/.. \
        --base_model .. \
        --output_dir model/.. \
        --lora_alpha 32
"""
import argparse
import importlib
import os
import re
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _local(value):
    """Extract a plain CPU tensor from a DTensor or regular tensor."""
    for mod_path in ('torch.distributed.tensor', 'torch.distributed._tensor'):
        try:
            DTensor = importlib.import_module(mod_path).DTensor
            if isinstance(value, DTensor):
                return value._local_tensor.detach().cpu()
        except Exception:
            pass
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _shard_dim(value):
    """Return the shard dimension if the tensor has Shard placement, else None."""
    for mod_path in ('torch.distributed.tensor', 'torch.distributed._tensor'):
        try:
            mod = importlib.import_module(mod_path)
            if isinstance(value, mod.DTensor):
                for p in value.placements:
                    if type(p).__name__ == 'Shard':
                        return p.dim
                return None
        except Exception:
            pass
    return None


def load_and_assemble_state_dict(actor_dir: str) -> dict:
    """
    Load all per-rank FSDP checkpoint shards and reassemble into a full state dict.
    Sharded tensors are concatenated across ranks; replicated tensors use rank 0.
    """
    rank_files = {}
    for fname in os.listdir(actor_dir):
        m = re.match(r'model_world_size_\d+_rank_(\d+)\.pt', fname)
        if m:
            rank_files[int(m.group(1))] = os.path.join(actor_dir, fname)

    assert rank_files, f'No model_world_size_*_rank_*.pt files found in {actor_dir}'
    print(f'Found {len(rank_files)} rank file(s): ranks {sorted(rank_files)}')

    shards = {}
    for rank in sorted(rank_files):
        print(f'  Loading rank {rank}: {rank_files[rank]}')
        shards[rank] = torch.load(rank_files[rank], map_location='cpu', weights_only=False)

    assembled = {}
    for key in shards[0]:
        rank0_val = shards[0][key]
        dim = _shard_dim(rank0_val)
        if dim is not None and len(shards) > 1:
            parts = [_local(shards[r][key]) for r in sorted(shards)]
            assembled[key] = torch.cat(parts, dim=dim)
        else:
            assembled[key] = _local(rank0_val)

    return assembled


def merge_lora_state_dict(state_dict: dict, lora_alpha: float = None) -> dict:
    PREFIX = 'base_model.model.'

    base_weights = {}
    lora_a = {}
    lora_b = {}
    merged = {}

    for key, value in state_dict.items():
        hf_key = key[len(PREFIX):] if key.startswith(PREFIX) else key

        m_base = re.match(r'^(.*?)\.base_layer\.weight$', hf_key)
        m_base_bias = re.match(r'^(.*?)\.base_layer\.bias$', hf_key)
        m_a = re.match(r'^(.*?)\.lora_A\.\w+\.weight$', hf_key)
        m_b = re.match(r'^(.*?)\.lora_B\.\w+\.weight$', hf_key)

        if m_base:
            base_weights[m_base.group(1)] = value
        elif m_base_bias:
            merged[m_base_bias.group(1) + '.bias'] = value
        elif m_a:
            lora_a[m_a.group(1)] = value
        elif m_b:
            lora_b[m_b.group(1)] = value
        elif re.search(r'\.(lora_|base_layer\.)', hf_key):
            pass  # skip other LoRA-specific keys
        else:
            merged[hf_key] = value

    for mod_key, w_base in base_weights.items():
        if mod_key in lora_a and mod_key in lora_b:
            a = lora_a[mod_key]
            b = lora_b[mod_key]
            rank = a.shape[0]
            alpha = lora_alpha if lora_alpha is not None else rank
            scale = alpha / rank
            dtype = w_base.dtype
            delta = (b.to(dtype) @ a.to(dtype)) * scale
            merged[mod_key + '.weight'] = w_base.add_(delta)
            print(f'  merged {mod_key} (rank={rank}, alpha={alpha}, scale={scale:.4f})')
        else:
            merged[mod_key + '.weight'] = w_base

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_dir', required=True,
                        help='Path to the actor checkpoint dir (contains model_world_size_*.pt)')
    parser.add_argument('--base_model', required=True,
                        help='HuggingFace model name or local path for config/tokenizer')
    parser.add_argument('--output_dir', required=True,
                        help='Where to save the merged HuggingFace model')
    parser.add_argument('--lora_alpha', type=float, default=None,
                        help='LoRA alpha (defaults to lora_rank, giving scale=1.0)')
    args = parser.parse_args()

    print(f'Loading and assembling shards from {args.actor_dir}')
    state_dict = load_and_assemble_state_dict(args.actor_dir)
    print(f'Assembled {len(state_dict)} keys. Sample: {list(state_dict.keys())[:3]}')

    print('Merging LoRA weights...')
    merged = merge_lora_state_dict(state_dict, lora_alpha=args.lora_alpha)
    print(f'Merged dict has {len(merged)} keys.')

    print('Loading base model config...')
    config = AutoConfig.from_pretrained(args.base_model)

    print('Instantiating empty model...')
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    print(f'Saving merged model to {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, state_dict=merged)

    print('Saving tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_dir)

    print('Done.')


if __name__ == '__main__':
    main()
