import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pathlib import Path
import shutil
from tqdm import tqdm
from safetensors.torch import save_file

def convert_fsdp_checkpoint(checkpoint_dir, output_dir):
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hf_dir = checkpoint_path / "huggingface"
    if not hf_dir.exists():
        hf_dir = checkpoint_path

    files_to_copy = [
        'config.json', 'tokenizer.json', 'tokenizer_config.json',
        'vocab.json', 'merges.txt', 'special_tokens_map.json',
        'added_tokens.json', 'generation_config.json', 'chat_template.jinja'
    ]

    print("Copying config and tokenizer files...")
    for file_name in files_to_copy:
        src = hf_dir / file_name
        if src.exists():
            shutil.copy2(src, output_path / file_name)
            print(f"Copied {file_name}")

    fsdp_files = sorted(checkpoint_path.glob("model_world_size_*.pt"))
    if not fsdp_files:
        print(f"No FSDP checkpoint files found in {checkpoint_path}")
        return

    world_size_from_filename = int(fsdp_files[0].stem.split('_')[3])
    num_ranks = len([f for f in fsdp_files if f'world_size_{world_size_from_filename}' in f.name])
    world_size = num_ranks

    print(f"Loading model weights (world_size={world_size})...")
    state_dicts = []
    for rank in range(world_size):
        model_file = checkpoint_path / f"model_world_size_{world_size}_rank_{rank}.pt"
        print(f"Loading {model_file.name}")
        state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
        state_dicts.append(state_dict)

    print("Merging model shards...")
    merged_state_dict = {}
    all_keys = set(state_dicts[0].keys())

    for key in tqdm(all_keys, desc="Merging parameters"):
        tensors = []
        for state_dict in state_dicts:
            tensor = state_dict[key]
            if hasattr(tensor, '_local_tensor'):
                tensors.append(tensor._local_tensor.to(torch.bfloat16))
            else:
                tensors.append(tensor.to(torch.bfloat16))

        if len(tensors) > 1:
            merged_state_dict[key] = torch.cat(tensors, dim=0).contiguous()
        else:
            merged_state_dict[key] = tensors[0]

    print("Saving model...")
    torch.save(merged_state_dict, output_path / "pytorch_model.bin")

    try:
        save_file(merged_state_dict, output_path / "model.safetensors")
        print("Saved model.safetensors")
    except Exception as e:
        print(f"Warning: Could not save safetensors: {e}")

    print(f"\nModel successfully converted to {output_path}")
    print(f"Files created:")
    for file_path in sorted(output_path.iterdir()):
        size_mb = file_path.stat().st_size / 1024 / 1024
        print(f"  {file_path.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    checkpoint_dir = "/datadrive/difan/verl-llm-tandem/scratch/checkpoints/tandem_grpo_gsm8k_Qwen3-0.6B/global_step_180/actor"
    output_dir = "/datadrive/difan/verl-llm-tandem/scratch/hf/Qwen3-0.6B-gsm8k-tandem-step180"
    
    convert_fsdp_checkpoint(checkpoint_dir, output_dir)