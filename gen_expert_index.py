#!/usr/bin/env python3
#!/usr/bin/env python3
"""
gen_expert_index.py
A setup script added in this fork to dynamically map Qwen3.5-122B-A10B safetensor
indexes to the format expected by the metal_infer C engine.
"""
import json
import os
import argparse
from pathlib import Path
import struct
import re

def parse_safetensors_header(filepath):
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, default='expert_index.json')
    args = parser.parse_args()

    model_path = Path(args.model)
    index_path = model_path / 'model.safetensors.index.json'
    
    with open(index_path) as f:
        weight_map = json.load(f)['weight_map']

    # Expert components in Qwen3.5: .switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}
    expr = re.compile(r'model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)')
    
    # Organize by layer
    expert_tensors = {} # layer -> component_name -> filename
    for name, filename in weight_map.items():
        match = expr.search(name)
        if match:
            layer_idx = int(match.group(1))
            comp_name = f"{match.group(2)}.{match.group(3)}"
            if layer_idx not in expert_tensors: expert_tensors[layer_idx] = {}
            expert_tensors[layer_idx][comp_name] = filename

    # Parse headers to get absolute offsets
    header_cache = {}
    expert_reads = {}
    
    for layer_idx in sorted(expert_tensors.keys()):
        layer_reads = {}
        for comp_name, filename in expert_tensors[layer_idx].items():
            if filename not in header_cache:
                header_cache[filename] = parse_safetensors_header(model_path / filename)
            
            header, data_start = header_cache[filename]
            
            # Try both names
            keys_to_try = [
                f"language_model.model.layers.{layer_idx}.mlp.switch_mlp.{comp_name}",
                f"model.layers.{layer_idx}.mlp.switch_mlp.{comp_name}"
            ]
            
            meta = None
            for k in keys_to_try:
                if k in header:
                    meta = header[k]
                    break
            
            if meta is None:
                print(f"ERROR: No match for layer {layer_idx} component {comp_name} in {filename}")
                continue
                
            offsets = meta['data_offsets']
            abs_offset = data_start + offsets[0]
            total_size = offsets[1] - offsets[0]
            
            # For 122B: num_experts = 256
            num_experts = 256
            expert_size = total_size // num_experts
            
            layer_reads[comp_name] = {
                "file": filename,
                "abs_offset": abs_offset,
                "expert_stride": expert_size,
                "expert_size": expert_size
            }
        expert_reads[layer_idx] = layer_reads

    # Final structure expected by repack_experts.py
    index = {
        "model_path": str(model_path.absolute()),
        "expert_reads": expert_reads
    }
    
    with os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), 'w') as f:
        json.dump(index, f, indent=2)
    print(f"Generated {args.output} for {len(expert_reads)} layers")

if __name__ == '__main__':
    main()
