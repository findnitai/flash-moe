#!/usr/bin/env python3
"""Repack expert weights from scattered safetensors into contiguous per-layer binary files.

Creates one binary file per layer: packed_experts/layer_XX.bin
Each file = 512 experts x 7,077,888 bytes = ~3.63 GB
Expert E starts at byte offset E * 7,077,888

Within each expert block, 9 components packed in fixed order:
  gate_proj.weight, gate_proj.scales, gate_proj.biases,
  up_proj.weight,   up_proj.scales,   up_proj.biases,
  down_proj.weight,  down_proj.scales,  down_proj.biases

Usage:
    python repack_experts.py                    # repack all 60 layers
    python repack_experts.py --layers 0-4       # repack layers 0-4
    python repack_experts.py --layers 0,5,10    # repack specific layers
    python repack_experts.py --dry-run           # verify without writing
    python repack_experts.py --verify-only 0     # verify layer 0 against originals
"""

import argparse
import json
import os
import time
import sys

def get_components(hidden_size, moe_intermediate_size):
    k_dim = hidden_size // 8
    g_dim = hidden_size // 64
    d_k_dim = moe_intermediate_size // 8
    d_g_dim = moe_intermediate_size // 64

    # Size formulas: weight=U32 (4 bytes), scales/biases=BF16 (2 bytes)
    gate_wt_sz = moe_intermediate_size * k_dim * 4
    gate_sc_sz = moe_intermediate_size * g_dim * 2
    
    down_wt_sz = hidden_size * d_k_dim * 4
    down_sc_sz = hidden_size * d_g_dim * 2

    comps = [
        {"name": "gate_proj.weight",  "size": gate_wt_sz, "dtype": "U32"},
        {"name": "gate_proj.scales",  "size": gate_sc_sz, "dtype": "BF16"},
        {"name": "gate_proj.biases",  "size": gate_sc_sz, "dtype": "BF16"},
        {"name": "up_proj.weight",    "size": gate_wt_sz, "dtype": "U32"},
        {"name": "up_proj.scales",    "size": gate_sc_sz, "dtype": "BF16"},
        {"name": "up_proj.biases",    "size": gate_sc_sz, "dtype": "BF16"},
        {"name": "down_proj.weight",  "size": down_wt_sz, "dtype": "U32"},
        {"name": "down_proj.scales",  "size": down_sc_sz, "dtype": "BF16"},
        {"name": "down_proj.biases",  "size": down_sc_sz, "dtype": "BF16"},
    ]
    
    offset = 0
    for c in comps:
        c["offset"] = offset
        offset += c["size"]
    return comps, offset

NUM_EXPERTS = 256


def parse_layers(spec, num_layers=48):
    """Parse layer specification like '0-4' or '0,5,10' or 'all'."""
    if spec is None or spec == 'all':
        return list(range(num_layers))
    layers = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_index(index_path):
    """Load expert_index.json and return expert_reads dict + model_path + config."""
    with open(index_path) as f:
        idx = json.load(f)
    model_path = idx['model_path']
    config = {}
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f).get('text_config', {})
    return idx['expert_reads'], model_path, config


def verify_component_sizes(expert_reads, components):
    """Verify that component sizes in the index match expected sizes."""
    expected = {c['name']: c['size'] for c in components}
    for layer_key, comps in expert_reads.items():
        for comp_name, info in comps.items():
            if comp_name not in expected:
                print(f"WARNING: unknown component {comp_name} in layer {layer_key}")
                # The original diff had an issue here, `comp_info` was not defined.
                # If we don't expect it, we should probably just skip it or warn.
                continue
            if info['expert_size'] != expected[comp_name]:
                print(f"MISMATCH: layer {layer_key}, {comp_name}: "
                      f"index says {info['expert_size']}, expected {expected[comp_name]}")
                return False
    print("Component sizes verified: all match expected layout")
    return True


def open_source_files(expert_reads, model_path, layers):
    """Open all needed safetensors files, return {filename: fd}."""
    needed_files = set()
    for layer_idx in layers:
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            print(f"WARNING: layer {layer_idx} not found in expert_reads")
            continue
        for info in expert_reads[layer_key].values():
            needed_files.add(info['file'])

    fds = {}
    for fname in sorted(needed_files):
        path = os.path.join(model_path, fname)
        fds[fname] = os.open(path, os.O_RDONLY)
    print(f"Opened {len(fds)} source safetensors files")
    return fds


def pack_layer(layer_idx, expert_reads, components, expert_size, fds, output_dir, dry_run=False):
    """Repack all 512 experts for one layer into a contiguous binary file.

    Returns (bytes_written, elapsed_seconds).
    """
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: NOT FOUND in index, skipping")
        return 0, 0.0

    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")
    layer_total_size = NUM_EXPERTS * expert_size

    if dry_run:
        # Just verify we can compute all offsets
        for expert_idx in range(NUM_EXPERTS):
            for comp in components:
                info = layer_info[comp['name']]
                src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
                dst_offset = expert_idx * expert_size + comp['offset']
        print(f"  Layer {layer_idx:2d}: DRY RUN OK — would write {layer_total_size:,} bytes to {out_path}")
        return layer_total_size, 0.0

    t0 = time.monotonic()

    # Pre-allocate output file with zeros
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, layer_total_size)

    bytes_written = 0

    # Build read plan: group reads by source file for better locality
    # Each entry: (src_fd, src_offset, dst_offset, size)
    read_plan = []
    for expert_idx in range(NUM_EXPERTS):
        for comp in components:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * expert_size + comp['offset']
            read_plan.append((src_fd, src_offset, dst_offset, comp['size']))

    # Sort by (src_fd, src_offset) for sequential read locality
    read_plan.sort(key=lambda x: (x[0], x[1]))

    # Execute reads and writes
    for src_fd, src_offset, dst_offset, size in read_plan:
        data = os.pread(src_fd, size, src_offset)
        if len(data) != size:
            raise IOError(f"Short read: expected {size}, got {len(data)} "
                          f"at offset {src_offset}")
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += size

    os.close(fd_out)
    elapsed = time.monotonic() - t0

    return bytes_written, elapsed


def verify_layer(layer_idx, expert_reads, components, expert_size, fds, output_dir):
    """Read back expert 0 from packed file and compare to originals."""
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: NOT FOUND in index, skipping verification")
        return False

    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if not os.path.exists(out_path):
        print(f"  Layer {layer_idx}: packed file not found")
        return False

    fd_packed = os.open(out_path, os.O_RDONLY)

    mismatches = 0
    # spot check several experts: first, middle, last
    for expert_idx in [0, 1, NUM_EXPERTS // 2, NUM_EXPERTS - 1]:
        for comp in components:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * expert_size + comp['offset']

            original = os.pread(src_fd, comp['size'], src_offset)
            packed = os.pread(fd_packed, comp['size'], dst_offset)

            if original != packed:
                print(f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {comp['name']}")
                mismatches += 1

    os.close(fd_packed)

    if mismatches == 0:
        print(f"  Layer {layer_idx}: verification PASSED (experts 0, 1, {NUM_EXPERTS // 2}, {NUM_EXPERTS - 1})")
    else:
        print(f"  Layer {layer_idx}: verification FAILED ({mismatches} mismatches)")

    return mismatches == 0


def write_layout(output_dir, components, expert_size, num_layers):
    """Write layout.json describing the packed format."""
    layout = {
        "expert_size": expert_size,
        "num_layers": num_layers,
        "num_experts": NUM_EXPERTS,
        "components": components,
    }
    path = os.path.join(output_dir, "layout.json")
    with open(path, 'w') as f:
        json.dump(layout, f, indent=2)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Repack expert weights into contiguous per-layer binary files")
    parser.add_argument('--index', default='/Users/danielwoods/Workspace/ane-research/expert_index.json',
                        help='Path to expert_index.json')
    parser.add_argument('--layers', default=None,
                        help='Layer spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Verify offsets without writing')
    parser.add_argument('--verify-only', type=int, default=None, metavar='LAYER',
                        help='Verify a specific layer against originals')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for packed expert files (default: packed_experts/ inside model dir)')
    args = parser.parse_args()

    print("Loading expert index...")
    expert_reads, model_path, config = load_index(args.index)
    
    hidden_size = config.get('hidden_size', 3072)
    moe_intermediate_size = config.get('moe_intermediate_size', 1024)
    components, expert_size = get_components(hidden_size, moe_intermediate_size)
    num_layers_total = config.get('num_hidden_layers', 48) # Renamed to avoid conflict with 'layers' list

    output_dir = args.output if args.output else os.path.join(model_path, "packed_experts")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine which layers to process
    layers_to_process = parse_layers(args.layers, num_layers_total)

    print(f"Layers to process: {layers_to_process[0]}-{layers_to_process[-1]} ({len(layers_to_process)} layers)")

    # Open source files once for all layers
    fds = open_source_files(expert_reads, model_path, layers_to_process)

    if args.verify_only is not None:
        if args.verify_only not in layers_to_process:
            print(f"WARNING: Layer {args.verify_only} not in specified layers, adding for verification.")
            layers_to_process = sorted(list(set(layers_to_process + [args.verify_only])))
        
        print(f"Verifying layer {args.verify_only}...")
        verify_layer(args.verify_only, expert_reads, components, expert_size, fds, output_dir)
        for fd in fds.values():
            os.close(fd)
        sys.exit(0)

    if not verify_component_sizes(expert_reads, components):
        print("ABORTING: component size mismatch")
        sys.exit(1)

    layer_total_size = NUM_EXPERTS * expert_size
    if not args.dry_run:
        total_bytes_to_write = len(layers_to_process) * layer_total_size
        print(f"Total data to write: {total_bytes_to_write / (1024**3):.1f} GB")

        # Check free disk space
        stat = os.statvfs(output_dir)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)
        needed_gb = total_bytes_to_write / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB, needed: {needed_gb:.1f} GB")
        if free_bytes < total_bytes_to_write:
            print(f"WARNING: Not enough free space! Need {needed_gb:.1f} GB but only {free_gb:.1f} GB free.")
            print(f"Hint: use --layers to process a subset, e.g. --layers 0-{int(free_gb / 3.63) - 1}")
            sys.exit(1)

    # Open source files once for all layers
    fds = open_source_files(expert_reads, model_path, layers_to_process)

    if args.verify_only is not None:
        verify_layer(args.verify_only, expert_reads, components, expert_size, fds, output_dir)
        for fd in fds.values():
            os.close(fd)
        return

    # Write layout.json
    write_layout(output_dir, components, expert_size, num_layers_total)

    # Repack each layer
    t_start = time.monotonic()
    total_written = 0
    layer_size = NUM_EXPERTS * expert_size

    for i, layer_idx in enumerate(layers_to_process):
        t_layer = time.monotonic()
        bytes_written, elapsed = pack_layer(
            layer_idx, expert_reads, components, expert_size, fds, output_dir, dry_run=args.dry_run
        )
        total_written += bytes_written

        if not args.dry_run and bytes_written > 0:
            throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else float('inf')
            overall_elapsed = time.monotonic() - t_start
            overall_throughput = total_written / overall_elapsed / (1024**3) if overall_elapsed > 0 else 0
            eta = (len(layers_to_process) - i - 1) * (overall_elapsed / (i + 1))
            print(f"  Layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s "
                  f"({throughput:.1f} GB/s) | "
                  f"Total: {total_written/1024**3:.1f}/{len(layers_to_process)*layer_size/1024**3:.1f} GB "
                  f"({overall_throughput:.1f} GB/s avg) | "
                  f"ETA: {eta:.0f}s")

            if not verify_layer(layer_idx, expert_reads, components, expert_size, fds, output_dir):
                print(f"ABORTING: verification failed for layer {layer_idx}")
                sys.exit(1)

    # Close source files
    for fd in fds.values():
        os.close(fd)

    # Final summary
    total_elapsed = time.monotonic() - t_start
    if not args.dry_run and total_written > 0:
        print(f"\n{'='*60}")
        print(f"DONE: {total_written:,} bytes ({total_written/1024**3:.1f} GB) written")
        print(f"Time: {total_elapsed:.1f}s")
        print(f"Throughput: {total_written/total_elapsed/1024**3:.1f} GB/s")
        print(f"Output: {output_dir}")
    elif args.dry_run:
        print(f"\nDRY RUN complete: {len(layers_to_process)} layers validated")


if __name__ == '__main__':
    main()
