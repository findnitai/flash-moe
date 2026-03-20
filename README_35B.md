# Flash-MoE: Qwen3.5-35B-A3B Support

This document covers the specific setup and usage for the **Qwen3.5-35B-A3B** model in Flash-MoE.

## Overview

The Qwen3.5-35B-A3B model is a Mixture-of-Experts (MoE) model with:
- **Total Parameters**: ~35B
- **Active Parameters**: ~3B per token
- **Architecture**: 40 layers, 16 heads, hidden dim 2048, 256 experts.
- **Performance**: ~11-21 tokens/second on M4 Pro (24GB RAM).

## Requirements

- **Memory**: 16GB RAM minimum (24GB recommended).
- **Disk Space**: ~20GB for packed weights.
- **Metal Support**: Apple Silicon Mac (M1/M2/M3/M4).

## Setup Instructions

### 1. Download the Weights

Download the 4-bit MLX version of the model:

```bash
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit --local-dir qwen-35b-4bit
```

### 2. Process the Weights

Run the conversion pipeline to extract non-expert weights and repack experts into the optimized binary format:

```bash
# Extract non-expert weights (projections, norms, etc.)
python3 extract_weights.py --model qwen-35b-4bit

# Generate expert index
python3 gen_expert_index.py --model qwen-35b-4bit > expert_index_35b.json

# Pack experts into per-layer binary files
python3 repack_experts.py --model qwen-35b-4bit --index expert_index_35b.json --output qwen-35b-4bit/packed_experts

# Generate binary vocabulary
python3 gen_simple_vocab.py qwen-35b-4bit/tokenizer.json qwen-35b-4bit/vocab.bin
```

### 3. Build the Engine

Compile the dedicated 35B inference binary:

```bash
cd metal_infer
make 35b
```

This creates a separate binary `infer_35b` so you can keep the 122B engine (`infer`) side-by-side.

## Usage

Run inference with the `infer_35b` binary:

```bash
./metal_infer/infer_35b \
    --weights qwen-35b-4bit/model_weights.bin \
    --manifest qwen-35b-4bit/model_weights.json \
    --vocab qwen-35b-4bit/vocab.bin \
    --model qwen-35b-4bit \
    --k 8 \
    --tokens 100 \
    --quiet \
    --prompt "Explain quantum entanglement in simple terms."
```

### Options

- `--k <N>`: Set number of active experts per layer (default 8). Lower k (e.g., 4) is faster but less accurate.
- `--tokens <N>`: Maximum tokens to generate.
- `--quiet`: Only output the generated text.
- `--chat`: (Planned) Interactive chat mode.

## Architecture & Optimizations

- **Fused in_proj_qkv**: The model uses a fused projection for query, key, and value in the linear attention layers.
- **Explicit HEAD_DIM**: Unlike smaller models, 35B uses an explicit `head_dim=256` which is not derived purely from `hidden_size / num_heads`.
- **Smaller Expert Size**: Experts are ~1.7MB each (4-bit), significantly smaller than the 122B model's 5.1MB experts.

## Performance & Results

Tested on **Apple M4 Pro (24GB RAM)** with **k=8** (Max Accuracy mode):

- **Throughput**: **16.34 tokens/second**
- **TTFT**: ~3.7s (including prefill)

### Sample Output

**Prompt**: *"Write a detailed explanation of how a Mixture-of-Experts (MoE) transformer architecture works, including the gating mechanism and expert routing."*

**Response**:
> "The **Mixture-of-Experts (MoE)** architecture is a specialized design within the Transformer family that aims to drastically increase model capacity... It achieves this by activating only a subset of the model's parameters for any given input token...
>
> 1. **Core Concept: Sparse Activation**
> In an MoE Transformer, the standard FFN layer is replaced by an MoE Layer. This layer consists of N distinct sub-networks, called Experts... For any specific input token, the model does not use all N experts. Instead, it selects only the top-K experts..."
