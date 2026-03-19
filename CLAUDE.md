# Flash-MoE: High-Performance MoE on Mac (397B & 122B Models)

> **Note**: This is a fork of the original [Flash-MoE](https://github.com/danveloper/flash-moe) repository, specifically adapted to run the **Qwen3.5-122B-A10B** model on memory-constrained Macs (e.g., M4 Pro with 24GB RAM).
> 
> **[Read the original paper](paper/flash_moe.pdf)** — Full technical details, 90+ experiments, and the story of how an AI and a human built this in 24 hours.

Pure C/Metal inference engine that runs **Qwen3.5-397B-A17B** and **Qwen3.5-122B-A10B** (Mixture-of-Experts models) on a MacBook Pro with 24GB-48GB RAM at up to **7.6+ tokens/second** with production-quality output.

The models stream from SSD through a custom Metal compute pipeline. No Python in the inference loop. Just C, Objective-C, and hand-tuned Metal shaders.

## Results

| Model | Configuration | tok/s | Quality | Notes |
|-------|---------------|-------|---------|-------|
| 122B-A10B | **2-bit experts (M4 Pro)** | **7.62** | Perfect | 35GB on disk. 24GB RAM. |
| 122B-A10B | 4-bit experts (M4 Pro) | **6.12** | Perfect | 61GB on disk. SSD-bound. |
| 397B-A17B | **2-bit experts (M3 Max)** | **5.55** | Excellent | 120GB on disk. 48GB RAM. |
| 397B-A17B | 4-bit experts (M3 Max) | 4.80 | Excellent | 209GB on disk. |

## Hardware

- **Machine A**: MacBook Pro, Apple M3 Max (48GB RAM, 1TB SSD)
- **Machine B**: MacBook Pro, Apple M4 Pro (24GB RAM, 512GB SSD)
- **Measured SSD Read**: up to **17.5 GB/s** (M3 Max) and **~6.5 GB/s** (M4 Pro)

## Architecture

- **For 397B-A17B**: 60 layers (45 GatedDeltaNet + 15 standard attention). Hidden dim 4096. 512 experts.
- **For 122B-A10B**: 48 layers (36 GatedDeltaNet + 12 standard attention). Hidden dim 3072. 256 experts.
- **Expert Streaming**: K=4 experts/layer read from SSD via `pread()` (mmap for weights).
- **2-bit Quantization**: Custom affine dequantization in Metal (16 values/U32).
- **F_NOCACHE**: Mandatory for memory-constrained machines (24GB) to avoid OS page thrash.

### Key Techniques

1. **SSD Expert Streaming** — Expert weights (120GB total at 2-bit) are read from NVMe SSD on demand via parallel `pread()`. Only the K=4 active experts per layer are loaded (~3.9MB each). Inspired by Apple's "LLM in a Flash" paper.

2. **2-bit Expert Quantization** — Custom requantization from MLX's 4-bit affine format to 2-bit affine (16 values per uint32). 44% size reduction with RMSE ~0.001. Quality preserved across math, code, and reasoning tasks.

3. **Metal Compute Shaders** — Hand-written Metal kernels for:
   - 4-bit and 2-bit dequantized matrix-vector multiply (tiled, SIMD-reduced, shared input cache)
   - Fused SwiGLU activation
   - RMS normalization (two-pass: sum-of-squares reduction + apply)
   - Batched GPU attention (Q@K^T, softmax, scores@V) for full attention layers
   - GPU RoPE (fused with Q deinterleave and K normalization)
   - MoE combine + residual + sigmoid gate (fused kernel)

4. **Deferred GPU Expert Compute** — CMD3 (expert forward pass) is submitted without waiting. The GPU executes it while the CPU prepares the next layer. The combine + residual + norm are also on GPU, feeding directly into the next layer's attention projections.

5. **Accelerate BLAS for Linear Attention** — The GatedDeltaNet recurrence uses `cblas_sscal`, `cblas_sgemv`, and `cblas_sger` for the 64-head × 128×128 state matrix update. 64% faster than scalar code.

6. **F_NOCACHE for Direct SSD Access** — Bypasses the OS page cache for expert files when using 2-bit mode. With 120GB >> 35GB available cache, page caching thrashes. Direct I/O avoids eviction overhead.

### Pipeline Per Layer (3.14ms average at 2-bit)

```
CMD3(prev) → CMD1: attention projections  [0.87ms GPU]
           → CPU: GatedDeltaNet / full attention  [0.27ms CPU+BLAS]
           → CMD2: o_proj + residual + norm + routing + shared expert  [0.45ms GPU]
           → CPU: softmax + topK routing  [0.003ms]
           → I/O: parallel pread K=4 experts  [1.49ms SSD]
           → CMD3: expert forward + combine + norm (DEFERRED)  [0.03ms encode]
```

## Quick Start (Shortcut)

The easiest way to run the 122B model on your Mac is using the `flash.sh` shortcut:

```bash
cd metal_infer
make
cd ..

# Start the server (Speed mode: 2-bit, K=4)
./flash.sh server speed

# Open a new terminal to start the chat client
./flash.sh chat speed
```

*For maximum accuracy (4-bit, K=8), use `./flash.sh server accuracy` and `./flash.sh chat accuracy`.*

## Project Structure

```
metal_infer/
  infer.m              # Complete inference engine (~6800 lines)
  shaders.metal        # Metal compute kernels (~1200 lines)
  chat.m               # Interactive TUI chat client (TUI + Markdown)
  tokenizer.h          # Pure C BPE Tokenizer (GPT-2/Qwen style)
  Makefile             # Build system (make chat)
  gen_expert_index.py  # Maps weights to layers from safetensors index
  export_tokenizer.py  # Binary BPET export from tokenizer.json
  gen_simple_vocab.py  # Decodes BPE byte-level strings for TUI
  repack_experts_2bit.py  # 4-bit -> 2-bit expert requantization
  repack_experts.py    # 4-bit expert packing from safetensors
  flash.sh             # Convenience script for quick launch
```

## What We Tried (and What Worked)

| Approach | Result | Verdict |
|----------|--------|---------|
| 2-bit expert quantization | +95% speed, quality preserved | **KEEP** |
| GPU combine+norm in CMD3 | Eliminates CPU round-trip | **KEEP** |
| BLAS delta-net (Accelerate) | cpu_attn 0.78→0.28ms | **KEEP** |
| F_NOCACHE for 2-bit | +3% from avoiding page thrash | **KEEP** |
| GPU fused attention (RoPE kernels) | +2% for full-attn layers | **KEEP** |
| Pre-allocated Metal LRU cache (500) | 35% hit rate, marginal for 2-bit | Neutral |
| mmap expert files | 5x SLOWER (page fault overhead) | Reverted |
| Metal cache >500 entries | GPU memory pressure kills perf | Reverted |
| Malloc zero-copy cache (17GB) | Slower than Metal LRU | Reverted |
| Speculative early routing | Cache pollution + overhead | Reverted |
| GPU delta-net (195MB state) | Memory pressure > compute savings | Disabled |
| CMD1+CMD2 merge via GPU RoPE | Dispatch overhead > sync savings | Reverted |
| Reduced `MAX_SEQ_LEN` | Prevents OOM by limiting KV cache on 24GB Macs | **KEEP** |
| BPE Byte-Decoding Layer | Fixes garbage chars (Ġ, Ċ) in chat | **KEEP** |
| Custom Expert Indexer | Handles Qwen3.5 122B naming variations | **KEEP** |
| SSD pread() multi-expert | Much faster than mmap on memory-constrained Macs | **KEEP** |

## Adaptation for Qwen3.5-122B-A10B

1. **Architecture**: Adjusted `HIDDEN_DIM`, `NUM_LAYERS`, and expert counts in `infer.m`.
2. **Quantization**: Calculated new offsets for 2-bit expert packing to handle 1024-intermediate dim.
3. **KV Cache**: Reduced `MAX_SEQ_LEN` from 1M to 32k to fit within the 24GB unified memory.
4. **Tokenizer**: Expanded `vocab.bin` to include special tokens like `<think>` and `</think>`.

## Safety

This is a primary development machine. The engine explicitly controls memory:
- Non-expert weights: 5.5GB (mmap'd, read-only)
- Metal scratch buffers: ~200MB
- Expert cache (optional): 0-3.5GB
- Total: 6-9GB, leaving 39-42GB for OS + page cache
- No OOM risk. Expert data streams from SSD on demand.
