# GPU Expert Slot Benchmark Session

## Instruction

Before testing, rebuild the binaries with:

```bash
./0_harness.sh
```

Then run the comparable GPU expert slot sweep with:

```bash
GPU_EXPERT_SLOT_NUMS='-1 16 32 64 72' N_PARALLEL=2 N_PROMPTS=16 ./2_benchmark.sh
```

Cases:

- `-1`: slot mode disabled; normal llama.cpp placement.
- `16`, `32`, `64`, `72`: layer-local GPU expert slots per MoE layer.

Record throughput, prompt latency, peak GPU memory, and the llama.cpp CUDA memory breakdown for each case. Do not report CPU memory unless the benchmark script has a process RSS sampler.

Each case writes results to `benchmark-results/gpu_expert_slot_<N>/`: `server.log`, `server-bench.txt`, `server-bench.log`, `gpu-memory.csv`, and `gpu-memory-summary.txt`.

Keep this file table-first: Instruction, Summary, Throughput, GPU Resource, Bench Details, then Test Setup.

Do not update this Instruction section. Only update the rest of this file with the latest benchmark data.

## Summary

| GPU expert slots | GPU type | Max GPU MiB | Output tok/s | Req/s | Prompt latency ms | Status |
|---:|---|---:|---:|---:|---:|---|
| -1 | NVIDIA A100-SXM4-40GB | 38929 | 26.90 | 0.42 | 813.69 | OK |
| 16 | NVIDIA A100-SXM4-40GB | 12759 | 7.75 | 0.12 | 2744.25 | OK |
| 32 | NVIDIA A100-SXM4-40GB | 18519 | 8.55 | 0.13 | 1920.61 | OK |
| 64 | NVIDIA A100-SXM4-40GB | 30039 | 7.87 | 0.12 | 2794.93 | OK |
| 72 | NVIDIA A100-SXM4-40GB | 32919 | 7.91 | 0.12 | 2783.81 | OK |

## Throughput

| GPU expert slots | Duration s | Req/s | Req/min | Prompt latency ms | Prompt tok/s | Generated tokens | Generation tok/s | Generation tok/s/slot |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -1 | 38.07 | 0.42 | 25.22 | 813.69 | 78.65 | 1024 | 26.90 | 13.45 |
| 16 | 132.14 | 0.12 | 7.27 | 2744.25 | 23.32 | 1024 | 7.75 | 3.87 |
| 32 | 119.79 | 0.13 | 8.01 | 1920.61 | 33.32 | 1024 | 8.55 | 4.27 |
| 64 | 130.19 | 0.12 | 7.37 | 2794.93 | 22.90 | 1024 | 7.87 | 3.93 |
| 72 | 129.38 | 0.12 | 7.42 | 2783.81 | 22.99 | 1024 | 7.91 | 3.96 |

## GPU Resource

| GPU expert slots | Peak used MiB | Total VRAM MiB | Free at peak MiB | Model MiB | Context MiB | Compute MiB | Expert slot buffers | Materialized slots | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -1 | 38929 | 40960 | 2031 | 36510.29 | 880.00 | 1019.52 | 0.00 MiB | 0 | 50 |
| 16 | 12759 | 40960 | 28201 | 4590.35 | 880.00 | 1013.01 | ~5443 MiB | 480 | 131 |
| 32 | 18519 | 40960 | 22441 | 4590.35 | 880.00 | 1013.01 | ~10886 MiB | 960 | 119 |
| 64 | 30039 | 40960 | 10921 | 4590.35 | 880.00 | 1013.01 | ~21773 MiB | 1920 | 134 |
| 72 | 32919 | 40960 | 8041 | 4590.35 | 880.00 | 1013.01 | ~24494 MiB | 2160 | 133 |

## Bench Details

### GPU expert slots -1

- Status: OK
- Benchmark duration: `38.07 s`
- Request throughput: `0.42 req/s`
- Request throughput: `25.22 req/min`
- Total prompt tokens: `1024`
- Average prompt tokens: `64.00`
- Average prompt latency: `813.69 ms`
- Average prompt speed: `78.65 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `96.50 tokens`
- Output token throughput: `26.90 tok/s`
- Output token throughput per slot: `13.45 tok/s/slot`
- Peak GPU memory used: `38929 / 40960 MiB`
- Peak GPU memory free: `2031 MiB`
- llama.cpp CUDA breakdown: `36510.29 MiB model + 880.00 MiB context + 1019.52 MiB compute`
- Materialized GPU expert slots: `0`
- Result: `benchmark-results/gpu_expert_slot_-1/server-bench.txt`
- GPU memory summary: `benchmark-results/gpu_expert_slot_-1/gpu-memory-summary.txt`

### GPU expert slots 16

- Status: OK
- Benchmark duration: `132.14 s`
- Request throughput: `0.12 req/s`
- Request throughput: `7.27 req/min`
- Total prompt tokens: `1024`
- Average prompt tokens: `64.00`
- Average prompt latency: `2744.25 ms`
- Average prompt speed: `23.32 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `96.50 tokens`
- Output token throughput: `7.75 tok/s`
- Output token throughput per slot: `3.87 tok/s/slot`
- Peak GPU memory used: `12759 / 40960 MiB`
- Peak GPU memory free: `28201 MiB`
- llama.cpp CUDA breakdown: `4590.35 MiB model + 880.00 MiB context + 1013.01 MiB compute`
- Slot normalization: `requested=16 active=8 total=128 effective=16`
- Materialized GPU expert slots: `480` (`30` MoE layers x `16` layer-local slots)
- Estimated GPU expert slot buffers: `~5443 MiB`
- Result: `benchmark-results/gpu_expert_slot_16/server-bench.txt`
- GPU memory summary: `benchmark-results/gpu_expert_slot_16/gpu-memory-summary.txt`

### GPU expert slots 32

- Status: OK
- Benchmark duration: `119.79 s`
- Request throughput: `0.13 req/s`
- Request throughput: `8.01 req/min`
- Total prompt tokens: `1024`
- Average prompt tokens: `64.00`
- Average prompt latency: `1920.61 ms`
- Average prompt speed: `33.32 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `96.50 tokens`
- Output token throughput: `8.55 tok/s`
- Output token throughput per slot: `4.27 tok/s/slot`
- Peak GPU memory used: `18519 / 40960 MiB`
- Peak GPU memory free: `22441 MiB`
- llama.cpp CUDA breakdown: `4590.35 MiB model + 880.00 MiB context + 1013.01 MiB compute`
- Slot normalization: `requested=32 active=8 total=128 effective=32`
- Materialized GPU expert slots: `960` (`30` MoE layers x `32` layer-local slots)
- Estimated GPU expert slot buffers: `~10886 MiB`
- Result: `benchmark-results/gpu_expert_slot_32/server-bench.txt`
- GPU memory summary: `benchmark-results/gpu_expert_slot_32/gpu-memory-summary.txt`

### GPU expert slots 64

- Status: OK
- Benchmark duration: `130.19 s`
- Request throughput: `0.12 req/s`
- Request throughput: `7.37 req/min`
- Total prompt tokens: `1024`
- Average prompt tokens: `64.00`
- Average prompt latency: `2794.93 ms`
- Average prompt speed: `22.90 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `96.50 tokens`
- Output token throughput: `7.87 tok/s`
- Output token throughput per slot: `3.93 tok/s/slot`
- Peak GPU memory used: `30039 / 40960 MiB`
- Peak GPU memory free: `10921 MiB`
- llama.cpp CUDA breakdown: `4590.35 MiB model + 880.00 MiB context + 1013.01 MiB compute`
- Slot normalization: `requested=64 active=8 total=128 effective=64`
- Materialized GPU expert slots: `1920` (`30` MoE layers x `64` layer-local slots)
- Estimated GPU expert slot buffers: `~21773 MiB`
- Result: `benchmark-results/gpu_expert_slot_64/server-bench.txt`
- GPU memory summary: `benchmark-results/gpu_expert_slot_64/gpu-memory-summary.txt`

### GPU expert slots 72

- Status: OK
- Benchmark duration: `129.38 s`
- Request throughput: `0.12 req/s`
- Request throughput: `7.42 req/min`
- Total prompt tokens: `1024`
- Average prompt tokens: `64.00`
- Average prompt latency: `2783.81 ms`
- Average prompt speed: `22.99 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `96.50 tokens`
- Output token throughput: `7.91 tok/s`
- Output token throughput per slot: `3.96 tok/s/slot`
- Peak GPU memory used: `32919 / 40960 MiB`
- Peak GPU memory free: `8041 MiB`
- llama.cpp CUDA breakdown: `4590.35 MiB model + 880.00 MiB context + 1013.01 MiB compute`
- Slot normalization: `requested=72 active=8 total=128 effective=72`
- Materialized GPU expert slots: `2160` (`30` MoE layers x `72` layer-local slots)
- Estimated GPU expert slot buffers: `~24494 MiB`
- Result: `benchmark-results/gpu_expert_slot_72/server-bench.txt`
- GPU memory summary: `benchmark-results/gpu_expert_slot_72/gpu-memory-summary.txt`

## Test Setup

GPU expert slot benchmark sweep for `--moe-gpu-expert-slot-num` after per-layer GPU expert slot buffer materialization.

Common settings:

- Run date: `2026-05-02`
- GPU: `0`
- GPU type: `NVIDIA A100-SXM4-40GB`
- Model: `/models/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it.gguf`
- Server script: `./1_serving.sh`
- Benchmark script: `./2_benchmark.sh`
- Server URL: `http://127.0.0.1:8071`
- Benchmark mode: `throughput`
- Benchmark prompts: `16`
- Parallel requests: `2`
- Prompt source: `rng-64-64`
- Input length: `64`
- Output length: `64`
- Context size: `4096`
- CUDA devices: `CUDA_VISIBLE_DEVICES=0`
- GPU memory sampling: `nvidia-smi --id=0`, 1 second interval
- CPU memory sampling: not collected

Build command:

```bash
./0_harness.sh
```

Benchmark command:

```bash
GPU_EXPERT_SLOT_NUMS='-1 16 32 64 72' N_PARALLEL=2 N_PROMPTS=16 ./2_benchmark.sh
```

Notes:

- Slot `-1` is the baseline with GPU expert slot mode disabled. It follows normal llama.cpp MoE placement and uses `--n-gpu-layers`.
- Slot mode normalizes `MOE_GPU_EXPERT_SLOT_NUM` against the model metadata. For this model, `active=8` and `total=128`, so the tested values normalize to the same effective counts: `16`, `32`, `64`, and `72`.
- Slot `-1` uses normal llama.cpp GPU layer offload and keeps most model weights on GPU.
- Slot mode keeps the CPU-mapped expert pool as the source of truth and materializes per-layer GPU expert slot buffers at startup.
- The materialized slot count is `30` MoE layers times the effective slot count.
- Each materialized expert slot is logged as `11.34 MiB`; the slot buffer totals in the table are estimates from that rounded log value.
- `2_benchmark.sh` now saves `server.log`, `server-bench.txt`, `server-bench.log`, `gpu-memory.csv`, and `gpu-memory-summary.txt` in each slot directory.
