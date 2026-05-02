# GPU Expert Slot Benchmark Session

GPU expert slot benchmark sweep for `--moe-gpu-expert-slot-num` with fixed llama.cpp server load.

Common settings:

- GPU: `0`
- Device: `NVIDIA A100-SXM4-40GB`
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

Command:

```bash
GPU_EXPERT_SLOT_NUMS='0 32 64 72' N_PARALLEL=2 N_PROMPTS=16 ./2_benchmark.sh
```

Notes:

- Slot `0` is the baseline with the GPU expert slot cache disabled.
- The script default sweep is `0 16 32 64 72`; this table records the completed comparable run set `0 32 64 72`.
- A previous `N_PARALLEL=4 N_PROMPTS=32` run aborted mid-benchmark, so the recorded table uses the stable `N_PARALLEL=2 N_PROMPTS=16` load.
- `2_benchmark.sh` captures `server-bench.py` stderr with stdout so the metric block is saved in each `server-bench.txt`.

## Summary

| GPU expert slots | Output tok/s | Req/s | Prompt latency ms | Status |
|---:|---:|---:|---:|---|
| 0 | 27.15 | 0.42 | 700.15 | OK |
| 32 | 25.59 | 0.40 | 908.67 | OK |
| 64 | 25.25 | 0.39 | 885.48 | OK |
| 72 | 24.89 | 0.39 | 917.90 | OK |

## Bench Details

### GPU expert slots 0

- Status: OK
- Benchmark duration: `37.72 s`
- Request throughput: `0.42 req/s`
- Request throughput: `25.45 req/min`
- Total prompt tokens: `1024`
- Average prompt tokens: `64.00`
- Average prompt latency: `700.15 ms`
- Average prompt speed: `91.41 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `96.50 tokens`
- Output token throughput: `27.15 tok/s`
- Output token throughput per slot: `13.57 tok/s/slot`
- Result: `benchmark-results/gpu_expert_slot_0/server-bench.txt`

### GPU expert slots 32

- Status: OK
- Benchmark duration: `40.02 s`
- Request throughput: `0.40 req/s`
- Request throughput: `23.99 req/min`
- Total prompt tokens: `1024`
- Average prompt tokens: `64.00`
- Average prompt latency: `908.67 ms`
- Average prompt speed: `70.43 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `96.50 tokens`
- Output token throughput: `25.59 tok/s`
- Output token throughput per slot: `12.79 tok/s/slot`
- Result: `benchmark-results/gpu_expert_slot_32/server-bench.txt`

### GPU expert slots 64

- Status: OK
- Benchmark duration: `40.56 s`
- Request throughput: `0.39 req/s`
- Request throughput: `23.67 req/min`
- Total prompt tokens: `1024`
- Average prompt tokens: `64.00`
- Average prompt latency: `885.48 ms`
- Average prompt speed: `72.28 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `96.50 tokens`
- Output token throughput: `25.25 tok/s`
- Output token throughput per slot: `12.62 tok/s/slot`
- Result: `benchmark-results/gpu_expert_slot_64/server-bench.txt`

### GPU expert slots 72

- Status: OK
- Benchmark duration: `41.15 s`
- Request throughput: `0.39 req/s`
- Request throughput: `23.33 req/min`
- Total prompt tokens: `1024`
- Average prompt tokens: `64.00`
- Average prompt latency: `917.90 ms`
- Average prompt speed: `69.72 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `96.50 tokens`
- Output token throughput: `24.89 tok/s`
- Output token throughput per slot: `12.44 tok/s/slot`
- Result: `benchmark-results/gpu_expert_slot_72/server-bench.txt`
