# GPU Expert Slot Locality Benchmark Session

## Instruction

Before testing, rebuild the binaries with:

```bash
./0_harness.sh
```

Then run the repeated-prompt locality sweep with:

```bash
GPU_EXPERT_SLOT_NUMS='-1 16 72' REPEATS_PER_PROMPT_LIST='16 32' ./3_locality_benchmark.sh
```

Cases:

- `-1`: slot mode disabled; normal llama.cpp placement.
- `16`, `72`: layer-local GPU expert slots per MoE layer.
- Repeat counts `16`, `32`: repeated adjacent requests for each prompt group.

Record throughput, prompt latency, peak GPU memory, and the llama.cpp CUDA memory breakdown for each GPU slot case and repeat count. Do not report CPU memory unless the benchmark script has a process RSS sampler.

Each slot case writes server-level files to `locality-results/gpu_expert_slot_<N>/`: `server.log`, `gpu-memory.csv`, and `gpu-memory-summary.txt`.

Each repeat case writes benchmark files to `locality-results/gpu_expert_slot_<N>/repeat_<R>/`: `locality-bench.txt`, `locality-results.jsonl`, `repeated-prompts.jsonl`, `gen_rate.png`, and `prompt_time_by_repeat.png`.

Keep this file table-first: Instruction, Summary, Throughput, GPU Resource, Bench Details, then Test Setup.

Do not update this Instruction section. Only update the rest of this file with the latest locality benchmark data.

## Summary

| GPU expert slots | Repeat count | GPU type | Max GPU MiB | Output tok/s | Req/s | Prompt latency ms | Status |
|---:|---:|---|---:|---:|---:|---:|---|
| -1 | 16 | NVIDIA A100-SXM4-40GB | 38929 | 27.68 | 0.43 | 646.61 | OK |
| -1 | 32 | NVIDIA A100-SXM4-40GB | 38929 | 26.89 | 0.42 | 692.51 | OK |
| 16 | 16 | NVIDIA A100-SXM4-40GB | 12759 | 8.16 | 0.13 | 1887.25 | OK |
| 16 | 32 | NVIDIA A100-SXM4-40GB | 12759 | 8.29 | 0.13 | 1955.05 | OK |
| 72 | 16 | NVIDIA A100-SXM4-40GB | 32919 | 8.50 | 0.13 | 1999.73 | OK |
| 72 | 32 | NVIDIA A100-SXM4-40GB | 32919 | 8.18 | 0.13 | 1946.72 | OK |

## Throughput

| GPU expert slots | Repeat count | Requests | Duration s | Req/s | Req/min | Prompt latency ms | Prompt tok/s | Generated tokens | Generation tok/s | Generation tok/s/slot |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -1 | 16 | 16 | 37.00 | 0.43 | 25.95 | 646.61 | 58.77 | 1024 | 27.68 | 13.84 |
| -1 | 32 | 32 | 76.15 | 0.42 | 25.21 | 692.51 | 54.87 | 2048 | 26.89 | 13.45 |
| 16 | 16 | 16 | 125.44 | 0.13 | 7.65 | 1887.25 | 20.14 | 1024 | 8.16 | 4.08 |
| 16 | 32 | 32 | 247.08 | 0.13 | 7.77 | 1955.05 | 19.44 | 2048 | 8.29 | 4.14 |
| 72 | 16 | 16 | 120.42 | 0.13 | 7.97 | 1999.73 | 19.00 | 1024 | 8.50 | 4.25 |
| 72 | 32 | 32 | 250.52 | 0.13 | 7.66 | 1946.72 | 19.52 | 2048 | 8.18 | 4.09 |

## GPU Resource

| GPU expert slots | Peak used MiB | Total VRAM MiB | Free at peak MiB | Model MiB | Context MiB | Compute MiB | Expert slot buffers | Materialized slots | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -1 | 38929 | 40960 | 2031 | 36510.29 | 880.00 | 1019.52 | 0.00 MiB | 0 | 286 |
| 16 | 12759 | 40960 | 28201 | 4590.35 | 880.00 | 1013.01 | ~5443 MiB | 480 | 345 |
| 72 | 32919 | 40960 | 8041 | 4590.35 | 880.00 | 1013.01 | ~24494 MiB | 2160 | 346 |

## Bench Details

### GPU expert slots -1, repeats 16

- Status: OK
- Prompt groups: `1`
- Repeats per prompt: `16`
- Total requests: `16`
- Parallel requests: `2`
- Benchmark duration: `37.00 s`
- Request throughput: `0.43 req/s`
- Request throughput: `25.95 req/min`
- Total prompt tokens: `608`
- Average prompt tokens: `38.00`
- Average prompt latency: `646.61 ms`
- Average prompt speed: `58.77 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `70.50 tokens`
- Output token throughput: `27.68 tok/s`
- Output token throughput per slot: `13.84 tok/s/slot`
- Peak GPU memory used: `38929 / 40960 MiB`
- Result: `locality-results/gpu_expert_slot_-1/repeat_16/locality-bench.txt`

### GPU expert slots -1, repeats 32

- Status: OK
- Prompt groups: `1`
- Repeats per prompt: `32`
- Total requests: `32`
- Parallel requests: `2`
- Benchmark duration: `76.15 s`
- Request throughput: `0.42 req/s`
- Request throughput: `25.21 req/min`
- Total prompt tokens: `1216`
- Average prompt tokens: `38.00`
- Average prompt latency: `692.51 ms`
- Average prompt speed: `54.87 tok/s`
- Total generated tokens: `2048`
- Average generation depth: `70.50 tokens`
- Output token throughput: `26.89 tok/s`
- Output token throughput per slot: `13.45 tok/s/slot`
- Peak GPU memory used: `38929 / 40960 MiB`
- Result: `locality-results/gpu_expert_slot_-1/repeat_32/locality-bench.txt`

### GPU expert slots 16, repeats 16

- Status: OK
- Prompt groups: `1`
- Repeats per prompt: `16`
- Total requests: `16`
- Parallel requests: `2`
- Benchmark duration: `125.44 s`
- Request throughput: `0.13 req/s`
- Request throughput: `7.65 req/min`
- Total prompt tokens: `608`
- Average prompt tokens: `38.00`
- Average prompt latency: `1887.25 ms`
- Average prompt speed: `20.14 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `70.50 tokens`
- Output token throughput: `8.16 tok/s`
- Output token throughput per slot: `4.08 tok/s/slot`
- Peak GPU memory used: `12759 / 40960 MiB`
- Slot normalization: `requested=16 active=8 total=128 effective=16`
- Materialized GPU expert slots: `480` (`30` MoE layers x `16` layer-local slots)
- Result: `locality-results/gpu_expert_slot_16/repeat_16/locality-bench.txt`

### GPU expert slots 16, repeats 32

- Status: OK
- Prompt groups: `1`
- Repeats per prompt: `32`
- Total requests: `32`
- Parallel requests: `2`
- Benchmark duration: `247.08 s`
- Request throughput: `0.13 req/s`
- Request throughput: `7.77 req/min`
- Total prompt tokens: `1216`
- Average prompt tokens: `38.00`
- Average prompt latency: `1955.05 ms`
- Average prompt speed: `19.44 tok/s`
- Total generated tokens: `2048`
- Average generation depth: `70.50 tokens`
- Output token throughput: `8.29 tok/s`
- Output token throughput per slot: `4.14 tok/s/slot`
- Peak GPU memory used: `12759 / 40960 MiB`
- Slot normalization: `requested=16 active=8 total=128 effective=16`
- Materialized GPU expert slots: `480` (`30` MoE layers x `16` layer-local slots)
- Result: `locality-results/gpu_expert_slot_16/repeat_32/locality-bench.txt`

### GPU expert slots 72, repeats 16

- Status: OK
- Prompt groups: `1`
- Repeats per prompt: `16`
- Total requests: `16`
- Parallel requests: `2`
- Benchmark duration: `120.42 s`
- Request throughput: `0.13 req/s`
- Request throughput: `7.97 req/min`
- Total prompt tokens: `608`
- Average prompt tokens: `38.00`
- Average prompt latency: `1999.73 ms`
- Average prompt speed: `19.00 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `70.50 tokens`
- Output token throughput: `8.50 tok/s`
- Output token throughput per slot: `4.25 tok/s/slot`
- Peak GPU memory used: `32919 / 40960 MiB`
- Slot normalization: `requested=72 active=8 total=128 effective=72`
- Materialized GPU expert slots: `2160` (`30` MoE layers x `72` layer-local slots)
- Result: `locality-results/gpu_expert_slot_72/repeat_16/locality-bench.txt`

### GPU expert slots 72, repeats 32

- Status: OK
- Prompt groups: `1`
- Repeats per prompt: `32`
- Total requests: `32`
- Parallel requests: `2`
- Benchmark duration: `250.52 s`
- Request throughput: `0.13 req/s`
- Request throughput: `7.66 req/min`
- Total prompt tokens: `1216`
- Average prompt tokens: `38.00`
- Average prompt latency: `1946.72 ms`
- Average prompt speed: `19.52 tok/s`
- Total generated tokens: `2048`
- Average generation depth: `70.50 tokens`
- Output token throughput: `8.18 tok/s`
- Output token throughput per slot: `4.09 tok/s/slot`
- Peak GPU memory used: `32919 / 40960 MiB`
- Slot normalization: `requested=72 active=8 total=128 effective=72`
- Materialized GPU expert slots: `2160` (`30` MoE layers x `72` layer-local slots)
- Result: `locality-results/gpu_expert_slot_72/repeat_32/locality-bench.txt`

## Test Setup

Repeated-prompt locality benchmark sweep for `--moe-gpu-expert-slot-num`.

Common settings:

- Run date: `2026-05-02`
- GPU: `0`
- GPU type: `NVIDIA A100-SXM4-40GB`
- Model: `/models/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it.gguf`
- Server script: `./1_serving.sh`
- Benchmark script: `./3_locality_benchmark.sh`
- Server URL: `http://127.0.0.1:8071`
- Benchmark mode: `locality`
- Prompt groups: `1`
- Repeat counts: `16`, `32`
- Output length: `64`
- Parallel requests: `2`
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
GPU_EXPERT_SLOT_NUMS='-1 16 72' REPEATS_PER_PROMPT_LIST='16 32' ./3_locality_benchmark.sh
```

Notes:

- The locality script generates deterministic prompt groups and sends repeated adjacent requests for each prompt.
- The request order is not shuffled, so repeated prompts stay adjacent in the submitted work queue.
- This benchmark is intended to test temporal locality effects in the MoE expert slot cache.
- Slot mode keeps the CPU-mapped expert pool as the source of truth and materializes per-layer GPU expert slot buffers at startup.
- The total request count is `prompt_groups * repeats_per_prompt`; this run uses `1` prompt group, so the repeat count equals total requests.
