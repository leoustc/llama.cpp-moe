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
| -1 | 16 | NVIDIA A100-SXM4-40GB | 38929 | 25.05 | 0.39 | 778.55 | OK |
| -1 | 32 | NVIDIA A100-SXM4-40GB | 38929 | 25.40 | 0.40 | 705.03 | OK |
| 16 | 16 | NVIDIA A100-SXM4-40GB | 12459 | 7.78 | 0.12 | 2793.27 | OK |
| 16 | 32 | NVIDIA A100-SXM4-40GB | 12459 | 7.74 | 0.12 | 2736.26 | OK |
| 72 | 16 | NVIDIA A100-SXM4-40GB | 31539 | 7.93 | 0.12 | 2045.50 | OK |
| 72 | 32 | NVIDIA A100-SXM4-40GB | 31539 | 7.79 | 0.12 | 2022.55 | OK |

## Throughput

| GPU expert slots | Repeat count | Requests | Duration s | Req/s | Req/min | Prompt latency ms | Prompt tok/s | Generated tokens | Generation tok/s | Generation tok/s/slot |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -1 | 16 | 16 | 40.89 | 0.39 | 23.48 | 778.55 | 48.81 | 1024 | 25.05 | 12.52 |
| -1 | 32 | 32 | 80.64 | 0.40 | 23.81 | 705.03 | 53.90 | 2048 | 25.40 | 12.70 |
| 16 | 16 | 16 | 131.65 | 0.12 | 7.29 | 2793.27 | 13.60 | 1024 | 7.78 | 3.89 |
| 16 | 32 | 32 | 264.59 | 0.12 | 7.26 | 2736.26 | 13.89 | 2048 | 7.74 | 3.87 |
| 72 | 16 | 16 | 129.16 | 0.12 | 7.43 | 2045.50 | 18.58 | 1024 | 7.93 | 3.96 |
| 72 | 32 | 32 | 262.97 | 0.12 | 7.30 | 2022.55 | 18.79 | 2048 | 7.79 | 3.89 |

## GPU Resource

| GPU expert slots | Peak used MiB | Total VRAM MiB | Free at peak MiB | Model MiB | Context MiB | Compute MiB | Expert slot buffers | Materialized slots | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -1 | 38929 | 40960 | 2031 | 36510.29 | 880.00 | 1019.52 | 0.00 MiB | 0 | 126 |
| 16 | 12459 | 40960 | 28501 | 4590.35 | 880.00 | 1013.01 | ~5445.00 MiB | 480 | 369 |
| 72 | 31539 | 40960 | 9421 | 4590.35 | 880.00 | 1013.01 | ~24502.50 MiB | 2160 | 365 |

## Bench Details

### GPU expert slots -1, repeats 16

- Status: OK
- Prompt groups: `1`
- Repeats per prompt: `16`
- Total requests: `16`
- Parallel requests: `2`
- Benchmark duration: `40.89 s`
- Request throughput: `0.39 req/s`
- Request throughput: `23.48 req/min`
- Total prompt tokens: `608`
- Average prompt tokens: `38.00`
- Average prompt latency: `778.55 ms`
- Average prompt speed: `48.81 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `70.50 tokens`
- Output token throughput: `25.05 tok/s`
- Output token throughput per slot: `12.52 tok/s/slot`
- Peak GPU memory used: `38929 / 40960 MiB`
- Result: `locality-results/gpu_expert_slot_-1/repeat_16/locality-bench.txt`

### GPU expert slots -1, repeats 32

- Status: OK
- Prompt groups: `1`
- Repeats per prompt: `32`
- Total requests: `32`
- Parallel requests: `2`
- Benchmark duration: `80.64 s`
- Request throughput: `0.40 req/s`
- Request throughput: `23.81 req/min`
- Total prompt tokens: `1216`
- Average prompt tokens: `38.00`
- Average prompt latency: `705.03 ms`
- Average prompt speed: `53.90 tok/s`
- Total generated tokens: `2048`
- Average generation depth: `70.50 tokens`
- Output token throughput: `25.40 tok/s`
- Output token throughput per slot: `12.70 tok/s/slot`
- Peak GPU memory used: `38929 / 40960 MiB`
- Result: `locality-results/gpu_expert_slot_-1/repeat_32/locality-bench.txt`

### GPU expert slots 16, repeats 16

- Status: OK
- Prompt groups: `1`
- Repeats per prompt: `16`
- Total requests: `16`
- Parallel requests: `2`
- Benchmark duration: `131.65 s`
- Request throughput: `0.12 req/s`
- Request throughput: `7.29 req/min`
- Total prompt tokens: `608`
- Average prompt tokens: `38.00`
- Average prompt latency: `2793.27 ms`
- Average prompt speed: `13.60 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `70.50 tokens`
- Output token throughput: `7.78 tok/s`
- Output token throughput per slot: `3.89 tok/s/slot`
- Peak GPU memory used: `12459 / 40960 MiB`
- Slot normalization: `requested=16 active=8 total=128 effective=16`
- Materialized GPU expert slots: `480` (`30` MoE layers x `16` layer-local slots)
- Partial-slot compute path: conservative CPU fallback because this model has `128` total experts; bank-backed compute is enabled only when slots cover the full expert dimension.
- Result: `locality-results/gpu_expert_slot_16/repeat_16/locality-bench.txt`

### GPU expert slots 16, repeats 32

- Status: OK
- Prompt groups: `1`
- Repeats per prompt: `32`
- Total requests: `32`
- Parallel requests: `2`
- Benchmark duration: `264.59 s`
- Request throughput: `0.12 req/s`
- Request throughput: `7.26 req/min`
- Total prompt tokens: `1216`
- Average prompt tokens: `38.00`
- Average prompt latency: `2736.26 ms`
- Average prompt speed: `13.89 tok/s`
- Total generated tokens: `2048`
- Average generation depth: `70.50 tokens`
- Output token throughput: `7.74 tok/s`
- Output token throughput per slot: `3.87 tok/s/slot`
- Peak GPU memory used: `12459 / 40960 MiB`
- Slot normalization: `requested=16 active=8 total=128 effective=16`
- Materialized GPU expert slots: `480` (`30` MoE layers x `16` layer-local slots)
- Partial-slot compute path: conservative CPU fallback because this model has `128` total experts; bank-backed compute is enabled only when slots cover the full expert dimension.
- Result: `locality-results/gpu_expert_slot_16/repeat_32/locality-bench.txt`

### GPU expert slots 72, repeats 16

- Status: OK
- Prompt groups: `1`
- Repeats per prompt: `16`
- Total requests: `16`
- Parallel requests: `2`
- Benchmark duration: `129.16 s`
- Request throughput: `0.12 req/s`
- Request throughput: `7.43 req/min`
- Total prompt tokens: `608`
- Average prompt tokens: `38.00`
- Average prompt latency: `2045.50 ms`
- Average prompt speed: `18.58 tok/s`
- Total generated tokens: `1024`
- Average generation depth: `70.50 tokens`
- Output token throughput: `7.93 tok/s`
- Output token throughput per slot: `3.96 tok/s/slot`
- Peak GPU memory used: `31539 / 40960 MiB`
- Slot normalization: `requested=72 active=8 total=128 effective=72`
- Materialized GPU expert slots: `2160` (`30` MoE layers x `72` layer-local slots)
- Partial-slot compute path: conservative CPU fallback because this model has `128` total experts; bank-backed compute is enabled only when slots cover the full expert dimension.
- Result: `locality-results/gpu_expert_slot_72/repeat_16/locality-bench.txt`

### GPU expert slots 72, repeats 32

- Status: OK
- Prompt groups: `1`
- Repeats per prompt: `32`
- Total requests: `32`
- Parallel requests: `2`
- Benchmark duration: `262.97 s`
- Request throughput: `0.12 req/s`
- Request throughput: `7.30 req/min`
- Total prompt tokens: `1216`
- Average prompt tokens: `38.00`
- Average prompt latency: `2022.55 ms`
- Average prompt speed: `18.79 tok/s`
- Total generated tokens: `2048`
- Average generation depth: `70.50 tokens`
- Output token throughput: `7.79 tok/s`
- Output token throughput per slot: `3.89 tok/s/slot`
- Peak GPU memory used: `31539 / 40960 MiB`
- Slot normalization: `requested=72 active=8 total=128 effective=72`
- Materialized GPU expert slots: `2160` (`30` MoE layers x `72` layer-local slots)
- Partial-slot compute path: conservative CPU fallback because this model has `128` total experts; bank-backed compute is enabled only when slots cover the full expert dimension.
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
- Repeat count did not materially change throughput in the `-1` baseline (`25.05 tok/s` at repeat `16`, `25.40 tok/s` at repeat `32`). This suggests the llama.cpp server path is not aggregating repeated identical prompts into shared pipeline work for this benchmark shape.
- Continuous batching may still batch active generation work, but this run did not show vLLM-style request locality or prompt aggregation gains from adjacent repeated prompts.
- Slot mode keeps the CPU-mapped expert pool as the source of truth and materializes per-layer GPU expert slot buffers at startup.
- Gemma4-26B-A4B has `128` total experts and `8` active experts, so slot counts `16` and `72` are partial-slot cases, not full expert-bank cases.
- Runtime partial-slot remapping was tested and rejected for this run because it can reuse a slot while a single `mul_mat_id` graph still references that expert. The stable path keeps partial-slot cases on the CPU expert compute path and uses bank-backed GPU compute only when the slot count covers the full expert dimension.
- To benchmark cache-first GPU expert compute for this model, include `MOE_GPU_EXPERT_SLOT_NUM=128` after confirming available VRAM.
- The total request count is `prompt_groups * repeats_per_prompt`; this run uses `1` prompt group, so the repeat count equals total requests.
