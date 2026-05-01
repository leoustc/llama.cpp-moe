# llama.cpp-moe

`llama.cpp-moe` follows the same practical spirit as `llama.cpp`: keep inference local, efficient, and understandable.

This repository emphasizes a **Mixture-of-Experts (MoE)** oriented workflow while preserving the lightweight, hackable, and portable approach that makes `llama.cpp` useful across many environments.

## Design Philosophy

### 1) Local-first and user-owned
- Run models where your data already lives.
- Favor offline and self-hosted execution paths.
- Keep developer control over binaries, model files, and runtime behavior.

### 2) Performance through simplicity
- Prefer clear, low-overhead implementations that are easy to profile.
- Optimize critical paths (memory layout, quantization, kernels, scheduling) before adding complexity.
- Treat throughput, latency, and startup time as first-class constraints.

### 3) Portability over platform lock-in
- Build on commodity hardware and across operating systems.
- Keep dependencies minimal and predictable.
- Preserve a straightforward build and deployment story for both experimentation and production.

### 4) MoE pragmatism
- Support MoE use cases with practical defaults and transparent controls.
- Expose expert routing behavior and trade-offs to users instead of hiding them behind opaque abstractions.
- Balance quality gains from expert specialization against compute and memory cost.

### 5) Transparency and debuggability
- Prefer explicit configuration over hidden magic.
- Make runtime behavior observable and reproducible.
- Keep features explainable so contributors can reason about correctness and performance.

### 6) Incremental, maintainable evolution
- Favor small, reviewable changes over large rewrites.
- Reuse established `llama.cpp` patterns where possible.
- Optimize for long-term maintainability by humans.

## Repository Notes

- The previous project README has been preserved as [`README_OLD.md`](./README_OLD.md).
- Use that file as a historical and technical reference while this new README defines project intent and guiding principles.
