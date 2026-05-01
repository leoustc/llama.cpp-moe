# MoE Expert Paging Stage 6 Design

This note captures safe implementation hooks for real MoE expert paging data movement.

## Where selected expert IDs are available

Router-selected experts are produced during MoE forward graph construction in model-specific MoE builders.
A generic hook point should be introduced after top-k routing indices are computed and before expert FFN matmuls are enqueued.

## How expert tensors are stored

Current architectures in `llama.cpp` use multiple patterns:
- Separate expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`)
- Fused tensors (`ffn_gate_up_exps`)
- Optional expert bias/scale variants

Many tensors are stored packed across expert dimension (typically the last dimension).

## GPU slot tensor allocation

Target design for miss handling:
1. Keep packed source expert tensors resident on CPU buffers.
2. Allocate GPU-side slot tensors sized for **single-expert slices** per role (gate/up/down or gate_up).
3. Maintain mapping:
   - logical `(layer_id, expert_id)` -> physical `slot_id`
4. Reuse existing slot buffers across misses; only overwrite content.

## Safe graph patching strategy

Two feasible directions:

1. **View + copy path (preferred incremental)**
   - Create CPU views over packed source tensors for selected expert slices.
   - Copy slice into slot tensor on miss.
   - Build expert compute from slot tensors (not full packed expert tensors).

2. **Runtime indirection path**
   - Extend MoE kernels/builders to resolve slot tensors by `(layer_id, expert_id)`.
   - Requires stable API for slot lookup during graph construction.

## Backend APIs required

- `ggml_backend_tensor_set` for host->device expert-slice transfer
- Backend buffer allocation hooks for slot tensors per device
- Tensor buffer type override path to keep expert source tensors on host when slot mode is enabled

## Rollout plan

1. Keep current stage-4 accounting hook and stage-5 host-residency policy logs.
2. Add model-arch-specific pilot for one MoE layout (separate gate/up/down) first.
3. Validate slot hit/miss/evict counters against router selections.
4. Expand to fused layouts (`ffn_gate_up_exps`) and bias/scale variants.

## Non-goals

- No change to router semantics.
- No change to top-k selection behavior.
- No layer-level offload redesign.
