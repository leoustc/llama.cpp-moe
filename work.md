# llama.cpp-moe: MoE GPU Expert Slot Roadmap

This document defines the staged implementation plan for router-aware MoE expert paging in `llama.cpp-moe`.

The goal is to make sparse MoE models run better on memory-constrained GPUs by treating GPU memory as a bounded cache of physical GPU expert slots.

The key idea is simple:

- Do not blindly move full MoE layers to GPU.
- Keep the full expert pool available in CPU memory.
- Let the router select logical experts as usual.
- Check whether each selected expert is already resident in a GPU expert slot.
- If an expert is already resident, use the GPU expert slot and do not copy from CPU again.
- If an expert is missing, page only that missing expert from CPU into a physical GPU slot.
- Reuse hot experts when possible.

This is expert-level paging, not layer-level offload. When GPU expert slot mode is enabled for an MoE model, slot-mode placement owns the MoE model residency policy instead of `n_gpu_layers`.

Conceptual mapping:

    logical expert id -> physical GPU expert slot
    physical GPU expert slot -> logical expert id

The implementation must maintain both directions:

- expert-to-slot map: `(layer_id, expert_id) -> (layer_id, slot_id)`
- slot-to-expert map: `(layer_id, slot_id) -> (layer_id, expert_id)`

Slot ids are layer-local. Slot `3` in layer `L0` and slot `3` in layer `L1` are different physical expert slots and may hold different logical experts at the same time.

The model behavior must not change. The router, top-k expert selection, output math, and model semantics remain the same. Only expert weight residency changes.

## User-facing flag

Add and use this CLI flag:

    --moe-gpu-expert-slot-num N

Meaning:

    Configure the maximum number of GPU-resident physical GPU expert slots used for router-aware MoE expert paging.

Default:

    -1

Behavior:

    -1 disables GPU expert slot mode and preserves existing llama.cpp behavior.
    0 enables GPU expert slot mode with the minimum useful slot count: the active expert count.
    N > 0 enables GPU expert slot mode with N requested slots, normalized at startup.

Internal field name:

    n_moe_gpu_expert_slot_num

Naming convention:

    Use "GPU expert slot" in user-facing text and `gpu_expert_slot` in internal names.

Important distinction:

    n_gpu_layers controls layer-level GPU offload.
    n_moe_gpu_expert_slot_num controls expert-level GPU residency capacity.

These two knobs must remain conceptually separate in normal mode. In MoE GPU expert slot mode, `n_gpu_layers` is intentionally ignored for MoE model placement and the slot-mode policy decides what is CPU-resident and what is GPU-resident.

## Runtime placement policy

There are three runtime cases:

| Model type | `--moe-gpu-expert-slot-num` | Placement policy |
|---|---:|---|
| Non-MoE | `-1` | Normal llama.cpp behavior. |
| Non-MoE | `>= 0` | Ignore GPU expert slot flag, log once, use normal llama.cpp behavior. |
| MoE | `-1` | Normal llama.cpp behavior; `n_gpu_layers` controls layer offload. |
| MoE | `0` | Enable GPU expert slot mode with `active_experts` slots; ignore `n_gpu_layers` for MoE placement. |
| MoE | `> 0` | Enable GPU expert slot mode with normalized slot count; ignore `n_gpu_layers` for MoE placement. |

For MoE models with `--moe-gpu-expert-slot-num N >= 0`:

- Keep the complete MoE expert weight pool CPU-resident.
- Put activation, dense/shared, attention, norm, embedding, and output tensors on GPU when memory allows.
- Put router/gate tensors on GPU.
- Normalize the requested slot count at startup before allocation.
- Preload the normalized number of logical experts into physical GPU expert slots during model startup.
- During inference, use router-selected expert IDs to hit resident slots or page missing experts from CPU into GPU slots.

This policy is deliberate. Setting `n_gpu_layers = 0` is not the default slot-mode behavior because it would keep dense/shared model weights on CPU and would not represent the intended GPU expert slot design.

## Non-MoE model behavior

If `--moe-gpu-expert-slot-num N` is passed for a non-MoE model:

- Do not fail.
- Do not allocate GPU expert slots.
- Do not change tensor placement.
- Do not change inference behavior.
- Ignore the flag safely.
- Print at most one informational log:

    MoE GPU expert slot mode requested, but model has no MoE experts; ignoring

The raw user parameter may remain stored, but the effective runtime GPU expert slot count must become 0 for non-MoE models.

## GPU expert slot count normalization

For MoE models with `--moe-gpu-expert-slot-num N >= 0`, normalize the effective runtime slot count during model startup.

If `N == -1`, do not normalize. Slot mode is disabled and the model uses normal llama.cpp MoE behavior.
If `N == 0`, enable slot mode with the minimum useful GPU expert slot count, equal to `active_experts`.

Definitions:

- `requested_slots`: raw user value from `n_moe_gpu_expert_slot_num`
- `active_experts`: number of experts used per token by the router, for example `n_expert_used`
- `total_model_experts`: total expert count advertised by the model/router, for example `n_expert`
- `effective_slots`: normalized slot count used to allocate and preload physical GPU expert slots

Normalization:

    if requested_slots == -1:
        effective_slots = 0
    else if requested_slots == 0:
        effective_slots = active_experts
    else:
        effective_slots = clamp(requested_slots, active_experts, total_model_experts)

Behavior:

- If `requested_slots == -1`, disable slot mode and do not normalize.
- If `requested_slots == 0`, enable slot mode and use `active_experts`.
- If `requested_slots < active_experts`, raise it to `active_experts`.
- If `requested_slots > total_model_experts`, cap it to `total_model_experts`.
- If `requested_slots` is already inside the valid range, keep it unchanged.
- If the model is non-MoE, skip this normalization and set the effective slot count to `0`.

Expected logs:

    MoE GPU expert slot count normalized: requested=R active=A total=T effective=E

Rationale:

- The slot cache should have enough capacity for at least one token's active expert set.
- The slot cache should never exceed the model's expert count.
- The raw user value can remain available for diagnostics, but allocation and preload must use the effective normalized value.

## Stage 1: CLI and parameter plumbing

Goal: add the CLI flag and propagate it through llama.cpp parameter structures. Do not change runtime behavior.

Files to inspect:

- `common/common.h`
- `common/arg.cpp`
- `include/llama.h`
- `src/llama.cpp`
- `common/common.cpp`
- `common/arg.h`

Required changes:

1. Add a field to `struct common_params` near the existing offload parameters:

    int32_t n_moe_gpu_expert_slot_num = -1;

Suggested placement near:

    n_gpu_layers
    main_gpu
    tensor_split
    split_mode

Suggested comment:

    number of MoE GPU expert slots; -1 disables, 0 uses active expert count

2. Add a field to `struct llama_model_params` in `include/llama.h` near `n_gpu_layers`:

    int32_t n_moe_gpu_expert_slot_num;

Keep the public C API struct initializer order consistent.

3. Initialize the new field in `llama_model_default_params()` in `src/llama.cpp`:

    n_moe_gpu_expert_slot_num = -1

4. Add CLI option in `common/arg.cpp` near existing GPU/offload options:

    --moe-gpu-expert-slot-num N

Parser behavior:

- Accept integer N.
- Reject values less than -1.
- -1 means disabled.
- 0 means minimum slot mode using `active_experts`.
- Positive value enables GPU expert slot mode with normalization.

Handler logic:

    if value < -1:
        throw invalid_argument
    params.n_moe_gpu_expert_slot_num = value

Help text:

    number of MoE GPU expert slots for router-aware expert paging (default: -1, disabled)

5. Propagate from `common_params` to `llama_model_params` wherever common params are converted to model params:

    mparams.n_moe_gpu_expert_slot_num = params.n_moe_gpu_expert_slot_num;

This should be near existing assignments for:

    n_gpu_layers
    main_gpu
    split_mode
    tensor_split

Validation:

    cmake --build build -j
    ./build/bin/llama-cli --help | grep moe-gpu-expert-slot-num

Smoke test:

    ./build/bin/llama-cli -m /path/to/model.gguf --moe-gpu-expert-slot-num 32 -p "hello" -n 8

Expected result:

- CLI accepts the flag.
- Default behavior is unchanged when the flag is 0.
- Non-MoE models do not fail merely because the flag exists.
- No tensor placement changes yet.

Suggested commit message:

    llama: add MoE GPU expert slot CLI option

## Stage 2: GPU Expert Slot Cache Metadata Skeleton

Goal: add internal metadata structures for GPU expert slots. Do not move tensors yet.

Runtime concept:

    physical GPU expert slot -> logical layer and expert identity

Preferred location:

- `src/llama-model.h`
- `src/llama-model.cpp`

Suggested structures:

    struct llama_moe_gpu_expert_slot {
        int32_t layer_id = -1;
        int32_t expert_id = -1;
        int64_t last_used = 0;
        bool resident = false;
    };

    struct llama_moe_gpu_expert_cache {
        int32_t n_slots = 0;
        std::unordered_map<int32_t, std::vector<llama_moe_gpu_expert_slot>> slots_by_layer;
        std::unordered_map<uint64_t, int32_t> expert_to_slot;

        void init(int32_t n);
        void clear();
        int32_t find(int32_t layer_id, int32_t expert_id) const;
        int32_t find_free(int32_t layer_id) const;
        int32_t find_lru_victim(int32_t layer_id) const;
        void assign_slot(int32_t slot_id, int32_t layer_id, int32_t expert_id, int64_t step);
        bool enabled() const;
        int32_t size() const;
    };

The cache must maintain two maps:

- `expert_to_slot`: logical expert id `(layer_id, expert_id)` to layer-local physical GPU slot id.
- `slots_by_layer`: `(layer_id, slot_id)` to logical expert id `(layer_id, expert_id)` and slot storage.

When assigning or evicting a slot, update both directions in the same helper. Never update only one map.

Add this field to `struct llama_model`:

    llama_moe_gpu_expert_cache moe_gpu_expert_cache;

Initialization policy:

- If `n_moe_gpu_expert_slot_num < 0`, cache is disabled.
- If `n_moe_gpu_expert_slot_num >= 0` and model is not MoE, ignore the flag and keep cache disabled.
- If `n_moe_gpu_expert_slot_num >= 0` and model is MoE, normalize the requested slot count, enable GPU expert slot mode, initialize `effective_slots` physical GPU expert slots, and preload a deterministic initial GPU expert slot set at startup.

MoE detection helper:

    static bool llama_model_has_moe_experts(const llama_model & model) {
        for (const auto & layer : model.layers) {
            if (layer.ffn_gate_exps ||
                layer.ffn_down_exps ||
                layer.ffn_up_exps ||
                layer.ffn_gate_up_exps ||
                layer.ffn_gate_exps_b ||
                layer.ffn_down_exps_b ||
                layer.ffn_up_exps_b ||
                layer.ffn_gate_up_exps_b) {
                return true;
            }
        }
        return false;
    }

Use tensor-pointer detection after layer tensors are loaded or assigned enough for the check to be meaningful.

Expected logs:

For non-MoE models with the flag set:

    MoE GPU expert slot mode requested, but model has no MoE experts; ignoring

For MoE models with the flag set:

    MoE GPU expert slot mode enabled; ignoring n_gpu_layers for MoE placement
    MoE GPU expert slot count normalized: requested=R active=A total=T effective=E
    initialized MoE GPU expert slot cache with N slots
    MoE GPU expert slot preload: layer=L expert=E slot=S

Validation:

    cmake --build build -j

Non-MoE smoke test:

    ./build/bin/llama-cli -m /path/to/non-moe-model.gguf --moe-gpu-expert-slot-num 16 -p "hello" -n 8

Expected:

- no crash
- inference works
- flag is ignored
- no GPU expert slots allocated

MoE smoke test:

    ./build/bin/llama-cli -m /path/to/moe-model.gguf --moe-gpu-expert-slot-num 16 -p "hello" -n 8

Expected:

- no crash
- inference works
- cache initializes with 16 GPU expert slots and preloads the startup GPU expert slot set
- no inference behavior change yet

Suggested commit message:

    llama: add MoE GPU expert slot cache metadata

## Stage 3: MoE expert tensor indexing

Goal: identify MoE expert tensors and build a lightweight logical expert index.

Do not move tensors yet.

Existing MoE tensor fields in `struct llama_layer` may include:

- `ffn_gate_inp`
- `ffn_gate_inp_s`
- `ffn_gate_exps`
- `ffn_down_exps`
- `ffn_up_exps`
- `ffn_gate_up_exps`
- `ffn_gate_inp_b`
- `ffn_gate_exps_b`
- `ffn_down_exps_b`
- `ffn_up_exps_b`
- `ffn_gate_up_exps_b`
- `ffn_gate_exps_s`
- `ffn_down_exps_s`
- `ffn_up_exps_s`

The index should capture:

- layer id
- expert id
- tensor role: gate, up, down, gate_up, bias, scale
- layout type: packed, separate, fused, unknown

Important caution:

Different MoE architectures may store experts differently:

- all experts packed in one tensor
- separate gate/up/down tensors
- fused gate_up tensors
- per-expert bias tensors
- per-expert scale tensors
- architecture-specific Gemma4 layout

Do not assume one layout unless guarded.

Deliverable:

Add debug or info logs such as:

    layer X: MoE tensors found
    layer X: expert tensor layout = packed/separate/fused/unknown
    layer X: n_experts = Y

If `n_experts` cannot be derived reliably, add conservative placeholder metadata and clear TODO comments.

Suggested commit message:

    llama: index MoE expert tensors for GPU expert slot paging

## Stage 4: GPU Expert Slot Hit/Miss Accounting

Goal: implement GPU expert slot lookup and hit/miss accounting. Do not copy tensors yet.

Add cache operations:

    int find(layer_id, expert_id) const
    int find_free(int layer_id) const
    int find_lru_victim(int layer_id) const
    int get_or_assign_slot(layer_id, expert_id, step)

Add counters:

    n_hit
    n_miss
    n_evict

Expected debug logs:

    MoE GPU expert slot hit: layer=L expert=E slot=S
    MoE GPU expert slot miss: layer=L expert=E slot=S
    MoE GPU expert slot evict: slot=S old_layer=L0 old_expert=E0 new_layer=L1 new_expert=E1

Find where router-selected expert IDs become available in graph construction or MoE forward code.

If selected expert IDs are not easily available yet, add a narrow internal hook or synthetic cache test path, but do not force a large graph refactor.

Suggested commit message:

    llama: add MoE GPU expert slot hit miss accounting

## Stage 5: Slot-mode tensor placement policy

Goal: implement the explicit tensor placement policy used when GPU expert slot mode is enabled for an MoE model.

Design target:

- Non-MoE models keep normal llama.cpp placement, even if the user passed a positive GPU expert slot count.
- MoE models with GPU expert slot mode disabled keep normal llama.cpp placement.
- MoE models with GPU expert slot mode enabled ignore `n_gpu_layers` for model placement.
- Activation, dense/shared, attention, norm, embedding, output, and router/gate tensors are placed on GPU when memory allows.
- The complete MoE expert weight pool is kept CPU-resident.
- The configured number of logical experts is preloaded into physical GPU expert slots.
- Router-selected missing experts will later be copied from the CPU expert pool into GPU expert slots.

Add policy helpers such as:

    bool llama_moe_gpu_expert_slot_mode_enabled(const llama_model & model, const llama_model_params & params);
    bool llama_moe_tensor_is_expert(const llama_model & model, const ggml_tensor * tensor, const char * tensor_name);
    bool llama_moe_tensor_is_router(const llama_model & model, const ggml_tensor * tensor, const char * tensor_name);
    bool llama_moe_tensor_prefers_gpu_in_slot_mode(const llama_model & model, const ggml_tensor * tensor, const char * tensor_name);
    bool llama_moe_tensor_prefers_cpu_in_slot_mode(const llama_model & model, const ggml_tensor * tensor, const char * tensor_name);

Behavior:

- If `n_moe_gpu_expert_slot_num < 0`, do not use slot-mode placement.
- If the model is not MoE, do not use slot-mode placement.
- If the model is MoE and `n_moe_gpu_expert_slot_num >= 0`, ignore `n_gpu_layers` for placement.
- If tensor is an MoE expert tensor, keep it CPU-resident as the authoritative source.
- If tensor is a router/gate tensor, prefer GPU placement.
- If tensor is activation, dense/shared, attention, norm, embedding, or output, prefer GPU placement.
- If memory fitting cannot satisfy this policy, fail clearly or degrade only through an explicit documented fallback. Do not silently revert to layer-level `n_gpu_layers` behavior.

At this stage, add logs first and avoid broad tensor placement changes until tensor identification is reliable.

Example logs:

    MoE GPU expert slot mode enabled; ignoring n_gpu_layers for MoE placement
    MoE GPU expert slot mode: expert tensor CPU-resident source: NAME
    MoE GPU expert slot mode: router tensor placed on GPU: NAME
    MoE GPU expert slot mode: dense/shared tensor placed on GPU: NAME

Suggested commit message:

    llama: add MoE GPU expert slot placement policy

## Stage 6: Startup GPU expert slot preload

Goal: allocate GPU expert slot storage and preload the first deterministic set of logical experts during model startup.

Before allocation:

- Compute `active_experts` from model metadata, for example `n_expert_used`.
- Compute `total_model_experts` from model metadata or the MoE layer/expert index, for example `n_expert`.
- Normalize `requested_slots` into `effective_slots`.
- Allocate physical GPU expert slots using `effective_slots`, not the raw requested value.

Preload policy:

- Use the CPU-resident expert pool as the source of truth.
- Select logical experts deterministically, for example layer-major order: `(layer 0, expert 0)`, `(layer 1, expert 0)`, then continue through available layer/expert IDs until `N` slots are filled.
- Copy selected expert weights into physical GPU expert slot buffers.
- Record the logical layer/expert identity in the slot metadata.
- If the same logical expert is already resident in a materialized GPU slot, treat it as a hit and skip the CPU-to-GPU copy.
- Do not count startup preload as runtime hit/miss/evict traffic.

Expected logs:

    MoE GPU expert slot preload: layer=L expert=E slot=S
    MoE GPU expert slot preload reuse: layer=L expert=E slot=S
    MoE GPU expert slot preload replace: slot=S old_layer=L0 old_expert=E0 new_layer=L1 new_expert=E1
    MoE GPU expert slot hit: slot=S layer=L expert=E tensors=T; skip CPU copy

Validation:

- Startup succeeds for the target MoE model.
- GPU expert slot count matches the normalized effective slot count.
- A request below `active_experts` is raised to `active_experts`.
- A request above `total_model_experts` is capped to `total_model_experts`.
- CPU expert pool remains available after preload.
- Normal non-MoE models are unchanged.

Suggested commit message:

    llama: preload MoE GPU expert slots at startup

## Stage 7: Real expert paging design hooks

Goal: prepare real data movement hooks. Implement actual movement only if safe.

Final target on expert miss:

1. Check the GPU expert slot table for `(layer, expert)`.
2. On hit, use the resident GPU slot tensors and do not copy from CPU.
3. On miss, select a free or LRU physical GPU expert slot.
4. Evict previous logical expert metadata if necessary.
5. Copy selected expert weights from CPU-resident source tensor/storage into the GPU slot buffer.
6. Update slot table.
7. Execute expert compute using slot-resident tensors.

Before coding real movement, inspect:

- `ggml_backend_tensor_set`
- `ggml_backend_tensor_get`
- backend buffer allocation logic
- tensor buffer type override logic
- `model.load_tensors()`
- how packed MoE expert tensors are represented
- how CUDA backend handles tensor slices and views

Possible complication:

If all experts are packed into one large tensor, physical GPU expert slots may require slicing source expert regions from the packed CPU tensor and copying them into smaller GPU expert slot tensors. If the graph expects a packed tensor, GPU expert slot replacement may require graph rewrite.

If implementation is unsafe, stop and write a design note instead:

    docs/moe-expert-paging-stage7-design.md

The design note should explain:

- where selected expert IDs are available
- how expert tensors are packed
- how to allocate GPU expert slot tensors
- how to patch graph inputs safely
- what backend APIs are needed

Suggested commit message for design-only stage:

    docs: describe MoE expert paging data movement design

Suggested commit message for safe hook implementation:

    llama: add initial MoE expert paging data movement hooks

## General rules

1. Keep each stage small and compilable.

2. Never change default behavior when:

    --moe-gpu-expert-slot-num -1

3. Non-MoE models must continue to work.

4. Passing a non-negative GPU expert slot number to a non-MoE model must be ignored safely.

5. Do not rename existing llama.cpp flags.

6. Do not remove existing offload behavior for normal mode.

7. Do not implement layer-level offload. This feature is expert-level paging.

8. Keep this distinction clear:

    normal mode: n_gpu_layers controls layer residency/offload.
    MoE GPU expert slot mode: n_gpu_layers is ignored for MoE placement.
    n_moe_gpu_expert_slot_num controls bounded GPU expert slot residency.

9. Add comments where useful:

    MoE expert paging
    logical expert id -> physical GPU expert slot
    -1 disables GPU expert slot mode
    0 uses active_experts GPU expert slots

10. After each stage, run:

    cmake --build build -j

11. If the build path differs, inspect the repo README or CMake presets and use the correct build command.

## Expected final architecture

    llama CLI
        --moe-gpu-expert-slot-num N
            -> common_params.n_moe_gpu_expert_slot_num
            -> llama_model_params.n_moe_gpu_expert_slot_num
            -> llama_model.moe_gpu_expert_cache
            -> slot-mode tensor placement policy
            -> MoE expert tensor index
            -> startup GPU expert slot preload
            -> router-selected expert IDs
            -> GPU expert slot lookup
            -> hit: use resident GPU expert slot
            -> miss: page missing expert from CPU to GPU slot
            -> expert compute

This follows the same high-level design as the previous vllm-moe Case 2 work:

- GPU memory is a bounded GPU expert slot cache.
- CPU memory holds the full expert pool.
- Router-selected missing experts are moved on demand.
- Hot experts remain resident and are reused.
- The router and model semantics are unchanged.
