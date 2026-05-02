# llama.cpp-moe: MoE GPU Expert Slot Roadmap

This document defines the staged implementation plan for router-aware MoE expert paging in `llama.cpp-moe`.

The goal is to make sparse MoE models run better on memory-constrained GPUs by treating GPU memory as a bounded cache of physical GPU expert slots.

The key idea is simple:

- Do not blindly move full MoE layers to GPU.
- Keep the full expert pool available in CPU memory.
- Let the router select logical experts as usual.
- Check whether each selected expert is already resident in a GPU expert slot.
- If an expert is missing, page only that missing expert into a physical GPU slot.
- Reuse hot experts when possible.

This is expert-level paging, not layer-level offload.

Conceptual mapping:

    logical expert id -> physical GPU expert slot

The model behavior must not change. The router, top-k expert selection, output math, and model semantics remain the same. Only expert weight residency changes.

## User-facing flag

Add and use this CLI flag:

    --moe-gpu-expert-slot-num N

Meaning:

    Configure the maximum number of GPU-resident physical GPU expert slots used for router-aware MoE expert paging.

Default:

    0

Behavior:

    0 disables GPU expert slot mode and preserves existing llama.cpp behavior.

Internal field name:

    n_moe_gpu_expert_slot_num

Naming convention:

    Use "GPU expert slot" in user-facing text and `gpu_expert_slot` in internal names.

Important distinction:

    n_gpu_layers controls layer-level GPU offload.
    n_moe_gpu_expert_slot_num controls expert-level GPU residency capacity.

These two knobs must remain conceptually separate.

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

    int32_t n_moe_gpu_expert_slot_num = 0;

Suggested placement near:

    n_gpu_layers
    main_gpu
    tensor_split
    split_mode

Suggested comment:

    number of MoE GPU expert slots, 0 disables GPU expert slot mode

2. Add a field to `struct llama_model_params` in `include/llama.h` near `n_gpu_layers`:

    int32_t n_moe_gpu_expert_slot_num;

Keep the public C API struct initializer order consistent.

3. Initialize the new field in `llama_model_default_params()` in `src/llama.cpp`:

    n_moe_gpu_expert_slot_num = 0

4. Add CLI option in `common/arg.cpp` near existing GPU/offload options:

    --moe-gpu-expert-slot-num N

Parser behavior:

- Accept integer N.
- Reject negative values.
- 0 means disabled.
- Positive value enables future GPU expert slot mode.

Handler logic:

    if value < 0:
        throw invalid_argument
    params.n_moe_gpu_expert_slot_num = value

Help text:

    number of MoE GPU expert slots for router-aware expert paging (default: 0, disabled)

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
        std::vector<llama_moe_gpu_expert_slot> slots;

        void init(int32_t n);
        void clear();
        bool enabled() const;
        int32_t size() const;
    };

Add this field to `struct llama_model`:

    llama_moe_gpu_expert_cache moe_gpu_expert_cache;

Initialization policy:

- If `n_moe_gpu_expert_slot_num <= 0`, cache is disabled.
- If `n_moe_gpu_expert_slot_num > 0` and model is not MoE, ignore the flag and keep cache disabled.
- If `n_moe_gpu_expert_slot_num > 0` and model is MoE, initialize N physical GPU expert slots and preload a deterministic initial GPU expert slot set at startup.

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
    int find_free() const
    int find_lru_victim() const
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

## Stage 5: CPU-resident expert policy preparation

Goal: prepare policy for keeping MoE expert tensors host-resident when GPU expert slot mode is enabled.

Design target:

- router and non-expert/shared tensors can stay under normal llama.cpp offload behavior
- MoE expert tensors can be kept CPU-resident by default when GPU expert slot mode is enabled
- selected missing experts will later be copied into GPU expert slots

Add a policy helper such as:

    bool llama_moe_should_keep_expert_on_host(const llama_model_params & params, const ggml_tensor * tensor, const char * tensor_name);

Behavior:

- if `n_moe_gpu_expert_slot_num <= 0`, return false
- if tensor is an MoE expert tensor, return true
- otherwise return false

At this stage, add logs first and avoid broad tensor placement changes until tensor identification is reliable.

Example log:

    MoE GPU expert slot mode: expert tensor candidate kept host-resident: NAME

Suggested commit message:

    llama: prepare host residency policy for MoE expert tensors

## Stage 6: Real expert paging design hooks

Goal: prepare real data movement hooks. Implement actual movement only if safe.

Final target on expert miss:

1. Select physical GPU expert slot.
2. Evict previous logical expert if necessary.
3. Copy selected expert weights from CPU-resident source tensor/storage into GPU slot buffer.
4. Update slot table.
5. Execute expert compute using slot-resident tensors.

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

    docs/moe-expert-paging-stage6-design.md

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

    --moe-gpu-expert-slot-num 0

3. Non-MoE models must continue to work.

4. Passing a positive GPU expert slot number to a non-MoE model must be ignored safely.

5. Do not rename existing llama.cpp flags.

6. Do not remove existing offload behavior.

7. Do not implement layer-level offload. This feature is expert-level paging.

8. Keep this distinction clear:

    n_gpu_layers controls layer residency/offload.
    n_moe_gpu_expert_slot_num controls bounded GPU expert slot residency.

9. Add comments where useful:

    MoE expert paging
    logical expert id -> physical GPU expert slot
    0 disables GPU expert slot mode

10. After each stage, run:

    cmake --build build -j

11. If the build path differs, inspect the repo README or CMake presets and use the correct build command.

## Expected final architecture

    llama CLI
        --moe-gpu-expert-slot-num N
            -> common_params.n_moe_gpu_expert_slot_num
            -> llama_model_params.n_moe_gpu_expert_slot_num
            -> llama_model.moe_gpu_expert_cache
            -> MoE expert tensor index
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
