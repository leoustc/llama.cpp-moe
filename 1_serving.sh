#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
MODEL_DIR="${MODEL_DIR:-/models/gemma-4-26B-A4B-it}"
MODEL_GGUF="${MODEL_GGUF:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8071}"
HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
MOE_GPU_EXPERT_SLOT_NUM="${MOE_GPU_EXPERT_SLOT_NUM:--1}"
CTX_SIZE="${CTX_SIZE:-4096}"

export CUDA_VISIBLE_DEVICES

SERVER_BIN="${BUILD_DIR}/bin/llama-server"
if [[ ! -x "${SERVER_BIN}" ]]; then
    SERVER_BIN="${BUILD_DIR}/bin/${BUILD_TYPE}/llama-server"
fi
if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "error: llama-server binary not found under ${BUILD_DIR}/bin; run ./0_harness.sh first" >&2
    exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "error: model directory not found: ${MODEL_DIR}" >&2
    exit 1
fi

if [[ -z "${MODEL_GGUF}" ]]; then
    MODEL_GGUF="$(find "${MODEL_DIR}" -maxdepth 1 -type f -name '*.gguf' | sort | head -n 1)"
fi

if [[ -z "${MODEL_GGUF}" || ! -f "${MODEL_GGUF}" ]]; then
    echo "error: no GGUF model found in ${MODEL_DIR}" >&2
    echo "llama-server serves GGUF files; ${MODEL_DIR} currently contains HF safetensors." >&2
    echo "convert first, for example:" >&2
    echo "  python3 convert_hf_to_gguf.py ${MODEL_DIR} --outfile ${MODEL_DIR}/gemma-4-26B-A4B-it.gguf --outtype bf16" >&2
    echo "then rerun with:" >&2
    echo "  MODEL_GGUF=${MODEL_DIR}/gemma-4-26B-A4B-it.gguf ./1_serving.sh" >&2
    exit 1
fi

stop_existing_server() {
    local pids
    pids="$(lsof -ti "tcp:${PORT}" 2>/dev/null || true)"
    if [[ -z "${pids}" ]]; then
        return 0
    fi

    if ! curl -fsS "http://${HEALTH_HOST}:${PORT}/health" >/dev/null 2>&1; then
        echo "error: port ${PORT} is already in use, but http://${HEALTH_HOST}:${PORT}/health is not healthy" >&2
        echo "refusing to stop an unrelated process: ${pids}" >&2
        exit 1
    fi

    echo "stopping existing llama-server on port ${PORT}: ${pids}"
    kill ${pids} 2>/dev/null || true

    for _ in $(seq 1 20); do
        pids="$(lsof -ti "tcp:${PORT}" 2>/dev/null || true)"
        if [[ -z "${pids}" ]]; then
            return 0
        fi
        sleep 0.5
    done

    echo "force stopping process on port ${PORT}: ${pids}"
    kill -KILL ${pids} 2>/dev/null || true
}

stop_existing_server

echo "serving ${MODEL_GGUF} on ${HOST}:${PORT} with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
exec "${SERVER_BIN}" \
    --model "${MODEL_GGUF}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --ctx-size "${CTX_SIZE}" \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --moe-gpu-expert-slot-num "${MOE_GPU_EXPERT_SLOT_NUM}"
