#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8071}"
MODE="${MODE:-throughput}"
NAME="${NAME:-gemma4-26B-A4B-it-${MODE}}"
OUT_DIR="${OUT_DIR:-benchmark-results}"
GPU_EXPERT_SLOT_NUMS="${GPU_EXPERT_SLOT_NUMS:--1 16 32 64 72}"
SERVING_SCRIPT="${SERVING_SCRIPT:-./1_serving.sh}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-5}"
GPU_MEMORY_POLL_SECONDS="${GPU_MEMORY_POLL_SECONDS:-1}"
GPU_MEMORY_INDEX="${GPU_MEMORY_INDEX:-0}"
GPU_MEMORY_SAMPLING="${GPU_MEMORY_SAMPLING:-1}"

case "${MODE}" in
    throughput)
        N_PARALLEL="${N_PARALLEL:-4}"
        N_PROMPTS="${N_PROMPTS:-32}"
        PROMPT_SOURCE="${PROMPT_SOURCE:-rng-64-64}"
        N_PREDICT="${N_PREDICT:-64}"
        N_PREDICT_MIN="${N_PREDICT_MIN:-64}"
        ;;
    smoke)
        N_PARALLEL="${N_PARALLEL:-4}"
        N_PROMPTS="${N_PROMPTS:-16}"
        PROMPT_SOURCE="${PROMPT_SOURCE:-rng-128-256}"
        N_PREDICT="${N_PREDICT:-128}"
        N_PREDICT_MIN="${N_PREDICT_MIN:-64}"
        ;;
    *)
        echo "error: unsupported MODE=${MODE}; use MODE=throughput or MODE=smoke" >&2
        exit 1
        ;;
esac

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required but was not found on PATH" >&2
    exit 1
fi

python3 - <<'PY'
import importlib.util
import sys

missing = [name for name in ("requests", "numpy", "matplotlib", "datasets", "tqdm") if importlib.util.find_spec(name) is None]
if missing:
    print("error: missing Python benchmark dependencies: " + ", ".join(missing), file=sys.stderr)
    print("install with: python3 -m pip install -r requirements/requirements-server-bench.txt", file=sys.stderr)
    sys.exit(1)
PY

mkdir -p "${OUT_DIR}"

echo "benchmark mode: ${MODE}"
echo "server url:     ${SERVER_URL}"
echo "parallel:       ${N_PARALLEL}"
echo "prompts:        ${N_PROMPTS}"
echo "prompt source:  ${PROMPT_SOURCE}"
echo "predict tokens: ${N_PREDICT_MIN}-${N_PREDICT}"
echo "output dir:     ${OUT_DIR}"
echo "slot sweep:     ${GPU_EXPERT_SLOT_NUMS}"

SERVER_PID=""
GPU_MEMORY_PID=""
GPU_MEMORY_CSV=""

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
    stop_gpu_memory_sampler
}

trap stop_server EXIT

start_gpu_memory_sampler() {
    local slot_out_dir="$1"
    GPU_MEMORY_CSV="${slot_out_dir}/gpu-memory.csv"

    if [[ "${GPU_MEMORY_SAMPLING}" != "1" ]]; then
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "status: unavailable (nvidia-smi not found)" > "${slot_out_dir}/gpu-memory-summary.txt"
        return 0
    fi

    echo "timestamp,memory_used_mib,memory_total_mib" > "${GPU_MEMORY_CSV}"
    (
        while true; do
            nvidia-smi \
                --id="${GPU_MEMORY_INDEX}" \
                --query-gpu=timestamp,memory.used,memory.total \
                --format=csv,noheader,nounits \
                >> "${GPU_MEMORY_CSV}" 2>/dev/null || true
            sleep "${GPU_MEMORY_POLL_SECONDS}"
        done
    ) &
    GPU_MEMORY_PID="$!"
}

stop_gpu_memory_sampler() {
    if [[ -n "${GPU_MEMORY_PID}" ]] && kill -0 "${GPU_MEMORY_PID}" 2>/dev/null; then
        kill "${GPU_MEMORY_PID}" 2>/dev/null || true
        wait "${GPU_MEMORY_PID}" 2>/dev/null || true
    fi
    GPU_MEMORY_PID=""

    if [[ -n "${GPU_MEMORY_CSV}" && -f "${GPU_MEMORY_CSV}" ]]; then
        awk -F',' '
            NR > 1 {
                gsub(/^[ \t]+|[ \t]+$/, "", $2)
                gsub(/^[ \t]+|[ \t]+$/, "", $3)
                if ($2 + 0 > max_used) { max_used = $2 + 0 }
                if ($3 + 0 > total) { total = $3 + 0 }
                samples += 1
            }
            END {
                if (samples == 0) {
                    print "status: unavailable (no GPU memory samples)"
                } else {
                    printf "max_memory_used_mib: %d\n", max_used
                    printf "memory_total_mib: %d\n", total
                    printf "samples: %d\n", samples
                }
            }
        ' "${GPU_MEMORY_CSV}" > "${GPU_MEMORY_CSV%/*}/gpu-memory-summary.txt"
    fi
    GPU_MEMORY_CSV=""
}

wait_for_health() {
    local elapsed=0
    local url="${SERVER_URL%/}/health"

    while ! curl -fsS --max-time 5 "${url}" >/dev/null 2>&1; do
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "error: llama-server exited before becoming healthy" >&2
            wait "${SERVER_PID}" 2>/dev/null || true
            exit 1
        fi
        if (( elapsed >= HEALTH_TIMEOUT_SECONDS )); then
            echo "error: llama-server did not become healthy at ${url} within ${HEALTH_TIMEOUT_SECONDS}s" >&2
            exit 1
        fi
        echo "waiting for llama-server health at ${url}... ${elapsed}s"
        sleep "${HEALTH_POLL_SECONDS}"
        elapsed=$((elapsed + HEALTH_POLL_SECONDS))
    done
}

wait_for_completion_ready() {
    local elapsed=0
    local url="${SERVER_URL%/}/v1/completions"
    local payload='{"prompt":"hello","max_tokens":1,"temperature":0,"stream":false}'

    while ! curl -fsS --max-time 30 \
            -H 'Content-Type: application/json' \
            -d "${payload}" \
            "${url}" >/dev/null 2>&1; do
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "error: llama-server exited before completions became ready" >&2
            wait "${SERVER_PID}" 2>/dev/null || true
            exit 1
        fi
        if (( elapsed >= HEALTH_TIMEOUT_SECONDS )); then
            echo "error: llama-server completions endpoint did not become ready at ${url} within ${HEALTH_TIMEOUT_SECONDS}s" >&2
            exit 1
        fi
        echo "waiting for llama-server completions at ${url}... ${elapsed}s"
        sleep "${HEALTH_POLL_SECONDS}"
        elapsed=$((elapsed + HEALTH_POLL_SECONDS))
    done
}

run_one_slot() {
    local slot_num="$1"
    local slot_out_dir="${OUT_DIR}/gpu_expert_slot_${slot_num}"

    stop_server
    mkdir -p "${slot_out_dir}"

    echo
    echo "============================================================"
    echo "benchmarking MOE_GPU_EXPERT_SLOT_NUM=${slot_num}"
    echo "============================================================"

    MOE_GPU_EXPERT_SLOT_NUM="${slot_num}" "${SERVING_SCRIPT}" > "${slot_out_dir}/server.log" 2>&1 &
    SERVER_PID="$!"
    start_gpu_memory_sampler "${slot_out_dir}"
    wait_for_health
    wait_for_completion_ready

    (
        cd "${slot_out_dir}"
        LLAMA_ARG_N_PARALLEL="${N_PARALLEL}" \
            python3 ../../scripts/server-bench.py \
                --path_server "${SERVER_URL}" \
                --path_log server-bench.log \
                --name "${NAME}-gpu-expert-slot-${slot_num}" \
                --prompt_source "${PROMPT_SOURCE}" \
                --n_prompts "${N_PROMPTS}" \
                --n_predict "${N_PREDICT}" \
                --n_predict_min "${N_PREDICT_MIN}" \
                2>&1 | tee server-bench.txt
    )
    stop_gpu_memory_sampler
}

for slot_num in ${GPU_EXPERT_SLOT_NUMS}; do
    run_one_slot "${slot_num}"
done

echo
echo "slot sweep complete. Results:"
for slot_num in ${GPU_EXPERT_SLOT_NUMS}; do
    echo "  ${OUT_DIR}/gpu_expert_slot_${slot_num}/server-bench.txt"
done
