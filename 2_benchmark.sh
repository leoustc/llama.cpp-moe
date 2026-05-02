#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8071}"
MODE="${MODE:-throughput}"
NAME="${NAME:-gemma4-26B-A4B-it-${MODE}}"
OUT_DIR="${OUT_DIR:-benchmark-results}"
GPU_EXPERT_SLOT_NUMS="${GPU_EXPERT_SLOT_NUMS:-0 16 32 64 72}"
SERVING_SCRIPT="${SERVING_SCRIPT:-./1_serving.sh}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-5}"

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

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}

trap stop_server EXIT

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

    echo
    echo "============================================================"
    echo "benchmarking MOE_GPU_EXPERT_SLOT_NUM=${slot_num}"
    echo "============================================================"

    MOE_GPU_EXPERT_SLOT_NUM="${slot_num}" "${SERVING_SCRIPT}" &
    SERVER_PID="$!"
    wait_for_health
    wait_for_completion_ready

    mkdir -p "${slot_out_dir}"
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
}

for slot_num in ${GPU_EXPERT_SLOT_NUMS}; do
    run_one_slot "${slot_num}"
done

echo
echo "slot sweep complete. Results:"
for slot_num in ${GPU_EXPERT_SLOT_NUMS}; do
    echo "  ${OUT_DIR}/gpu_expert_slot_${slot_num}/server-bench.txt"
done
