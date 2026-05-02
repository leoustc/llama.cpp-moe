#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8071}"
NAME="${NAME:-gemma4-26B-A4B-it-locality}"
OUT_DIR="${OUT_DIR:-locality-results}"
GPU_EXPERT_SLOT_NUMS="${GPU_EXPERT_SLOT_NUMS:--1 16 32 64 72}"
REPEATS_PER_PROMPT_LIST="${REPEATS_PER_PROMPT_LIST:-16 32}"
PROMPT_GROUPS="${PROMPT_GROUPS:-1}"
SERVING_SCRIPT="${SERVING_SCRIPT:-./1_serving.sh}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-5}"
GPU_MEMORY_POLL_SECONDS="${GPU_MEMORY_POLL_SECONDS:-1}"
GPU_MEMORY_INDEX="${GPU_MEMORY_INDEX:-0}"
GPU_MEMORY_SAMPLING="${GPU_MEMORY_SAMPLING:-1}"
N_PARALLEL="${N_PARALLEL:-2}"
N_PREDICT="${N_PREDICT:-64}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED="${SEED:-0}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required but was not found on PATH" >&2
    exit 1
fi

python3 - <<'PY'
import importlib.util
import sys

missing = [name for name in ("requests", "matplotlib") if importlib.util.find_spec(name) is None]
if missing:
    print("error: missing Python locality benchmark dependencies: " + ", ".join(missing), file=sys.stderr)
    print("install with: python3 -m pip install requests matplotlib", file=sys.stderr)
    sys.exit(1)
PY

mkdir -p "${OUT_DIR}"

echo "benchmark mode: locality"
echo "server url:     ${SERVER_URL}"
echo "parallel:       ${N_PARALLEL}"
echo "prompt groups:  ${PROMPT_GROUPS}"
echo "repeat sweep:   ${REPEATS_PER_PROMPT_LIST}"
echo "predict tokens: ${N_PREDICT}"
echo "output dir:     ${OUT_DIR}"
echo "slot sweep:     ${GPU_EXPERT_SLOT_NUMS}"

SERVER_PID=""
GPU_MEMORY_PID=""
GPU_MEMORY_CSV=""

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

run_locality_case() {
    local slot_num="$1"
    local repeats="$2"
    local slot_out_dir="${OUT_DIR}/gpu_expert_slot_${slot_num}"
    local repeat_out_dir="${slot_out_dir}/repeat_${repeats}"

    mkdir -p "${repeat_out_dir}"
    echo
    echo "------------------------------------------------------------"
    echo "locality benchmark: MOE_GPU_EXPERT_SLOT_NUM=${slot_num}, repeats=${repeats}"
    echo "------------------------------------------------------------"

    (
        cd "${repeat_out_dir}"
        python3 - "${SERVER_URL}" "${NAME}-gpu-expert-slot-${slot_num}-repeat-${repeats}" \
            "${PROMPT_GROUPS}" "${repeats}" "${N_PARALLEL}" "${N_PREDICT}" "${TEMPERATURE}" "${SEED}" \
            2>&1 <<'PY' | tee locality-bench.txt
import concurrent.futures
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import requests


server_url = sys.argv[1].rstrip("/")
name = sys.argv[2]
prompt_groups = int(sys.argv[3])
repeats_per_prompt = int(sys.argv[4])
n_parallel = int(sys.argv[5])
n_predict = int(sys.argv[6])
temperature = float(sys.argv[7])
seed = int(sys.argv[8])

topics = [
    "sparse mixture-of-experts routing",
    "GPU memory pressure during inference",
    "KV cache allocation",
    "expert cache locality",
    "CPU pinned memory transfers",
    "router logits and top-k experts",
    "batch scheduling for LLM serving",
    "token bucket packing",
    "prefetch latency hiding",
    "active expert eviction",
]


def make_prompt(group_id: int) -> str:
    topic = topics[group_id % len(topics)]
    return (
        f"Locality group {group_id:03d}: Explain {topic} for a llama.cpp "
        "MoE inference server in one concise paragraph. Include one practical "
        "observation about performance."
    )


requests_data = []
for group_id in range(prompt_groups):
    prompt = make_prompt(group_id)
    for repeat_id in range(repeats_per_prompt):
        requests_data.append({
            "prompt": prompt,
            "group_id": group_id,
            "repeat_id": repeat_id,
            "request_id": len(requests_data),
        })

dataset_path = Path("repeated-prompts.jsonl")
with dataset_path.open("w", encoding="utf-8") as f:
    for row in requests_data:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def tokenize_prompt(session: requests.Session, prompt: str) -> int:
    response = session.post(
        f"{server_url}/tokenize",
        json={"content": prompt, "add_special": True},
        timeout=30,
    )
    response.raise_for_status()
    return len(response.json()["tokens"])


def send_prompt(row: dict) -> dict:
    session = requests.Session()
    try:
        request_seed = seed + row["request_id"]
        payload = {
            "prompt": row["prompt"],
            "ignore_eos": True,
            "max_tokens": n_predict,
            "temperature": temperature,
            "seed": request_seed,
            "stream": True,
        }

        t_submit = time.time()
        response = session.post(f"{server_url}/v1/completions", json=payload, stream=True, timeout=600)
        response.raise_for_status()

        token_arrival_times = []
        first_token_time = None
        for line in response.iter_lines(decode_unicode=False):
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            now = time.time()
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            text = choices[0].get("text", "") if choices else ""
            if text:
                if first_token_time is None:
                    first_token_time = now
                token_arrival_times.append(now)

        if first_token_time is None:
            first_token_time = time.time()

        return {
            "request_id": row["request_id"],
            "group_id": row["group_id"],
            "repeat_id": row["repeat_id"],
            "t_submit": t_submit,
            "first_token_time": first_token_time,
            "token_arrival_times": token_arrival_times,
            "generated_tokens": len(token_arrival_times),
            "error": None,
        }
    except Exception as exc:
        return {
            "request_id": row["request_id"],
            "group_id": row["group_id"],
            "repeat_id": row["repeat_id"],
            "t_submit": time.time(),
            "first_token_time": time.time(),
            "token_arrival_times": [],
            "generated_tokens": 0,
            "error": repr(exc),
        }
    finally:
        session.close()


with requests.Session() as session:
    prompt_lengths = [tokenize_prompt(session, make_prompt(group_id)) for group_id in range(prompt_groups)]

prompt_length_by_group = {group_id: prompt_lengths[group_id] for group_id in range(prompt_groups)}

print("Generating repeated prompts...")
print("Starting the locality benchmark...\n")
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
    results = list(executor.map(send_prompt, requests_data))

errors = [r for r in results if r["error"]]
if errors:
    for error in errors[:5]:
        print(f"error: request {error['request_id']} failed: {error['error']}", file=sys.stderr)
    raise SystemExit(f"{len(errors)} locality requests failed")

token_times = [t for result in results for t in result["token_arrival_times"]]
if not token_times:
    raise SystemExit("no tokens generated")

duration = max(token_times) - t0
total_prompt_tokens = sum(prompt_length_by_group[r["group_id"]] for r in results)
total_generated_tokens = sum(r["generated_tokens"] for r in results)
prompt_latencies = [r["first_token_time"] - r["t_submit"] for r in results]
depth_sum = 0
for result in results:
    prompt_len = prompt_length_by_group[result["group_id"]]
    n_tokens = result["generated_tokens"]
    depth_sum += n_tokens * prompt_len
    depth_sum += n_tokens * (n_tokens + 1) // 2

request_throughput = len(results) / duration
prompt_latency_avg = sum(prompt_latencies) / len(prompt_latencies)
prompt_speed = total_prompt_tokens / sum(prompt_latencies)
generation_speed = total_generated_tokens / duration

print(f"Benchmark name:                    {name}")
print(f"Prompt groups:                     {prompt_groups}")
print(f"Repeats per prompt:                {repeats_per_prompt}")
print(f"Total requests:                    {len(results)}")
print(f"Parallel requests:                 {n_parallel}")
print(f"Benchmark duration:                {duration:.2f} s")
print(f"Request throughput:                {request_throughput:.2f} requests/s = {request_throughput * 60:.2f} requests/min")
print(f"Total prompt length:               {total_prompt_tokens} tokens")
print(f"Average prompt length:             {total_prompt_tokens / len(results):.2f} tokens")
print(f"Average prompt latency:            {1e3 * prompt_latency_avg:.2f} ms")
print(f"Average prompt speed:              {prompt_speed:.2f} tokens/s")
print(f"Total generated tokens:            {total_generated_tokens}")
print(f"Average generation depth:          {depth_sum / total_generated_tokens:.2f} tokens")
print(f"Average total generation speed:    {generation_speed:.2f} tokens/s")
print(f"Average generation speed per slot: {generation_speed / n_parallel:.2f} tokens/s / slot")

with Path("locality-results.jsonl").open("w", encoding="utf-8") as f:
    for result in results:
        row = dict(result)
        row["prompt_length"] = prompt_length_by_group[result["group_id"]]
        row["token_arrival_times"] = [t - t0 for t in result["token_arrival_times"]]
        row["first_token_latency_ms"] = 1e3 * (result["first_token_time"] - result["t_submit"])
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

plt.figure()
plt.scatter(
    [r["repeat_id"] for r in results],
    [1e3 * (r["first_token_time"] - r["t_submit"]) for r in results],
    s=10.0,
    marker=".",
    alpha=0.35,
)
plt.title(name)
plt.xlabel("Repeat id within prompt group")
plt.ylabel("Time to first token [ms]")
plt.savefig("prompt_time_by_repeat.png", dpi=240)

relative_token_times = [t - t0 for t in token_times]
bin_max = int(duration) + 2
plt.figure()
plt.hist(relative_token_times, range(0, bin_max + 1))
plt.xlim(0, bin_max + 1)
plt.title(name)
plt.xlabel("Time [s]")
plt.ylabel("Num. tokens generated per second")
plt.savefig("gen_rate.png", dpi=240)
PY
    )
}

run_slot() {
    local slot_num="$1"
    local slot_out_dir="${OUT_DIR}/gpu_expert_slot_${slot_num}"

    stop_server
    mkdir -p "${slot_out_dir}"

    echo
    echo "============================================================"
    echo "starting locality server with MOE_GPU_EXPERT_SLOT_NUM=${slot_num}"
    echo "============================================================"

    MOE_GPU_EXPERT_SLOT_NUM="${slot_num}" "${SERVING_SCRIPT}" > "${slot_out_dir}/server.log" 2>&1 &
    SERVER_PID="$!"
    start_gpu_memory_sampler "${slot_out_dir}"
    wait_for_health
    wait_for_completion_ready

    for repeats in ${REPEATS_PER_PROMPT_LIST}; do
        run_locality_case "${slot_num}" "${repeats}"
    done

    stop_server
}

for slot_num in ${GPU_EXPERT_SLOT_NUMS}; do
    run_slot "${slot_num}"
done

echo
echo "locality sweep complete. Results:"
for slot_num in ${GPU_EXPERT_SLOT_NUMS}; do
    for repeats in ${REPEATS_PER_PROMPT_LIST}; do
        echo "  ${OUT_DIR}/gpu_expert_slot_${slot_num}/repeat_${repeats}/locality-bench.txt"
    done
done

exit 0
