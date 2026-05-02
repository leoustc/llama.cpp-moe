#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

if ! command -v cmake >/dev/null 2>&1; then
    echo "error: cmake is required but was not found on PATH" >&2
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1 && [[ -z "${CUDAToolkit_ROOT:-}" ]]; then
    echo "error: CUDA toolkit was not found; install CUDA or set CUDAToolkit_ROOT before running this harness" >&2
    exit 1
fi

cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGGML_CUDA=ON

if ! grep -q '^GGML_CUDA:BOOL=ON$' "${BUILD_DIR}/CMakeCache.txt"; then
    echo "error: CUDA support was not enabled in ${BUILD_DIR}/CMakeCache.txt" >&2
    exit 1
fi

cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --target llama-cli llama-server -j "${JOBS}"

CLI_BIN="${BUILD_DIR}/bin/llama-cli"
if [[ ! -x "${CLI_BIN}" ]]; then
    CLI_BIN="${BUILD_DIR}/bin/${BUILD_TYPE}/llama-cli"
fi
if [[ ! -x "${CLI_BIN}" ]]; then
    echo "error: llama-cli binary not found under ${BUILD_DIR}/bin" >&2
    exit 1
fi

SERVER_BIN="${BUILD_DIR}/bin/llama-server"
if [[ ! -x "${SERVER_BIN}" ]]; then
    SERVER_BIN="${BUILD_DIR}/bin/${BUILD_TYPE}/llama-server"
fi
if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "error: llama-server binary not found under ${BUILD_DIR}/bin" >&2
    exit 1
fi

"${CLI_BIN}" --help | tee "${BUILD_DIR}/llama-cli-help.txt"

if ! grep -q -- '--moe-gpu-expert-slot-num' "${BUILD_DIR}/llama-cli-help.txt"; then
    echo "error: llama-cli --help did not list --moe-gpu-expert-slot-num" >&2
    exit 1
fi

"${SERVER_BIN}" --help | tee "${BUILD_DIR}/llama-server-help.txt"

if ! grep -q -- '--moe-gpu-expert-slot-num' "${BUILD_DIR}/llama-server-help.txt"; then
    echo "error: llama-server --help did not list --moe-gpu-expert-slot-num" >&2
    exit 1
fi

echo "build complete"
echo "llama-cli: ${CLI_BIN}"
echo "llama-server: ${SERVER_BIN}"
echo "help outputs:"
echo "  ${BUILD_DIR}/llama-cli-help.txt"
echo "  ${BUILD_DIR}/llama-server-help.txt"
