#!/bin/bash
# WSL ext4 构建入口；所有目录都可按任务隔离覆盖。
set -euo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda-13.3/bin:$PATH
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-nightly-2026-08-28}"
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-14/lib}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/root/work/target-lbrr}"
cd "${LBRR_WSL_ROOT:-/root/work/lbrr}"
exec cargo oxide "${LBRR_CARGO_ACTION:-run}" "$@"
