#!/bin/bash
# 远程独立构建入口；保留既有目录默认值。
set -euo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda-13.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-nightly-2026-08-28}"
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-19/lib}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/data/dsh/target-lbrr}"
cd "${LBRR_REMOTE_ROOT:-/data/dsh/logic-bs-roformer-rs}"
exec cargo oxide "${LBRR_CARGO_ACTION:-run}" "$@"
