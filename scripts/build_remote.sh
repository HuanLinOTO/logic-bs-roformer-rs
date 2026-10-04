#!/bin/bash
# 远程（<REMOTE_NODE_NAME> / Debian 13 / RTX 3080 / CUDA 13.3）构建入口
# 用法: bash scripts/build_remote.sh [extra cargo-oxide args...]
set -e
export PATH=/root/.cargo/bin:/usr/local/cuda-13.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export RUSTUP_TOOLCHAIN=nightly-2026-08-28
export LIBCLANG_PATH=/usr/lib/llvm-19/lib
export CUDA_HOME=/usr/local/cuda-13.3
export CARGO_TARGET_DIR=/data/dsh/target-lbrr
cd /data/dsh/logic-bs-roformer-rs
exec cargo oxide run "$@"
