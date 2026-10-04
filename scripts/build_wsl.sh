#!/bin/bash
# WSL 构建入口：cargo oxide run（standalone 项目）
set -e
export PATH=/root/.cargo/bin:/usr/local/cuda-12.6/bin:$PATH
export RUSTUP_TOOLCHAIN=nightly-2026-08-28
export LIBCLANG_PATH=/usr/lib/llvm-14/lib
export CUDA_HOME=/usr/local/cuda-12.6
export CARGO_TARGET_DIR=/root/work/target-lbrr
cd /root/work/lbrr
exec cargo oxide run "$@"