#!/bin/bash
cd /data/dsh/logic-bs-roformer-rs
export PATH=/root/.cargo/bin:/usr/local/cuda-13.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export RUSTUP_TOOLCHAIN=nightly-2026-08-28 LIBCLANG_PATH=/usr/lib/llvm-19/lib CUDA_HOME=/usr/local/cuda-13.3 CARGO_TARGET_DIR=/data/dsh/target-lbrr
cargo oxide build -- --release 2>&1 | tail -4
/data/dsh/target-lbrr/release/lbrr --bench --iters 30 --model-dir assets 2>&1 | grep -E "BENCH|SNR|panic|rror"
