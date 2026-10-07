#!/bin/bash
# 本地 CI：overlay -> sync -> release 构建 -> GPU 自检 -> golden SNR 门槛
# 用法（Windows 侧一键）:
#   wsl -d Ubuntu-22.04 -u root -- bash -lc 'bash /mnt/d/Projects/logic-bs-roformer-rs/scripts/ci.sh'
# 或 WSL 内: bash scripts/ci.sh
# 门槛: SNR >= 60 dB（BENCH final SNR vs ref_output）
set -euo pipefail

SRC_DIR=$(realpath "${LBRR_SOURCE_DIR:-$(dirname "$0")/..}")
WSL_ROOT="${LBRR_WSL_ROOT:-/root/work/lbrr}"
export LBRR_SOURCE_DIR="$SRC_DIR" LBRR_WSL_ROOT="$WSL_ROOT"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/root/work/target-lbrr}"
SNR_GATE=60

echo "== [1/5] vendor overlay (cuda-oxide-codegen src/target) =="
# 上游 gitignore 吞掉了这个源码目录（不入库），fresh checkout 必须从
# vendor-overlay/ 恢复，否则 cuda-oxide-codegen 报 E0583。
mkdir -p "$SRC_DIR/vendor/cuda-rust/cuda-oxide/crates/cuda-oxide-codegen/src"
cp -r "$SRC_DIR/vendor-overlay/cuda-oxide-codegen/src/target" \
      "$SRC_DIR/vendor/cuda-rust/cuda-oxide/crates/cuda-oxide-codegen/src/"

echo "== [2/5] sync -> ext4 =="
bash "$SRC_DIR/scripts/sync_wsl.sh"

echo "== [3/5] build (release) =="
LBRR_CARGO_ACTION=build bash "$SRC_DIR/scripts/build_wsl.sh" -- --release

BIN="${CARGO_TARGET_DIR:-/root/work/target-lbrr}/release/lbrr"

echo "== [4/5] GPU self-test =="
cd "$SRC_DIR" && "$BIN" --self-test

echo "== [5/5] golden SNR gate (>= ${SNR_GATE}dB) =="
if [ ! -f "$SRC_DIR/assets/model.safetensors" ]; then
  echo "SKIP: assets/model.safetensors 不存在（需要预置模型权重才能跑 SNR 门槛）"
else
  cd "$SRC_DIR"
  "$BIN" --bench --iters 1 --model-dir assets > /tmp/ci_bench.log 2>&1 || { tail -5 /tmp/ci_bench.log; exit 1; }
  tail -3 /tmp/ci_bench.log
  snr=$(grep -oP 'BENCH final SNR vs ref_output: \K[0-9.]+' /tmp/ci_bench.log)
  [ -n "$snr" ] || { echo "FAIL: SNR 行缺失"; exit 1; }
  awk -v s="$snr" -v g="$SNR_GATE" 'BEGIN { if (s+0 < g+0) { printf "FAIL: SNR %s < %s\n", s, g; exit 1 } else printf "PASS: SNR %s >= %s\n", s, g }'
fi

echo "== CI OK =="
