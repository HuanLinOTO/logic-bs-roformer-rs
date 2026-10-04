#!/bin/bash
# 同步 Windows 侧项目到 WSL ext4（构建宿主）
# 用法: bash /mnt/d/Projects/logic-bs-roformer-rs/scripts/sync_wsl.sh
set -e
SRC=/mnt/d/Projects/logic-bs-roformer-rs
DST=/root/work/lbrr

mkdir -p $DST
rsync -a --delete \
  --exclude '/target' --exclude '/.git' --exclude '/vendor' \
  --exclude '/assets' --exclude '/out' --exclude '*.npz' \
  $SRC/ $DST/

mkdir -p $DST/vendor
rsync -a \
  --exclude '.git' --exclude 'target' --exclude 'target-oxide' \
  $SRC/vendor/cuda-rust/ $DST/vendor/cuda-rust/

echo "sync done -> $DST"
