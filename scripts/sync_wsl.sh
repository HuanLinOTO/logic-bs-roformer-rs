#!/bin/bash
# Windows 源码同步到专用 WSL ext4 目录；先验证目标再执行 --delete。
set -euo pipefail
SRC=$(realpath "${LBRR_SOURCE_DIR:-$(dirname "$0")/..}")
DST=$(realpath -m "${LBRR_WSL_ROOT:-/root/work/lbrr}")
python3 - "$SRC" "$DST" <<'PY'
import pathlib, sys
src, dst = (pathlib.Path(p).resolve() for p in sys.argv[1:])
if not (src / 'Cargo.toml').is_file() or not (src / 'src/main.rs').is_file():
    raise SystemExit('拒绝同步：源目录不是 lbrr 源码目录')
if dst == src or dst in src.parents or src in dst.parents or len(dst.parts) < 4 or str(dst).startswith('/mnt/'):
    raise SystemExit('拒绝同步：目标必须是与源分离的专用 ext4 任务目录')
marker = dst / '.lbrr-sync-source'
if marker.exists():
    if marker.read_text().strip() != str(src):
        raise SystemExit('拒绝同步：目标属于其他源码工作区')
elif dst.exists() and any(dst.iterdir()):
    if str(dst) != '/root/work/lbrr' or not (dst / 'Cargo.toml').is_file():
        raise SystemExit('拒绝同步：非空目标没有此任务的归属标记')
dst.mkdir(parents=True, exist_ok=True)
marker.write_text(str(src) + chr(10))
print(f'已核验同步路径：{src} -> {dst}')
PY
rsync -a --delete \
  --exclude '/.lbrr-sync-source' --exclude '/.worktrees' --exclude '/.codegraph' \
  --exclude '/target*' --exclude '/.git' --exclude '/vendor' \
  --exclude '/assets' --exclude '/out' --exclude '/output' --exclude '/separated*' --exclude '*.npz' \
  "$SRC/" "$DST/"
mkdir -p "$DST/vendor"
rsync -a --exclude '.git' --exclude 'target' --exclude 'target-oxide' \
  "$SRC/vendor/cuda-rust/" "$DST/vendor/cuda-rust/"
echo "sync done -> $DST"
