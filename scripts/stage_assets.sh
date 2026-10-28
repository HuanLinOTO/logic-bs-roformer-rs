#!/usr/bin/env bash
# Stage the model assets for packaging: DEST/assets/{model.safetensors,logic_bs_roformer.yaml}
#
# Sources, first hit wins:
#   $LBRR_ASSETS_DIR    directory holding model.safetensors (self-hosted runner)
#   <repo>/assets       local working copy (gitignored)
#   /opt/lbrr-assets    the mount the GPU validation step already probes
#   $LBRR_ASSETS_URL    archive (.zip/.tar.gz) or a bare model.safetensors
#
# A missing checkpoint is a warning, not a failure: hosted runners have no
# weights, and the binary bundle itself does not need them.
#
# Usage: scripts/stage_assets.sh third_party/assets
set -euo pipefail

dest=${1:?usage: stage_assets.sh DEST_DIR}
dest="$dest/assets"
root=$(cd "$(dirname "$0")/.." && pwd)
model=""
config=""

warn() { echo "::warning::assets: $*"; }

try_dir() {
  local dir=$1
  [ -n "$dir" ] && [ -d "$dir" ] || return 1
  [ -f "$dir/model.safetensors" ] || return 1
  model="$dir/model.safetensors"
  local candidate
  for candidate in "$dir/logic_bs_roformer.yaml" "$dir/logic_pro_config_v1.yaml"; do
    if [ -f "$candidate" ]; then
      config="$candidate"
      break
    fi
  done
  return 0
}

for candidate in "${LBRR_ASSETS_DIR:-}" "$root/assets" /opt/lbrr-assets; do
  if try_dir "$candidate"; then
    echo "assets: using $candidate"
    break
  fi
done

if [ -z "$model" ] && [ -n "${LBRR_ASSETS_URL:-}" ]; then
  echo "assets: downloading $LBRR_ASSETS_URL"
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  curl -fL --retry 3 --retry-delay 5 -o "$tmp/download" "$LBRR_ASSETS_URL"
  case "$LBRR_ASSETS_URL" in
    *.zip)
      mkdir -p "$tmp/extract"
      python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
        "$tmp/download" "$tmp/extract"
      ;;
    *.tar.gz | *.tgz)
      mkdir -p "$tmp/extract"
      tar -xzf "$tmp/download" -C "$tmp/extract"
      ;;
    *)
      mkdir -p "$tmp/extract"
      mv "$tmp/download" "$tmp/extract/model.safetensors"
      ;;
  esac
  model=$(find "$tmp/extract" -name model.safetensors -type f -print -quit)
  if [ -n "$model" ]; then
    config=$(find "$(dirname "$model")" -maxdepth 1 -name '*.yaml' -type f -print -quit)
  fi
fi

# The YAML is small and versioned, so it can always come from the repo.
if [ ! -f "${config:-/nonexistent}" ] && [ -f "$root/assets/logic_bs_roformer.yaml" ]; then
  config="$root/assets/logic_bs_roformer.yaml"
fi

if [ -z "$model" ] || [ ! -f "$model" ]; then
  warn "model.safetensors not found (set LBRR_ASSETS_URL or LBRR_ASSETS_DIR); bundle will omit weights"
  exit 0
fi

mkdir -p "$dest"
cp -f "$model" "$dest/model.safetensors"
if [ -f "${config:-/nonexistent}" ]; then
  cp -f "$config" "$dest/logic_bs_roformer.yaml"
else
  warn "logic_bs_roformer.yaml missing; separation needs the model config"
fi
echo "assets: staged model.safetensors ($(du -h "$dest/model.safetensors" | cut -f1)) into $dest"
