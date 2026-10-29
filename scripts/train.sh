#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -lt 3 ]]; then
  echo 'Usage: bash scripts/train.sh CODE_DIR DATA_NPZ OUTPUT_DIR [extra allmimic flags]'
  exit 2
fi
code_dir="$(realpath "$1")"
motions="$(realpath "$2")"
output="$(realpath -m "$3")"
shift 3
expected=cb334ff674b2f57248f33f8c21b7fb526fbdf8cc
actual="$(git -C "$code_dir" rev-parse HEAD)"
[[ "$actual" == "$expected" ]] || { echo "Unexpected upstream commit: $actual"; exit 1; }
[[ ! -e "$output" ]] || { echo "Refusing to reuse output path: $output"; exit 1; }
export PYTHONPATH="$code_dir:${PYTHONPATH:-}"
cd "$code_dir"
python -u -m allmimic.train --preset recommended --motions "$motions" --out "$output" \
  --seed 0 --num-envs 4096 --iterations 320000 --wall-hours 2 \
  --ckpt-every 4000 --eval-every 4000 --eval-clips 0 --eval-envs 40 "$@"
