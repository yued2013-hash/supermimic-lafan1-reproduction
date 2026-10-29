#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 4 ]]; then
  echo 'Usage: bash scripts/evaluate.sh CODE_DIR CHECKPOINT LIBRARY_DIR OUTPUT_DIR'
  exit 2
fi
code_dir="$(realpath "$1")"
checkpoint="$(realpath "$2")"
library="$(realpath "$3")"
output="$(realpath -m "$4")"
expected=cb334ff674b2f57248f33f8c21b7fb526fbdf8cc
actual="$(git -C "$code_dir" rev-parse HEAD)"
[[ "$actual" == "$expected" ]] || { echo "Unexpected upstream commit: $actual"; exit 1; }
[[ -f "$library/lafan1.npz" ]] || { echo 'Missing LIBRARY_DIR/lafan1.npz'; exit 1; }
[[ ! -e "$output" ]] || { echo "Refusing to overwrite evaluation output: $output"; exit 1; }
export PYTHONPATH="$code_dir:${PYTHONPATH:-}"
cd "$code_dir"
python -u tools/eval_allmimic.py --ckpt "$checkpoint" --lib-dir "$library" \
  --testsets lafan1 --num-envs 40 --out-dir "$output"
