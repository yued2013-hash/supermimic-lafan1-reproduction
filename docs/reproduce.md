# Reproduction guide

## 1. Inspect published evidence without a GPU

Run `python scripts/verify_results.py` and `python -m unittest discover -s tests -v`. These require only Python's standard library. Plotting additionally needs matplotlib.

## 2. Obtain the actual training implementation

```bash
git clone https://github.com/Shimingyi/allmimic.git ../allmimic
git -C ../allmimic checkout --detach cb334ff674b2f57248f33f8c21b7fb526fbdf8cc
```

Read its README, requirements and source. The pinned commit is the experiment implementation; this evidence repository does not replace it. Review its licensing before further redistribution.

## 3. Independent environment

Use a separate Linux Python3.11 environment. Actual experiment versions are recorded in `environment/versions.json` and `environment/pip-freeze.txt`. The experiment used an existing torch2.14.1+cu126 environment, then installed the pinned implementation's dependencies while retaining torch. This repository does not claim that a generic unpinned `pip install` on a fresh machine reproduces that environment; do not blindly upgrade or mix it into another robotics project's environment.

Verify torch CUDA availability, mjlab, mujoco, mujoco_warp and warp versions before GPU tests. No Isaac Sim installation is required for this allmimic backend.

## 4. Download and process the 40 clips

```bash
python scripts/download_lafan1.py --output ../experiment_data/raw
CODE_DIR="$(realpath ../allmimic)"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"
python "$CODE_DIR/scripts/prepare_motions.py" --format lafan   --fps-in 30 --fps-out 50 --input ../experiment_data/raw/g1   --output ../experiment_data/lafan1_50hz.npz --workers 4
```

The downloader pins the dataset revision and validates every CSV with SHA256. It skips an existing valid file and refuses to replace a different one. The complete library must contain40 unique names,441120 frames,50Hz. Use the implementation's motion-library loader to validate dimensions before training.

## 5. Start a bounded training run

```bash
bash scripts/train.sh "$CODE_DIR" "$(realpath ../experiment_data/lafan1_50hz.npz)"   "$(pwd)/runs/lafan1_seed0"
```

This starts a real GPU workload: 4096 environments, recommended preset,2-hour accumulated budget,seed0,checkpoint/eval every4000 steps and all40 eval clips. First run the implementation's CPU/GPU sanity checks and a small smoke test on your machine. The published run reached30665 iterations before the budget, not320000. Memory/time depend on installed versions and hardware.

## 6. Independent full-sequence FP32 evaluation

Place the same processed motion file at `LIBRARY_DIR/lafan1.npz`, then:

```bash
bash scripts/evaluate.sh "$CODE_DIR" "$CHECKPOINT" "$LIBRARY_DIR" "$EVAL_OUTPUT_DIR"
```

`tools/eval_allmimic.py` disables network AMP for fp32 and writes per-clip JSON. In-run evaluation uses bf16; do not merge precision labels. Confirm40 unique sequences,zero failures, and each sequence `steps == frames - 1`.

## 7. Reading the saved experiment

`results/lafan1_seed0/checkpoint_metadata.json` stores the actual cfg and counters; the final checkpoint SHA256 is in `results/summary.json`. Weights and raw motion data are not in Git. The checkpoint contains the world model,policy,critic and optimizer states; the replay buffer is not saved by the implementation.

The specialized compiled-BPTT consistency harness previously timed out after900 seconds (exit124). Basic tests and completed training do not establish that missing validation; rerun it separately if strict compilation-gradient parity is required.
