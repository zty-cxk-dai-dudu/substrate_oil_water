#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_train.sh --data-dir DIR --output DIR [--mace-run-train EXE] [--dry-run]

Train the CaF2/oil/water MACE model with the original seed and hyperparameters.
DIR must contain train_weighted_high0p25.xyz, validation_base403.xyz and
evaluation_high1432.xyz. Outputs and training.complete are written to --output.
The executable defaults to MACE_RUN_TRAIN or mace_run_train on PATH.
--dry-run prints the command without creating files or starting training.
EOF
}

data_dir=""
run_dir=""
train_exe="${MACE_RUN_TRAIN:-mace_run_train}"
dry_run=false
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --data-dir|--output|--mace-run-train)
      if (($# < 2)); then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
      case "$1" in
        --data-dir) data_dir="$2" ;;
        --output) run_dir="$2" ;;
        --mace-run-train) train_exe="$2" ;;
      esac
      shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$data_dir" || -z "$run_dir" ]]; then usage >&2; exit 2; fi
[[ "$data_dir" = /* ]] || data_dir="$PWD/$data_dir"
[[ "$run_dir" = /* ]] || run_dir="$PWD/$run_dir"
if [[ "$train_exe" == */* && "$train_exe" != /* ]]; then train_exe="$PWD/$train_exe"; fi

train_command=("$train_exe" \
  --name=caf2_mace_weight0p25_seed20260805 \
  --train_file="$data_dir/train_weighted_high0p25.xyz" \
  --valid_file="$data_dir/validation_base403.xyz" \
  --test_file="$data_dir/evaluation_high1432.xyz" \
  --energy_key=energy \
  --forces_key=forces \
  --E0s=average \
  --model=MACE \
  --pair_repulsion \
  --loss=universal \
  --num_interactions=2 \
  --num_channels=64 \
  --max_L=1 \
  --correlation=3 \
  --r_max=6.0 \
  --energy_weight=10 \
  --forces_weight=1000 \
  --batch_size=2 \
  --valid_batch_size=2 \
  --max_num_epochs=45 \
  --eval_interval=1 \
  --patience=15 \
  --scheduler_patience=5 \
  --lr=0.005 \
  --weight_decay=1e-8 \
  --ema \
  --ema_decay=0.99 \
  --amsgrad \
  --clip_grad=10.0 \
  --default_dtype=float32 \
  --device=cuda \
  --seed=20260805 \
  --num_workers=4 \
  --error_table=PerAtomMAE \
  --restart_latest \
  --keep_checkpoints \
  --save_cpu)

if "$dry_run"; then
  printf 'cd %q\n' "$run_dir"
  printf '%q ' "${train_command[@]}"
  printf '\n'
  exit 0
fi
for input_name in train_weighted_high0p25.xyz validation_base403.xyz evaluation_high1432.xyz; do
  if [[ ! -r "$data_dir/$input_name" ]]; then
    printf 'Training input not found: %s\n' "$data_dir/$input_name" >&2
    exit 2
  fi
done
if ! command -v "$train_exe" >/dev/null; then
  printf 'MACE training executable not found: %s\n' "$train_exe" >&2
  exit 2
fi
mkdir -p "$run_dir"
cd "$run_dir"
"${train_command[@]}"
touch "$run_dir/training.complete"
