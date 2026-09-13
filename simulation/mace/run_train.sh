#!/usr/bin/env bash
set -euo pipefail

WORK=/mnt/d/jacs/caf2_mace_weight0p25_20260805
VENV="$WORK/.venv_mace"
DATA="$WORK/data"
RUN="$WORK/train_seed20260805"
mkdir -p "$RUN"
cd "$RUN"

"$VENV/bin/mace_run_train" \
  --name=caf2_mace_weight0p25_seed20260805 \
  --train_file="$DATA/train_weighted_high0p25.xyz" \
  --valid_file="$DATA/validation_base403.xyz" \
  --test_file="$DATA/evaluation_high1432.xyz" \
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
  --save_cpu

touch "$WORK/training.complete"
