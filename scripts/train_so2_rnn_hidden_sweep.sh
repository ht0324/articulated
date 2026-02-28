#!/bin/bash
# Train identical SO(2) RNN models while sweeping hidden size.
#
# Usage:
#   bash scripts/train_so2_rnn_hidden_sweep.sh 0
#   bash scripts/train_so2_rnn_hidden_sweep.sh 0 "64 128"
#   bash scripts/train_so2_rnn_hidden_sweep.sh 0 "64 128" data/estimation/trajectories_so2.pt

set -euo pipefail

GPUS="${1:-0}"
HIDDEN_SIZES="${2:-64 128}"
DATA_PATH="${3:-data/estimation/trajectories_so2.pt}"

# Fixed training setup (identical across runs except hidden size)
MODEL_TYPE="rnn"
MANIFOLD="so2"
EPOCHS=200
BATCH_SIZE=128
LR=1e-3
N_PLACE_CELLS=64
DROPOUT=0.5
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SEED=42
NUM_WORKERS=4
WANDB_PROJECT="articulated-estimation"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON="python"
fi

if [ "$GPUS" = "all" ]; then
    NUM_DEVICES=-1
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES="$GPUS"
    NUM_DEVICES=$(echo "$GPUS" | awk -F',' '{print NF}')
fi

if [ "$NUM_DEVICES" -gt 1 ] || [ "$NUM_DEVICES" -eq -1 ]; then
    STRATEGY="ddp"
else
    STRATEGY="auto"
fi

echo "============================================"
echo "  SO(2) RNN Hidden-Size Sweep"
echo "============================================"
echo "GPUs:         ${GPUS} (devices=${NUM_DEVICES}, strategy=${STRATEGY})"
echo "Data:         ${DATA_PATH}"
echo "Hidden sizes: ${HIDDEN_SIZES}"
echo "============================================"
echo ""

for HIDDEN_SIZE in $HIDDEN_SIZES; do
    OUTPUT_DIR="logs/estimation/rnn_so2_h${HIDDEN_SIZE}"

    echo "---- Training hidden_size=${HIDDEN_SIZE} ----"
    "$PYTHON" "$SCRIPT_DIR/train.py" \
        --data_path "$DATA_PATH" \
        --model_type "$MODEL_TYPE" \
        --manifold "$MANIFOLD" \
        --input_size 2 \
        --init_pos_size 4 \
        --hidden_size "$HIDDEN_SIZE" \
        --n_place_cells "$N_PLACE_CELLS" \
        --dropout "$DROPOUT" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --grad_clip "$GRAD_CLIP" \
        --epochs "$EPOCHS" \
        --accelerator gpu \
        --devices "$NUM_DEVICES" \
        --strategy "$STRATEGY" \
        --seed "$SEED" \
        --output_dir "$OUTPUT_DIR" \
        --wandb \
        --project "$WANDB_PROJECT"
    echo ""
done
