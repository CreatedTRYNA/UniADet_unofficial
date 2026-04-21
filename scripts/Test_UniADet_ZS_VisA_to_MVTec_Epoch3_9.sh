#!/usr/bin/env bash

set -e

GPU=cuda:1
TEST_DATASET="mvtec"
TEST_PATH="/mnt/nvme-data1/pzh_proj/datasets/mvtec"
# CHECKPOINT_DIR="/mnt/nvme-data1/pzh_proj/my_uniadet/experiments/uniadet_zs/visa_to_mvtec/20260419_195439/checkpoints"
# CHECKPOINT_DIR="/mnt/nvme-data1/pzh_proj/my_uniadet/experiments/uniadet_zs_late_fusion/visa_to_mvtec/checkpoints"
# CHECKPOINT_DIR="/mnt/nvme-data1/pzh_proj/my_uniadet/experiments/uniadet_zs_loss_cls0.5_seg1.0/visa_to_mvtec/20260418_225355/checkpoints"
CHECKPOINT_DIR="/mnt/nvme-data1/pzh_proj/my_uniadet/experiments/uniadet_zs_loss_cls1.0_seg0.5/visa_to_mvtec/20260418_225513/checkpoints"
EPOCH_START=3
EPOCH_END=9
RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULT_ROOT="$(dirname "${CHECKPOINT_DIR}")/results_epoch_sweep_3_9/${RUN_TIMESTAMP}"

echo "================================================================"
echo "UniADet Zero-Shot Checkpoint Sweep Test"
echo "================================================================"
echo "Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "Test Dataset:   ${TEST_DATASET}"
echo "Test Path:      ${TEST_PATH}"
echo "GPU:            ${GPU}"
echo "Epoch Range:    ${EPOCH_START} -> ${EPOCH_END}"
echo "Result Root:    ${RESULT_ROOT}"
echo "================================================================"
echo ""

for epoch in $(seq "${EPOCH_START}" "${EPOCH_END}"); do
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/epoch_${epoch}.pth"
    SAVE_PATH="${RESULT_ROOT}/epoch_${epoch}"

    if [ ! -f "${CHECKPOINT_PATH}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip epoch ${epoch}: checkpoint not found at ${CHECKPOINT_PATH}"
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing epoch ${epoch}..."
    python test_uniadet_zs.py \
        --test_data_path "${TEST_PATH}" \
        --test_dataset "${TEST_DATASET}" \
        --data_mode test \
        --checkpoint_path "${CHECKPOINT_PATH}" \
        --save_path "${SAVE_PATH}" \
        --device "${GPU}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished epoch ${epoch}. Results: ${SAVE_PATH}"
    echo ""
done

echo "================================================================"
echo "Checkpoint sweep completed."
echo "================================================================"
