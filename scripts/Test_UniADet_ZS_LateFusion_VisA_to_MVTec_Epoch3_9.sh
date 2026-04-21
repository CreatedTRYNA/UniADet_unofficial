#!/usr/bin/env bash

set -e

GPU=cuda:6
TEST_DATASET="mvtec"
TEST_PATH="/mnt/nvme-data1/pzh_proj/datasets/mvtec"
CHECKPOINT_DIR="/mnt/nvme-data1/pzh_proj/my_uniadet/experiments/uniadet_zs_late_fusion/visa_to_mvtec/checkpoints"
EPOCH_START=3
EPOCH_END=9
RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULT_ROOT="$(dirname "${CHECKPOINT_DIR}")/results_epoch_sweep_3_9/${RUN_TIMESTAMP}"

echo "================================================================"
echo "UniADet Zero-Shot Late-Fusion Checkpoint Sweep Test"
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

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing late-fusion epoch ${epoch}..."
    python test_uniadet_zs_late_fusion.py \
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
echo "Late-fusion checkpoint sweep completed."
echo "================================================================"
