#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-cuda:0}"
TEST_DATASET="${TEST_DATASET:-mvtec}"
TEST_PATH="${TEST_PATH:-/mnt/nvme-data1/pzh_proj/datasets/mvtec}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/nvme-data1/pzh_proj/my_uniadet/experiments/uniadet_zs_dinov2/visa_to_mvtec/checkpoints}"
EPOCH_START="${EPOCH_START:-1}"
EPOCH_END="${EPOCH_END:-14}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RESULT_ROOT="${RESULT_ROOT:-$(dirname "${CHECKPOINT_DIR}")/results_epoch_sweep_${EPOCH_START}_${EPOCH_END}/${RUN_TIMESTAMP}}"

mkdir -p "${RESULT_ROOT}"

echo "================================================================"
echo "UniADet Zero-Shot DINOv2 Checkpoint Sweep Test"
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
