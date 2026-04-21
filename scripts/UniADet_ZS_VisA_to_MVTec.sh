#!/usr/bin/env bash

set -e

BATCH_SIZE=8
GPU=cuda:6
BACKBONE="ViT-L/14@336px"
IMAGE_SIZE=518
TEMPERATURE=0.07
SCORE_FUSION_WEIGHT=0.5
FEATURE_LAYERS=(12 15 18 21 24)
ENABLE_CAA=1
EVAL_INTERVAL=3
EVAL_METRIC="mean_auc"
EPOCHS=15
RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

TRAIN_DATASET="visa"
TEST_DATASET="mvtec"
TRAIN_PATH="/mnt/nvme-data1/pzh_proj/datasets/visa"
TEST_PATH="/mnt/nvme-data1/pzh_proj/datasets/mvtec"

EXP_NAME="${TRAIN_DATASET}_to_${TEST_DATASET}"
EXP_DIR="./experiments/uniadet_zs/${EXP_NAME}/${RUN_TIMESTAMP}"
CHECKPOINT_DIR="${EXP_DIR}/checkpoints"
RESULT_DIR="${EXP_DIR}/results"

echo "================================================================"
echo "UniADet Zero-Shot Single-Direction Evaluation - CLIP"
echo "================================================================"
echo "Experiment: ${EXP_NAME}"
echo "Batch Size: ${BATCH_SIZE}"
echo "GPU: ${GPU}"
echo "Backbone: ${BACKBONE}"
echo "Image Size: ${IMAGE_SIZE}"
echo "Temperature: ${TEMPERATURE}"
echo "Score Fusion Weight: ${SCORE_FUSION_WEIGHT}"
echo "Feature Layers: ${FEATURE_LAYERS[*]}"
echo "Epochs: ${EPOCHS}"
echo "Periodic Eval: every ${EVAL_INTERVAL} epochs"
echo "Best Checkpoint Metric: ${EVAL_METRIC}"
if [ "${ENABLE_CAA}" -eq 1 ]; then
    echo "CAA: ENABLED"
else
    echo "CAA: DISABLED"
fi
echo "Train Dataset Path: ${TRAIN_PATH}"
echo "Test Dataset Path:  ${TEST_PATH}"
echo "Run Timestamp: ${RUN_TIMESTAMP}"
echo "================================================================"
echo ""

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training on ${TRAIN_DATASET}..."
if [ "${ENABLE_CAA}" -eq 1 ]; then
    python train_uniadet_zs.py \
        --train_data_path "${TRAIN_PATH}" \
        --save_path "${CHECKPOINT_DIR}" \
        --train_dataset "${TRAIN_DATASET}" \
        --data_mode test \
        --backbone "${BACKBONE}" \
        --features_list "${FEATURE_LAYERS[@]}" \
        --epoch "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --image_size "${IMAGE_SIZE}" \
        --temperature "${TEMPERATURE}" \
        --score_fusion_weight "${SCORE_FUSION_WEIGHT}" \
        --test_data_path "${TEST_PATH}" \
        --test_dataset "${TEST_DATASET}" \
        --eval_interval "${EVAL_INTERVAL}" \
        --eval_metric "${EVAL_METRIC}" \
        --enable_caa \
        --device "${GPU}"
else
    python train_uniadet_zs.py \
        --train_data_path "${TRAIN_PATH}" \
        --save_path "${CHECKPOINT_DIR}" \
        --train_dataset "${TRAIN_DATASET}" \
        --data_mode test \
        --backbone "${BACKBONE}" \
        --features_list "${FEATURE_LAYERS[@]}" \
        --epoch "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --image_size "${IMAGE_SIZE}" \
        --temperature "${TEMPERATURE}" \
        --score_fusion_weight "${SCORE_FUSION_WEIGHT}" \
        --test_data_path "${TEST_PATH}" \
        --test_dataset "${TEST_DATASET}" \
        --eval_interval "${EVAL_INTERVAL}" \
        --eval_metric "${EVAL_METRIC}" \
        --device "${GPU}"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed!"
echo ""

SELECTED_CHECKPOINT="${CHECKPOINT_DIR}/best_model.pth"
SELECTED_NAME="best"
if [ ! -f "${SELECTED_CHECKPOINT}" ]; then
    SELECTED_CHECKPOINT="${CHECKPOINT_DIR}/final_model.pth"
    SELECTED_NAME="final"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing on ${TEST_DATASET} (${SELECTED_NAME} checkpoint)..."
python test_uniadet_zs.py \
    --test_data_path "${TEST_PATH}" \
    --test_dataset "${TEST_DATASET}" \
    --data_mode test \
    --checkpoint_path "${SELECTED_CHECKPOINT}" \
    --save_path "${RESULT_DIR}/${SELECTED_NAME}" \
    --device "${GPU}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Test completed!"
echo "Results saved in: ${RESULT_DIR}"
echo "================================================================"
echo "VISA -> MVTEC EXPERIMENT COMPLETED!"
echo "================================================================"
