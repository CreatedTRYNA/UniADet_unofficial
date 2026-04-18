#!/usr/bin/env bash

set -e

BATCH_SIZE=8
GPU=cuda:6
BACKBONE="DINOv2-R ViT-L/14"
IMAGE_SIZE=518
TEMPERATURE=0.07
SCORE_FUSION_WEIGHT=0.5
FEATURE_LAYERS=(12 15 18 21 24)
ENABLE_CAA=1

MVTEC_PATH="/mnt/nvme-data1/pzh_proj/datasets/mvtec"
VISA_PATH="/mnt/nvme-data1/pzh_proj/datasets/visa"

echo "================================================================"
echo "UniADet Zero-Shot Cross-Dataset Evaluation - DINOv2-R ViT-L/14"
echo "================================================================"
echo "Batch Size: ${BATCH_SIZE}"
echo "GPU: ${GPU}"
echo "Backbone: ${BACKBONE}"
echo "Image Size: ${IMAGE_SIZE}"
echo "Temperature: ${TEMPERATURE}"
echo "Score Fusion Weight: ${SCORE_FUSION_WEIGHT}"
echo "Feature Layers: ${FEATURE_LAYERS[*]}"
if [ "${ENABLE_CAA}" -eq 1 ]; then
    echo "CAA: ENABLED"
else
    echo "CAA: DISABLED"
fi
echo "Dataset Paths:"
echo "  - MVTec: ${MVTEC_PATH}"
echo "  - VisA:  ${VISA_PATH}"
echo "Epoch Configuration:"
echo "  - MVTec -> VisA: 15 epochs"
echo "  - VisA -> MVTec: 15 epochs"
echo "================================================================"
echo ""

run_uniadet_zs_experiment() {
    local train_dataset=$1
    local test_dataset=$2
    local train_path=$3
    local test_path=$4
    local epochs=$5

    local exp_name="${train_dataset}_to_${test_dataset}"
    local exp_dir="./experiments/uniadet_zs_dinov2/${exp_name}"
    local checkpoint_dir="${exp_dir}/checkpoints"
    local result_dir="${exp_dir}/results"

    echo ""
    echo "=========================================="
    echo "Experiment: ${exp_name}"
    echo "Training on: ${train_dataset} (${epochs} epochs)"
    echo "Testing on: ${test_dataset}"
    echo "=========================================="

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training on ${train_dataset}..."
    if [ "${ENABLE_CAA}" -eq 1 ]; then
        python train_uniadet_zs.py \
            --train_data_path "${train_path}" \
            --save_path "${checkpoint_dir}" \
            --train_dataset "${train_dataset}" \
            --data_mode test \
            --backbone "${BACKBONE}" \
            --features_list "${FEATURE_LAYERS[@]}" \
            --epoch "${epochs}" \
            --batch_size "${BATCH_SIZE}" \
            --image_size "${IMAGE_SIZE}" \
            --temperature "${TEMPERATURE}" \
            --score_fusion_weight "${SCORE_FUSION_WEIGHT}" \
            --enable_caa \
            --device "${GPU}"
    else
        python train_uniadet_zs.py \
            --train_data_path "${train_path}" \
            --save_path "${checkpoint_dir}" \
            --train_dataset "${train_dataset}" \
            --data_mode test \
            --backbone "${BACKBONE}" \
            --features_list "${FEATURE_LAYERS[@]}" \
            --epoch "${epochs}" \
            --batch_size "${BATCH_SIZE}" \
            --image_size "${IMAGE_SIZE}" \
            --temperature "${TEMPERATURE}" \
            --score_fusion_weight "${SCORE_FUSION_WEIGHT}" \
            --device "${GPU}"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed!"
    echo ""

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing on ${test_dataset} (epoch ${epochs})..."
    python test_uniadet_zs.py \
        --test_data_path "${test_path}" \
        --test_dataset "${test_dataset}" \
        --data_mode test \
        --checkpoint_path "${checkpoint_dir}/epoch_${epochs}.pth" \
        --save_path "${result_dir}/epoch_${epochs}" \
        --device "${GPU}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Test completed!"
    echo "Results saved in: ${result_dir}"
    echo ""
}

echo ""
echo "################################################################"
echo "# Experiment 1/2: MVTec -> VisA"
echo "################################################################"

run_uniadet_zs_experiment \
    "mvtec" \
    "visa" \
    "${MVTEC_PATH}" \
    "${VISA_PATH}" \
    15

echo ""
echo "################################################################"
echo "# Experiment 2/2: VisA -> MVTec"
echo "################################################################"

run_uniadet_zs_experiment \
    "visa" \
    "mvtec" \
    "${VISA_PATH}" \
    "${MVTEC_PATH}" \
    15

echo ""
echo "================================================================"
echo "ALL UNIADet DINOv2-R EXPERIMENTS COMPLETED!"
echo "================================================================"
