#!/bin/bash
# ============================================
# Author: T. S. Liang @ Rama Alpaca
# Emails: tsliang2001@gmail.com | shuangliang@ramaalpaca.com | sliang57@connect.hku.hk
# Date: Aug. 2025
# Description: Launch script for evaluating SDPose on pose estimation benchmarks.
# License: MIT License (see LICENSE for details)
# ============================================

LAUNCH_EVAL() {

    cd ..

    # ------------------ Configurable Parameters ------------------
    dataset_name='COCO'
    keypoint_scheme='body'  # or 'wholebody'
    dataset_root='/data/coding/datasets'
    ann_file='/data/coding/datasets/HumanArt/annotations/validation_humanart.json'
    checkpoint_path='/data/coding/outputs/COCO_813'
    eval_batch_size=16
    dataloader_num_workers=16
    timestep=999

    # ------------------ Runtime Environment ------------------
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --mixed_precision="bf16"\
        --multi_gpu \
        eval/launch_evaluator.py \
        --dataset_name $dataset_name \
        --keypoint_scheme $keypoint_scheme \
        --dataset_root $dataset_root \
        --checkpoint_path $checkpoint_path \
        --ann_file $ann_file \
        --eval_batch_size $eval_batch_size \
        --dataloader_num_workers $dataloader_num_workers \
        --enable_xformers_memory_efficient_attention \
        --timestep $timestep
}

# ------------- Entry -------------
LAUNCH_EVAL
