# Author: T. S. Liang @ Rama Alpaca
# Emails: tsliang2001@gmail.com | shuangliang@ramaalpaca.com | sliang57@connect.hku.hk
# Date: Feb. 2025
# License: MIT License (see LICENSE for details)

from SDPose_Evaluator import SDPose_Evaluator
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Evaluate SDPose on Pose Estimation Benchmarks")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="The dataset name for the evaluation benchmark.",
    )

    parser.add_argument(
        "--keypoint_scheme",
        type=str,
        default="body",
        required=True,
        help="The keypoint scheme for the pose estimation model, please fill in body (COCO-17) or wholebody (COCOWholebody-133).",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        required=True,
        help="The root path for the datasets.",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        required=True,
        help="Path to the SDPose checkpoint.",
    )

    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="Batch size (per device) for the evaluation dataloader."
    )

    parser.add_argument(
        "--ann_file", type=str, help="The path for the annotation file."
    )

    parser.add_argument(
        "--timestep",
        type=int,
        default=999,
    )
    
    # dataloaderes
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    
    # using xformers for efficient training and inference
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()

    return args


args = parse_args()
Trainer = SDPose_Evaluator(args)

Trainer.evaluate()