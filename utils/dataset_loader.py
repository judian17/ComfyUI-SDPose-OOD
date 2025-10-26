# Author: T. S. Liang @ Rama Alpaca
# Emails: tsliang2001@gmail.com | shuangliang@ramaalpaca.com
# Date: Feb. 2025
# Description: The dataloader for the training and evaluation protocol.
# License: MIT License (see LICENSE for details)

from torch.utils.data import DataLoader
from mmengine.dataset import default_collate 
from mmengine.registry import init_default_scope
from mmpose.registry import DATASETS

init_default_scope('mmpose')

def prepare_dataset(batch_size, dataset_name, dataset_root, mode = "val", num_workers = 4):

    image_size = [768, 1024] ## width x height
    sigma = 6 ## sigma is 2 for 256
    scale = 4

    codec = dict(
        type='UDPHeatmap', input_size=(image_size[0], image_size[1]), heatmap_size=(int(image_size[0]/scale), int(image_size[1]/scale)), sigma=sigma) 
    
    train_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='RandomFlip', direction='horizontal'),
        dict(type='RandomHalfBody'),
        dict(type='RandomBBoxTransform'),
        dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
        dict(type='PhotometricDistortion'),
        dict(type='CacheRGBTarget'),
        dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
        dict(type='GenerateTarget', encoder=codec),
        dict(type='PackPoseInputs'),
    ]

    val_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
        dict(type='CacheRGBTarget'),
        dict(type='PackPoseInputs'),
    ]

    train_cfg = dict(
        type='CocoDataset',
        data_root=dataset_root,
        data_mode='topdown',
        ann_file='COCO/annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='COCO/train2017/'),
        pipeline=train_pipeline,
    )

    if dataset_name == "COCO_OOD":

        print("Running Validation on COCO_OOD.")

        val_cfg = dict(
            type='CocoDataset',
            data_root=dataset_root,
            data_mode='topdown',
            ann_file='COCO/annotations/person_keypoints_val2017.json',
            data_prefix=dict(img='COCO/val2017oil/'),
            test_mode=True,
            pipeline=val_pipeline,
            bbox_file='COCO/person_detection_results/COCO_val2017_detections_AP_H_70_person.json',
        )
    
    elif dataset_name == "Humanart":

        print("Running Validation on Humanart.")

        val_cfg = dict(
            type='HumanArtDataset',
            data_root=dataset_root,
            data_mode='topdown',
            ann_file='HumanArt/annotations/validation_humanart.json',
            test_mode=True,
            pipeline=val_pipeline,
        )

    elif dataset_name == "COCO":

        print("Running Validation on COCO.")

        val_cfg = dict(
            type='CocoDataset',
            data_root=dataset_root,
            data_mode='topdown',
            ann_file='COCO/annotations/person_keypoints_val2017.json',
            data_prefix=dict(img='COCO/val2017/'),
            test_mode=True,
            pipeline=val_pipeline,
            bbox_file='COCO/person_detection_results/COCO_val2017_detections_AP_H_70_person.json',
        )
    
    elif dataset_name == "COCOWholebody":

        print("Running Validation on COCO Wholebody.")

        val_cfg = dict(
            type='CocoWholeBodyDataset',
            data_root=dataset_root,
            data_mode='topdown',
            ann_file='COCO/annotations/coco_wholebody_val_v1.0.json',
            data_prefix=dict(img='COCO/val2017/'),
            test_mode=True,
            pipeline=val_pipeline,
            bbox_file='COCO/person_detection_results/COCO_val2017_detections_AP_H_70_person.json',
        )

    elif dataset_name == "COCO-OOD_Wholebody":

        print("Running Validation on COCO-OOD Wholebody.")

        val_cfg = dict(
            type='CocoWholeBodyDataset',
            data_root=dataset_root,
            data_mode='topdown',
            ann_file='COCO/annotations/coco_wholebody_val_v1.0.json',
            data_prefix=dict(img='COCO/val2017oil/'),
            test_mode=True,
            pipeline=val_pipeline,
            bbox_file='COCO/person_detection_results/COCO_val2017_detections_AP_H_70_person.json',
        )

    if mode == "train":

        train_ds = DATASETS.build(train_cfg)
        val_ds = DATASETS.build(val_cfg)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=default_collate,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=default_collate,
            pin_memory=True,
        )

        return train_loader, val_loader

    elif mode == "val":

        val_ds = DATASETS.build(val_cfg)

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=default_collate,
            pin_memory=True,
        )

        return val_loader
