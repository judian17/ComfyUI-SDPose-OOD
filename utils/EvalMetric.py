# Author: T. S. Liang @ Rama Alpaca
# Emails: tsliang2001@gmail.com | shuangliang@ramaalpaca.com
# Date: Feb. 2025
# Description: Build COCO-style keypoint metrics for SDPose evaluation.
# License: MIT License (see LICENSE for details)


from mmpose.evaluation.metrics import CocoMetric, CocoWholeBodyMetric

def get_metric(ann_file, mode = "body"):

    if mode == "body":

        metric = CocoMetric(iou_type='keypoints', ann_file = ann_file)

    elif mode == "wholebody":

        metric = CocoWholeBodyMetric(iou_type='keypoints', ann_file = ann_file)

    else:

        print("Error, please specify the correct type of keypoint scheme.")

    return metric