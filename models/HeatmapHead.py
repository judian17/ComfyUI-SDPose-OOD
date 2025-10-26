# Author: T. S. Liang @ Rama Alpaca
# Emails: tsliang2001@gmail.com | shuangliang@ramaalpaca.com | sliang57@connect.hku.hk
# Date: Jun. 2025
# Description: SDPose heatmap head factory using mmpose `HeatmapHead` + `UDPHeatmap` codec.
#   - mode="body"     → 17 keypoints, in_channels=320
#   - mode="wholebody"→ 133 keypoints, in_channels=640
#   Defaults: image_size=(768,1024), scale=4 (→ heatmap size = (w/4, h/4)), sigma=6.
# License: MIT License (see LICENSE for details)

from mmpose.registry import MODELS

def get_heatmap_head(mode = "body"):

    if mode == "body":

        image_size = [768, 1024] ## width x height
        sigma = 6 ## sigma is 2 for 256
        scale = 4

        embed_dim = 320
        num_keypoints = 17

        codec = dict(
            type='UDPHeatmap', input_size=(image_size[0], image_size[1]), heatmap_size=(int(image_size[0]/scale), int(image_size[1]/scale)), sigma=sigma) ## sigma is 2 for 256

        head_cfg = dict(
            type='HeatmapHead',
            in_channels=embed_dim,
            out_channels=num_keypoints,
            deconv_out_channels=(320,),
            deconv_kernel_sizes=(4,),
            conv_out_channels=(320,),
            conv_kernel_sizes=(1,),
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder=codec
        )

    elif mode == "wholebody":

        image_size = [768, 1024] ## width x height
        sigma = 6 ## sigma is 2 for 256
        scale = 4

        embed_dim = 640
        num_keypoints = 133

        codec = dict(
            type='UDPHeatmap', input_size=(image_size[0], image_size[1]), heatmap_size=(int(image_size[0]/scale), int(image_size[1]/scale)), sigma=sigma) ## sigma is 2 for 256

        head_cfg = dict(
            type='HeatmapHead',
            in_channels=embed_dim,
            out_channels=num_keypoints,
            deconv_out_channels=(640,),
            deconv_kernel_sizes=(4,),
            conv_out_channels=(640,),
            conv_kernel_sizes=(1,),
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder=codec
        )

    head = MODELS.build(head_cfg)
    
    return head