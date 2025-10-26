# Author T. S. Liang @ Rama Alpaca, July. 2024.
# Email: tsliang2001@gmail.com & shuangliang@ramaalpaca.com
# Adapted from Marigold: https://github.com/prs-eth/marigold and MMPose: https://github.com/open-mmlab/mmpose
# License: MIT License (see LICENSE for details)

"""
    Description:
    
    This script is the pipeline utilizing Stable Diffusion Model for the pose perception.
    Adaptations have been made to ensure full compatibility with MMPose-style
    decoder heads, enabling seamless integration with existing heatmap-based
    keypoint decoders and evaluation protocols.
"""

from typing import Union, Optional

import torch
import torch.nn as nn

from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
    DDPMScheduler,
)

from PIL import Image

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from itertools import zip_longest
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)

from mmpose.models.heads.heatmap_heads.heatmap_head import HeatmapHead
    
class SDPose_D_Pipeline(DiffusionPipeline):
     
    """
    A diffusion-based human pose estimation pipeline that reconstructs pose heatmaps
    from RGB images using a Stable-Diffusion-style latent denoising process.

    NOTE:
        This pipeline assumes the scheduler/unet are configured for x0 prediction,
        i.e., `scheduler.config.prediction_type == "sample"`. In this setting the UNet
        directly predicts the clean latent x0 at a fixed diffusion step.

    Args:
        unet (`UNet2DConditionModel`):
            Conditional UNet used to predict the clean latent `x0`.
            Inputs: noisy pose/image latents and conditioning (image latent, text embeds).
        vae (`AutoencoderKL`):
            Variational Autoencoder for encoding RGB images to latents and for decoding
            latents (image/heatmap) back to pixel space as needed.
        text_encoder (`CLIPTextModel`):
            Text encoder to obtain conditioning embeddings. For text-free pose tasks,
            empty prompts are typically used.
        tokenizer (`CLIPTokenizer`):
            Tokenizer paired with the text encoder (tokenizes empty/neutral prompts).
        decoder (`HeatmapHead`):
            A lightweight head mapping denoised latent features to pose heatmaps,
            typically with shape `[B, num_keypoints, H, W]`.
        scheduler (`Union[DDIMScheduler, DDPMScheduler]`):
            Diffusion scheduler defining the diffusion process configuration
            (only used to set the fixed timestep).  
            Its `prediction_type` **must be `'sample'`** to ensure x₀ prediction.

    Returns:
        `SDPose_D_Pipeline`:
            Initialized pipeline ready for inference/training.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        decoder: HeatmapHead,
        scheduler: Union[DDIMScheduler, DDPMScheduler]
        ):

        super().__init__()
        
        self.register_modules(
            unet = unet,
            vae = vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            decoder = decoder,
            scheduler = scheduler
        )
        
        self.rgb_latent_scale_factor = 0.18215

        self.test_cfg = {
            'output_keypoint_indices': None,
            'flip_test': True,
            'flip_mode': 'heatmap',
            'shift_heatmap': False
        }

    @torch.no_grad()
    def __call__(
            self,
            rgb_in: Union[Image.Image, torch.Tensor],
            show_progress_bar: bool = True,
            timesteps = None,
            mode = None,
            data_samples = None,
            test_cfg = None
        ):
        
        """
        Run a forward inference through the SDPose-D pipeline.

        This function performs **x₀ prediction** using a Stable-Diffusion-style UNet
        conditioned on the RGB image latent, and optionally returns decoded pose heatmaps
        or full structured pose predictions (MMPose-compatible).

        Args:
            rgb_in (`PIL.Image.Image` or `torch.Tensor`):
                Input RGB image or batch tensor of shape `[B, 3, H, W]`.
                If a PIL image is provided, it should be preprocessed to a normalized tensor
                compatible with the VAE encoder input range.
            show_progress_bar (`bool`, optional, defaults to `True`):
                Whether to display a progress bar during diffusion inference.
            timesteps (`int` or `list[int]`, optional):
                Diffusion timestep(s) to use.  
            mode (`str`, optional):
                - `"predict"`: Use the decoder's `predict()` method to generate keypoint
                  predictions and attach them to `data_samples` (for evaluation).  
                - `None`: Decode the denoised latent directly to a heatmap tensor `[B, K, H, W]`
                  without MMPose post-processing.
            data_samples (`SampleList`, optional):
                A list of MMPose `PoseDataSample` objects required when `mode="predict"`.
            test_cfg (`dict`, optional):
                Configuration overrides for test-time behavior.  
                Examples: `{'flip_test': True, 'shift_heatmap': False}`.

        Returns:
            - If `mode == "predict"`:
                `SampleList`:  
                The input `data_samples` with updated `pred_instances` and optional `pred_fields`
                containing the decoded keypoints and heatmaps.
            - Else:
                `torch.Tensor`:  
                The predicted pose heatmap tensor of shape `[B, num_keypoints, H, W]`.

        Notes:
            - This pipeline operates in **x₀ prediction mode**, where the UNet directly predicts
              the clean latent representation instead of ε or v.  
            - The pose diffusion process uses a **single, fixed timestep** (default t=999),
              meaning there is no iterative denoising loop.
            - `test_cfg` supports `flip_test` inference (horizontal ensemble of original and flipped images).
            - `scheduler.config.prediction_type` must be set to `"sample"` for correct operation.
        """
            
        # ------------- Predicting pose ---------------
            
        # Predict pose images

        self.encode_empty_text()
        
        # Update test_cfg if provided
        if test_cfg is not None:
            self.test_cfg.update(test_cfg)
            
        heatmap_pred = self.single_infer(
            rgb_in = rgb_in,
            show_pbar = show_progress_bar,
            timesteps = timesteps,
            mode = mode,
            data_samples = data_samples
            )
            
        return heatmap_pred
    
    @torch.no_grad()
    def single_infer(
            self,
            rgb_in: torch.Tensor,
            timesteps,
            show_pbar: bool,
            mode = None,
            data_samples = None
        )-> torch.Tensor:

        device = self.unet.device
        rgb_in = rgb_in.to(device)
        bsz = rgb_in.shape[0]
        
        self.scheduler.set_timesteps(timesteps = timesteps, device=device)
        
        timesteps = torch.tensor(timesteps, device=device).long()

        # STEP 1: Encode image
        if self.test_cfg.get('flip_test', False):
            # Flip test: process both original and flipped images
            rgb_in_flipped = torch.flip(rgb_in, dims=[3])  # Flip horizontally (W dimension)
            
            # Encode both original and flipped images
            rgb_latent = self.encode_rgb(rgb_in)
            rgb_latent_flipped = self.encode_rgb(rgb_in_flipped)
            
            # Process both latents
            feats = []
            for rgb_latent_curr in [rgb_latent, rgb_latent_flipped]:
                # Denoising loop for current latent
                if show_pbar:
                    iterable = tqdm(
                        enumerate(timesteps),
                        total=len(timesteps),
                        leave=False,
                        desc=" " * 4 + "Diffusion denoising",
                    )
                else:
                    iterable = enumerate(timesteps)

                self.encode_empty_text()
                text_embed = self.empty_text_embed.repeat((bsz, 1, 1))

                for i, t in iterable:
                    task_emb_anno = torch.tensor([1, 0]).float().unsqueeze(0).to(device)
                    task_emb_anno = torch.cat([torch.sin(task_emb_anno), torch.cos(task_emb_anno)], dim=-1).repeat(bsz, 1)

                    if self.scheduler.config.prediction_type == "sample":
                        feat = self.unet(
                            rgb_latent_curr, t, text_embed, class_labels=task_emb_anno, return_dict=False, return_decoder_feats=True
                        )
                    else:
                        raise
                
                feats.append((feat,))  # Wrap each feat in a tuple to match expected format

        else:
            # Original single image processing
            rgb_latent = self.encode_rgb(rgb_in)

            # Denoising loop
            if show_pbar:
                iterable = tqdm(
                    enumerate(timesteps),
                    total=len(timesteps),
                    leave=False,
                    desc=" " * 4 + "Diffusion denoising",
                )
            else:
                iterable = enumerate(timesteps)

            self.encode_empty_text()
            text_embed = self.empty_text_embed.repeat((bsz, 1, 1))

            for i, t in iterable:
                task_emb_anno = torch.tensor([1, 0]).float().unsqueeze(0).to(device)
                task_emb_anno = torch.cat([torch.sin(task_emb_anno), torch.cos(task_emb_anno)], dim=-1).repeat(bsz, 1)

                if self.scheduler.config.prediction_type == "sample":
                    feat = self.unet(
                        rgb_latent, t, text_embed, class_labels=task_emb_anno, return_dict=False, return_decoder_feats=True
                    )
                else:
                    raise
            
            feats = (feat,)  # Wrap single tensor in tuple to match expected format for prediction mode
            feat_for_training = feat  # Keep original for training mode
        
        if mode == "predict":

            if isinstance(self.decoder, (nn.parallel.DistributedDataParallel, )):

                preds = self.decoder.module.predict(feats, data_samples, test_cfg=self.test_cfg)
            
            else:

                preds = self.decoder.predict(feats, data_samples, test_cfg=self.test_cfg)

            if isinstance(preds, tuple):
                batch_pred_instances, batch_pred_fields = preds
            else:
                batch_pred_instances = preds
                batch_pred_fields = None

            return self.add_pred_to_datasample(batch_pred_instances=batch_pred_instances, batch_data_samples=data_samples, batch_pred_fields=batch_pred_fields)

        elif mode == "inference":
            
            preds = self.decoder.predict(feats, data_samples, test_cfg=self.test_cfg)

            return preds
            
        else:
            if self.test_cfg.get('flip_test', False):
                
                if isinstance(feats, list):
                    # feats is [(feat_orig,), (feat_flipped,)], extract the first one for training
                    heatmap_pred = self.decoder(feats[0][0])  # Use feat_orig for training
                else:
                    # feats is a single tuple (feat,), extract the tensor
                    heatmap_pred = self.decoder(feats[0])
            else:
                # In training mode, decoder expects raw features, not wrapped in tuples
                heatmap_pred = self.decoder(feat_for_training)  # Use the original feat tensor

            return heatmap_pred

    def encode_empty_text(self):

        """
        Encode text embedding for empty prompt
        """

        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0]
    
    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        encode RGB image into latent.
            
        Args:
            
        rgb_in (`torch.Tensor`):
            Input RGB image to be encoded.
            
        Returns:
            `torch.Tensor`: Image latent.
        """
        
        rgb_latent = self.vae.encode(rgb_in).latent_dist.sample()
        rgb_latent = rgb_latent * self.rgb_latent_scale_factor
        
        return rgb_latent

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        
        # from https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/mmpose/models/pose_estimators/topdown.py#L122
        
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            bbox_centers = gt_instances.bbox_centers
            bbox_scales = gt_instances.bbox_scales
            input_size = data_sample.metainfo['input_size']

            pred_instances.keypoints = pred_instances.keypoints / input_size \
                * bbox_scales + bbox_centers - 0.5 * bbox_scales

            if output_keypoint_indices is not None:
                
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)
            
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
