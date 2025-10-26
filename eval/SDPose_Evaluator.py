# Author: T. S. Liang @ Rama Alpaca
# Emails: tsliang2001@gmail.com | shuangliang@ramaalpaca.com | sliang57@connect.hku.hk
# Date: Feb. 2025
# Description: Evaluation protocol for fine-tuned Stable Diffusion models on pose estimation benchmarks.
# License: MIT License (see LICENSE for details)

import os
import torch
from tqdm import tqdm
from safetensors.torch import load_file

from accelerate import Accelerator
from accelerate.logging import get_logger
from packaging import version
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils.import_utils import is_xformers_available

from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler
)

from pipelines.SDPose_D_Pipeline import SDPose_D_Pipeline

from models.HeatmapHead import get_heatmap_head
from models.ModifiedUNet import Modified_forward

from utils.EvalMetric import get_metric
from utils.dataset_loader import prepare_dataset


class SDPose_Evaluator:
    
    def __init__(
        self,
        args
        ):
        
        self.logger = get_logger(__name__, log_level="INFO")
        # -------------------- Device --------------------
        
        self.args = args
        self.accelerator = Accelerator()

        unet = UNet2DConditionModel.from_pretrained(
            self.args.checkpoint_path, subfolder="unet", revision=None,
            class_embed_type="projection", projection_class_embeddings_input_dim=4,
            low_cpu_mem_usage=False, device_map=None,
        )

        unet = Modified_forward(unet, keypoint_scheme = self.args.keypoint_scheme)
        
        vae = AutoencoderKL.from_pretrained(self.args.checkpoint_path,
                                            subfolder='vae')

        tokenizer = CLIPTokenizer.from_pretrained(self.args.checkpoint_path, subfolder='tokenizer')
        text_encoder = CLIPTextModel.from_pretrained(self.args.checkpoint_path, subfolder='text_encoder')

        # Loading Heatmap Autoencoder
        dec_path = os.path.join(self.args.checkpoint_path, "decoder", "decoder.safetensors")
        hm_decoder = get_heatmap_head(mode = self.args.keypoint_scheme)
        hm_decoder.load_state_dict(load_file(dec_path, device="cpu"), strict=True)

        noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.checkpoint_path, subfolder='scheduler'
        )
        
        self.model = SDPose_D_Pipeline(unet = unet, vae = vae, tokenizer = tokenizer, text_encoder = text_encoder, decoder = hm_decoder, scheduler = noise_scheduler)
        
        # using xformers for efficient attentions.
        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.model.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.decoder.requires_grad_(False)
        self.model.unet.requires_grad_(False)

        self.model.vae.eval()
        self.model.text_encoder.eval()
        self.model.decoder.eval()
        self.model.unet.eval()

    @torch.no_grad()
    def evaluate(self):

        val_loader = prepare_dataset(batch_size = self.args.eval_batch_size, dataset_name = self.args.dataset_name, dataset_root = self.args.dataset_root, mode = "val", num_workers = self.args.dataloader_num_workers)


        self.model.unet, self.model.vae, self.model.decoder, val_loader, self.model.text_encoder = self.accelerator.prepare(self.model.unet, self.model.vae, self.model.decoder, val_loader, self.model.text_encoder)
        
        device = self.accelerator.device

        metric = get_metric(ann_file = self.args.ann_file, mode = self.args.keypoint_scheme)
        
        metric.dataset_meta = val_loader.dataset.metainfo
        
        iterator = tqdm(val_loader, desc="Validating", leave=False) if self.accelerator.is_main_process else val_loader
        
        for batch in iterator:

            rgb_in = batch["inputs"] # [B, 3, H, W]
            rgb_in = rgb_in[:, [2, 1, 0], ...]
            rgb_in = rgb_in.float() / 255.0

            rgb_in = rgb_in * 2.0 - 1.0

            rgb_in = rgb_in.to(device)  # [B, 3, H, W]

            ds_list = batch['data_samples']

            # Enable flip test by setting test_cfg
            test_cfg = {'flip_test': True}  # Re-enabled to test our fixes
            
            # Run inference with flip test enabled
            timesteps = [self.args.timestep]
            outputs = self.model(rgb_in, mode = "predict", data_samples=batch['data_samples'], timesteps = timesteps, test_cfg=test_cfg)
            
            samples_for_metric = [s.to_dict() for s in outputs]
            
            metric.process(data_samples = samples_for_metric, data_batch = batch)

        self.accelerator.wait_for_everyone()
        res = metric.evaluate(len(val_loader.dataset))

        if self.accelerator.is_main_process:
                
            print(res)