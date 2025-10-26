# Author: T. S. Liang @ Rama Alpaca
# Emails: tsliang2001@gmail.com | shuangliang@ramaalpaca.com
# Date: Feb. 2025

"""

Description:

    SDPose UNet Forward Modifier

    Temporarily overrides `UNet.forward` to expose decoder features via forward hooks.
    - keypoint_scheme: "body" (COCO-17) or "wholebody" (COCO-WholeBody-133)
    - return_decoder_feats: if True, returns a selected decoder feature (Tensor);
    otherwise returns the original UNet output.
"""

def new_forward_kpt17(self, *args, return_decoder_feats=False, **kwargs):
    self._decoder_feats = []

    def hook_fn(module, input, output):
        self._decoder_feats.append(output)

    handles = [blk.register_forward_hook(hook_fn) for blk in self.up_blocks]

    out = self._old_forward(*args, **kwargs)

    for h in handles:
        h.remove()

    if return_decoder_feats:

        feats = self._decoder_feats[::-1]
        return feats[0]
    else:
        return out

def new_forward_kpt133(self, *args, return_decoder_feats=False, **kwargs):
    self._decoder_feats = []

    def hook_fn(module, input, output):
        self._decoder_feats.append(output)

    handles = [blk.register_forward_hook(hook_fn) for blk in self.up_blocks]

    out = self._old_forward(*args, **kwargs)

    for h in handles:
        h.remove()

    if return_decoder_feats:

        feats = self._decoder_feats[::-1]
        return feats[1]
    else:
        return out

def Modified_forward(unet, keypoint_scheme = "body"):

    if keypoint_scheme == "body":

        unet._old_forward = unet.forward
        unet.forward = new_forward_kpt17.__get__(unet, unet.__class__)

    elif keypoint_scheme == "wholebody":

        unet._old_forward = unet.forward
        unet.forward = new_forward_kpt133.__get__(unet, unet.__class__)

    return unet