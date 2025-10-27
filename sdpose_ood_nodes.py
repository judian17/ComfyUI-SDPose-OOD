import os
import warnings
import logging

# Suppress common warnings to reduce noise
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

# Set logging level to reduce verbose output
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("deepspeed").setLevel(logging.ERROR)

# Environment detection
IS_COMFYUI_ENV = False
try:
    import comfy
    IS_COMFYUI_ENV = True
except ImportError:
    pass

# Additional warning suppression for specific environments
if IS_COMFYUI_ENV:
    # In ComfyUI environment, suppress more warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)

import torch
import numpy as np
from PIL import Image
import json

# Try to import folder_paths from ComfyUI
try:
    import folder_paths
except ImportError:
    # If not available, create a mock object for testing
    class MockFolderPaths:
        models_dir = os.path.expanduser("~/models")
        supported_pt_extensions = {".pt", ".pth", ".ckpt", ".safetensors"}
        folder_names_and_paths = {}
    folder_paths = MockFolderPaths()

from huggingface_hub import snapshot_download
import sys
from pathlib import Path
import cv2
import math
import matplotlib.colors
import tempfile
from torchvision import transforms

# Try to import model_management from ComfyUI
try:
    import model_management
except ImportError:
    # If not available, create a mock object for testing
    class MockModelManagement:
        @staticmethod
        def get_torch_device():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        @staticmethod
        def soft_empty_cache():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    model_management = MockModelManagement()

# Add the project root to Python path to allow direct imports
sys.path.append(str(Path(__file__).parent))

# --- Imports from the original SDPose project ---
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

# Try relative imports first, then absolute imports
try:
    from .models.HeatmapHead import get_heatmap_head
    from .models.ModifiedUNet import Modified_forward
    from .pipelines.SDPose_D_Pipeline import SDPose_D_Pipeline
except ImportError:
    # Fallback to absolute imports
    from models.HeatmapHead import get_heatmap_head
    from models.ModifiedUNet import Modified_forward
    from pipelines.SDPose_D_Pipeline import SDPose_D_Pipeline

from safetensors.torch import load_file

try:
    # Temporarily suppress ultralytics warnings during import
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("SDPose Node: ultralytics library not found. YOLO detection will be disabled.")
except Exception as e:
    YOLO_AVAILABLE = False
    print(f"SDPose Node: Failed to import ultralytics: {e}. YOLO detection will be disabled.")

# --- Add custom folder paths to ComfyUI ---
SDPOSE_MODEL_DIR = os.path.join(folder_paths.models_dir, "SDPose_OOD")
YOLO_MODEL_DIR = os.path.join(folder_paths.models_dir, "yolo")
folder_paths.folder_names_and_paths["SDPose_OOD"] = ([SDPOSE_MODEL_DIR], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["yolo"] = ([YOLO_MODEL_DIR], {".pt"})

os.makedirs(SDPOSE_MODEL_DIR, exist_ok=True)
os.makedirs(YOLO_MODEL_DIR, exist_ok=True)

# --- Define path for the local empty embedding ---
NODE_DIR = Path(__file__).parent
EMPTY_EMBED_DIR = os.path.join(NODE_DIR, "empty_text_encoder")
os.makedirs(EMPTY_EMBED_DIR, exist_ok=True)

# --- Helper functions for tensor/image conversion ---
def tensor_to_pil(tensor):
    """Converts a torch tensor (CHW, float32, 0-1) to a PIL Image (RGB)."""
    return Image.fromarray((tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

def pil_to_tensor(image):
    """Converts a PIL Image (RGB) to a torch tensor (BCHW, float32, 0-1)."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def numpy_to_tensor(img_np):
    """Converts a numpy array (HWC, uint8, 0-255) to a torch tensor (BCHW, float32, 0-1)."""
    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)

# --- Drawing functions (copied from SDPose_gradio.py) ---
# (Functions draw_body17_keypoints_openpose_style and draw_wholebody_keypoints_openpose_style are omitted for brevity but would be pasted here)
def draw_body17_keypoints_openpose_style(canvas, keypoints, scores=None, threshold=0.3, overlay_mode=False, overlay_alpha=0.6):
    H, W, C = canvas.shape
    if len(keypoints) >= 7:
        neck = (keypoints[5] + keypoints[6]) / 2
        neck_score = min(scores[5], scores[6]) if scores is not None else 1.0
        candidate = np.zeros((18, 2))
        candidate_scores = np.zeros(18)
        candidate[0] = keypoints[0]; candidate[1] = neck; candidate[2] = keypoints[6]; candidate[3] = keypoints[8]; candidate[4] = keypoints[10]; candidate[5] = keypoints[5]; candidate[6] = keypoints[7]; candidate[7] = keypoints[9]; candidate[8] = keypoints[12]; candidate[9] = keypoints[14]; candidate[10] = keypoints[16]; candidate[11] = keypoints[11]; candidate[12] = keypoints[13]; candidate[13] = keypoints[15]; candidate[14] = keypoints[2]; candidate[15] = keypoints[1]; candidate[16] = keypoints[4]; candidate[17] = keypoints[3]
        if scores is not None:
            candidate_scores[0] = scores[0]; candidate_scores[1] = neck_score; candidate_scores[2] = scores[6]; candidate_scores[3] = scores[8]; candidate_scores[4] = scores[10]; candidate_scores[5] = scores[5]; candidate_scores[6] = scores[7]; candidate_scores[7] = scores[9]; candidate_scores[8] = scores[12]; candidate_scores[9] = scores[14]; candidate_scores[10] = scores[16]; candidate_scores[11] = scores[11]; candidate_scores[12] = scores[13]; candidate_scores[13] = scores[15]; candidate_scores[14] = scores[2]; candidate_scores[15] = scores[1]; candidate_scores[16] = scores[4]; candidate_scores[17] = scores[3]
    else: return canvas
    avg_size = (H + W) / 2; stickwidth = max(1, int(avg_size / 256)); circle_radius = max(2, int(avg_size / 192))
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(len(limbSeq)):
        index = np.array(limbSeq[i]) - 1
        if index[0] >= len(candidate) or index[1] >= len(candidate): continue
        if scores is not None and (candidate_scores[index[0]] < threshold or candidate_scores[index[1]] < threshold): continue
        Y = candidate[index.astype(int), 0]; X = candidate[index.astype(int), 1]; mX = np.mean(X); mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        if length < 1: continue
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
    for i in range(18):
        if scores is not None and candidate_scores[i] < threshold: continue
        x, y = int(candidate[i][0]), int(candidate[i][1])
        if x < 0 or y < 0 or x >= W or y >= H: continue
        cv2.circle(canvas, (x, y), circle_radius, colors[i % len(colors)], thickness=-1)
    return canvas

def draw_wholebody_keypoints_openpose_style(canvas, keypoints, scores=None, threshold=0.3, overlay_mode=False, overlay_alpha=0.6):
    H, W, C = canvas.shape; stickwidth = 4
    body_limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
    hand_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    if len(keypoints) >= 18:
        for i, limb in enumerate(body_limbSeq):
            idx1, idx2 = limb[0] - 1, limb[1] - 1
            if idx1 >= 18 or idx2 >= 18: continue
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold): continue
            Y = np.array([keypoints[idx1][0], keypoints[idx2][0]]); X = np.array([keypoints[idx1][1], keypoints[idx2][1]]); mX = np.mean(X); mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            if length < 1: continue
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
        for i in range(18):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), 4, colors[i % len(colors)], thickness=-1)
    if len(keypoints) >= 24:
        for i in range(18, 24):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), 4, colors[i % len(colors)], thickness=-1)
    if len(keypoints) >= 113:
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 92 + edge[0], 92 + edge[1]
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold): continue
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1]); x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            if x1 > 0.01 and y1 > 0.01 and x2 > 0.01 and y2 > 0.01 and 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                color = matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0]) * 255
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        for i in range(92, 113):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > 0.01 and y > 0.01 and 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    if len(keypoints) >= 134:
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 113 + edge[0], 113 + edge[1]
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold): continue
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1]); x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            if x1 > 0.01 and y1 > 0.01 and x2 > 0.01 and y2 > 0.01 and 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                color = matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0]) * 255
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        for i in range(113, 134):
            if scores is not None and i < len(scores) and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > 0.01 and y > 0.01 and 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    if len(keypoints) >= 92:
        for i in range(24, 92):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > 0.01 and y > 0.01 and 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas

# --- Processing functions (adapted from SDPose_gradio.py) ---
# (Functions detect_person_yolo, preprocess_image_for_sdpose, restore_keypoints_to_original, convert_to_openpose_json are omitted for brevity but would be pasted here)
def detect_person_yolo(image, yolo_model_path, confidence_threshold=0.5):
    if not YOLO_AVAILABLE: return [[0, 0, image.shape[1], image.shape[0]]], False
    try:
        model = YOLO(yolo_model_path)
        results = model(image, verbose=False)
        person_bboxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > confidence_threshold:
                        person_bboxes.append(box.xyxy[0].cpu().numpy().tolist())
        if person_bboxes: return person_bboxes, True
        else: return [[0, 0, image.shape[1], image.shape[0]]], False
    except Exception as e:
        print(f"SDPose Node: YOLO detection failed: {e}")
        return [[0, 0, image.shape[1], image.shape[0]]], False

def preprocess_image_for_sdpose(image, bbox=None, input_size=(768, 1024)):
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else: pil_image = image
    original_size = pil_image.size
    crop_info = (0, 0, pil_image.width, pil_image.height)
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        if x2 > x1 and y2 > y1:
            pil_image = pil_image.crop((x1, y1, x2, y2))
            crop_info = (x1, y1, x2 - x1, y2 - y1)
    
    transform = transforms.Compose([
        transforms.Resize((input_size[1], input_size[0]), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(pil_image).unsqueeze(0)
    return input_tensor, original_size, crop_info

def restore_keypoints_to_original(keypoints, crop_info, input_size, original_size):
    x1, y1, crop_w, crop_h = crop_info
    scale_x = crop_w / input_size[0]; scale_y = crop_h / input_size[1]
    keypoints_restored = keypoints.copy()
    keypoints_restored[:, 0] = keypoints[:, 0] * scale_x + x1
    keypoints_restored[:, 1] = keypoints[:, 1] * scale_y + y1
    return keypoints_restored

def convert_to_loader_json(all_keypoints, all_scores, image_width, image_height, keypoint_scheme="body", threshold=0.3):
    """
    Converts keypoints to the "Loader" JSON format (OpenPose-18 body only)
    regardless of the input scheme (Body or WholeBody).
    Applies a score threshold, setting keypoints below it to [0, 0, 0].
    """
    people = []
    for keypoints, scores in zip(all_keypoints, all_scores):
        person_data = {}
        pose_kpts_18 = []

        if keypoint_scheme == "body":
            # Input is COCO-17, we must convert it to OpenPose-18 by adding a 'Neck'
            if len(keypoints) < 17 or len(scores) < 17:
                continue # Skip if data is incomplete

            # 1. Calculate Neck
            neck = (keypoints[5] + keypoints[6]) / 2
            neck_score = min(scores[5], scores[6])

            # 2. Create 18-point arrays from 17-point arrays
            op_keypoints = np.zeros((18, 2))
            op_scores = np.zeros(18)
            
            coco_to_op_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            
            for i_op in range(18):
                if i_op == 1: # Neck keypoint
                    op_keypoints[i_op] = neck
                    op_scores[i_op] = neck_score
                else:
                    i_coco = coco_to_op_map[i_op]
                    if i_coco >= len(keypoints): continue
                    op_keypoints[i_op] = keypoints[i_coco]
                    op_scores[i_op] = scores[i_coco]
            
            # 3. Flatten to [x, y, score, ...] with thresholding
            for i in range(18):
                score = float(op_scores[i])
                if score < threshold:
                    pose_kpts_18.extend([0.0, 0.0, 0.0])
                else:
                    pose_kpts_18.extend([float(op_keypoints[i, 0]), float(op_keypoints[i, 1]), score])

        else: # "wholebody"
            # Input is already OpenPose-18 + face/hands/feet. We just take the first 18 points.
            if len(keypoints) < 18 or len(scores) < 18:
                continue # Skip if data is incomplete
                
            # Flatten to [x, y, score, ...] with thresholding
            for i in range(18):
                score = float(scores[i])
                if score < threshold:
                    pose_kpts_18.extend([0.0, 0.0, 0.0])
                else:
                    pose_kpts_18.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), score])

        person_data["pose_keypoints_2d"] = pose_kpts_18
        people.append(person_data)

    # Use 'width' and 'height' keys as per the target format
    return {"width": int(image_width), "height": int(image_height), "people": people}

def convert_to_openpose_json(all_keypoints, all_scores, image_width, image_height, keypoint_scheme="body"):
    people = []
    for keypoints, scores in zip(all_keypoints, all_scores):
        person_data = {}
        if keypoint_scheme == "body":
            # Body only: 17 keypoints
            pose_kpts = [v for i in range(min(17, len(keypoints))) for v in [float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])]]
            person_data["pose_keypoints_2d"] = pose_kpts
        else: # wholebody - 期望 134 点输入
            # Body: 0-17 (18 keypoints including neck at index 17)
            pose_kpts = [v for i in range(min(18, len(keypoints))) for v in [float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])]]
            # Foot: 18-23 (6 keypoints)
            foot_kpts = [v for i in range(18, min(24, len(keypoints))) for v in [float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])]]
            # Face: 24-91 (68 keypoints)
            face_kpts = [v for i in range(24, min(92, len(keypoints))) for v in [float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])]]
            # Right hand: 92-112 (21 keypoints)
            right_hand_kpts = [v for i in range(92, min(113, len(keypoints))) for v in [float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])]]
            # Left hand: 113-133 (21 keypoints)
            left_hand_kpts = [v for i in range(113, min(134, len(keypoints))) for v in [float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])]]
            
            person_data.update({"pose_keypoints_2d": pose_kpts, "foot_keypoints_2d": foot_kpts, "face_keypoints_2d": face_kpts, "hand_right_keypoints_2d": right_hand_kpts, "hand_left_keypoints_2d": left_hand_kpts})
        people.append(person_data)
    return {"people": people, "canvas_width": int(image_width), "canvas_height": int(image_height)}


class YOLOModelLoader:
    """ComfyUI node to load a YOLO model for object detection."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("yolo"),),
            }
        }

    RETURN_TYPES = ("YOLO_MODEL",)
    FUNCTION = "load_yolo"
    CATEGORY = "SDPose"

    def load_yolo(self, model_name):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics library is not available. Please install it to use YOLO models.")
        
        model_path = folder_paths.get_full_path("yolo", model_name)
        if not model_path:
            raise FileNotFoundError(f"YOLO model not found: {model_name}")
            
        print(f"SDPose Node: Loading YOLO model from {model_path}")
        model = YOLO(model_path)
        return (model,)


class SDPoseOODLoader:
    """ComfyUI node to load SDPose models and dependencies."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["Body", "WholeBody"],),
                "unet_precision": (["fp32", "fp16", "bf16"],),
                "device": (["auto", "cuda", "cpu"],),
                "unload_on_finish": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SDPOSE_MODEL",)
    FUNCTION = "load_sdpose_model"
    CATEGORY = "SDPose"

    def load_sdpose_model(self, model_type, unet_precision, device, unload_on_finish):
        repo_id = {
            "Body": "teemosliang/SDPose-Body",
            "WholeBody": "teemosliang/SDPose-Wholebody"
        }[model_type]
        
        keypoint_scheme = model_type.lower()
        model_path = os.path.join(SDPOSE_MODEL_DIR, repo_id.split('/')[-1])
        
        if not os.path.exists(os.path.join(model_path, "unet")):
            print(f"SDPose Node: Downloading model from {repo_id} to {model_path}")
            snapshot_download(repo_id=repo_id, local_dir=model_path, local_dir_use_symlinks=False)

        if device == "auto":
            device = model_management.get_torch_device()
        else:
            device = torch.device(device)

        # --- Precision Handling ---
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[unet_precision]
        if device.type == 'cpu' and dtype == torch.float16:
            print("SDPose Node Warning: FP16 is not supported on CPU. Falling back to FP32.")
            dtype = torch.float32

        print(f"SDPose Node: Loading model on device: {device} with UNet precision: {unet_precision}")
        
        # --- Load Empty Embedding ---
        embed_path = os.path.join(EMPTY_EMBED_DIR, "empty_embedding.safetensors")
        if not os.path.exists(embed_path):
            raise FileNotFoundError(f"Empty embedding not found at '{embed_path}'. Please run 'generate_empty_embedding.py' script first.")
        empty_text_embed = load_file(embed_path)["empty_text_embed"].to(device)

        # --- Load Models ---
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype).to(device)
        unet = Modified_forward(unet, keypoint_scheme=keypoint_scheme)
        vae = AutoencoderKL.from_pretrained(model_path, subfolder='vae').to(device) # VAE is small, keep fp32
        
        dec_path = os.path.join(model_path, "decoder", "decoder.safetensors")
        hm_decoder = get_heatmap_head(mode=keypoint_scheme).to(device)
        if os.path.exists(dec_path):
            hm_decoder.load_state_dict(load_file(dec_path, device=str(device)), strict=True)
        hm_decoder = hm_decoder.to(dtype)
        
        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder='scheduler')

        sdpose_model = {
            "unet": unet,
            "vae": vae,
            "empty_text_embed": empty_text_embed,
            "decoder": hm_decoder,
            "scheduler": noise_scheduler,
            "keypoint_scheme": keypoint_scheme,
            "device": device,
            "unload_on_finish": unload_on_finish
        }
        return (sdpose_model,)

class SDPoseOODProcessor:
    """ComfyUI node to run the SDPose inference pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sdpose_model": ("SDPOSE_MODEL",),
                "image": ("IMAGE",),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.9, "step": 0.05}),
                "overlay_alpha": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "yolo_model": ("YOLO_MODEL",),
                "save_for_editor": ("BOOLEAN", {"default": False}),
                "filename_prefix_edit": ("STRING", {"default": "poses/pose_edit"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "pose_json")
    FUNCTION = "process"
    CATEGORY = "SDPose"

    def process(self, sdpose_model, image, score_threshold, overlay_alpha, yolo_model=None, save_for_editor=False, filename_prefix_edit="poses/pose_edit"):
        device = sdpose_model["device"]
        keypoint_scheme = sdpose_model["keypoint_scheme"]
        input_size = (768, 1024)

        # --- Handle batch processing for video frames ---
        B, H, W, C = image.shape
        images_np = image.cpu().numpy()
        
        print(f"SDPose Node: Processing {B} frame(s) with resolution {W}x{H}")
        
        # Initialize progress tracking
        try:
            from comfy.utils import ProgressBar
            comfy_pbar = ProgressBar(B)
            progress = 0
        except ImportError:
            comfy_pbar = None
            progress = 0

        # --- Instantiate a mock pipeline for inference ---
        class MockPipeline:
            def __init__(self, model_dict):
                self.unet = model_dict["unet"]
                self.vae = model_dict["vae"]
                self.decoder = model_dict["decoder"]
                self.scheduler = model_dict["scheduler"]
                self.empty_text_embed = model_dict["empty_text_embed"]
                self.device = model_dict["device"]

            @torch.no_grad()
            def __call__(self, rgb_in, timesteps, test_cfg, **kwargs):
                unet_dtype = self.unet.dtype
                bsz = rgb_in.shape[0]
                
                # VAE is fp32, cast its output to the UNet's precision
                rgb_latent = self.vae.encode(rgb_in).latent_dist.sample() * 0.18215
                rgb_latent = rgb_latent.to(dtype=unet_dtype)

                t = torch.tensor(timesteps, device=self.device).long()
                
                # Cast other float inputs to the UNet's precision
                text_embed = self.empty_text_embed.repeat((bsz, 1, 1)).to(dtype=unet_dtype)
                task_emb_anno = torch.tensor([1, 0]).float().unsqueeze(0).to(self.device)
                task_emb_anno = torch.cat([torch.sin(task_emb_anno), torch.cos(task_emb_anno)], dim=-1).repeat(bsz, 1)
                task_emb_anno = task_emb_anno.to(dtype=unet_dtype)
                
                feat = self.unet(rgb_latent, t, text_embed, class_labels=task_emb_anno, return_dict=False, return_decoder_feats=True)
                return self.decoder.predict((feat,), None, test_cfg=test_cfg)

        pipeline = MockPipeline(sdpose_model)

        # --- Process all frames ---
        result_images = []
        all_frames_json_data = []
        
        for frame_idx in range(B):
            # --- Convert current frame to numpy array ---
            original_image_rgb = (images_np[frame_idx] * 255).astype(np.uint8)
            original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)

            # --- YOLO Detection for current frame ---
            if yolo_model is not None and YOLO_AVAILABLE:
                results = yolo_model(original_image_bgr, verbose=False)
                bboxes = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                                bboxes.append(box.xyxy[0].cpu().numpy().tolist())
                if not bboxes:
                    h, w = original_image_bgr.shape[:2]
                    bboxes = [[0, 0, w, h]]
            else:
                if yolo_model is not None and not YOLO_AVAILABLE and frame_idx == 0:
                    print("SDPose Node Warning: YOLO model provided, but 'ultralytics' is not installed. Processing full image.")
                h, w = original_image_bgr.shape[:2]
                bboxes = [[0, 0, w, h]]

            # --- Process each detected person in current frame ---
            pose_canvas = np.zeros_like(original_image_rgb)
            all_keypoints, all_scores = [], []

            for bbox in bboxes:
                input_tensor, _, crop_info = preprocess_image_for_sdpose(original_image_bgr, bbox, input_size)
                input_tensor = input_tensor.to(device)

                with torch.no_grad():
                    out = pipeline(input_tensor, timesteps=[999], test_cfg={'flip_test': False}, show_progress_bar=False, mode="inference")
                    # The MMPose decoder head already returns numpy arrays on CPU
                    keypoints = out[0].keypoints[0]
                    scores = out[0].keypoint_scores[0]

                keypoints_original = restore_keypoints_to_original(keypoints, crop_info, input_size, (original_image_rgb.shape[1], original_image_rgb.shape[0]))

                if keypoint_scheme == "body":
                    # Body 模式：直接追加和绘制
                    all_keypoints.append(keypoints_original)
                    all_scores.append(scores)
                    pose_canvas = draw_body17_keypoints_openpose_style(pose_canvas, keypoints_original, scores, threshold=score_threshold)
                
                else: # wholebody
                    # Wholebody 模式：应用 Gradio 的 133 -> 134 点转换逻辑
                    keypoints_with_neck = keypoints_original.copy()
                    scores_with_neck = scores.copy()
                    
                    if len(keypoints_original) >= 17:
                        neck = (keypoints_original[5] + keypoints_original[6]) / 2
                        neck_score = min(scores[5], scores[6]) if scores[5] > 0.3 and scores[6] > 0.3 else 0
                        
                        # 插入 neck 点，将 133 数组变为 134
                        keypoints_with_neck = np.insert(keypoints_original, 17, neck, axis=0)
                        scores_with_neck = np.insert(scores, 17, neck_score)
                        
                        # 重新映射身体关节
                        mmpose_idx = np.array([17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3])
                        openpose_idx = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17])
                        
                        temp_kpts = keypoints_with_neck.copy()
                        temp_scores = scores_with_neck.copy()
                        temp_kpts[openpose_idx] = keypoints_with_neck[mmpose_idx]
                        temp_scores[openpose_idx] = scores_with_neck[mmpose_idx]
                        
                        keypoints_with_neck = temp_kpts
                        scores_with_neck = temp_scores
                    
                    # 将处理后的 134 点数组用于 JSON 和绘图
                    all_keypoints.append(keypoints_with_neck)
                    all_scores.append(scores_with_neck)
                    pose_canvas = draw_wholebody_keypoints_openpose_style(
                        pose_canvas, keypoints_with_neck, scores_with_neck, 
                        threshold=score_threshold
                    )

            # --- Finalize current frame output ---
            result_image = cv2.addWeighted(original_image_rgb, 1.0 - overlay_alpha, pose_canvas, overlay_alpha, 0)
            result_images.append(result_image)
            
            image_width = original_image_rgb.shape[1]
            image_height = original_image_rgb.shape[0]

            # 1. 生成当前帧的标准 OpenPose JSON
            frame_json_data = convert_to_openpose_json(all_keypoints, all_scores, image_width, image_height, keypoint_scheme)
            all_frames_json_data.append(frame_json_data)
            
            # 2. (可选) 生成并保存用于编辑器的 JSON 文件
            if save_for_editor:
                # 生成编辑器格式的 Python 字典
                data_to_save = convert_to_loader_json(all_keypoints, all_scores, image_width, image_height, keypoint_scheme, score_threshold)
                
                # --- 借鉴 SavePoseToJson 的文件保存逻辑 ---
                try:
                    output_dir = folder_paths.get_output_directory()
                    full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix_edit, output_dir, image_width, image_height)
                    
                    counter = frame_idx + 1
                    try:
                        # 检查已存在的文件
                        existing_files = [f for f in os.listdir(full_output_folder) if f.startswith(filename + "_") and f.endswith(".json")]
                        if existing_files:
                            max_counter = 0
                            for f in existing_files:
                                try:
                                    num_str = f[len(filename)+1:-5] # 从 "filename_00001.json" 中提取 "00001"
                                    num = int(num_str)
                                    if num > max_counter:
                                        max_counter = num
                                except ValueError:
                                    continue # 忽略不匹配的文件
                            counter = max_counter + frame_idx + 1
                    except FileNotFoundError:
                        os.makedirs(full_output_folder, exist_ok=True) # 如果 "poses" 目录不存在，则创建它
                    
                    final_filename = f"{filename}_{counter:05d}.json"
                    file_path = os.path.join(full_output_folder, final_filename)

                    with open(file_path, 'w') as f:
                        json.dump(data_to_save, f, indent=4)
                    
                    if frame_idx == 0:  # Only print once for the first frame
                        print(f"SDPose Node: Saving editor-compatible poses to {full_output_folder}")

                except Exception as e:
                    if frame_idx == 0:  # Only print once for the first frame
                        print(f"SDPose Node: ERROR Failed to save editor-compatible pose JSON: {e}")
            
            # Update progress
            progress += 1
            if comfy_pbar is not None:
                comfy_pbar.update_absolute(progress)

        # --- Combine all frames ---
        result_images_np = np.stack(result_images, axis=0)  # Shape: (B, H, W, C)
        
        # Convert to tensor format expected by ComfyUI (B, H, W, C) -> (B, H, W, C) float32 0-1
        result_tensor = torch.from_numpy(result_images_np.astype(np.float32) / 255.0)
        
        # For JSON output, we'll return the data for all frames
        combined_json_data = {
            "frames": all_frames_json_data,
            "total_frames": B
        }
        
        # --- Unload models if requested ---
        if sdpose_model.get("unload_on_finish", False):
            print("SDPose Node: Unloading models from VRAM to CPU.")
            offload_device = torch.device("cpu")
            for key in ["unet", "vae", "decoder"]:
                if key in sdpose_model and hasattr(sdpose_model[key], 'to'):
                    sdpose_model[key].to(offload_device)
            model_management.soft_empty_cache()

        return (result_tensor, json.dumps(combined_json_data, indent=2))

# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "SDPoseOODLoader": SDPoseOODLoader,
    "SDPoseOODProcessor": SDPoseOODProcessor,
    "YOLOModelLoader": YOLOModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDPoseOODLoader": "Load SDPose Model",
    "SDPoseOODProcessor": "Run SDPose Estimation",
    "YOLOModelLoader": "Load YOLO Model",
}
