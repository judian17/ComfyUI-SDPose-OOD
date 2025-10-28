[[中文说明]](README-ZH.md)



# SDPose-OOD for ComfyUI



## Introduction



This project is a ComfyUI custom node implementation of [SDPose-OOD](https://github.com/T-S-Liang/SDPose-OOD).



It brings the powerful and robust pose estimation capabilities of SDPose-OOD into the ComfyUI workflow, allowing for the extraction of high-quality body or whole-body poses from images.



## Features



* **Two Model Types**:

&nbsp;   * **Body**: 17 keypoints, body-only pose.

&nbsp;   * **WholeBody**: 133 keypoints, including body, face, and hands.

* **Automatic Model Download**: Nodes will automatically download the required models on first use.

* **YOLO Integration**: Optional YOLO-based person detection for precise multi-person pose estimation.

* **Advanced BBox Detection**: In addition to YOLO, now supports precise person detection using **ComfyUI-Florence2** (https://github.com/kijai/ComfyUI-Florence2) or **GroundingDINO**(from ComfyUI-SAM2: ) (https://github.com/neverbiasu/ComfyUI-SAM2).

* **Keypoint Filtering**: Options to keep or remove keypoints for the face, hands, and feet to suit different needs.

* **Precision Control**: Supports `bf16`, `fp16`, and `fp32` inference to balance speed and VRAM usage.

* **OpenPose Editor Compatibility**: One-click save poses as JSON files for import and editing in the [ComfyUI-OpenPose-Editor](https://github.com/judian17/ComfyUI-OpenPose-Editor-jd).



## Installation



1.  **Install this node**:

&nbsp;   * Clone this repository into your `ComfyUI/custom_nodes/` directory using `git clone`.

&nbsp;   * `cd ComfyUI/custom_nodes/SDPose-OOD-ComfyUI`

&nbsp;   * `pip install -r requirements.txt`



2.  **Install Models (Automatic or Manual)**:

&nbsp;   * **Automatic (Recommended)**: The nodes (`Load SDPose Model`, `Load YOLO Model`) will automatically download models from Hugging Face and Github on first run and place them in the correct directories.

&nbsp;   * **Manual**: You can also download the models yourself and place them in the corresponding folders:

&nbsp;       * **SDPose Models**: Download and place in `ComfyUI/models/SDPose_OOD/`

&nbsp;           * SDPose-Body: [huggingface.co/teemosliang/SDPose-Body](https://huggingface.co/teemosliang/SDPose-Body)

&nbsp;           * SDPose-Wholebody: [huggingface.co/teemosliang/SDPose-Wholebody](https://huggingface.co/teemosliang/SDPose-Wholebody)

&nbsp;       * **YOLO Models**: Download and place in `ComfyUI/models/yolo/`

&nbsp;           * YOLOv8: [yolov8n-pose.pt](https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n-pose.pt)

&nbsp;           * YOLOv11: [yolo11x.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) (or other YOLO models)

&nbsp;           * Default YOLO models (like `yolov8n-pose.pt`) are better suited for detecting **real people**.

&nbsp;           * SDPose-OOD itself handles anime characters well (even without YOLO).

&nbsp;           * If you don't want to use Florence2/GroundingDINO, you can consider this detection model which performs excellently for both real and anime characters: [real_person_detection](https://huggingface.co/deepghs/real_person_detection). Download your preferred model version (.pt file) from the repo and place it in the `models/yolo` folder.

&nbsp;       * **groundingdino model**: Download the model file and py file for either of the two models listed below, and place them in the modelsgrounding-dino directory. Usually, the model will be downloaded automatically.

&nbsp;           * groundingdino_swint_ogc: [groundingdino_swint_ogc](https://huggingface.co/ShilongLiu/GroundingDINO/blob/main/groundingdino_swint_ogc.pth) and [GroundingDINO_SwinT_OGC.cfg.py](https://huggingface.co/ShilongLiu/GroundingDINO/blob/main/GroundingDINO_SwinT_OGC.cfg.py) 

&nbsp;           * ggroundingdino_swinb_cogcoor: [groundingdino_swinb_cogcoor](https://huggingface.co/ShilongLiu/GroundingDINO/blob/main/groundingdino_swinb_cogcoor.pth) and [GroundingDINO_SwinB.cfg](https://huggingface.co/ShilongLiu/GroundingDINO/blob/main/GroundingDINO_SwinB.cfg.py) 



3.  **Install Optional Dependencies (for Advanced BBox Detection)**:

&nbsp;   * For Florence2, install [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2).

&nbsp;   * For GroundingDINO, install [ComfyUI-SAM2](https://github.com/neverbiasu/ComfyUI-SAM2) (code is sourced from here).



## Node Descriptions



### 1. `Load SDPose Model`



* **`model_type`**: Choose `Body` (17 keypoints) or `WholeBody` (133 keypoints).

* **`unet_precision`**: Precision selection.

&nbsp;   * `bf16`: Recommended for 30-series GPUs and newer.

&nbsp;   * `fp16`: Recommended for 20-series GPUs and older.

&nbsp;   * `fp32`: For CPU inference or when maximum precision is required.

&nbsp;   * *Note: `bf16`/`fp16` significantly reduce VRAM usage with almost identical speed to `fp32`.*

* **`device`**: `auto` will detect CUDA. CPU inference only supports `fp32`.



### 2. `Load YOLO Model` (Optional)



* Loads a YOLO model (e.g., `yolov8n-pose.pt`).

* Connect the `YOLO_MODEL` output of this node to the `Run SDPose Estimation` node.



### 3. `Run SDPose Estimation`



* **`sdpose_model`**: The model from `Load SDPose Model`.

* **`image`**: Your input image.

* **`yolo_model` (Optional)**:

&nbsp;   * **Not connected (and `bbox_input` also not connected)**: The node will process the entire image. Suitable for single-person images.

&nbsp;   * **Connected**: The node will first use YOLO to detect all persons in the image, then process each detected person individually.

* **`bbox_input` (Optional)**: New input. Can be connected to the BBox output from **ComfyUI-Florence2** or **GroundingDINO** (ComfyUI-SAM2).

&nbsp;   * *Note: **Detection Priority***: If `yolo_model` and (`grounding_dino_model` or `data_from_florence2`) are connected simultaneously, the node will prioritize `data_from_florence2` first, followed by GroundingDINO. (This implies `yolo_model` is the lowest priority).

* **`score_threshold`**: Keypoint confidence threshold. Points below this score will be ignored.

* **`keep_face` (bool)**: `True` to keep face keypoints (only effective for `WholeBody`).

* **`keep_hands` (bool)**: `True` to keep hand keypoints (only effective for `WholeBody`).

* **`keep_feet` (bool)**: `True` to keep feet keypoints.

* **`overlay_alpha`**: Controls the visibility of the output image.

&nbsp;   * `0.0`: 100% original image.

&nbsp;   * `0.6`: Blend of original image and pose map (original image at 60% brightness).

&nbsp;   * `1.0`: 100% pose map (pure black background).

* **`save_for_editor` (bool)**:

&nbsp;   * **`True`**: Saves a JSON file to the `ComfyUI/output/poses/` directory with the prefix `pose_edit`.

&nbsp;   * This JSON file can be loaded by the [ComfyUI-OpenPose-Editor](https://github.com/judian17/ComfyUI-OpenPose-Editor-jd) node for further editing.

* **`image` (Output)**: The final visualized image (based on `overlay_alpha` setting).

* **`pose_json` (Output)**:

&nbsp;   * **Important**: This is a **string** output containing the *original* JSON format from the SDPose-OOD project.

&nbsp;   * This format is **different** from the JSON saved by `save_for_editor` and cannot be used with the OpenPose editor node.



## Workflow Example



[workflow](./workflow/sdpose_ood.json)



![Example](./example.png)



1.  `Load Image` to load your source image.

2.  `Load SDPose Model` and select `WholeBody` and `bf16`.

3.  `Load YOLO Model` and select `yolov8n-pose.pt`. (Or use Florence2 / GroundingDINO and connect to the appropriate input).

4.  Connect all three (or replace #3 with a BBox node) to `Run SDPose Estimation`.

5.  Set `overlay_alpha` to `1.0` (if you want a pure pose map).

6.  Set `save_for_editor` to `True` (if you want to edit it later).

7.  Connect a `Preview Image` node to the `image` output.

