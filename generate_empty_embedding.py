import torch
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import save_file
import os
from pathlib import Path
# No longer need folder_paths for this script

# --- Configuration ---
# We only need to do this once, so we can hardcode the repo ID.
# The empty embedding is identical for both Body and WholeBody models.
REPO_ID = "teemosliang/SDPose-Body"
# Get the directory of the current script and build the path relative to it
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "empty_text_encoder")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "empty_embedding.safetensors")

def generate_and_save_empty_embedding():
    """
    Loads a CLIP text model, generates an embedding for an empty prompt,
    and saves it to a safetensors file.
    """
    print("--- SDPose Empty Embedding Generator ---")

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory prepared: {OUTPUT_DIR}")

    # Check if the file already exists
    if os.path.exists(OUTPUT_FILE):
        print(f"'{OUTPUT_FILE}' already exists. Skipping generation.")
        return

    try:
        # 1. Load tokenizer and text encoder from Hugging Face
        print(f"Loading CLIP model from '{REPO_ID}'...")
        tokenizer = CLIPTokenizer.from_pretrained(REPO_ID, subfolder='tokenizer')
        text_encoder = CLIPTextModel.from_pretrained(REPO_ID, subfolder='text_encoder')
        print("Model loaded successfully.")

        # 2. Generate the empty text embedding
        print("Generating empty text embedding...")
        prompt = ""
        text_inputs = tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        empty_text_embed = text_encoder(text_input_ids)[0]
        print(f"Embedding generated with shape: {empty_text_embed.shape}")

        # 3. Save the embedding to a safetensors file
        print(f"Saving embedding to '{OUTPUT_FILE}'...")
        tensor_dict = {"empty_text_embed": empty_text_embed}
        save_file(tensor_dict, OUTPUT_FILE)
        print("--- Success! ---")
        print("Empty embedding has been saved. You no longer need to run this script.")

    except Exception as e:
        print(f"\n--- An error occurred ---")
        print(f"Error: {e}")
        print("Please ensure you have an internet connection and the 'transformers' library is installed.")

if __name__ == "__main__":
    generate_and_save_empty_embedding()
