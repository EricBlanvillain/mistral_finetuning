# merge_adapter.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import shutil

# --- Configuration ---
base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# *** CONFIRM this is the adapter path from your BEST run ***
# Path relative to the script's location (scripts/)
adapter_path = "../models/mistral-qlora-adapter" # Adjust if your adapter dir name is different
# *** This directory will be created inside models/ ***
merged_model_path = "../models/merged_mistral_adapter"

# --- Ensure paths exist ---
# We need to check the path relative to the script's execution directory (project root)
# when running `python scripts/merge_adapter.py`
# However, os.path.isdir() inside the script will resolve relative to the script itself.
# Let's adjust the check slightly for clarity or rely on downstream errors.
# A better approach might use absolute paths or pass paths as arguments.
# For now, let's assume the script is run from the project root and check relative to that.
# This check might be less reliable now. Let's simplify the check logic slightly
# and rely more on the loading process to fail if the path is wrong.

# Let's keep the original check logic but be aware it might behave differently
# depending on *how* the script is run vs where paths are interpreted.
if not os.path.isdir(adapter_path):
    # Construct the path expected when running from project root for the error message
    expected_path_from_root = os.path.join(os.path.dirname(__file__), adapter_path)
    # It's better to check the absolute path derived from the script location
    absolute_adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), adapter_path))
    print(f"Checking for adapter at resolved path: {absolute_adapter_path}")
    if not os.path.isdir(absolute_adapter_path):
        print(f"Error: Adapter path not found at '{absolute_adapter_path}'")
        print(f"(Derived from relative path '{adapter_path}' in script {__file__})")
        print("Please ensure the adapter files are in the correct directory relative to the script.")
        exit(1)
    else:
        # If the absolute path IS found, update adapter_path to be absolute for loading
        adapter_path = absolute_adapter_path

# Ensure the output directory exists (using the relative path is fine here)
# os.makedirs will create it relative to the script's location if it doesn't exist.
# Let's resolve the output path to be absolute as well for consistency
absolute_merged_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), merged_model_path))
merged_model_path = absolute_merged_model_path # Use absolute path for saving

print(f"Loading base model: {base_model_name}")
# Load base model - use appropriate dtype and device map for Colab GPU
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16, # Should work on A100/T4
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loading adapter from: {adapter_path}") # Log the path being used
# Load the LoRA adapter onto the base model
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging adapter...")
# Merge the adapter weights into the base model
model = model.merge_and_unload()
print("Merge complete.")

print(f"Saving merged model to: {merged_model_path}") # Log the path being used
# Ensure target directory exists using the now absolute path
os.makedirs(merged_model_path, exist_ok=True)
# Save the merged model
model.save_pretrained(merged_model_path)

print("Loading tokenizer...")
# Load and save the tokenizer associated with the base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.save_pretrained(merged_model_path)

print("Merged model and tokenizer saved successfully.")

# Optional: Clean up memory if needed in Colab
# import gc
# del model
# del base_model
# gc.collect()
# if torch.cuda.is_available(): torch.cuda.empty_cache()
