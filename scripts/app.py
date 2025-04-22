import os
import torch # Keep for device checks, maybe ctransformers uses it?
import gradio as gr
# from threading import Thread # No longer needed for ctransformers streaming
# Keep transformers for tokenizer, remove model/peft/bnb imports
from transformers import AutoTokenizer #, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel
import psutil # To display RAM usage
from ctransformers import AutoModelForCausalLM # Import from ctransformers

# --- Configuration ---
# Path to your CONVERTED GGUF model file (relative to this script's location)
# Assumes model is in 'models/' directory one level up.
GGUF_MODEL_PATH = "../models/mistral-7b-jn-Q4_K_M.gguf"

# Tokenizer still needed, usually from the original base model
TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# CTransformers configuration
N_GPU_LAYERS = 0 # Number of layers to offload to MPS. Set to 0 for CPU only.
                 # Set to 0 for CPU only. Increase cautiously if you have more RAM.
N_CTX = 2048 # Context length for the model. Adjust if your GGUF model uses a different size.

# Check if MPS is available
# if torch.backends.mps.is_available():
#     DEVICE = "mps"
#     print("Using MPS (Apple Silicon GPU) via ctransformers.")
# else:
DEVICE = "cpu" # Force CPU
print("MPS not available or disabled. Using CPU via ctransformers.")
N_GPU_LAYERS = 0 # Force CPU if MPS not found or explicitly set

# --- Model Loading (ctransformers) ---
print(f"Loading GGUF model from: {GGUF_MODEL_PATH}")

if not os.path.exists(GGUF_MODEL_PATH):
    print(f"Error: GGUF model file not found at {GGUF_MODEL_PATH}")
    print("Please ensure you have converted your fine-tuned model to GGUF format")
    print("and placed it in the correct directory, then update GGUF_MODEL_PATH in app.py.")
    exit(1)

try:
    # Load the GGUF model using ctransformers
    model = AutoModelForCausalLM.from_pretrained(
        GGUF_MODEL_PATH,
        model_type="mistral", # Specify model type
        gpu_layers=N_GPU_LAYERS,
        context_length=N_CTX,
        # local_files_only=False, # Set to True if you don't want it checking Hub
    )
    print(f"GGUF model loaded. Offloading {N_GPU_LAYERS} layers to MPS.")
except Exception as e:
    print(f"Fatal Error: Could not load GGUF model: {e}")
    print("Check the GGUF file path, model_type, and ctransformers installation.")
    exit(1)

# --- Tokenizer Loading (transformers) ---
print(f"Loading tokenizer: {TOKENIZER_NAME}")
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Set pad token if needed
    print("Tokenizer loaded.")
except Exception as e:
    print(f"Fatal Error: Could not load tokenizer: {e}")
    exit(1)

# No adapter loading needed for GGUF

# --- RAM Usage ---
def get_ram_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return f"RAM Used: {mem_info.rss / (1024**3):.2f} GB"

# --- Gradio Interface ---
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

def chat_stream(message: str, history: list, system_prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, repetition_penalty: float = 1.1):
    """Generates a response using the loaded CTransformers GGUF model (non-streaming)."""
    if not system_prompt:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # Format prompt for Mistral Instruct GGUF - may need slight adjustments
    # based on how the GGUF was created, but standard format is usually preserved.
    formatted_history = ""
    for user_msg, model_answer in history:
         # Append user message and assistant response correctly formatted
         formatted_history += f"<s>[INST] {user_msg} [/INST] {model_answer}</s>" # Ensure EOS token is present

    # Construct the final prompt including the system prompt and current message
    # Note: Some GGUF implementations might handle system prompts differently.
    # This assumes the standard Mistral Instruct format works.
    prompt = f"{formatted_history}<s>[INST] {system_prompt}\\n{message} [/INST]"

    print(f"\\n--- Prompt ---\\n{prompt}\\n-----------\\n")

    # Use the model's __call__ method for generation (non-streaming)
    try:
        # Call the model directly
        response = model(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["</s>"], # Stop generation at the end-of-sequence token
            repetition_penalty=repetition_penalty, # Add repetition penalty
            min_length=32, # Add minimum length constraint
            # stream=False # Default is usually False
        )
        print(f"\\n--- Response ---\\n{response}\\n-----------\\n")
        yield response # Yield the complete response at once
    except Exception as e:
        print(f"Error during CTransformers model generation: {e}")
        yield "[Error during generation]"


# Build Gradio UI
with gr.Blocks(theme=gr.themes.Base()) as demo:
    ram_usage_display = gr.Textbox(label="Resource Usage", value=get_ram_usage, interactive=False, every=5) # Update every 5s

    gr.Markdown(
        """
        # Mistral-7B Instruct (Fine-tuned GGUF)
        Chat with the model running via CTransformers.
        *Optimized for limited RAM (like M1 Mac).*
        """
    )
    chatbot = gr.ChatInterface(
        fn=chat_stream,
        additional_inputs=[
            gr.Textbox(label="System Prompt", value=DEFAULT_SYSTEM_PROMPT),
            gr.Slider(minimum=32, maximum=1024, step=32, value=512, label="Max New Tokens"),
            gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperature"),
            gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.9, label="Top-P"),
            gr.Slider(minimum=1.0, maximum=1.5, step=0.05, value=1.1, label="Repetition Penalty")
        ],
        title="Mistral-7B GGUF Chat (ctransformers)",
        description="Enter your message and interact with the fine-tuned GGUF model.",
        examples=[["Explain the concept of progressive overload."], ["Summarize the key points about nutrient timing."]],
        # undo_btn="Delete Previous Turn", # Removed unsupported argument
        # clear_btn="Clear Chat", # Removed unsupported argument
    )

if __name__ == "__main__":
    print("Launching Gradio interface with CTransformers GGUF model...")
    # Check model path again before launch
    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"FATAL: GGUF model not found at {GGUF_MODEL_PATH} before launch.")
    else:
        demo.queue().launch(share=False) # Set share=True for public link (use with caution)
