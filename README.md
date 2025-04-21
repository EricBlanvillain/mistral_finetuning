# Mistral-7B YouTube Transcript Finetuning Pipeline

This project provides an end-to-end pipeline to:
1.  Fetch transcripts from YouTube videos (`scripts/fetch_channel_videos.py`, `scripts/data_gen.py`).
2.  Process transcripts into an instruction-following dataset (`scripts/data_gen.py`).
3.  Fine-tune `mistralai/Mistral-7B-Instruct-v0.3` using QLoRA on Google Colab (`notebooks/mistral_qlora_youtube.ipynb`).
4.  Merge the adapter with the base model (`scripts/merge_adapter.py`).
5.  Convert the merged model to GGUF format (using `llama.cpp`).
6.  Deploy the fine-tuned GGUF model locally using a Gradio web interface (`scripts/app.py`).

This pipeline is designed to run the fine-tuning on a free Colab GPU (T4/A10) and the final inference on a local machine (tested on MacBook Air M1 8GB RAM, though performance may be limited).

## Project Structure

```
.
├── data/                   # Stores raw/processed data (ignored by git)
│   ├── raw/
│   └── processed/
├── models/                 # Stores adapters, merged models, GGUF files (ignored by git)
│   ├── mistral-qlora-adapter/
│   ├── merged_mistral_adapter/
│   └── *.gguf
├── notebooks/              # Jupyter/Colab notebooks
│   ├── mistral_qlora_youtube.ipynb
│   └── colab_gradio_qlora.ipynb # Example Gradio app notebook for Colab
├── scripts/                # Python scripts
│   ├── fetch_channel_videos.py
│   ├── data_gen.py
│   ├── merge_adapter.py
│   └── app.py
├── venv/                   # Python virtual environment (ignored by git)
├── .env                    # Environment variables (ignored by git)
├── .gitignore              # Specifies intentionally untracked files
├── LICENSE                 # Project license (e.g., MIT)
├── README.md               # This file
├── requirements.txt        # Python dependencies for local scripts
└── videos.yaml             # Input list of YouTube video IDs/URLs
```

## Setup

**1. Clone the Repository:**

```bash
git clone <your-repo-url>
cd mistral_finetuning
```

**2. Create a Virtual Environment (Recommended):**

```bash
python3 -m venv venv  # Use python3 or specific version like python3.11
source venv/bin/activate # On macOS/Linux
# venv\\Scripts\\activate # On Windows
```

**3. Install Dependencies:**

*   **PyTorch:** Install PyTorch separately first if needed, following official instructions ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)). For M1/M2 Macs, ensure MPS support is installed if desired.
*   **Other Dependencies:** Install requirements for local scripts and `llama.cpp` conversion/inference:
    ```bash
    pip install -r requirements.txt
    ```
*   **`llama.cpp`:** Clone and build `llama.cpp` for GGUF conversion and potentially faster inference (instructions assume it's cloned *outside* this project directory, e.g., alongside it):
    ```bash
    # Navigate to where you want to clone llama.cpp (e.g., parent directory of mistral_finetuning)
    cd ..
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    # Build with Metal support for Mac (recommended)
    mkdir build
    cd build
    cmake .. -DLLAMA_METAL=ON
    cmake --build . --config Release
    cd ../.. # Return to the directory containing mistral_finetuning
    # Verify build by checking for executables like build/bin/llama-quantize
    ```

**4. YouTube Data API Key (Optional but Recommended):**

The `youtube-transcript-api` library often works without an API key for public videos. However, for reliability or to access private/caption data, you might need a Google Cloud API key with the YouTube Data API v3 enabled.

*   Follow Google's documentation to create an API key: [https://developers.google.com/youtube/v3/getting-started](https://developers.google.com/youtube/v3/getting-started)
*   **Note:** The current `scripts/fetch_channel_videos.py` *doesn't* explicitly use an API key, relying on the library's default behavior. If you encounter fetching issues, you might need to modify the script to use `google-api-python-client` and your key.

**5. Hugging Face Account & Token:**

You'll need a Hugging Face account to:
*   Download the base Mistral model.
*   (Optional) Push your fine-tuned adapter to the Hub.
    *   Create an account: [https://huggingface.co/join](https://huggingface.co/join)
    *   Create an access token (with `write` permissions if pushing): [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    *   You'll log in using this token within the Colab notebook.

## Workflow

**Step 1: Prepare Video List**

Edit `videos.yaml` and add the YouTube video IDs or full URLs you want to process. Optionally run `scripts/fetch_channel_videos.py` first if you have a channel ID and API key configured.

**Step 2: Generate Training Data**

Run the data generation script locally from the `mistral_finetuning` directory:

```bash
python scripts/data_gen.py
```

This will:
*   Fetch transcripts for videos in `videos.yaml`.
*   Save raw transcripts to `data/raw/`.
*   Chunk transcripts and create instruction pairs.
*   Save the final dataset as `data/processed/train.jsonl`.

**Step 3: Fine-tune on Google Colab**

1.  Open Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)
2.  Upload the `notebooks/mistral_qlora_youtube.ipynb` notebook.
3.  Upload the generated `data/processed/train.jsonl` file (or configure loading from Drive/HF).
4.  **Configure Notebook:** Set HF username, token, etc.
5.  **Run Notebook Cells:** Follow instructions within the notebook.
6.  **Download Adapter:** After training, download the adapter folder (e.g., `mistral-qlora-adapter`) and place it inside the local `models/` directory (e.g., `mistral_finetuning/models/mistral-qlora-adapter/`).

**Step 4: Merge Adapter (Colab or Local High-RAM Machine)**

1.  Ensure the base model (`mistralai/Mistral-7B-Instruct-v0.3`) is accessible (requires HF login).
2.  Ensure the downloaded adapter is in the correct location (e.g., `models/mistral-qlora-adapter/`).
3.  Run the merge script (adjust paths inside the script if needed):
    ```bash
    # If running locally (needs high RAM):
    python scripts/merge_adapter.py
    # Or run equivalent steps in a Colab notebook
    ```
4.  This script saves the merged model to a directory (e.g., `models/merged_mistral_adapter/`).

**Step 5: Convert Merged Model to GGUF (Local)**

1.  Make sure `llama.cpp` is built (Setup Step 3).
2.  Ensure the merged model directory (`models/merged_mistral_adapter/`) exists and contains all necessary files (weights `.safetensors`, `tokenizer.model`, configs).
3.  Run the conversion scripts from the parent directory containing `mistral_finetuning` and `llama.cpp`:
    ```bash
    # Step 5a: Convert HF to FP16 GGUF
    python llama.cpp/convert_hf_to_gguf.py mistral_finetuning/models/merged_mistral_adapter \
      --outfile mistral_finetuning/models/mistral-7b-jn-f16.gguf \
      --outtype f16

    # Step 5b: Quantize FP16 GGUF to Q4_K_M (or desired type)
    llama.cpp/build/bin/llama-quantize mistral_finetuning/models/mistral-7b-jn-f16.gguf \
      mistral_finetuning/models/mistral-7b-jn-Q4_K_M.gguf \
      q4_k_m
    ```
    *Note: Adjust paths if your `llama.cpp` location or model/output names differ.*

**Step 6: Deploy Locally with Gradio**

1.  Ensure the final GGUF file exists (e.g., `models/mistral-7b-jn-Q4_K_M.gguf`).
2.  Ensure the path in `scripts/app.py` points correctly to the GGUF file relative to the script (e.g., `../models/mistral-7b-jn-Q4_K_M.gguf`).
3.  Launch the app from the `mistral_finetuning` directory:
    ```bash
    python scripts/app.py
    ```
4.  **Access UI:** Open your web browser to the local URL provided (usually `http://127.0.0.1:7860`).

## Hardware Notes & Performance

*   **Fine-tuning:** Requires a GPU with sufficient VRAM (>=12GB recommended). Free Colab T4 (15GB) or A10 (24GB) GPUs should work.
*   **Local Inference:** The `scripts/app.py` uses `ctransformers` to load the GGUF file. Performance depends heavily on CPU/RAM and `N_GPU_LAYERS` setting in the script if using MPS. 8GB RAM is very limiting for a 7B model.

## Optional: Dockerization

A sample `Dockerfile` is provided to containerize the Gradio application. Note that building this image might be complex due to PyTorch and `bitsandbytes` dependencies.

```Dockerfile
# Use a base image with Python 3.11 and appropriate CUDA version if targeting GPU
# For CPU/MPS, a standard Python image is sufficient
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt requirements.txt

# Install PyTorch (CPU version here, adjust for CUDA if needed)
# Refer to PyTorch website for specific wheels/commands
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
# Note: bitsandbytes might fail or require specific build steps without CUDA
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model adapter
COPY scripts/app.py .
COPY mistral-qlora-adapter/ ./mistral-qlora-adapter/

EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
```

**Build and Run (CPU Example):**

```bash
docker build -t mistral-youtube-app .
docker run -p 7860:7860 -it mistral-youtube-app
```

*(Adjust Dockerfile and build process significantly if targeting GPU)*
