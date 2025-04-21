import os
import re
import random
import json
import yaml
import time # Added for rate limiting
import math # Added for splitting
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator

# Third-party libraries
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
except ImportError:
    print("Please install youtube_transcript_api: pip install youtube-transcript-api")
    exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Please install transformers: pip install transformers")
    exit(1)

# Added for OpenAI API call
try:
    import openai
    from dotenv import load_dotenv
except ImportError:
    print("Please install openai and python-dotenv: pip install openai python-dotenv")
    exit(1)


# --- Configuration ---
VIDEO_LIST_PATH = Path("videos.yaml")
RAW_DATA_DIR = Path("./data/raw")
PROCESSED_DATA_DIR = Path("./data/processed")
OUTPUT_JSONL_PATH = PROCESSED_DATA_DIR / "train.jsonl"
MAX_CHUNK_TOKENS = 512 # Max tokens per chunk (adjust based on model context length)
TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.3" # For accurate token counting

# --- OpenAI API Setup ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
    print("Error: OpenAI API Key not found or not set in .env file.")
    print("Please add your key to mistral_finetuning/.env")
    exit(1)

# Use a recommended model like gpt-4-turbo or gpt-3.5-turbo if gpt-4.1-mini is unavailable
# Check OpenAI documentation for the latest suitable models
OPENAI_MODEL = os.getenv("OPENAI_GENERATION_MODEL", "gpt-4.1-mini") # Default to gpt-4-turbo
print(f"Using OpenAI model: {OPENAI_MODEL} for data generation.")

# Initialize OpenAI client (ensure openai library version >= 1.0.0)
try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
except TypeError: # Handle older openai library versions if needed, though >=1.0.0 is recommended
    openai.api_key = OPENAI_API_KEY
    client = openai # Use the module directly for older versions (less preferred)
    print("Warning: Using older OpenAI library structure. Recommend upgrading to >= 1.0.0.")


# Rate limiting parameters
MAX_RETRIES = 5
INITIAL_DELAY_SECONDS = 1


# --- Helper Functions ---

def get_video_id(video_input: str) -> Optional[str]:
    """Extracts YouTube video ID from URL or returns ID directly."""
    video_id = None
    # Regex to find YouTube video ID from various URL formats
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, video_input)
        if match:
            video_id = match.group(1)
            break
    if not video_id and re.match(r"^[0-9A-Za-z_-]{11}$", video_input):
        # Assume it's already a valid ID
        video_id = video_input
    return video_id

def fetch_transcript(video_id: str) -> Optional[List[Dict]]:
    """Fetches transcript using youtube-transcript-api."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try fetching manually created transcripts first, specifying language codes
        transcript = transcript_list.find_manually_created_transcript(['en']) # Look for English first
        print(f"Found manually created English transcript for {video_id}.")
        return transcript.fetch()
    except NoTranscriptFound:
        # If manual English not found, try auto-generated English
        try:
            transcript = transcript_list.find_generated_transcript(['en'])
            print(f"Found auto-generated English transcript for {video_id}.")
            return transcript.fetch()
        except NoTranscriptFound:
            print(f"No English transcript found for video {video_id}.")
            return None
    except TranscriptsDisabled:
        print(f"Transcripts are disabled for video {video_id}.")
        return None
    except Exception as e:
        print(f"An error occurred fetching transcript for {video_id}: {e}")
        return None

def chunk_transcript(transcript_data: List[Dict], tokenizer, max_tokens: int) -> Generator[str, None, None]:
    """Chunks transcript text based on max token count."""
    current_chunk = ""
    current_token_count = 0

    for item in transcript_data:
        # Directly access the .text attribute
        segment_text = item.text + " " # Add space between segments

        if not segment_text.strip():
            continue # Skip empty segments

        segment_tokens = tokenizer.encode(segment_text, add_special_tokens=False)
        segment_token_count = len(segment_tokens)

        if current_token_count + segment_token_count <= max_tokens:
            current_chunk += segment_text
            current_token_count += segment_token_count
        else:
            # Yield the current chunk if it's not empty
            if current_chunk.strip():
                yield current_chunk.strip()

            # Start a new chunk, handling segments longer than max_tokens
            if segment_token_count > max_tokens:
                # Split the long segment itself (simplified)
                # A more robust approach might use overlapping windows or sentence splitting
                words = segment_text.split()
                temp_chunk = ""
                temp_count = 0
                for word in words:
                    word_tokens = tokenizer.encode(word + " ", add_special_tokens=False)
                    word_token_count = len(word_tokens)
                    if temp_count + word_token_count <= max_tokens:
                        temp_chunk += word + " "
                        temp_count += word_token_count
                    else:
                        if temp_chunk.strip(): yield temp_chunk.strip()
                        temp_chunk = word + " "
                        temp_count = word_token_count
                if temp_chunk.strip(): yield temp_chunk.strip()

                current_chunk = "" # Reset after handling long segment
                current_token_count = 0
            else:
                # Start the new chunk with the current segment
                current_chunk = segment_text
                current_token_count = segment_token_count

    # Yield the last remaining chunk
    if current_chunk.strip():
        yield current_chunk.strip()

def generate_instruction_pair_with_llm(chunk: str) -> Optional[Dict[str, str]]:
    """Generates an instruction and answer pair using OpenAI API based on the chunk."""
    delay = INITIAL_DELAY_SECONDS
    prompt_template = f"""
You are an expert data creator for instruction fine-tuning LLMs, specializing in fitness and exercise science content like that from Jeff Nippard.
Based *only* on the following video transcript segment, please generate:
1. A concise and relevant instruction that asks a question or requests an action related to the segment's content. The instruction should be answerable *solely* from the provided text. Examples: "Summarize the key point about X.", "Explain the concept of Y mentioned here.", "What does this segment say about Z?".
2. A concise answer to that instruction, derived *strictly* from the information within the segment. Do not add any external knowledge, opinions, or information not present in the text. The answer should directly address the instruction.

Transcript Segment:
---
{chunk}
---

Output the result ONLY as a valid JSON object with two keys: "instruction" and "answer". Do not include any other text before or after the JSON object.

Example Input Segment:
"So, when we talk about progressive overload, the key is consistency. You need to gradually increase the demands on your muscles over time. This could be more weight, more reps, or more sets. But don't jump too quickly, or you risk injury."

Example Output JSON:
{{
  "instruction": "Based on the segment, what is the key to progressive overload and how is it achieved?",
  "answer": "The key to progressive overload is consistency in gradually increasing demands on muscles over time, which can be done by adding more weight, reps, or sets, while avoiding rapid increases to prevent injury."
}}

Now, generate the JSON for the provided transcript segment.
"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert data creator for instruction fine-tuning LLMs, specializing in fitness and exercise science content."},
                    {"role": "user", "content": prompt_template}
                ],
                temperature=0.5, # Adjust temperature for creativity vs. faithfulness
                max_tokens=256,  # Increased token limit
                response_format={ "type": "json_object" }, # Request JSON output directly if model supports it
            )

            # Extract JSON content
            content = response.choices[0].message.content
            if not content:
                print(f"Warning: Received empty content from OpenAI API for chunk: {chunk[:50]}...")
                return None

            # Parse the JSON response
            try:
                qa_pair = json.loads(content)
                if "instruction" in qa_pair and "answer" in qa_pair:
                     # Basic validation
                    if not qa_pair["instruction"] or not qa_pair["answer"]:
                        print(f"Warning: Generated pair has empty instruction or answer. Skipping.")
                        return None
                    # Return the final structure including the original chunk as input
                    return {
                        "instruction": qa_pair["instruction"].strip(),
                        "input": chunk,
                        "output": qa_pair["answer"].strip()
                    }
                else:
                    print(f"Warning: Received malformed JSON (missing keys) from OpenAI API: {content}")
                    return None # Skip this chunk if JSON is malformed
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode JSON from OpenAI API response: {content}")
                # Attempt to extract JSON from potential markdown code fences
                match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if match:
                    try:
                        qa_pair = json.loads(match.group(1))
                        if "instruction" in qa_pair and "answer" in qa_pair:
                            if not qa_pair["instruction"] or not qa_pair["answer"]:
                                print(f"Warning: Generated pair has empty instruction or answer (after extraction). Skipping.")
                                return None
                            return {
                                "instruction": qa_pair["instruction"].strip(),
                                "input": chunk,
                                "output": qa_pair["answer"].strip()
                            }
                        else:
                            print(f"Warning: Extracted JSON missing keys: {match.group(1)}")
                            return None
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to decode extracted JSON: {match.group(1)}")
                        return None
                else:
                    print("Warning: No valid JSON found in response.")
                    return None # Skip if JSON cannot be parsed

        except openai.RateLimitError:
            print(f"Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2 # Exponential backoff
        except openai.APIError as e:
            print(f"OpenAI API error: {e}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"An unexpected error occurred during OpenAI API call: {e}")
            return None # Skip chunk on unexpected errors

    print(f"Failed to generate instruction pair for chunk after {MAX_RETRIES} retries: {chunk[:50]}...")
    return None # Return None if all retries fail

# --- Main Execution ---

def main():
    print("Starting data generation process (using OpenAI)...")

    # --- 0. Setup ---
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_LIST_PATH.exists():
        print(f"Error: Video list file not found at {VIDEO_LIST_PATH}")
        print("Please create videos.yaml with a list of YouTube video IDs or URLs.")
        return

    print(f"Loading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    # --- 1. Load Video List ---
    print(f"Loading video list from {VIDEO_LIST_PATH}...")
    with open(VIDEO_LIST_PATH, 'r') as f:
        try:
            video_config = yaml.safe_load(f)
            video_inputs = video_config.get('videos', [])
            if not video_inputs:
                print("Warning: No videos found in the YAML file.")
                return
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return

    # --- 2. Process Each Video ---
    all_instruction_pairs = []
    total_videos = len(video_inputs)
    processed_videos = 0
    skipped_chunks = 0
    generated_pairs = 0
    print(f"Found {total_videos} video entries to process.")

    for i, video_input in enumerate(video_inputs):
        print(f"\n--- Processing video {i+1}/{total_videos}: {video_input} ---")
        video_id = get_video_id(video_input)

        if not video_id:
            print(f"Could not extract video ID from input: {video_input}. Skipping.")
            continue

        # --- 2a. Fetch Transcript ---
        # Caching logic removed for simplicity, re-fetching structured data
        print(f"Fetching transcript for {video_id}...")
        transcript_data = fetch_transcript(video_id)

        if not transcript_data:
             print(f"Could not get transcript for {video_id}. Skipping video.")
             continue # Skip to the next video

        # Save raw transcript text (using direct attribute access)
        raw_transcript_path = RAW_DATA_DIR / f"{video_id}.txt"
        try:
             full_text = " ".join([item.text for item in transcript_data]) # Direct access
             if not raw_transcript_path.exists(): # Only save if not already there
                 print(f"Saving raw transcript to {raw_transcript_path}...")
                 with open(raw_transcript_path, 'w', encoding='utf-8') as f:
                     f.write(full_text)
        except Exception as e:
             print(f"Warning: Could not process/save raw transcript for {video_id}: {e}")
             # Continue processing chunks even if raw saving fails

        # --- 2b. Chunk and Create Pairs using LLM ---
        print(f"Chunking transcript and generating instruction pairs for {video_id} (max {MAX_CHUNK_TOKENS} tokens)...")
        chunk_count_for_video = 0
        pairs_for_video = 0
        start_time_video = time.time()

        for chunk_num, chunk in enumerate(chunk_transcript(transcript_data, tokenizer, MAX_CHUNK_TOKENS)):
            chunk_count_for_video += 1
            print(f"  Processing chunk {chunk_num + 1}...")
            # Generate pair using LLM
            instruction_pair = generate_instruction_pair_with_llm(chunk)

            if instruction_pair:
                all_instruction_pairs.append(instruction_pair)
                generated_pairs += 1
                pairs_for_video += 1
            else:
                skipped_chunks += 1
                print(f"  Skipped chunk {chunk_num + 1} due to generation failure.")
            # Optional: Add a small delay between chunks if needed, though API call retries handle rate limits
            # time.sleep(0.1)

        end_time_video = time.time()
        print(f"Finished video {video_id}: Processed {chunk_count_for_video} chunks, generated {pairs_for_video} pairs in {end_time_video - start_time_video:.2f} seconds.")
        processed_videos += 1


    # --- 3. Save Processed Data ---
    print("\n--- Aggregation Summary ---")
    print(f"Processed {processed_videos}/{total_videos} videos.")
    print(f"Total instruction pairs generated: {generated_pairs}")
    print(f"Total chunks skipped due to errors: {skipped_chunks}")

    if not all_instruction_pairs:
        print("No instruction pairs were generated. Exiting.")
        return

    # --- 3a. Shuffle and Split Data (80/20) ---
    print(f"\nShuffling {len(all_instruction_pairs)} generated pairs...")
    random.shuffle(all_instruction_pairs)

    split_index = math.ceil(len(all_instruction_pairs) * 0.8) # 80% for training
    train_pairs = all_instruction_pairs[:split_index]
    test_pairs = all_instruction_pairs[split_index:]
    print(f"Splitting into {len(train_pairs)} training pairs and {len(test_pairs)} test pairs.")

    # --- 3b. Save Train Data ---
    train_output_path = PROCESSED_DATA_DIR / "train.jsonl"
    print(f"Saving training data to {train_output_path}...")
    saved_train_count = 0
    with open(train_output_path, 'w', encoding='utf-8') as f:
        for pair in train_pairs:
            try:
                json_line = json.dumps(pair, ensure_ascii=False)
                f.write(json_line + '\n')
                saved_train_count += 1
            except TypeError as e:
                print(f"Skipping training pair due to JSON serialization error: {e} - Pair: {pair}")
    print(f"Saved {saved_train_count} training pairs.")

    # --- 3c. Save Test Data ---
    test_output_path = PROCESSED_DATA_DIR / "test.jsonl"
    print(f"Saving test data to {test_output_path}...")
    saved_test_count = 0
    with open(test_output_path, 'w', encoding='utf-8') as f:
        for pair in test_pairs:
            try:
                json_line = json.dumps(pair, ensure_ascii=False)
                f.write(json_line + '\n')
                saved_test_count += 1
            except TypeError as e:
                print(f"Skipping test pair due to JSON serialization error: {e} - Pair: {pair}")
    print(f"Saved {saved_test_count} test pairs.")

    print("-" * 20)
    print("Data generation and splitting complete!")
    print(f"Raw transcripts saved in: {RAW_DATA_DIR}")
    print(f"Processed training data saved to: {train_output_path}")
    print(f"Processed test data saved to: {test_output_path}")
    print("-" * 20)

if __name__ == "__main__":
    main()
