#!/usr/bin/env python
import os
import yaml
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
    print("Error: YouTube API Key not found or not set in .env file.")
    print("Please add your key to mistral_finetuning/.env")
    exit(1)

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Jeff Nippard Channel ID (verify if needed)
CHANNEL_ID = "UC68TLK0mAEzUyHx5x5k-S1Q"
OUTPUT_YAML = "videos.yaml"

# --- Main Logic ---
def get_channel_videos(api_key, channel_id):
    """Fetches all video IDs from a YouTube channel."""
    try:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=api_key)
    except Exception as e:
        print(f"Error building YouTube API client: {e}")
        return None

    video_ids = []
    next_page_token = None

    print(f"Fetching video uploads for channel ID: {channel_id}...")

    while True:
        try:
            # 1. Get channel's uploads playlist ID
            # This step is often needed, but sometimes directly searching works.
            # Let's try getting the playlist ID first for robustness.
            channel_response = youtube.channels().list(
                part="contentDetails",
                id=channel_id
            ).execute()

            if not channel_response.get("items"):
                 print(f"Error: Could not find channel details for ID {channel_id}.")
                 return None

            uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

            # 2. Get videos from the uploads playlist
            playlist_request = youtube.playlistItems().list(
                playlistId=uploads_playlist_id,
                part="contentDetails", # We only need videoId from contentDetails
                maxResults=50, # Max allowed by API
                pageToken=next_page_token
            )
            playlist_response = playlist_request.execute()

            # Extract video IDs
            for item in playlist_response.get("items", []):
                video_id = item.get("contentDetails", {}).get("videoId")
                if video_id:
                    video_ids.append(video_id)

            # Check for next page
            next_page_token = playlist_response.get("nextPageToken")
            print(f"Fetched {len(video_ids)} videos so far...")

            if not next_page_token:
                break # No more pages

        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
            # Handle specific errors like quota exceeded if necessary
            if e.resp.status == 403: # Often quota related
                print("Quota exceeded? Check your YouTube Data API v3 quota.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    print(f"Finished fetching. Found {len(video_ids)} total videos.")
    return video_ids

def save_videos_to_yaml(video_ids, filename):
    """Saves the list of video IDs to a YAML file."""
    data = {"videos": video_ids}
    try:
        with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"Successfully saved video IDs to {filename}")
    except Exception as e:
        print(f"Error saving video IDs to {filename}: {e}")

# --- Execution ---
if __name__ == "__main__":
    print("Starting script to fetch channel video IDs...")
    fetched_video_ids = get_channel_videos(API_KEY, CHANNEL_ID)

    if fetched_video_ids is not None:
        save_videos_to_yaml(fetched_video_ids, OUTPUT_YAML)
    else:
        print("Failed to fetch video IDs. No changes made to videos.yaml")

    print("Script finished.")
