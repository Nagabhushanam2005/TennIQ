import os
import cv2
import numpy as np
import subprocess
import argparse
import sys
from typing import Dict, Any

def load_config(config_file_path: str) -> Dict[str, Any]:
    config = {}
    int_keys = ['LIMIT_NUM_FRAMES', 'FRAME_START', 'FRAME_END']

    try:
        with open(config_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key in int_keys:
                            try:
                                config[key] = int(value)
                            except ValueError:
                                config[key] = value
                        else:
                            config[key] = value

    except FileNotFoundError:
        print(f"Error: Config file '{config_file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)
    
    return config

def download_youtube_video(url, output_path):
    if os.path.exists(output_path):
        print(f"Video already exists: {output_path}")
        return
    print(f"Downloading video to {output_path} ...")
    try:
        subprocess.run([
            'yt-dlp',
            '-f', 'best',
            '-o', output_path,
            url
            ], check=True)
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: yt-dlp not found. Please install it with: pip install yt-dlp")
        sys.exit(1)

def extract_frames(video_path, frames_dir, frame_start, frame_end):
    if os.path.exists(frames_dir):
        print(f"Frames directory '{frames_dir}' exists. Deleting and recreating for fresh extraction.")
        try:
            subprocess.run(['rm', '-rf', frames_dir], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error deleting old frames directory: {e}")
            sys.exit(1)
    os.makedirs(frames_dir)

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    
    # Calculate total frames for setting END if it's -1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start = max(0, frame_start) if frame_start != -1 else 0
    end = frame_end if frame_end != -1 else total_frames - 1
    
    if start >= total_frames:
        print(f"Error: Start frame {start} is beyond total frames {total_frames}.")
        cap.release()
        return

    # Set video position to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_count = start
    
    print(f"Starting frame extraction (range: {start} to {end})...")

    while True:
        if frame_count > end and end != -1:
            print(f"End frame ({end}) reached. Stopping extraction.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

        if (frame_count - start) % 1000 == 0:
            print(f"Extracted {frame_count - start} frames...")

    cap.release()
    print(f"Total frames extracted: {frame_count - start}")

def main():
    parser = argparse.ArgumentParser(description='Download YouTube video and extract frames')
    parser.add_argument('--config', '-c', 
                       default='inference/data-configs/data_config_test.txt',
                       help='Path to configuration file (default: inference/data-configs/data_config_test.txt)')

    args = parser.parse_args()
    config_defaults = {
        'FRAME_START': 0,
        'FRAME_END': 3000
    }

    config = load_config(args.config)
    final_config = {**config_defaults, **config}

    youtube_url = final_config.get('YOUTUBE_URL')
    video_output_path = final_config.get('VIDEO_OUTPUT_PATH')
    frames_dir = final_config.get('FRAMES_DIR')
    frame_start = final_config.get('FRAME_START')
    frame_end = final_config.get('FRAME_END')
    
    # Handle legacy/unused LIMIT_NUM_FRAMES if present in final_config but not used in logic
    if 'LIMIT_NUM_FRAMES' in final_config and 'LIMIT_NUM_FRAMES' not in config_defaults:
        # If the user provides FRAME_END, it takes precedence.
        # Otherwise, if they only provide the old LIMIT_NUM_FRAMES, use it to calculate FRAME_END.
        if frame_end == 3000: # Use 3000 as the marker for default state
            frame_end = final_config['LIMIT_NUM_FRAMES']


    if not all([youtube_url, video_output_path, frames_dir]):
        print("Error: Missing required configuration parameters.")
        print("Required: YOUTUBE_URL, VIDEO_OUTPUT_PATH, FRAMES_DIR")
        sys.exit(1)
    print("-" * 100)
    print(f"Configuration loaded from: {args.config}")
    print(f"YouTube URL: {youtube_url}")
    print(f"Video output path: {video_output_path}")
    print(f"Frames directory: {frames_dir}")
    print(f"Frame Range: {frame_start} to {frame_end}")
    print("-" * 100)

    download_youtube_video(youtube_url, video_output_path)
    extract_frames(video_output_path, frames_dir, frame_start, frame_end)

if __name__ == "__main__":
    main()