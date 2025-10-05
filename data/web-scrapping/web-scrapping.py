import os
import cv2
import numpy as np
import subprocess
import argparse
import sys

def load_config(config_file_path):
    """Load configuration from a text file"""
    config = {}
    try:
        with open(config_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip().strip('"').strip("'")
    except FileNotFoundError:
        print(f"Error: Config file '{config_file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)
    
    return config

def download_youtube_video(url, output_path):
    """Download YouTube video using yt-dlp"""
    if os.path.exists(output_path):
        print(f"Video already exists: {output_path}")
        return
    
    print(f"Downloading video to {output_path} ...")
    try:
        # subprocess.run(['yt-dlp', '-U'], check=True)

        cmd = [
            'yt-dlp',
            '-f', 'bv*+ba/best',
            '--merge-output-format', 'mp4',
            '-o', output_path,
            url
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("Primary download failed, retrying with fallback...")
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

def extract_frames(video_path, frames_dir):
    """Extract frames from video"""
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        if frame_count % 5000 == 0:
            print(f"Extracted {frame_count} frames...")
    cap.release()
    print(f"Total frames extracted: {frame_count}")

def main():
    parser = argparse.ArgumentParser(description='Download YouTube video and extract frames')
    parser.add_argument('--config', '-c', 
                       default='inference/data-configs/data_config_alcaraz.txt',
                       help='Path to configuration file (default: inference/data-configs/data_config_alcaraz.txt)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    youtube_url = config.get('YOUTUBE_URL')
    video_output_path = config.get('VIDEO_OUTPUT_PATH')
    frames_dir = config.get('FRAMES_DIR')
    
    if not all([youtube_url, video_output_path, frames_dir]):
        print("Error: Missing required configuration parameters.")
        print("Required: YOUTUBE_URL, VIDEO_OUTPUT_PATH, FRAMES_DIR")
        sys.exit(1)
    
    print(f"Configuration loaded from: {args.config}")
    print(f"YouTube URL: {youtube_url}")
    print(f"Video output path: {video_output_path}")
    print(f"Frames directory: {frames_dir}")
    print("-" * 100)
    
    # Download video and extract frames
    download_youtube_video(youtube_url, video_output_path)
    extract_frames(video_output_path, frames_dir)
    print("Stub: Court and ball detection not yet implemented.")

if __name__ == "__main__":
    main()