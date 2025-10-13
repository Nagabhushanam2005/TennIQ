import os
import pandas as pd
import random
import cv2
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
IMAGE_EXTENSION = '.jpg'
BASE_DIR = 'dataset/Dataset'
OUTPUT_DIR = 'yolo_ball_dataset'

BALL_SIZE_PIXELS = 12
BALL_WIDTH_PIXELS = BALL_SIZE_PIXELS
BALL_HEIGHT_PIXELS = BALL_SIZE_PIXELS

BALL_WIDTH_NORM = BALL_WIDTH_PIXELS / IMAGE_WIDTH
BALL_HEIGHT_NORM = BALL_HEIGHT_PIXELS / IMAGE_HEIGHT

CLASS_ID = 0 
CLASS_NAMES = ['ball'] 
# --- END CONFIGURATION ---

def setup_directories():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    for subset in ['train', 'val']:
        Path(OUTPUT_DIR, 'images', subset).mkdir(parents=True, exist_ok=True)
        Path(OUTPUT_DIR, 'labels', subset).mkdir(parents=True, exist_ok=True)

def create_yaml():
    yaml_content = f"""
train: {OUTPUT_DIR}/images/train
val: {OUTPUT_DIR}/images/val

nc: {len(CLASS_NAMES)}

names: {CLASS_NAMES}
"""
    with open(Path(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print(f"\nGenerated data.yaml at {OUTPUT_DIR}/data.yaml")

def _save_dataset_from_metadata(samples_meta, subset_name):
    count = 0
    print(f"Writing {len(samples_meta)} {subset_name} samples to disk...")
    
    for sample in samples_meta:
        try:
            img_prev = cv2.imread(str(sample['paths'][0]))
            img_curr = cv2.imread(str(sample['paths'][1]))
            img_next = cv2.imread(str(sample['paths'][2]))

            if img_prev is None or img_curr is None or img_next is None:
                continue
                
            img_prev = img_prev.astype(np.float32)
            img_curr = img_curr.astype(np.float32)
            img_next = img_next.astype(np.float32)

            stacked_image = (img_prev + img_curr + img_next) / 3.0
            
            stacked_image = stacked_image.astype(np.uint8)
            
            clip_name = sample['clip_dir'].name
            game_name = sample['clip_dir'].parent.name
            
            new_name_base = f"{game_name}_{clip_name}_{sample['frame_num']:04d}_stacked"
            
            image_dst_path = Path(OUTPUT_DIR, 'images', subset_name, f"{new_name_base}{IMAGE_EXTENSION}")
            label_dst_path = Path(OUTPUT_DIR, 'labels', subset_name, f"{new_name_base}.txt")
            
            cv2.imwrite(str(image_dst_path), stacked_image)

            with open(label_dst_path, 'w') as f:
                f.write(sample['yolo_lines'])
            
            count += 1
            
        except Exception as e:
             print(f"Error saving sample for frame {sample['frame_num']} in clip {sample['clip_dir'].name}: {e}")

    return count

def _generate_yolo_line(frame_num, frame_map):
    if frame_num in frame_map and frame_map[frame_num]['visibility'] == 1:
        x_center_norm = frame_map[frame_num]['x-coordinate'] / IMAGE_WIDTH
        y_center_norm = frame_map[frame_num]['y-coordinate'] / IMAGE_HEIGHT
        
        yolo_line = (
            f"{CLASS_ID} {x_center_norm:.6f} {y_center_norm:.6f} "
            f"{BALL_WIDTH_NORM:.6f} {BALL_HEIGHT_NORM:.6f}"
        )
        return yolo_line
    return None

def process_all_clips():
    csv_paths = list(Path(BASE_DIR).rglob('Label.csv'))
    if not csv_paths:
        print(f"Error: No 'Label.csv' files found under {BASE_DIR}.")
        return

    print(f"Found {len(csv_paths)} 'Label.csv' files. Starting metadata collection...")

    all_frame_data = {}
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            clip_dir = csv_path.parent
            
            data_map = {int(Path(row['file name']).stem): row for _, row in df.iterrows()}
            all_frame_data[clip_dir] = data_map
            
        except Exception as e:
            print(f"Warning: Could not process {csv_path}: {e}")
            
    processed_samples_metadata = [] 

    for clip_dir, frame_map in all_frame_data.items():
        sorted_frames = sorted(frame_map.keys())
        
        for i in range(len(sorted_frames) - 2):
            f_prev_num, f_curr_num, f_next_num = sorted_frames[i], sorted_frames[i+1], sorted_frames[i+2]
            
            img_path_prev = Path(clip_dir, f"{f_prev_num:04d}{IMAGE_EXTENSION}")
            img_path_curr = Path(clip_dir, f"{f_curr_num:04d}{IMAGE_EXTENSION}")
            img_path_next = Path(clip_dir, f"{f_next_num:04d}{IMAGE_EXTENSION}")
            
            if not all(p.exists() for p in [img_path_prev, img_path_curr, img_path_next]):
                continue
            
            yolo_lines = []
            
            line_prev = _generate_yolo_line(f_prev_num, frame_map)
            if line_prev: yolo_lines.append(line_prev)
            
            line_curr = _generate_yolo_line(f_curr_num, frame_map)
            if line_curr: yolo_lines.append(line_curr)
            
            line_next = _generate_yolo_line(f_next_num, frame_map)
            if line_next: yolo_lines.append(line_next)
            
            if yolo_lines:
                processed_samples_metadata.append({
                    'paths': [img_path_prev, img_path_curr, img_path_next],
                    'yolo_lines': "\n".join(yolo_lines) + "\n",
                    'clip_dir': clip_dir,
                    'frame_num': f_curr_num
                })

    random.seed(42)
    random.shuffle(processed_samples_metadata)
    split_idx = int(len(processed_samples_metadata) * 0.8)
    train_meta = processed_samples_metadata[:split_idx]
    val_meta = processed_samples_metadata[split_idx:]
    
    processed_train = _save_dataset_from_metadata(train_meta, 'train')
    processed_val = _save_dataset_from_metadata(val_meta, 'val')
    
    total_processed = processed_train + processed_val
    print("\nDataset preparation finished.")
    print(f"Total stacked images processed: {total_processed}")
    print(f"  - Training samples: {processed_train}")
    print(f"  - Validation samples: {processed_val}")
    print(f"\nDataset structure created in the '{OUTPUT_DIR}' directory.")


if __name__ == '__main__':
    setup_directories()
    process_all_clips()
    create_yaml()
