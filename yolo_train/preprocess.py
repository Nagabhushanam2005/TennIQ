import os
import pandas as pd
import yaml
import shutil
import random
from collections import defaultdict

# --- CONFIGURATION ---
IMAGE_WIDTH = 1280 
IMAGE_HEIGHT = 720
IMAGE_EXTENSION = '.jpg'

BASE_DIR = 'dataset/Dataset'
CSV_DIR = 'dataset/csv'
OUTPUT_DIR = 'yolo_dataset'

TRAIN_RATIO = 0.8
CLASS_NAMES = {
    1: 'person',
    2: 'person',
}
# --- END CONFIGURATION ---

def create_yolo_folders():
    """Creates the necessary YOLO directory structure."""
    print(f"Creating output directory structure at: {OUTPUT_DIR}")
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def normalize_bbox(row, img_w, img_h):
    """Converts pixel-based x1, y1, x2, y2 to normalized YOLO format (x_center, y_center, w, h)."""
    x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']

    w_px = x2 - x1
    h_px = y2 - y1

    x_center_px = x1 + w_px / 2
    y_center_px = y1 + h_px / 2

    x_center_norm = x_center_px / img_w
    y_center_norm = y_center_px / img_h
    w_norm = w_px / img_w
    h_norm = h_px / img_h

    # Class ID must be 0 for the first class ('player')
    class_id = 0

    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

def process_csv_files():
    """Processes all CSVs and returns annotations grouped by full image path."""
    all_annotations = defaultdict(list)
    
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('_players.csv')]
    
    if not csv_files:
        print(f"Error: No CSV files found in {CSV_DIR}. Please check your path.")
        return None

    print(f"Found {len(csv_files)} CSV annotation files. Starting conversion...")

    for csv_file in csv_files:
        csv_path = os.path.join(CSV_DIR, csv_file)
        
        parts = csv_file.replace('_players.csv', '').split('_')
        game_id = parts[0]
        clip_id = parts[1]
        
        clip_image_dir = os.path.join(BASE_DIR, game_id, clip_id)

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        for frame_num, frame_group in df.groupby('frame'):
            # Fix: Use 4-digit padding and remove 'frame_' prefix to match the dataset images (e.g., '0000.jpg')
            image_filename = f"{frame_num:04d}{IMAGE_EXTENSION}"
            image_src_path = os.path.join(clip_image_dir, image_filename)

            if not os.path.exists(image_src_path):
                continue

            full_key = os.path.join(game_id, clip_id, image_filename)

            yolo_annotations = frame_group.apply(
                lambda row: normalize_bbox(row, IMAGE_WIDTH, IMAGE_HEIGHT), axis=1
            ).tolist()

            all_annotations[full_key] = {
                'source_path': image_src_path,
                'yolo_labels': '\n'.join(yolo_annotations)
            }
            
    print(f"Conversion complete. Found {len(all_annotations)} annotated images.")
    return all_annotations

def split_and_save_data(all_annotations):
    """Splits data into train/val and saves images and labels to the YOLO format."""
    
    if not all_annotations:
        print("No valid annotations to save.")
        return

    all_keys = list(all_annotations.keys())
    random.shuffle(all_keys)

    train_split_index = int(len(all_keys) * TRAIN_RATIO)
    train_keys = all_keys[:train_split_index]
    val_keys = all_keys[train_split_index:]

    print(f"Splitting data: {len(train_keys)} for training, {len(val_keys)} for validation.")

    for split_name, keys in [('train', train_keys), ('val', val_keys)]:
        print(f"Saving {len(keys)} files to {split_name}...")
        for key in keys:
            data = all_annotations[key]
            
            # Create a unique filename for the output directory
            base_filename = key.replace(os.path.sep, '_').replace(IMAGE_EXTENSION, '')
            
            # Save Image
            img_src = data['source_path']
            img_dest_name = base_filename + IMAGE_EXTENSION
            img_dest_path = os.path.join(OUTPUT_DIR, 'images', split_name, img_dest_name)
            shutil.copyfile(img_src, img_dest_path)

            # Save Label
            label_content = data['yolo_labels']
            label_dest_name = base_filename + '.txt'
            label_dest_path = os.path.join(OUTPUT_DIR, 'labels', split_name, label_dest_name)
            
            with open(label_dest_path, 'w') as f:
                f.write(label_content)

def create_yaml_config():
    """Creates the data.yaml configuration file for Ultralytics."""
    yaml_config = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASS_NAMES),
        'names': [name for id, name in sorted(CLASS_NAMES.items())]
    }

    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, sort_keys=False)
    
    print(f"\nSuccessfully created YOLO configuration file at: {yaml_path}")
    print("This file is ready for use in your model.train() call.")

if __name__ == '__main__':
    create_yolo_folders()
    annotations = process_csv_files()
    if annotations:
        split_and_save_data(annotations)
        create_yaml_config()
    print("\nDataset preparation finished.")
    print("\nNext step: Run the YOLOv11 training command using the generated 'yolo_dataset/data.yaml' file.")
