import os
from ultralytics import YOLO

# --- CONFIGURATION ---
# Path to the data.yaml
DATA_YAML_PATH = 'yolo_ball_dataset/data.yaml' 

# MODEL_WEIGHTS = 'yolo12n.pt'
MODEL_WEIGHTS = 'ball_detection/yolo12n_ball_fine_tune/weights/best.pt'

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 12
# BATCH_SIZE = -1  # Use -1 for auto batch size
PROJECT_NAME = 'ball_detection'
EXPERIMENT_NAME = 'yolo12n_ball_fine_tune'
DEVICE = '0'
# --- END CONFIGURATION ---

def run_training():
    print(f"Starting YOLOv12 fine-tuning with weights: {MODEL_WEIGHTS}")
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: Data YAML not found at {DATA_YAML_PATH}.")
        print("Please run 'prepare_yolo_ball_dataset.py' first to generate the dataset structure.")
        return None

    model = YOLO(MODEL_WEIGHTS)

    print(f"Training started for {EPOCHS} epochs on device {DEVICE}...")
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        device=DEVICE, 
        # Additional recommended fine-tuning arguments:
        # lr0=0.001,
        # patience=50,
    )
    
    print("\n--- Training Complete ---")
    print(f"Results saved to: runs/detect/{EXPERIMENT_NAME}")
    return results

def run_inference(training_results):
    if training_results and training_results.save_dir:
        best_model_path = os.path.join(training_results.save_dir, 'weights', 'best.pt')
    else:
        print("Warning: Could not determine best model path from training results. Skipping inference.")
        return
        
    if not os.path.exists(best_model_path):
         print(f"Warning: Best model weights not found at {best_model_path}. Skipping inference.")
         return
         
    print(f"\n--- Running Inference (Test) on device {DEVICE} ---")
    model = YOLO(best_model_path)
    
    val_images_path = os.path.join(os.path.dirname(DATA_YAML_PATH), 'images', 'val')
    
    model.predict(
        source=val_images_path,
        conf=0.20,
        save=True,
        show=False,
        name=f'{EXPERIMENT_NAME}_inference',
        device=DEVICE 
    )
    print("Inference complete. Results saved in the 'runs/detect/' directory.")


if __name__ == '__main__':
    # Run training
    training_results = run_training()
    
    # # Run inference
    # if training_results:
    #     run_inference(training_results)
