import os
from ultralytics import YOLO

# --- CONFIGURATION ---
DATA_YAML_PATH = 'yolo_dataset/data.yaml' 
MODEL_WEIGHTS = 'yolo11n.pt' 

EPOCHS = 200
IMG_SIZE = 640
BATCH_SIZE = 16 
BATCH_SIZE = -1 # auto batch size  
PROJECT_NAME = 'player_detection'
EXPERIMENT_NAME = 'yolo11n_player_finetune'
# --- END CONFIGURATION ---

def run_training():
    print(f"Starting YOLOv11 fine-tuning: {MODEL_WEIGHTS}")
    
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: Data YAML not found at {DATA_YAML_PATH}.")
        return None

    model = YOLO(MODEL_WEIGHTS)

    print(f"Training started for {EPOCHS} epochs...")
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        # Additional recommended fine-tuning arguments:
        # lr0=0.001,       # Lower initial learning rate is common for fine-tuning
        # patience=50,      # Stop training early if mAP doesn't improve for 50 epochs
        # device='0',       # Specify GPU device ID
    )
    
    print("\nTraining Complete")
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
         
    print(f"\n--- Running Inference (Test) ---")
    model = YOLO(best_model_path)
    
    val_images_path = os.path.join(os.path.dirname(DATA_YAML_PATH), 'images', 'val')
    
    model.predict(
        source=val_images_path,
        conf=0.25, # Confidence threshold
        save=True,
        show=False,
        name=f'{EXPERIMENT_NAME}_inference'
    )
    print("Inference complete. Results saved in the 'runs/detect/' directory.")


if __name__ == '__main__':
    training_results = run_training()
    
    if training_results:
        run_inference(training_results)
