import os
import cv2
import csv
import argparse
from ultralytics import YOLO

def run_video_inference(video_path, model_weights, device='0', conf=0.25):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at {video_path}")
    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"Model weights not found at {model_weights}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = f"{video_name}_predict.mp4"
    output_csv_path = f"{video_name}_predict.csv"

    model = YOLO(model_weights)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    csv_file = open(output_csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'class', 'confidence', 'x_center', 'y_center', 'width', 'height'])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, device=device, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf_val = float(box.conf[0])
            x_center, y_center, w, h = map(float, box.xywh[0])
            csv_writer.writerow([frame_idx, cls, conf_val, x_center, y_center, w, h])

        frame_idx += 1
        print(f"Processed frame {frame_idx}/{total_frames}", end='\r')

    cap.release()
    out.release()
    csv_file.close()

    print(f"\nResults saved as:\n{output_video_path}\n{output_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--model_weights', type=str, required=True)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()

    run_video_inference(
        video_path=args.video_path,
        model_weights=args.model_weights,
        device=args.device,
        conf=args.conf
    )
