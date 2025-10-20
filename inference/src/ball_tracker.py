import logging
import cv2
import numpy as np
import torch
import threading
from collections import deque
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BallTracker:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.predicted_points_queue = deque([None] * 10, maxlen=10)
        self.lock = threading.Lock()
        self.width_ratio = 1.0
        self.height_ratio = 1.0
        self.frame_count = 0
        self.ball_positions = []
        self.model_loaded = False
        self.model_type = None
        self.conf_threshold = 0.05
        self.target_class_name = "ball"

    def update_config(self, config):
        model_weights = config.get("BALL_MODEL_WEIGHTS")
        model_name = config.get("BALL_MODEL_NAME", "TrackNetV4_TypeA")

        if model_weights and not self.model_loaded:
            # Determine model type based on file extension or model name
            if 'yolo' in model_name.lower():
                self._load_yolo_model(model_weights)
            else:
                self._load_tracknet_model(model_weights, model_name)

    def _load_yolo_model(self, model_weights):
        try:
            self.model = YOLO(model_weights)
            self.model_loaded = True
            self.model_type = 'yolo'
            logger.info(f"YOLO model loaded successfully from {model_weights}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")

    def _load_tracknet_model(self, model_weights, model_name):
        try:
            from TrackNetv4.src.util import get_model
            INPUT_HEIGHT = 288
            INPUT_WIDTH = 512

            self.model = get_model(model_name, INPUT_HEIGHT, INPUT_WIDTH)
            self.model.load_state_dict(torch.load(model_weights, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            self.model_type = 'tracknet'
            self.model = torch.jit.script(self.model)
            logger.info(f"TrackNet model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load TrackNet model: {e}")

    def update(self, frames):
        if not self.model_loaded:
            return

        if self.model_type == 'yolo':
            self._update_yolo(frames)
        else:
            self._update_tracknet(frames)

    def _update_yolo(self, frames):
        """YOLO-based ball tracking"""
        self.frame_count += 1

        stacked_image = np.mean(np.stack(frames, axis=0, dtype=np.float32), axis=0, dtype=np.float32)
        composite_frame = stacked_image.astype(np.uint8)
        
        results = self.model.predict(
            source=composite_frame,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            self.ball_positions.append(None)
            self.predicted_points_queue.appendleft(None)
            return

        boxes = results[0].boxes
        xywh = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        best_box, best_conf = None, 0.0
        for i in range(len(xywh)):
            cls_idx = int(classes[i])
            cls_name = results[0].names.get(cls_idx, "")
            if cls_name == self.target_class_name and confs[i] > best_conf:
                best_box, best_conf = xywh[i], confs[i]

        if best_box is None:
            self.ball_positions.append(None)
            self.predicted_points_queue.appendleft(None)
            return

        x_center, y_center, w, h = best_box
        ball_point = (int(x_center), int(y_center))

        with self.lock:
            self.predicted_points_queue.appendleft(ball_point)
            self.ball_positions.append(ball_point)

    def _update_tracknet(self, frames):
        """TrackNet-based ball tracking"""
        if self.width_ratio == 1.0:
            self._calculate_ratios(frames[2])

        h, w = 288, 512
        batch = []
        for f in frames:
            f = cv2.resize(f, (w, h))
            f = f[:, :, ::-1].astype(np.float32) / 255.0
            batch.append(torch.from_numpy(f).permute(2, 0, 1))

        input_tensor = torch.cat(batch, dim=0).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.no_grad():
            if hasattr(self.model, 'forward_ball_only'):
                ball_preds = self.model.forward_ball_only(input_tensor)
            else:
                ball_preds = self.model(input_tensor)

            if isinstance(ball_preds, tuple):
                ball_preds = ball_preds[0]
        ball_predictions = ball_preds.cpu()
        self._process_predictions(ball_predictions, frames[2])

    def _calculate_ratios(self, frame):
        frame_height, frame_width = frame.shape[:2]
        INPUT_WIDTH = 512
        INPUT_HEIGHT = 288
        self.width_ratio = frame_width / INPUT_WIDTH
        self.height_ratio = frame_height / INPUT_HEIGHT

    def _process_predictions(self, ball_predictions, current_frame):
        ball_heatmaps = (ball_predictions > 0.5).float()
        ball_binary_heatmaps = (ball_heatmaps[0] * 255).byte().numpy()

        if np.amax(ball_binary_heatmaps[2]) <= 0:
            self.ball_positions.append(None)
            return

        contours, _ = cv2.findContours(ball_binary_heatmaps[2].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.ball_positions.append(None)
            return

        largest_bounding_box = max([cv2.boundingRect(c) for c in contours], key=lambda r: r[2] * r[3])

        predicted_x_center = int(self.width_ratio * (largest_bounding_box[0] + largest_bounding_box[2] / 2))
        predicted_y_center = int(self.height_ratio * (largest_bounding_box[1] + largest_bounding_box[3] / 2))

        with self.lock:
            self.predicted_points_queue.appendleft((predicted_x_center, predicted_y_center))
            self.ball_positions.append((predicted_x_center, predicted_y_center))

    def draw_ball(self, frame):
        result_frame = frame.copy()
        with self.lock:
            points_snapshot = list(self.predicted_points_queue)

            for point in points_snapshot:
                if point is not None:
                    cv2.circle(result_frame, point, 5, (0, 255, 0), 2)
            current_pos = self.ball_positions[-1] if self.ball_positions else None
            if current_pos is not None:
                cv2.circle(result_frame, current_pos, 8, (0, 0, 255), -1)

        return result_frame

    def get_ball_history(self):
        return self.ball_positions