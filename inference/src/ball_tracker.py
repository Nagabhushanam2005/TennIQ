import logging
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BallTracker:
    def __init__(self):
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.predicted_points_queue = deque([None] * 10, maxlen=10)
        self.frame_count = 0
        self.ball_positions = []
        self.model_loaded = False
        self.target_class_name = "ball"
        self.conf_threshold = 0.25

    def update_config(self, config):
        model_weights = config.get("BALL_MODEL_WEIGHTS")
        if model_weights and not self.model_loaded:
            self._load_model(model_weights)

    def _load_model(self, model_weights):
        try:
            self.model = YOLO(model_weights)
            self.model_loaded = True
            logger.info(f"YOLOv12 model loaded successfully from {model_weights}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv12 model: {e}")

    def update(self, frames):
        """
        Perform YOLO-based ball tracking on a batch of 3 frames.
        frames: [prev_frame, curr_frame, next_frame]
        """
        if not self.model_loaded:
            logger.warning("YOLO model not loaded — skipping frame.")
            return

        # if len(frames) < 3:
        #     logger.warning("Need exactly 3 frames for YOLO-based tracking.")
        #     return

        self.frame_count += 1

        # img_prev = frames[0].astype(np.float32)
        # img_curr = frames[1].astype(np.float32)
        # img_next = frames[2].astype(np.float32)

        # stacked_image = (img_prev + img_curr + img_next) / 3.0
        # composite_frame = stacked_image.astype(np.uint8)

        composite_frame = frames[2]

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

        self.predicted_points_queue.appendleft(ball_point)
        self.ball_positions.append(ball_point)

    def draw_ball(self, frame):
        """Draw the trail and current position of the ball."""
        result_frame = frame.copy()
        for point in self.predicted_points_queue:
            if point is not None:
                cv2.circle(result_frame, point, 4, (0, 255, 0), 2)
        if self.ball_positions and self.ball_positions[-1] is not None:
            cv2.circle(result_frame, self.ball_positions[-1], 6, (0, 0, 255), -1)
        return result_frame

    def get_ball_history(self):
        """Return all tracked ball positions."""
        return self.ball_positions
