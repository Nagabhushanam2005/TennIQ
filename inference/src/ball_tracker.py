"""
Ball Tracking Module

Authors: N Nagabhushanam <thechosentwins2005@gmail.com>
"""

import logging
import numpy as np
import cv2

from models.tracknet import trackNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BallTracker:
    def __init__(self):

        self.confidence = 0.3
        self.max_trajectory_length = 30
        self.n_classes = 256
        self.height = 360
        self.width = 640
        self.circles = None

    def update_config(self, config):
        """Update tracker configuration"""
        if config:
            self.confidence = config.get('BALL_CONFIDENCE', 0.3)
            self.max_trajectory_length = config.get('BALL_TRAJECTORY_LENGTH', 30)
        self.model = self._load_model(config.get('MODEL_PATH'))

    def _load_model(self, path):
        m = trackNet(self.n_classes, input_height=self.height, input_width=self.width)
        m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        m.load_weights(path)
        return m
    def detect_ball(self, frame):
        """Detect tennis ball"""
        frame_transposed = np.transpose(frame, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        pr = self.model.predict(np.array([frame_transposed]))[0]
        pr = pr.reshape((self.height, self.width, self.n_classes)).argmax(axis=2)
        pr = pr.astype(np.uint8)
        heatmap = cv2.resize(pr, (self.width, self.height))
        ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
        self.circles = circles
        return ret, heatmap, self.circles

    def detect_ball_motion(self):
        # TODO:
        pass

    def update(self,frame):
        """Update ball tracking for current frame"""
        # TODO: Update ball tracking for current frame using YOLO-enhanced detection
        self.detect_ball(frame=frame)
    def detect_bounce(self):
        """Detect if ball has bounced based on velocity change"""
        # TODO: Implement bounce detection based on velocity changes
        pass

    def draw_ball(self, frame):
        """Draw ball position, trajectory, and forecast on frame"""
        if self.circles is not None:
            for circle in self.circles[0, :]:
                x, y, r = circle
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        return frame
