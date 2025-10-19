import logging
import cv2
import numpy as np
import pandas as pd
from collections import deque
from typing import Set, List, Tuple, Optional, Dict
import catboost as ctb
from scipy.interpolate import CubicSpline
from scipy.spatial import distance


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventDetector:
    def __init__(self, tracker, bounce_model_path: Optional[str] = None):
        self.tracker = tracker
        self.events: Dict[int, List[Dict]] = {}
        self.frame_count = 0

        self.model = None
        self.threshold = 0.45
        self.last_known_bounce_frames: Set[int] = set()

        self.load_bounce_model(bounce_model_path)
        
    def load_bounce_model(self, path_model: Optional[str]):
        if not path_model:
            logger.warning("No path provided for CatBoost model. Event detection is disabled.")
            self.model = None
            return

        try:
            self.model = ctb.CatBoostRegressor(verbose=0)
            self.model.load_model(path_model)
            logger.info(f"CatBoost Bounce Model loaded from {path_model}")
        except Exception as e:
            logger.error(f"Failed to load CatBoost model from {path_model}: {e}. Disabling model prediction.")
            self.model = None

    def update(self) -> List[Dict]:
        self.frame_count += 1

        history_positions = self.tracker.get_ball_history()
        
        x_ball = [pos[0] if pos else None for pos in history_positions]
        y_ball = [pos[1] if pos else None for pos in history_positions]

        new_bounce_frames = self._detect_bounce(x_ball, y_ball)
        
        new_events_list = []
        
        newly_detected_frames = new_bounce_frames - self.last_known_bounce_frames
        
        for frame_num in newly_detected_frames:
            if 0 <= frame_num < len(history_positions):
                bounce_pos = history_positions[frame_num]
                if bounce_pos:
                    event = {
                        "frame": frame_num,
                        "type": "BOUNCE",
                        "position": bounce_pos
                    }
                    self.events.setdefault(frame_num, []).append(event)
                    new_events_list.append(event)
                    logger.info(f"BOUNCE detected, position {bounce_pos}")
                        
        self.last_known_bounce_frames = new_bounce_frames 
        return new_events_list

    def _detect_bounce(self, x_ball: List[Optional[float]], y_ball: List[Optional[float]]) -> Set[int]:
        if self.model is None:
            return set()

        try:
            x_smooth, y_smooth = self._smooth_predictions(x_ball, y_ball)
            
            features, valid_frames = self._prepare_features(x_smooth, y_smooth)
            
            if features.empty:
                return set()
                
            preds = self.model.predict(features)
            
            ind_bounce = np.where(preds > self.threshold)[0]
            
            if len(ind_bounce) > 0:
                ind_bounce = self._postprocess(ind_bounce, preds)
            
            frames_bounce = [valid_frames[x] for x in ind_bounce]
            return set(frames_bounce)
            
        except Exception as e:
            logger.error(f"Bounce Detector prediction failed in _detect_bounce: {e}")
            return set()

    def _prepare_features(self, x_ball: List[Optional[float]], y_ball: List[Optional[float]]) -> Tuple[pd.DataFrame, List[int]]:
        valid_data = [(i, x, y) for i, (x, y) in enumerate(zip(x_ball, y_ball)) if x is not None and y is not None]
        
        if not valid_data:
            return pd.DataFrame(), []

        frames, x_valid, y_valid = zip(*valid_data)
        labels = pd.DataFrame({
            'frame': frames, 
            'x-coordinate': x_valid, 
            'y-coordinate': y_valid
        })
        
        num = 3 
        eps = 1e-15
        
        for i in range(1, num):
            labels['x_lag_{}'.format(i)] = labels['x-coordinate'].shift(i)
            labels['x_lag_inv_{}'.format(i)] = labels['x-coordinate'].shift(-i)
            labels['y_lag_{}'.format(i)] = labels['y-coordinate'].shift(i)
            labels['y_lag_inv_{}'.format(i)] = labels['y-coordinate'].shift(-i) 
            labels['x_diff_{}'.format(i)] = abs(labels['x_lag_{}'.format(i)] - labels['x-coordinate'])
            labels['y_diff_{}'.format(i)] = labels['y_lag_{}'.format(i)] - labels['y-coordinate']
            labels['x_diff_inv_{}'.format(i)] = abs(labels['x_lag_inv_{}'.format(i)] - labels['x-coordinate'])
            labels['y_diff_inv_{}'.format(i)] = labels['y_lag_inv_{}'.format(i)] - labels['y-coordinate']
            
            x_div_den = labels['x_diff_inv_{}'.format(i)] + eps
            y_div_den = labels['y_diff_inv_{}'.format(i)] + eps
            labels['x_div_{}'.format(i)] = labels['x_diff_{}'.format(i)] / x_div_den
            labels['y_div_{}'.format(i)] = labels['y_diff_{}'.format(i)] / y_div_den

        for i in range(1, num):
            labels = labels[labels['x_lag_{}'.format(i)].notna()]
            labels = labels[labels['x_lag_inv_{}'.format(i)].notna()]
        
        colnames_x = ['x_diff_{}'.format(i) for i in range(1, num)] + \
                     ['x_diff_inv_{}'.format(i) for i in range(1, num)] + \
                     ['x_div_{}'.format(i) for i in range(1, num)]
        colnames_y = ['y_diff_{}'.format(i) for i in range(1, num)] + \
                     ['y_diff_inv_{}'.format(i) for i in range(1, num)] + \
                     ['y_div_{}'.format(i) for i in range(1, num)]
        colnames = colnames_x + colnames_y

        features = labels[colnames]
        return features, list(labels['frame'])

    def _smooth_predictions(self, x_ball: List[Optional[float]], y_ball: List[Optional[float]]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        x_ball = list(x_ball)
        y_ball = list(y_ball)
        
        is_none = [x is None for x in x_ball]
        interp_window = 5
        max_interp_count = 3
        counter = 0
        
        for num in range(interp_window, len(x_ball) - 1):
            if is_none[num] and not any(is_none[num - interp_window:num]) and counter < max_interp_count:
                x_coords = [c for c in x_ball[num - interp_window:num] if c is not None]
                y_coords = [c for c in y_ball[num - interp_window:num] if c is not None]
                
                if len(x_coords) == interp_window:
                    x_ext, y_ext = self._extrapolate(x_coords, y_coords)
                    
                    x_ball[num] = x_ext
                    y_ball[num] = y_ext
                    is_none[num] = False 
                    
                    if not is_none[num + 1]:
                        dist_val = distance.euclidean((x_ext, y_ext), (x_ball[num + 1], y_ball[num + 1]))
                        if dist_val > 80:
                            x_ball[num + 1], y_ball[num + 1] = None, None
                            is_none[num + 1] = True
                    counter += 1
                else:
                    counter = 0
            else:
                counter = 0  
                
        return x_ball, y_ball

    def _extrapolate(self, x_coords: List[float], y_coords: List[float]) -> Tuple[float, float]:
        xs = list(range(len(x_coords)))
        
        func_x = CubicSpline(xs, x_coords, bc_type='natural')
        x_ext = func_x(len(x_coords)) 
        
        func_y = CubicSpline(xs, y_coords, bc_type='natural')
        y_ext = func_y(len(x_coords))
        
        return float(x_ext), float(y_ext)    

    def _postprocess(self, ind_bounce: np.ndarray, preds: np.ndarray) -> List[int]:
        if not ind_bounce.size:
            return []
            
        ind_bounce_filtered = [ind_bounce[0]]
        
        for i in range(1, len(ind_bounce)):
            if (ind_bounce[i] - ind_bounce[i-1]) != 1:
                cur_ind = ind_bounce[i]
                ind_bounce_filtered.append(cur_ind)
            elif preds[ind_bounce[i]] > preds[ind_bounce_filtered[-1]]:
                ind_bounce_filtered[-1] = ind_bounce[i]
                
        return ind_bounce_filtered

    def get_events(self) -> Dict[int, List[Dict]]:
        return self.events

    def get_bounce_positions(self) -> Dict[int, Tuple[int, int]]:
        bounces = {}
        for frame_num, event_list in self.events.items():
            for event in event_list:
                if event.get("type") == "BOUNCE" and event.get("position") is not None:
                    bounces[frame_num] = tuple(event.get("position"))
        return bounces

    def draw_events(self, frame: np.ndarray) -> np.ndarray:
        result_frame = frame.copy()

        latest_bounce_frame = max(self.events.keys()) if self.events else -1

        for frame_num, event_list in self.events.items():
            for event in event_list:
                if event["type"] == "BOUNCE":
                    pos = event["position"]
                    if pos is not None: 
                        if frame_num == latest_bounce_frame:
                            cv2.circle(result_frame, pos, 10, (0, 255, 255), 5)
                        else:
                            cv2.circle(result_frame, pos, 8, (0, 125, 255), -1)
                            
        return result_frame
