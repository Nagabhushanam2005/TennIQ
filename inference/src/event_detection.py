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
    def __init__(self, tracker, event_model_path: Optional[str] = None):
        self.tracker = tracker
        self.events: Dict[int, List[Dict]] = {}
        self.frame_count = 0
        self.last_input = None
        self.last_event = 0
        self.frame_width = 1280
        self.frame_height = 720
        self.model = None
        self.threshold = 0.85
        self.last_known_bounce_frames: Set[int] = set()
        self.eventstring =""

        self.load_event_model(event_model_path)
        
    def load_event_model(self, path_model: Optional[str]):
        if not path_model:
            logger.warning("No path provided for CatBoost model. Event detection is disabled.")
            self.model = None
            return

        try:
            self.model = ctb.CatBoostClassifier(verbose=0)
            self.model.load_model(path_model)
            logger.info(f"CatBoost Bounce Model loaded from {path_model}")
        except Exception as e:
            logger.error(f"Failed to load CatBoost model from {path_model}: {e}. Disabling model prediction.")
            self.model = None

    def update(self) -> List[Dict]:
        self.frame_count += 1

        if(self.last_event + 7 > self.frame_count):
            return []

        history_positions = self.tracker.get_ball_history()
        
        x_ball = [pos[0] if pos else None for pos in history_positions]
        y_ball = [pos[1] if pos else None for pos in history_positions]

        detected_events = self._detect_bounce_hit(x_ball, y_ball)

        new_events_list = []
        for frame_num, evt_type in detected_events.items():
            if 0 <= frame_num < len(history_positions):
                pos = history_positions[frame_num]
                if pos:
                    event = {
                        "frame": frame_num,
                        "type": evt_type,
                        "position": pos
                    }
                    self.events.setdefault(frame_num, []).append(event)
                    new_events_list.append(event)
                    self.last_event = self.frame_count
                    logger.info(f"{evt_type} detected at frame {frame_num}, position {pos}")
        return new_events_list



    def _detect_bounce_hit(self, x_ball: List[Optional[float]], y_ball: List[Optional[float]]) -> Dict[int, str]:
        if self.model is None:
            return {}

        if((x_ball, y_ball) == self.last_input):
            return {}
        self.last_input = (x_ball, y_ball)
        x_smooth, y_smooth = self._smooth_predictions(x_ball, y_ball)
        features, valid_frames = self._prepare_features(x_smooth, y_smooth)

        if features.size == 0:
            return {}
        probs = self.model.predict_proba(features)

        if(np.max(probs) < self.threshold):
            return {}
        preds = np.argmax(probs, axis=1)
        events = {}
        for idx, cls in enumerate(preds):
            if cls == 1:
                events[valid_frames[idx]] = "HIT"
            elif cls == 2:
                events[valid_frames[idx]] = "BOUNCE"
            if cls==1 or cls==2:
                self.eventstring += events[valid_frames[idx]][0]
                logger.info(self.eventstring+f" {np.max(probs)}")

        return events



    def _prepare_features(self, x_ball, y_ball):
        features = []
        frame_indices = []

        Xseq = x_ball[-5:]
        Yseq = y_ball[-5:]

        x_seq = np.array(Xseq, dtype=float)
        y_seq = np.array(Yseq, dtype=float)

        f = self._create_advanced_features(x_seq, y_seq)
        features.append(f)
        frame_indices.append(len(x_ball) - 3)

        if not features:
            return np.array([]), []

        return np.array(features, dtype=float), frame_indices

    def _create_advanced_features(self, x_seq, y_seq):
        def compute_block(xs, ys):
            f = []
            num = 3
            eps = 1e-15
            df = pd.DataFrame({"x": xs, "y": ys})
            for i in range(1, num):
                df[f'x_lag_{i}'] = df['x'].shift(i)
                df[f'y_lag_{i}'] = df['y'].shift(i)
                df[f'x_lag_inv_{i}'] = df['x'].shift(-i)
                df[f'y_lag_inv_{i}'] = df['y'].shift(-i)
                df[f'x_diff_{i}'] = abs(df[f'x_lag_{i}'] - df['x'])
                df[f'y_diff_{i}'] = df[f'y_lag_{i}'] - df['y']
                df[f'x_diff_inv_{i}'] = abs(df[f'x_lag_inv_{i}'] - df['x'])
                df[f'y_diff_inv_{i}'] = df[f'y_lag_inv_{i}'] - df['y']
                df[f'x_div_{i}'] = df[f'x_diff_{i}'] / (df[f'x_diff_inv_{i}'] + eps)
                df[f'y_div_{i}'] = df[f'y_diff_{i}'] / (df[f'y_diff_inv_{i}'] + eps)
            mid = len(df) // 2
            for i in range(1, num):
                for k in [f'x_diff_{i}', f'y_diff_{i}', f'x_diff_inv_{i}', f'y_diff_inv_{i}', f'x_div_{i}', f'y_div_{i}']:
                    v = df.iloc[mid][k]
                    f.append(0.0 if pd.isna(v) else v)
            return f
        
        features = []
        if len(x_seq) >= 3:
            f1 = compute_block(x_seq[:3], y_seq[:3])
            f2 = compute_block(x_seq[-3:], y_seq[-3:])
            
            features = f1 + f2
            vx = np.diff(x_seq)
            vy = np.diff(y_seq)
            ax = np.diff(vx) if len(vx) > 1 else [0]
            ay = np.diff(vy) if len(vy) > 1 else [0]
            features.extend([
                np.mean(vx)/self.frame_width if len(vx)>0 else 0,
                np.mean(vy)/self.frame_height if len(vy)>0 else 0,
                np.std(vx)/self.frame_width if len(vx)>0 else 0,
                np.std(vy)/self.frame_height if len(vy)>0 else 0,
                np.mean(ax)/self.frame_width if len(ax)>0 else 0,
                np.mean(ay)/self.frame_height if len(ay)>0 else 0
            ])
        return features

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

        all_events = [event for event_list in self.events.values() for event in event_list]
        all_events.sort(key=lambda e: e['frame'])

        last_10_events = all_events[-10:]

        for event in last_10_events:
            pos = event.get("position")
            if not pos:
                continue
            pos = tuple(pos[:2])
            frame_num = event["frame"]

            if event["type"] == "BOUNCE":
                    cv2.circle(result_frame, pos, 8, (0, 125, 255), -1)

            elif event["type"] == "HIT":
                cv2.circle(result_frame, pos, 8, (255, 0, 0), -1)

        return result_frame
