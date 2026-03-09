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


# Event type constants
EVENT_HIT = "HIT"
EVENT_BOUNCE = "BOUNCE"
EVENT_OUT = "OUT"
EVENT_NET = "NET"
EVENT_SERVE_FAULT = "SERVE_FAULT"

# BGR colours for drawing each event type
EVENT_COLORS = {
    EVENT_HIT: (255, 0, 0),           # Blue
    EVENT_BOUNCE: (0, 125, 255),       # Orange
    EVENT_OUT: (0, 0, 255),            # Red
    EVENT_NET: (0, 255, 255),          # Yellow
    EVENT_SERVE_FAULT: (128, 0, 128),  # Purple
}


class EventDetector:
    """
    Unified event detector for tennis analysis.

    Detects five event types:
      - HIT:         player strikes the ball           (CatBoost trajectory model)
      - BOUNCE:      ball bounces inside the court      (CatBoost + court boundary check)
      - OUT:         ball bounces outside the court      (CatBoost + court boundary check)
      - NET:         ball hits the net                   (NetDetector integration)
      - SERVE_FAULT: serve lands outside the correct
                     service box                         (CatBoost + service-box geometry)
    """

    def __init__(
        self,
        tracker,
        court_detector=None,
        net_detector=None,
        player_tracker=None,
        event_model_path: Optional[str] = None,
    ):
        self.tracker = tracker
        self.court_detector = court_detector
        self.net_detector = net_detector
        self.player_tracker = player_tracker

        self.events: Dict[int, List[Dict]] = {}
        self.frame_count = 0
        self.last_input = None
        self.last_event = 0
        self.frame_width = 1280
        self.frame_height = 720
        self.model = None
        self.threshold = 0.85
        self.last_known_bounce_frames: Set[int] = set()
        self.eventstring = ""

        # Court geometry caches (populated lazily from court_detector)
        self._court_polygon = None      # singles boundary for in/out
        self._service_boxes: Dict[str, np.ndarray] = {}
        self._net_y: Optional[int] = None

        # Serve context – set externally via set_serve_context() each frame
        self._serve_context: Optional[Dict] = None

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

    def set_serve_context(self, serve_context: Optional[Dict]):
        """
        Provide current serve context so SERVE_FAULT can be distinguished.

        Args:
            serve_context: dict with keys ``is_serve`` (bool),
                ``server_half`` ("far"|"near"), ``point_side`` ("deuce"|"ad"),
                ``serve_number`` (int).  Pass *None* when not serving.
        """
        self._serve_context = serve_context

    # ── Court geometry helpers ───────────────────────────────────────────

    def _ensure_court_geometry(self):
        """Lazily build / refresh court geometry from *court_detector*."""
        if self._court_polygon is not None:
            return
        if self.court_detector is None:
            return
        kp = self.court_detector.get_keypoints()
        if kp is None:
            return

        # Singles court polygon (4 corners: indices 4, 5, 6, 7)
        singles = self.court_detector.get_singles_boundaries()
        if singles is not None:
            self._court_polygon = singles.reshape((-1, 1, 2)).astype(np.float32)

        # Net y-coordinate
        net_pts = self.court_detector.get_net_points()
        if net_pts is not None:
            self._net_y = int((net_pts[0][1] + net_pts[1][1]) / 2)

        # Service-box polygons
        self._build_service_boxes(kp)
        logger.info("Court geometry initialised for EventDetector "
                     f"(net_y={self._net_y})")

    def _build_service_boxes(self, kp: np.ndarray):
        """
        Create the four service-box polygons from the 16 court keypoints.

        Layout (camera perspective, upper = far / lower = near):

          kp[8]──kp[12]──kp[9]   upper service line
            │  UL  │  UR  │
          net ── net_center ── net
            │  LL  │  LR  │
          kp[10]─kp[13]─kp[11]   lower service line
        """
        net_left, net_right = kp[14], kp[15]

        net_center = self._line_intersection(kp[12], kp[13],
                                             net_left, net_right)
        if net_center is None:
            net_center = (net_left + net_right) / 2

        net_sl = self._line_intersection(kp[4], kp[5],
                                         net_left, net_right)
        if net_sl is None:
            net_sl = np.array([kp[4][0],
                               (net_left[1] + net_right[1]) / 2],
                              dtype=np.float32)

        net_sr = self._line_intersection(kp[7], kp[6],
                                         net_left, net_right)
        if net_sr is None:
            net_sr = np.array([kp[7][0],
                               (net_left[1] + net_right[1]) / 2],
                              dtype=np.float32)

        def _poly(pts):
            return np.array(pts, dtype=np.float32).reshape((-1, 1, 2))

        self._service_boxes = {
            "upper_left":  _poly([kp[8],  kp[12], net_center, net_sl]),
            "upper_right": _poly([kp[12], kp[9],  net_sr,     net_center]),
            "lower_left":  _poly([net_sl, net_center, kp[13], kp[10]]),
            "lower_right": _poly([net_center, net_sr, kp[11], kp[13]]),
        }

    @staticmethod
    def _line_intersection(p1, p2, p3, p4):
        """Intersection of segment *p1–p2* with segment *p3–p4* (or None)."""
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        x3, y3 = float(p3[0]), float(p3[1])
        x4, y4 = float(p4[0]), float(p4[1])
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return np.array([ix, iy], dtype=np.float32)

    def _is_in_court(self, position) -> bool:
        """Check if *position* falls inside the singles court polygon."""
        self._ensure_court_geometry()
        if self._court_polygon is None:
            return True  # no court info → assume in-court
        result = cv2.pointPolygonTest(
            self._court_polygon, (float(position[0]), float(position[1])), False
        )
        return result >= 0

    def _service_box_for_point(self, position) -> Optional[str]:
        """Return the service-box name the point falls in, or *None*."""
        self._ensure_court_geometry()
        if not self._service_boxes:
            return None
        pt = (float(position[0]), float(position[1]))
        for name, polygon in self._service_boxes.items():
            if cv2.pointPolygonTest(polygon, pt, False) >= 0:
                return name
        return None

    def _is_valid_serve_landing(self, position) -> bool:
        """Return *True* if the bounce is in the correct service box."""
        if not self._serve_context or not self._serve_context.get("is_serve"):
            return True
        if not self._service_boxes:
            return True

        server_half = self._serve_context.get("server_half", "far")
        point_side = self._serve_context.get("point_side", "deuce")

        # Diagonal serve rule (camera perspective)
        if server_half == "far":                      # server is upper
            expected = "lower_right" if point_side == "deuce" else "lower_left"
        else:                                         # server is lower
            expected = "upper_left" if point_side == "deuce" else "upper_right"

        actual = self._service_box_for_point(position)
        return actual == expected

    # ── Bounce classification ────────────────────────────────────────────

    def _classify_bounce(self, position) -> Tuple[str, Optional[str]]:
        """
        Decide whether a CatBoost-detected bounce is BOUNCE, OUT, or
        SERVE_FAULT.

        Returns:
            (event_type, out_reason)   – *out_reason* is None for BOUNCE.
        """
        in_court = self._is_in_court(position)
        is_serve = (self._serve_context is not None
                    and self._serve_context.get("is_serve", False))

        if not in_court:
            if is_serve:
                return EVENT_SERVE_FAULT, "out"
            return EVENT_OUT, "out_of_bounds"

        # In-court during a serve → check correct service box
        if is_serve and not self._is_valid_serve_landing(position):
            return EVENT_SERVE_FAULT, "wrong_box"

        return EVENT_BOUNCE, None

    # ── Net-hit detection (via NetDetector) ──────────────────────────────

    def _get_player_boxes(self) -> List[List[int]]:
        """Collect bounding boxes from *player_tracker* for net masking."""
        if self.player_tracker is None:
            return []
        boxes: List[List[int]] = []
        for player in getattr(self.player_tracker, "active_players", []):
            _, _, box = player.current_position()
            if box is not None:
                boxes.append(list(box))
        return boxes

    def _detect_net_hit(self, frame: np.ndarray) -> Optional[Dict]:
        """Run the net detector on *frame* and return an event dict or None."""
        if self.net_detector is None:
            return None
        # Skip if net region has not been calibrated yet to avoid a crash
        # inside NetDetector._compute_adaptive_threshold.  The fallback
        # region is set lazily inside NetDetector.detect() on the very
        # first call, so subsequent frames will work fine.
        player_boxes = self._get_player_boxes()
        net_event = self.net_detector.detect(frame, player_boxes=player_boxes)
        if net_event is None:
            return None

        position = None
        if self.net_detector.net_region is not None:
            position = list(self.net_detector.net_region.center)

        event = {
            "frame": self.frame_count,
            "type": EVENT_NET,
            "position": position,
            "confidence": net_event.confidence,
        }
        self.events.setdefault(self.frame_count, []).append(event)
        self.last_event = self.frame_count
        logger.info(f"NET detected at frame {self.frame_count}, "
                     f"confidence {net_event.confidence:.2f}")
        return event

    # ── Main per-frame entry point ───────────────────────────────────────

    def update(self, frame=None) -> List[Dict]:
        """
        Detect events for the current frame.

        Args:
            frame: Current BGR frame (needed for NET detection).  May be
                   *None* if only CatBoost-based detection is desired.

        Returns:
            List of event dicts, each with at least *frame*, *type*, *position*.
        """
        self.frame_count += 1
        new_events: List[Dict] = []

        # 1. CatBoost HIT / BOUNCE / OUT / SERVE_FAULT
        if self.last_event + 7 <= self.frame_count:
            new_events.extend(self._detect_catboost_events())

        # 2. Net-hit detection (independent cooldown inside NetDetector)
        if frame is not None:
            net_evt = self._detect_net_hit(frame)
            if net_evt is not None:
                new_events.append(net_evt)

        return new_events

    def _detect_catboost_events(self) -> List[Dict]:
        """Run the CatBoost model and classify HIT / BOUNCE / OUT / SERVE_FAULT."""
        history_positions = self.tracker.get_ball_history()
        x_ball = [pos[0] if pos else None for pos in history_positions]
        y_ball = [pos[1] if pos else None for pos in history_positions]

        detected_raw = self._detect_bounce_hit(x_ball, y_ball)

        new_events: List[Dict] = []
        for frame_num, evt_type in detected_raw.items():
            if 0 <= frame_num < len(history_positions):
                pos = history_positions[frame_num]
                if pos:
                    final_type = evt_type
                    out_reason = None

                    if evt_type == "BOUNCE":
                        final_type, out_reason = self._classify_bounce(pos)

                    event: Dict = {
                        "frame": frame_num,
                        "type": final_type,
                        "position": pos,
                    }
                    if out_reason:
                        event["out_reason"] = out_reason

                    self.events.setdefault(frame_num, []).append(event)
                    new_events.append(event)
                    self.last_event = self.frame_count
                    logger.info(f"{final_type} detected at frame {frame_num}, "
                                f"position {pos}")
        return new_events



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

        per_sample_max = np.max(probs, axis=1)
        if np.max(per_sample_max) < self.threshold:
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

        all_events = [event for event_list in self.events.values()
                      for event in event_list]
        all_events.sort(key=lambda e: e['frame'])
        last_10_events = all_events[-10:]

        for event in last_10_events:
            pos = event.get("position")
            if not pos:
                continue
            pos = tuple(int(v) for v in pos[:2])
            evt_type = event["type"]
            color = EVENT_COLORS.get(evt_type, (255, 255, 255))

            if evt_type == EVENT_BOUNCE:
                cv2.circle(result_frame, pos, 8, color, -1)
            elif evt_type == EVENT_HIT:
                cv2.circle(result_frame, pos, 8, color, -1)
            elif evt_type == EVENT_OUT:
                # Red X marker
                cv2.drawMarker(result_frame, pos, color,
                               cv2.MARKER_TILTED_CROSS, 16, 2)
            elif evt_type == EVENT_NET:
                cv2.drawMarker(result_frame, pos, color,
                               cv2.MARKER_DIAMOND, 14, 2)
            elif evt_type == EVENT_SERVE_FAULT:
                cv2.drawMarker(result_frame, pos, color,
                               cv2.MARKER_TILTED_CROSS, 16, 2)
                cv2.circle(result_frame, pos, 12, color, 2)
            else:
                cv2.circle(result_frame, pos, 6, color, -1)

        return result_frame