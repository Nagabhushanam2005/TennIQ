import logging
import cv2
import numpy as np
import math
from collections import deque
from typing import Optional, List, Tuple, Dict, Any
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_distance(box1_center, box2_center):
    return math.sqrt((box1_center[0] - box2_center[0])**2 + (box1_center[1] - box2_center[1])**2)

def get_box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


class Player:
    """Tracks a single player's trajectory and movement patterns (renamed from PlayerTracker in gen_player_data.py)."""

    def __init__(self, initial_position: Tuple[int, int], player_id: int, initial_box: Tuple[int, int, int, int], initial_confidence: float, history_length: Optional[int] = None, exponential_prediction: float = 1.0):
        self.player_id = player_id
        self.position_history = deque([initial_position], maxlen=history_length)
        self.box_history = deque([initial_box], maxlen=history_length)
        self.confidence_history = deque([initial_confidence], maxlen=history_length)
        self.velocity_history = deque(maxlen=history_length-1 if history_length else None)
        self.frames_lost = 0
        self.total_movement = 0.0
        self.is_active = True
        self.exponential_prediction_weight = exponential_prediction 
        self.color = self._get_unique_color(player_id)

    def _get_unique_color(self, player_id):
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        return colors[player_id % len(colors)]

    def update_position(self, new_position: Tuple[int, int], confidence: float, box: Tuple[int, int, int, int], frame_index: int):
        if len(self.position_history) > 0:
            velocity = (new_position[0] - self.position_history[-1][0], 
                        new_position[1] - self.position_history[-1][1])
            self.velocity_history.append(velocity)
            
            movement = math.sqrt(velocity[0]**2 + velocity[1]**2)
            self.total_movement += movement
            
        self.position_history.append(new_position)
        self.confidence_history.append(confidence)
        self.box_history.append(box)
        self.frames_lost = 0

    def current_position(self) -> Tuple[Optional[Tuple[int, int]], Optional[float], Optional[Tuple[int, int, int, int]]]:
        """Return the last known position, confidence, and box of the player."""
        return (self.position_history[-1] if self.position_history else None), \
                (self.confidence_history[-1] if self.confidence_history else None), \
                (self.box_history[-1] if self.box_history else None)

    def predict_next_position(self) -> Tuple[float, float]:
        """Predict next position based on velocity history using exponential weighting."""
        if not self.position_history:
            return (0.0, 0.0)

        if not self.velocity_history:
            return (float(self.position_history[-1][0]), float(self.position_history[-1][1]))

        vel_hist = list(self.velocity_history)

        window = vel_hist[-50:] # Use last 50 velocities
        n = len(window)
        
        if self.exponential_prediction_weight > 0 and n > 0:
            weights = np.exp(np.linspace(-self.exponential_prediction_weight, 0, n))
            weights /= weights.sum()
            avg_velocity = np.average(window, axis=0, weights=weights)
        else:
            avg_velocity = window[-1] if n > 0 else (0, 0)


        predicted_pos = (
            self.position_history[-1][0] + avg_velocity[0],
            self.position_history[-1][1] + avg_velocity[1]
        )
        return predicted_pos
        
    def get_average_velocity(self):
        if len(self.velocity_history) == 0:
            return 0.0
        velocities = [math.sqrt(v[0]**2 + v[1]**2) for v in self.velocity_history]
        return sum(velocities) / len(velocities)
        
    def get_movement_consistency(self):
        if len(self.velocity_history) < 2:
            return 0.5
            
        angle_changes = []
        for i in range(1, len(self.velocity_history)):
            v1 = self.velocity_history[i-1]
            v2 = self.velocity_history[i]
            
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle)) 
                angle_changes.append(abs(math.acos(cos_angle)))
                
        if not angle_changes:
            return 0.5
            
        avg_angle_change = sum(angle_changes) / len(angle_changes)
        consistency = max(0, 1 - (avg_angle_change / math.pi))
        return consistency
        
    def is_likely_player(self, min_avg_velocity=2.0, min_movement_per_frame=10.0):
        """Determine if this tracker represents an active tennis player."""
        avg_vel = self.get_average_velocity()
        movement_per_frame = self.total_movement / max(1, len(self.position_history))
        
        return (avg_vel >= min_avg_velocity or 
                movement_per_frame >= min_movement_per_frame or
                self.get_movement_consistency() > 0.3)


class PlayerTracker:

    def __init__(self, model_path: str = 'yolo11n.pt', max_distance: int = 25, max_lost_frames: int = 10, exponential_prediction: float = 1.0):
        try:
            # Check for GPU availability
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO(model_path).to(device)
            logger.info(f"Loaded YOLO model from '{model_path}' on device: {device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

        self.model_path = model_path
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames
        self.exponential_prediction = exponential_prediction
        self.next_player_id = 0
        self.player_trackers: List[Player] = []
        self.active_players: List[Player] = [] 
        self.frame_count = 0
        self.calibration_done = False
        self.calibration_max_frames = 30 # Default, overridden by inference_main config
        self.max_players = 2

    def update(self, frame: np.ndarray, court_warp_matrix: Optional[np.ndarray] = None) -> List[Player]:
        """
        Processes a single frame for player detection and updates player trajectories.
        """
        if self.model is None:
            return []
            
        self.frame_count += 1
        
        # 1. Detection
        results = self.model.predict(frame, verbose=False, classes=[0], save=False) 
        
        person_detections = []
        if results and results[0].boxes.data.shape[0] > 0:
            boxes_data = results[0].boxes.data.cpu().numpy()
            
            for *box, conf, cls in boxes_data:
                if int(cls) == 0:
                    x1, y1, x2, y2 = map(int, box)
                    center = get_box_center((x1, y1, x2, y2))
                    person_detections.append((float(conf), (x1, y1, x2, y2), center))
        
        # 2. Tracking Logic
        
        if not self.calibration_done:
            self._calibration_update(person_detections)
            return []
        else:
            self._main_tracking_update(frame, person_detections)
            
            # TODO: Convert to court coordinates using court_warp_matrix if required
            # For now, just return the list of active players
            return self.active_players

    def _calibration_update(self, person_detections: List[Tuple[float, Tuple[int, int, int, int], Tuple[int, int]]]):
        used_detections = set()
        
        for tracker in self.player_trackers:
            best_detection = None
            min_dist = self.max_distance + 1
            best_idx = -1
            predicted_pos = tracker.predict_next_position() # Use prediction for matching
            
            for idx, (conf, box, center) in enumerate(person_detections):
                if idx in used_detections:
                    continue

                dist_to_current = calculate_distance(tracker.position_history[-1], center)
                dist_to_predicted = calculate_distance(predicted_pos, center)
                dist = min(dist_to_current, dist_to_predicted)

                if dist < min_dist:
                    min_dist = dist
                    best_detection = (conf, box, center)
                    best_idx = idx

            if best_detection is not None and min_dist <= self.max_distance:
                conf, box, center = best_detection
                tracker.update_position(center, conf, box, self.frame_count)
                used_detections.add(best_idx)
            else:
                tracker.frames_lost += 1

        for idx, (conf, box, center) in enumerate(person_detections):
            if idx not in used_detections:
                new_tracker = Player(center, self.next_player_id, box, conf, exponential_prediction=self.exponential_prediction)
                self.player_trackers.append(new_tracker)
                self.next_player_id += 1

        if self.frame_count >= self.calibration_max_frames:
            self._finalize_calibration()

    def _finalize_calibration(self):
        active_calib_trackers = [t for t in self.player_trackers if t.frames_lost < self.max_lost_frames]

        if not active_calib_trackers:
            logger.warning("No consistent persons tracked during calibration.")
            self.calibration_done = True
            return
        active_calib_trackers.sort(key=lambda t: t.total_movement, reverse=True)
        self.active_players = active_calib_trackers[:self.max_players]
        
        if self.active_players:
            logger.info(f"Calibration finalized. Selected {len(self.active_players)} active player(s).")
            for i, tracker in enumerate(self.active_players):
                logger.info(f"  Player {i+1}: ID {tracker.player_id}, Total Movement: {tracker.total_movement:.2f}")
        else:
            logger.warning("Calibration: No players selected despite detections.")
            
        self.calibration_done = True
    def _main_tracking_update(self, frame: np.ndarray, person_detections: List[Tuple[float, Tuple[int, int, int, int], Tuple[int, int]]]):

        if not self.active_players:
            if person_detections and self.frame_count % 30 == 0:
                person_detections.sort(reverse=True, key=lambda x: x[0])
                for i, (conf, box, center) in enumerate(person_detections[:self.max_players]):
                    new_player = Player(center, self.next_player_id, box, conf, exponential_prediction=self.exponential_prediction)
                    self.active_players.append(new_player)
                    self.next_player_id += 1
                    logger.info(f"Re-initialized player {i+1} with high-confidence detection.")
            return

        used_detections = set()
        new_active_players = []

        for tracker in self.active_players:
            predicted_pos = tracker.predict_next_position()
            best_detection = None
            best_score = -1
            best_idx = -1
            
            for idx, (confidence, box, center) in enumerate(person_detections):
                if idx in used_detections:
                    continue
                    
                distance_to_current = calculate_distance(tracker.position_history[-1], center)
                distance_to_predicted = calculate_distance(predicted_pos, center)
                effective_distance = min(distance_to_current, distance_to_predicted)
                
                if effective_distance <= self.max_distance:

                    confidence_score = confidence * 0.3
                    distance_score = (1 - effective_distance / self.max_distance) * 0.2
                    prediction_score = (1 - distance_to_predicted / self.max_distance) * 0.2
                    
                    movement_velocity = calculate_distance((0, 0), 
                        (center[0] - tracker.position_history[-1][0], 
                         center[1] - tracker.position_history[-1][1]))
                    movement_score = min(1.0, movement_velocity / 15.0) * 0.2
                    consistency_score = tracker.get_movement_consistency() * 0.1
                    
                    combined_score = (confidence_score + distance_score + 
                                    prediction_score + movement_score + consistency_score)

                    if combined_score > best_score:
                        best_score = combined_score
                        best_detection = (confidence, box, center)
                        best_idx = idx
            
            if best_detection is not None:
                conf, box, center = best_detection
                tracker.update_position(center, conf, box, self.frame_count)
                used_detections.add(best_idx)
                new_active_players.append(tracker)
            else:
                tracker.frames_lost += 1
                if tracker.frames_lost <= self.max_lost_frames:
                    predicted_pos = tracker.predict_next_position()
                    last_pos, last_conf, last_box = tracker.current_position()
                    if last_pos:
                        tracker.update_position(predicted_pos, last_conf, last_box, self.frame_count)
                        new_active_players.append(tracker)
                        logger.debug(f"Player {tracker.player_id} tracked via prediction in frame {self.frame_count}")
                    else:
                         logger.debug(f"Player {tracker.player_id} lost and no history in frame {self.frame_count}")

                logger.debug(f"Player {tracker.player_id} lost for {tracker.frames_lost}/{self.max_lost_frames} frames.")

        self.active_players = [t for t in new_active_players if t.frames_lost <= self.max_lost_frames]

        if len(self.active_players) < self.max_players:
            self._try_to_reintroduce_player(person_detections, used_detections)


    def _try_to_reintroduce_player(self, person_detections, used_detections):
        remaining_detections = [(conf, box, center) for idx, (conf, box, center) in enumerate(person_detections) 
                                if idx not in used_detections]
        
        if remaining_detections:
            remaining_detections.sort(reverse=True, key=lambda x: x[0])
            
            for conf, box, center in remaining_detections:
                if len(self.active_players) >= self.max_players:
                    break
                is_far_enough = True
                if conf < 0.7: continue

                for tracker in self.active_players:
                    if calculate_distance(center, tracker.position_history[-1]) < 50:
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    new_tracker = Player(center, self.next_player_id, box, conf, exponential_prediction=self.exponential_prediction)
                    self.active_players.append(new_tracker)
                    self.next_player_id += 1
                    logger.info(f"Re-introduced Player {self.next_player_id-1} due to slot availability and high confidence.")


    def draw_players(self, frame: np.ndarray) -> np.ndarray:
        """Draw player bounding boxes and trajectory trail on the frame."""
        result_frame = frame.copy()

        trackers_to_draw = self.active_players if self.calibration_done else self.player_trackers

        for i, tracker in enumerate(trackers_to_draw):
            if tracker.frames_lost == 0 or (self.calibration_done and tracker.frames_lost <= self.max_lost_frames):
                # Get the last known or predicted position/box
                _, confidence, box = tracker.current_position()
                if box is None: continue
                
                x1, y1, x2, y2 = box
                color = tracker.color
                
                # Draw bounding box
                thickness = 2
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)

                # Draw trajectory trail
                if len(tracker.position_history) > 1:
                    points = list(tracker.position_history)[-30:] 
                    for j in range(1, len(points)):
                        thickness_trail = 1
                        cv2.line(result_frame, 
                                (int(points[j-1][0]), int(points[j-1][1])), 
                                (int(points[j][0]), int(points[j][1])), 
                                color, thickness_trail)

                avg_vel = tracker.get_average_velocity()
                if self.calibration_done:
                    player_label = next((f"Player {j+1}" for j, p in enumerate(self.active_players) if p.player_id == tracker.player_id), "Unknown")
                    label = f"{player_label}: {confidence:.2f} (v:{avg_vel:.1f})"
                else:
                    label = f"T{tracker.player_id}: {confidence:.2f}"
                    
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20

                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                cv2.rectangle(result_frame, (text_x, text_y - text_size[1] - 5), 
                            (text_x + text_size[0] + 5, text_y + 5), color, -1)
                cv2.putText(result_frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                
        return result_frame

    def get_player_positions(self) -> Dict[int, Tuple[int, int]]:
        """Get current positions of all confirmed active players (Player 1 and 2)."""
        positions = {}
        for i, tracker in enumerate(self.active_players):
            if tracker.frames_lost <= self.max_lost_frames:
                pos, _, _ = tracker.current_position()
                if pos:
                    positions[i + 1] = pos 
        return positions
