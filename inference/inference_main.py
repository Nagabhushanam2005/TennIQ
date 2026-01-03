import argparse
import cv2
import numpy as np
import os
import sys
import glob
from typing import Dict, List, Optional, Tuple
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.src.court_detector import CourtDetector
from inference.src.court_reference import CourtReference

from inference.src.player_tracker import PlayerTracker 
from inference.src.ball_tracker import BallTracker
from inference.src.event_detection import EventDetector
from inference.src.scoreboard import Scoreboard

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class TennisAnalyzer:
    def __init__(self, config_path: str = None):
        """
        Initialize tennis analyzer with all tracking components

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}

        self.enable_ball_tracking = self.config.get("ENABLE_BALL_TRACKING", True)
        self.enable_player_tracking = self.config.get("ENABLE_PLAYER_TRACKING", True)
        self.enable_court_tracking = self.config.get("ENABLE_COURT_TRACKING", True)
        self.calib_frames = self.config.get("CALIB_FRAMES", 10)
        self.court_homography_path = self.config.get("COURT_HOMOGRAPHY_PATH", "court_homography_matrices.npz")
        
        self.player_max_distance = self.config.get("PLAYER_MAX_DISTANCE", 25)
        self.player_max_lost_frames = self.config.get("PLAYER_MAX_LOST_FRAMES", 10)
        self.player_exp_pred = self.config.get("PLAYER_EXPONENTIAL_PREDICTION", 1.0)
        self.player_model_path = self.config.get("PLAYER_MODEL_PATH", 'yolo11n.pt')
        self.event_model_path = self.config.get("BOUNCE_MODEL_PATH")
        self.enable_scoreboard = self.config.get("ENABLE_SCOREBOARD", True)

        self.enable_event_detection = True

        self.court_warp_matrix: Optional[np.ndarray] = None
        self.game_warp_matrix: Optional[np.ndarray] = None
        self.court_lines_frame_coords: Optional[np.ndarray] = None

        if self.enable_player_tracking:
            self.player_tracker = PlayerTracker(
                model_path=self.player_model_path,
                max_distance=self.player_max_distance,
                max_lost_frames=self.player_max_lost_frames,
                exponential_prediction=self.player_exp_pred
            )
        else:
            self.player_tracker = None

        if self.enable_ball_tracking:
            self.ball_tracker = BallTracker()
            ball_model_weights = self.config.get("BALL_MODEL_WEIGHTS")
            ball_model_name = self.config.get("BALL_MODEL_NAME", "TrackNetV4_TypeA")
            if ball_model_weights:
                ball_config = {
                    "BALL_MODEL_WEIGHTS": ball_model_weights,
                    "BALL_MODEL_NAME": ball_model_name
                }
                self.ball_tracker.update_config(ball_config)
        else:
            self.ball_tracker = None

        if self.enable_court_tracking:
            self.court_detector = CourtDetector(verbose=0)
        else:
            self.court_detector = None

        if self.enable_event_detection:
            self.event_detector = EventDetector(
                self.ball_tracker, 
                event_model_path=self.event_model_path
            )
        else:
            self.event_detector = None

        if self.enable_scoreboard:
            self.scoreboard = Scoreboard(
                frame_width=1280,
                frame_height=720,
                enable_auto_scoring=self.config.get("AUTO_SCORING", True),
                rally_timeout_frames=self.config.get("RALLY_TIMEOUT_FRAMES", 50)
            )
        else:
            self.scoreboard = None


        self.frame_count = 0
        self.fps = self.config.get("FPS", 30)
        self.prev_frame = None

        # Output
        self.show_display = self.config.get("SHOW_DISPLAY", True)
        self.save_output = self.config.get("SAVE_OUTPUT", False)
        self.output_path = self.config.get("OUTPUT_PATH", "/dev/null")
        self.out_writer: Optional[cv2.VideoWriter] = None
        self.output_dir: Optional[str] = None

        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        self.tracking_futures = {}
        self.result_lock = threading.Lock()
        
        self.presentation_fps = 24
        self.presentation_results = deque()
        self.last_presentation_time = 0
        self.frame_interval = 1.0 / self.presentation_fps

    def _load_config(self, config_path: str) -> Dict:
        """Load TennIQ configuration for inference from file"""
        config = {}
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            if "=" in line:
                                key, value = line.split("=", 1)
                                key = key.strip()
                                value = value.strip().strip('"')

                                # Convert boolean strings
                                if value.lower() in ["true", "false"]:
                                    config[key] = value.lower() == "true"
                                # Convert numeric strings (allows float for exp_pred)
                                elif value.replace(".", "").replace("-", "").isdigit():
                                    if "." in value:
                                        config[key] = float(value)
                                    else:
                                        config[key] = int(value)
                                else:
                                    config[key] = value

                logger.info(f"Loaded config from {config_path}")
            else:
                logger.warning(f"Config file not found: {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        return config

    def set_presentation_fps(self, fps: int):
        self.presentation_fps = fps
        self.frame_interval = 1.0 / fps
        logger.info(f"Presentation FPS set to {fps}, frame interval: {self.frame_interval:.3f}s")

    def presentation(self):
        if self.presentation_results:
            result_frame = self.presentation_results[-1]
            if self.out_writer is not None:
                try:
                    self.out_writer.write(result_frame)
                except Exception:
                    logger.exception("Failed to write frame to output writer")
                    return False
            max_results = self.presentation_fps * 2
            while len(self.presentation_results) > max_results:
                self.presentation_results.popleft()
        return True

    def _calibrate_court(self, cap: cv2.VideoCapture, total_frames: int) -> None:
        calib_frames = min(self.calib_frames, total_frames)
        if self.enable_player_tracking and self.player_tracker:
            self.player_tracker.calibration_max_frames = calib_frames
            logger.info(f"Setting player tracker calibration frames to {calib_frames}")

        if not self.enable_court_tracking or not self.court_detector:
            logger.info("Court tracking disabled. Skipping court calibration.")
            
            if self.enable_player_tracking and self.player_tracker:
                self._run_player_calibration(cap, calib_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 0 
            return

        logger.info(f'Starting court and player calibration on the first {calib_frames} frames...')
        
        court_calibrated = False
        if os.path.exists(self.court_homography_path):
            try:
                data = np.load(self.court_homography_path)
                self.court_warp_matrix = data['court_warp_matrix']
                self.game_warp_matrix = data['game_warp_matrix']
                
                ret, frame = cap.read()
                if ret:
                    court_ref = self.court_detector.court_reference
                    p = np.array(court_ref.get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
                    self.court_lines_frame_coords = cv2.perspectiveTransform(p, self.court_warp_matrix).reshape(-1)
                    
                    if self.enable_player_tracking and self.player_tracker:
                        self.player_tracker.update(frame) 
                
                logger.info(f"Loaded court homography from {self.court_homography_path}. Skipping court calibration.")
                court_calibrated = True
            except Exception as e:
                logger.warning(f"Failed to load homography from {self.court_homography_path}: {e}. Recalibrating.")

        
        if not court_calibrated:
            successful_detections = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            for frame_i in range(1, calib_frames + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_i == 1:
                    lines = self.court_detector.detect(frame)
                else:
                    lines = self.court_detector.track_court(frame)

                if self.court_detector.success_flag and lines is not None:
                    successful_detections += 1
                    self.court_lines_frame_coords = lines
                    self.court_warp_matrix = self.court_detector.court_warp_matrix[-1]
                    self.game_warp_matrix = self.court_detector.game_warp_matrix[-1]

                if self.enable_player_tracking and self.player_tracker:
                    self.player_tracker.update(frame)


                logger.info(f'Calibration Frame: {frame_i}/{calib_frames} (Court Successes: {successful_detections})')
            
            # Finalize court calibration result and save homography
            if self.court_warp_matrix is not None:
                logger.info("\nCourt calibration finalized. Saving homography.")
                try:
                    np.savez(self.court_homography_path, 
                            court_warp_matrix=self.court_warp_matrix, 
                            game_warp_matrix=self.game_warp_matrix,
                            best_conf=self.court_detector.best_conf)
                    logger.info(f"Final homography matrices saved to: {self.court_homography_path}")
                except Exception as e:
                    logger.error(f"Failed to save homography matrices: {e}")
            else:
                logger.error("\nCourt calibration failed on all frames.")
                self.enable_court_tracking = False
                self.enable_event_detection = False
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0

    
    def build_court_minimap(self, width_minimap: int = 200, height_minimap: int = 120) -> np.ndarray:
        """Build a small stylized court minimap image."""
        court_reference = CourtReference()
        court = court_reference.build_court_reference()
        court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
        court_img = (np.stack((court, court, court), axis=2) * 255).astype(np.uint8)
        minimap = cv2.resize(court_img, (width_minimap, height_minimap))
        return minimap

    def _update_player_tracking(self, frame: np.ndarray) -> None:
        """Update player tracking in parallel"""
        if self.enable_player_tracking and self.player_tracker:
            # start_time = time.time()
            # print(time.time() - start_time)
            self.player_tracker.update(frame)

    def _update_ball_tracking(self, frames: List[np.ndarray]) -> None:
        """Update ball tracking in parallel"""
        if self.enable_ball_tracking and self.ball_tracker:
            self.ball_tracker.update(frames)

    def _update_event_detection(self) -> None:
        """Update event detection in parallel"""
        if self.enable_event_detection and self.event_detector and self.ball_tracker:
            self.event_detector.update()

    def _draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """Draw all tracking annotations on frame"""
        result_frame = frame

        # 1. Mark ball position
        if self.enable_ball_tracking and self.ball_tracker:
            result_frame = self.ball_tracker.draw_ball(result_frame)
            
        # 2. Mark court lines (New Integration)
        if self.enable_court_tracking and self.court_lines_frame_coords is not None:
            lines = self.court_lines_frame_coords
            for i in range(0, len(lines), 4):
                x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                cv2.line(result_frame, (int(x1),int(y1)),(int(x2),int(y2)), (0, 255, 255), 2)

        # 3. Mark players
        if self.enable_player_tracking and self.player_tracker:
            result_frame = self.player_tracker.draw_players(result_frame)

        # 4. Mark events
        if self.enable_event_detection and self.event_detector:
            result_frame = self.event_detector.draw_events(result_frame)
        
        # 5. Draw scoreboard
        if self.enable_scoreboard and self.scoreboard:
            player_positions = None
            if self.enable_player_tracking and self.player_tracker:
                player_positions = self.player_tracker.get_player_positions()
            result_frame = self.scoreboard.draw_scoreboard(result_frame, player_positions)

        return result_frame



    def _analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Analyze single frame with all tracking components in parallel

        Args:
            frame: Input video frame

        Returns:
            Annotated frame with analysis results
        """
        frame_id = self.frame_count
        futures = []
        
        # Player tracking
        if self.enable_player_tracking:
            # player_future = self.thread_pool.submit(self._update_player_tracking, frame)
            # futures.append(player_future)
            self.player_tracker.update(frame)
        # Ball tracking
        if self.enable_ball_tracking:
            ball_future = self.thread_pool.submit(self._update_ball_tracking, frame)
            futures.append(ball_future)
        
        # Event detection
        new_events = []
        if self.enable_event_detection:
            # event_future = self.thread_pool.submit(self._update_event_detection)
            # futures.append(event_future)
            new_events = self.event_detector.update()
        
        for future in futures:
            future.result()
        
        # Update scoreboard
        if self.enable_scoreboard and self.scoreboard:
            ball_pos = None
            if self.enable_ball_tracking and self.ball_tracker:
                ball_history = self.ball_tracker.get_ball_history()
                if ball_history:
                    ball_pos = ball_history[-1]
            
            player_positions = None
            if self.enable_player_tracking and self.player_tracker:
                player_positions = self.player_tracker.get_player_positions()
            
            # Update court bounds if available
            if self.court_lines_frame_coords is not None and self.scoreboard.court_bounds is None:
                self.scoreboard.set_court_bounds(self.court_lines_frame_coords.reshape(-1, 2))
            
            # TODO
            self.scoreboard.set_court_bounds(None)

            self.scoreboard.update(
                ball_position=ball_pos,
                player_positions=player_positions,
                events=new_events
            )
        
        result_frame = self._draw_annotations(frame)

        result_frame = self._draw_info_overlay(result_frame)

        return result_frame

    def _draw_minimap(self, frame: np.ndarray) -> np.ndarray:
        """Draw court minimap on frame"""
        result_frame = frame

        # Draw court minimap with players and ball to the right side
        frame_h, frame_w = result_frame.shape[0], result_frame.shape[1]
        # minimap width as fraction of frame width
        width_minimap = min(400, int(frame_w * 0.28))
        height_minimap = frame_h

        minimap = self.build_court_minimap(width_minimap, height_minimap)

        player_positions = {}
        if self.enable_player_tracking and self.player_tracker:
            try:
                player_positions = self.player_tracker.get_player_positions()
            except Exception:
                player_positions = {}

        bounce_positions = {}
        if self.enable_event_detection and self.event_detector:
            try:
                bounce_positions = self.event_detector.get_bounce_positions()
            except Exception:
                bounce_positions = {}

        inv_mat = None
        if self.game_warp_matrix is not None:
            try:
                inv_mat = np.linalg.inv(self.game_warp_matrix)
            except Exception:
                inv_mat = None

        for pid, pos in player_positions.items():
            point = np.array([[ [float(pos[0]), float(pos[1])] ]], dtype=np.float32)
            mapped = None
            if inv_mat is not None:
                try:
                    mapped = cv2.perspectiveTransform(point, inv_mat)
                    mx = int(mapped[0, 0, 0])
                    my = int(mapped[0, 0, 1])
                    court_ref = CourtReference().build_court_reference()
                    court_h, court_w = court_ref.shape
                    if court_w > 0 and court_h > 0:
                        scale_x = width_minimap / court_w
                        scale_y = height_minimap / court_h
                        draw_x = int(mx * scale_x)
                        draw_y = int(my * scale_y)
                    else:
                        draw_x, draw_y = int(width_minimap * 0.5), int(height_minimap * 0.5)
                except Exception:
                    mapped = None

            if inv_mat is None or mapped is None:
                draw_x = int((pos[0] / max(1, frame_w)) * width_minimap)
                draw_y = int((pos[1] / max(1, frame_h)) * height_minimap)

            # choose color for player marker
            color = (0, 0, 255) if pid == 1 else (255, 0, 0)
            cv2.circle(minimap, (int(np.clip(draw_x, 0, width_minimap-1)), int(np.clip(draw_y, 0, height_minimap-1))),
                        radius=8, color=color, thickness=-1)
            cv2.putText(minimap, f'P{pid}', (int(np.clip(draw_x+10, 0, width_minimap-1)), int(np.clip(draw_y+10, 0, height_minimap-1))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        for frame_idx, bounce_pos in bounce_positions.items():
            # bounce_pos is in frame coords; map similar to players
            point_b = np.array([[ [float(bounce_pos[0]), float(bounce_pos[1])] ]], dtype=np.float32)
            mapped_b = None
            if inv_mat is not None:
                try:
                    mapped_b = cv2.perspectiveTransform(point_b, inv_mat)
                    bx = int(mapped_b[0, 0, 0])
                    by = int(mapped_b[0, 0, 1])
                    court_ref = CourtReference().build_court_reference()
                    court_h, court_w = court_ref.shape
                    if court_w > 0 and court_h > 0:
                        scale_x = width_minimap / court_w
                        scale_y = height_minimap / court_h
                        draw_bx = int(bx * scale_x)
                        draw_by = int(by * scale_y)
                    else:
                        draw_bx, draw_by = int(width_minimap * 0.5), int(height_minimap * 0.5)
                except Exception:
                    mapped_b = None

            if inv_mat is None or mapped_b is None:
                draw_bx = int((bounce_pos[0] / max(1, frame_w)) * width_minimap)
                draw_by = int((bounce_pos[1] / max(1, frame_h)) * height_minimap)

            cv2.circle(minimap, (int(np.clip(draw_bx, 0, width_minimap-1)), int(np.clip(draw_by, 0, height_minimap-1))),
                        radius=10, color=(0, 255, 255), thickness=-1)

        if minimap.shape[0] != frame_h:
            minimap = cv2.resize(minimap, (width_minimap, frame_h))

        combined_w = frame_w + minimap.shape[1]
        combined_h = frame_h
        combined = np.zeros((combined_h, combined_w, 3), dtype=result_frame.dtype)
        combined[:, :frame_w] = result_frame
        combined[:, frame_w:frame_w + minimap.shape[1]] = minimap

        return combined
    
    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        result_frame = frame

        # Info panel background
        overlay = result_frame
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)

        # Frame info
        y_offset = 30
        cv2.putText(
            result_frame,
            f"Frame: {self.frame_count}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        y_offset += 25
        status_color = (0, 255, 0) if self.enable_ball_tracking else (128, 128, 128)
        cv2.putText(
            result_frame,
            f"Ball Tracking: {'ON' if self.enable_ball_tracking else 'OFF'}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            1,
        )

        y_offset += 20
        status_color = (0, 255, 0) if self.enable_player_tracking else (128, 128, 128)
        cv2.putText(
            result_frame,
            f"Player Tracking: {'ON' if self.enable_player_tracking else 'OFF'}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            1,
        )

        y_offset += 20
        status_color = (0, 255, 0) if self.enable_court_tracking else (128, 128, 128)
        cv2.putText(
            result_frame,
            f"Court Tracking: {'ON' if self.enable_court_tracking else 'OFF'}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            1,
        )

        y_offset += 20
        status_color = (0, 255, 0) if self.enable_event_detection else (128, 128, 128)
        cv2.putText(
            result_frame,
            f"Event Detection: {'ON' if self.enable_event_detection else 'OFF'}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            1,
        )
        return result_frame
    
    def _print_analysis_summary(self):
        print("\n" + "=" * 50)
        print("TENNIS ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total frames analyzed: {self.frame_count}")
        print(f"Ball tracking: {self.enable_ball_tracking}")
        print(f"Player tracking: {self.enable_player_tracking}")
        print(f"Court tracking: {self.enable_court_tracking}")
        print(f"Event detection: {self.enable_event_detection}")

        if self.enable_court_tracking and self.court_detector:
            print("Court tracking done and calibrated.")

        if self.enable_player_tracking and self.player_tracker:
            print(f"Player tracking done (Calibration: {'Completed' if self.player_tracker.calibration_done else 'Running'})")

        if self.enable_ball_tracking and self.ball_tracker:
            print(f"Ball tracking done")

        if self.enable_event_detection and self.event_detector:
            print(f"Event detection done")

        print("=" * 50)

    def analyze_image_sequence(self, image_dir: str) -> None:
        if not os.path.exists(image_dir):
            logger.error(f"Image directory not found: {image_dir}")
            return
            
        # Get image files
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))

        image_files.sort()

        if not image_files:
            logger.error(f"No image files found in {image_dir}")
            return

        logger.info(f"Found {len(image_files)} images in {image_dir}")

        if self.enable_player_tracking and self.player_tracker:
            self.player_tracker.calibration_max_frames = min(self.calib_frames, len(image_files))

        
        # Create a separate thread pool for frame analysis
        analysis_pool = ThreadPoolExecutor(max_workers=2)
        out_writer = None
        frame_queue = queue.Queue()
        start_time = time.time()
        submitted_frames = 0
        processed_frames = 0
        processed_frames_1 = 0
        fps_avg = 0.0
        
        def process_frame(frame_data):
            i, image_path, frame = frame_data
            result_frame = self._analyze_frame(frame)
            return i, result_frame

        for i, image_path in enumerate(image_files):
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            
            # first frame
            if i == 0:
                # initialize writer on first available frame
                if self.save_output and self.output_path and self.out_writer is None:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        # result_frame shape is (H,W,3)
                        out_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (frame.shape[1], frame.shape[0]))
                        self.out_writer = out_writer
                    except Exception:
                        logger.exception("Failed to create VideoWriter for image sequence")
                
                if self.enable_court_tracking and self.court_warp_matrix is None:
                    # Court detection for first frame only
                    self.court_detector.detect(frame)
                    self.court_lines_frame_coords = self.court_detector.lines
                    self.court_warp_matrix = self.court_detector.court_warp_matrix[-1] if self.court_detector.court_warp_matrix else None
                    self.game_warp_matrix = self.court_detector.game_warp_matrix[-1] if self.court_detector.game_warp_matrix else None

            # Submit frame for parallel processing
            future = analysis_pool.submit(process_frame, (i, image_path, frame))
            frame_queue.put((i, future))
            submitted_frames += 1

            if frame_queue.qsize() >= 5:
                frame_idx, future = frame_queue.get()
                _, result_frame = future.result()
                self.presentation_results.append(result_frame)
                self.frame_count = frame_idx + 1
                processed_frames += 1
                
                progress = (processed_frames / len(image_files)) * 100

                if(time.time() - start_time >= .99):
                    start_time = time.time()
                    fps_avg = processed_frames - processed_frames_1
                    processed_frames_1 = processed_frames
                print(f"Progress: {progress:.1f}% ({processed_frames}/{len(image_files)}), Avg FPS: {fps_avg:.1f}", end="\r")

                if not self.presentation():
                    break

        analysis_pool.shutdown()

        # release writer
        if self.out_writer is not None:
            self.out_writer.release()

        if self.show_display:
            cv2.destroyAllWindows()

        self._print_analysis_summary()

    def analyze_video(self, video_path: str) -> None:
        """
        Analyze tennis video with comprehensive tracking

        Args:
            video_path: Path to input video file
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return

        # VideoCapture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Video: {frame_width}x{frame_height}, {self.fps} FPS, {total_frames} frames"
        )
        
        self._calibrate_court(cap, total_frames)

        # prepare output writer if requested
        if self.save_output and self.output_path:
            try:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.out_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (frame_width, frame_height))
            except Exception:
                logger.exception("Failed to create VideoWriter for video output")

        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                
                result_frame = self._analyze_frame(frame)
                
                # Add result to presentation queue
                self.presentation_results.append(result_frame)

                # Present at specified FPS
                if not self.presentation():
                    break

                # Display frame if enabled
                if self.show_display:
                    cv2.imshow("TennIQ Analysis", result_frame)
                    
                    # Handle keyboard input for manual scoring
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit requested by user")
                        break
                    elif key == ord('1') and self.scoreboard:
                        # Award point to upper player (rear)
                        from inference.src.scoreboard import Player
                        self.scoreboard.manual_award_point(Player.UPPER)
                    elif key == ord('2') and self.scoreboard:
                        # Award point to lower player (front)
                        from inference.src.scoreboard import Player
                        self.scoreboard.manual_award_point(Player.LOWER)
                    elif key == ord('r') and self.scoreboard:
                        # Reset score
                        self.scoreboard.reset_score()
                    elif key == ord('h'):
                        # Show help
                        logger.info("Keyboard controls: 1=Point to P1(rear), 2=Point to P2(front), r=Reset score, q=Quit")

                if self.out_writer:
                    self.out_writer.write(result_frame)

                progress = (self.frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_avg = self.frame_count / elapsed
                logger.info(
                    f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames}), "
                    f"Avg FPS: {fps_avg:.1f}"
                )

                self.prev_frame = frame

        except KeyboardInterrupt:
            logger.info("Analysis interrupted by user")

        finally:
            cap.release()
            if self.out_writer:
                try:
                    self.out_writer.release()
                except Exception:
                    logger.exception("Failed to release out_writer")
            if self.show_display:
                cv2.destroyAllWindows()

            total_time = time.time() - start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            logger.info(
                f"Analysis completed: {self.frame_count} frames in {total_time:.1f}s "
                f"(avg {avg_fps:.1f} FPS)"
            )

            self._print_analysis_summary()


def main():
    parser = argparse.ArgumentParser(description="TennIQ Tennis Analysis System")
    parser.add_argument(
        "--config",
        "-c",
        default="inference/data-configs/data_config_alcaraz.txt",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input video file or image directory path"
    )
    parser.add_argument("--output", "-o", help="Optional output video file path (overrides config)")
    parser.add_argument("--no-display", action="store_true", help="Disable display window")
    parser.add_argument(
        "--mode",
        choices=["video", "images"],
        default="video",
        help="Analysis mode: video file or image sequence",
    )

    args = parser.parse_args()

    # Prepare timestamped outputs directory and logging
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", ts)
    os.makedirs(out_dir, exist_ok=True)

    output_video_path = os.path.join(out_dir, "inference.mp4")
    output_log_path = os.path.join(out_dir, "inference.log")

    fh = logging.FileHandler(output_log_path)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    root_logger = logging.getLogger()
    root_logger.addHandler(fh)
    root_logger.setLevel(logging.INFO)
    logger.info(f"Logging initialized. Writing logs to: {output_log_path}")

    # Initialize analyzer with config file (the config should contain most runtime options)
    analyzer = TennisAnalyzer(args.config)

    # Use timestamped output directory by default; allow --output to override
    analyzer.output_dir = out_dir
    analyzer.output_path = output_video_path
    analyzer.save_output = True
    analyzer.show_display = not args.no_display
    if args.output:
        analyzer.save_output = True
        analyzer.output_path = args.output
    fps_value = analyzer.config.get("PRESENTATION_FPS", analyzer.config.get("FPS", analyzer.presentation_fps))
    try:
        analyzer.set_presentation_fps(int(fps_value))
    except Exception:
        logger.warning("Invalid presentation FPS in config; using default")

    # Run
    if args.mode == "video":
        analyzer.analyze_video(args.input)
    else:
        analyzer.analyze_image_sequence(args.input)


if __name__ == "__main__":
    main()