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
        
        self.player_max_distance = self.config.get("PLAYER_MAX_DISTANCE", 25)
        self.player_max_lost_frames = self.config.get("PLAYER_MAX_LOST_FRAMES", 10)
        self.player_exp_pred = self.config.get("PLAYER_EXPONENTIAL_PREDICTION", 1.0)
        self.player_model_path = self.config.get("PLAYER_MODEL_PATH", 'yolo11n.pt')
        self.event_model_path = self.config.get("BOUNCE_MODEL_PATH")
        self.enable_scoreboard = self.config.get("ENABLE_SCOREBOARD", True)

        self.enable_event_detection = self.enable_ball_tracking and self.enable_player_tracking and self.enable_court_tracking

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

    def _calibrate_court(self, cap, total_frames):
        """
        Calibrate court detection from the first frame of a video.
        
        Args:
            cap: OpenCV VideoCapture object
            total_frames: Total number of frames in the video
        """
        if not self.enable_court_tracking or self.court_detector is None:
            return
        
        # Save current position
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Seek to first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        
        if ret:
            logger.info("Calibrating court detection...")
            # Use resized=True for faster detection during calibration
            keypoints = self.court_detector.detect(frame, use_resized=True, verbose=1)
            if keypoints is not None:
                logger.info(f"Court calibrated with {len(keypoints)} keypoints")
            else:
                logger.warning("Court detection failed during calibration")
        
        # Reset to original position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

    def _draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """Draw all tracking annotations on frame"""
        result_frame = frame

        # 1. Mark ball position
        if self.enable_ball_tracking and self.ball_tracker:
            result_frame = self.ball_tracker.draw_ball(result_frame)
            
        # 2. Mark court lines
        if self.enable_court_tracking and self.court_detector:
            result_frame = self.court_detector.draw_court_overlay(result_frame)
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
            if self.court_detector and self.court_detector.get_keypoints() is not None and self.scoreboard.court_bounds is None:
                self.scoreboard.set_court_bounds(self.court_detector.get_keypoints())

            self.scoreboard.update(
                ball_position=ball_pos,
                player_positions=player_positions,
                events=new_events
            )
        
        result_frame = self._draw_annotations(frame)

        result_frame = self._draw_info_overlay(result_frame)

        return result_frame

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
                
                if self.enable_court_tracking and self.court_detector:
                    # Court detection for first frame only
                    logger.info("Calibrating court detection from first frame...")
                    keypoints = self.court_detector.detect(frame, use_resized=True, verbose=1)
                    if keypoints is not None:
                        logger.info(f"Court calibrated with {len(keypoints)} keypoints")
                    else:
                        logger.warning("Court detection failed during calibration")

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