import argparse
import cv2
import numpy as np
import os
import sys
import glob
from typing import Dict, List, Optional, Tuple
import logging
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.src.court_detector import CourtDetector
from inference.src.court_reference import CourtReference

from inference.src.player_tracker import PlayerTracker 
from inference.src.ball_tracker import BallTracker
from inference.src.tennis_event_detection import TennisEventDetector

logging.basicConfig(level=logging.INFO)
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
        self.calib_frames = self.config.get("CALIB_FRAMES", 30)
        self.court_homography_path = self.config.get("COURT_HOMOGRAPHY_PATH", "court_homography_matrices.npz")
        
        self.player_max_distance = self.config.get("PLAYER_MAX_DISTANCE", 25)
        self.player_max_lost_frames = self.config.get("PLAYER_MAX_LOST_FRAMES", 10)
        self.player_exp_pred = self.config.get("PLAYER_EXPONENTIAL_PREDICTION", 1.0)
        self.player_model_path = self.config.get("PLAYER_MODEL_PATH", 'yolo11n.pt')

        # Event detection only if all tracking is enabled
        self.enable_event_detection = (
            self.enable_ball_tracking
            and self.enable_player_tracking
            and self.enable_court_tracking
        )

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
            self.ball_tracker.update_config(self.config)
        else:
            self.ball_tracker = None

        if self.enable_court_tracking:
            self.court_detector = CourtDetector(verbose=0)
        else:
            self.court_detector = None

        if self.enable_event_detection:
            # TODO
            self.event_detector = TennisEventDetector()
        else:
            self.event_detector = None

        # State
        self.frame_count = 0
        self.fps = self.config.get("FPS", 30)
        self.prev_frame = None

        # Output
        self.show_display = self.config.get("SHOW_DISPLAY", True)
        self.save_output = self.config.get("SAVE_OUTPUT", False)
        self.output_path = self.config.get("OUTPUT_PATH", "/dev/null")

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

    def _calibrate_court(self, cap: cv2.VideoCapture, total_frames: int) -> None:
        """
        Calibrate the court and also perform player tracking calibration by processing the initial frames. 
        """
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


                logger.info(f'Calibration Frame: {frame_i}/{calib_frames} (Court Successes: {successful_detections})', end='\r')
            
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

        if self.enable_player_tracking and self.player_tracker and not self.player_tracker.calibration_done:
             self._run_player_calibration(cap, calib_frames)


    def _run_player_calibration(self, cap: cv2.VideoCapture, calib_frames: int):
        """Helper to run player calibration specifically."""
        logger.info(f'Starting dedicated player calibration on the first {calib_frames} frames...')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_i in range(1, calib_frames + 1):
            ret, frame = cap.read()
            if not ret:
                break
            self.player_tracker.update(frame)
            logger.info(f'Player Calibration Frame: {frame_i}/{calib_frames}', end='\r')
        logger.info("\nPlayer calibration finalized.")


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

        out_writer = None
        if self.save_output and self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (frame_width, frame_height)
            )

        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                
                result_frame = self._analyze_frame(frame)

                if self.show_display:
                    cv2.imshow("TennIQ Analysis", result_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        cv2.imwrite(f"frame_{self.frame_count:06d}.jpg", result_frame)
                        logger.info(f"Saved frame {self.frame_count}")

                if out_writer:
                    out_writer.write(result_frame)

                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    fps_avg = self.frame_count / elapsed
                    logger.info(
                        f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames}), "
                        f"Avg FPS: {fps_avg:.1f}"
                    )

                self.prev_frame = frame.copy()

        except KeyboardInterrupt:
            logger.info("Analysis interrupted by user")

        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            if self.show_display:
                cv2.destroyAllWindows()

            total_time = time.time() - start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            logger.info(
                f"Analysis completed: {self.frame_count} frames in {total_time:.1f}s "
                f"(avg {avg_fps:.1f} FPS)"
            )

            self._print_analysis_summary()


    def _analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Analyze single frame with all tracking components

        Args:
            frame: Input video frame

        Returns:
            Annotated frame with analysis results
        """
        result_frame = frame.copy()

        # 1. Player Tracking 
        if self.enable_player_tracking and self.player_tracker:
            # NOTE: PlayerTracker handles its own calibration state internally
            # It needs to know the court matrix primarily for calculating court-view positions,
            # but the raw update can happen without it. Passing it if available.
            self.player_tracker.update(frame, self.court_warp_matrix)

        # 2. Ball Tracking
        if self.enable_ball_tracking and self.ball_tracker:
            self.ball_tracker.update(frame) 

        # 3. Event Detection
        if self.enable_event_detection and self.event_detector:
            # TODO: Add event detection call here
            self.event_detector.update()

        result_frame = self._draw_annotations(result_frame)

        result_frame = self._draw_info_overlay(result_frame)

        return result_frame

    def _draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """Draw all tracking annotations on frame"""
        result_frame = frame.copy()

        # 1. Mark ball position
        if self.enable_ball_tracking and self.ball_tracker:
            result_frame = self.ball_tracker.draw_ball(result_frame)
            
        # 2. Mark court lines (New Integration)
        if self.enable_court_tracking and self.court_lines_frame_coords is not None:
            lines = self.court_lines_frame_coords
            for i in range(0, len(lines), 4):
                x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                # Draw lines in White (255, 255, 255)
                cv2.line(result_frame, (int(x1),int(y1)),(int(x2),int(y2)), (255, 255, 255), 2)
                
        # 3. Mark players
        if self.enable_player_tracking and self.player_tracker:
            result_frame = self.player_tracker.draw_players(result_frame)


        return result_frame
    
    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        result_frame = frame.copy()

        # Info panel background
        overlay = result_frame.copy()
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

        for i, image_path in enumerate(image_files):
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            self.frame_count = i + 1
            
            if self.enable_court_tracking and self.court_warp_matrix is None:
                self.court_detector.detect(frame)
                self.court_lines_frame_coords = self.court_detector.lines

                self.court_warp_matrix = self.court_detector.court_warp_matrix[-1] if self.court_detector.court_warp_matrix else None
                self.game_warp_matrix = self.court_detector.game_warp_matrix[-1] if self.court_detector.game_warp_matrix else None

            result_frame = self._analyze_frame(frame)

            if self.show_display:
                cv2.imshow("TennIQ Analysis", result_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break

            self.prev_frame = frame.copy()

        if self.show_display:
            cv2.destroyAllWindows()

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
    parser.add_argument("--output", "-o", help="Output video file path (optional)")
    parser.add_argument(
        "--no-display", action="store_true", help="Disable display window"
    )
    parser.add_argument(
        "--mode",
        choices=["video", "images"],
        default="video",
        help="Analysis mode: video file or image sequence",
    )

    parser.add_argument(
        "--calib-frames",
        type=int,
        default=None,
        help="Override the number of calibration frames specified in the config",
    )

    parser.add_argument(
        "--exp-pred",
        type=float,
        default=None,
        help="Override the exponential prediction weight for player tracking (default is 1.0)",
    )

    parser.add_argument(
        "--player-model",
        type=str,
        default=None,
        help="Path to a custom fine-tuned YOLO model (.pt) for player tracking.",
    )
    args = parser.parse_args()

    # Initialize
    analyzer = TennisAnalyzer(args.config)
    analyzer.show_display = not args.no_display

    if args.output:
        analyzer.save_output = True
        analyzer.output_path = args.output
    if args.calib_frames is not None:
        analyzer.calib_frames = args.calib_frames
    
    if args.exp_pred is not None and analyzer.player_tracker:
        analyzer.player_tracker.exponential_prediction = args.exp_pred
        logger.info(f"Overriding PlayerTracker exponential_prediction to {args.exp_pred}")

    if args.player_model is not None and analyzer.player_tracker:
        analyzer.player_model_path = args.player_model
        analyzer.player_tracker = PlayerTracker(model_path=args.player_model, max_distance=analyzer.player_max_distance, max_lost_frames=analyzer.player_max_lost_frames, exponential_prediction=analyzer.player_exp_pred)

    # Run
    try:
        if args.mode == "video":
            analyzer.analyze_video(args.input)
        else:
            analyzer.analyze_image_sequence(args.input)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()