"""
TennIQ Main Inference Module
Comprehensive tennis analysis with player tracking, ball tracking, and event detection
"""

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

        # Initialize config
        self.enable_ball_tracking = self.config.get("ENABLE_BALL_TRACKING", True)
        self.enable_player_tracking = self.config.get("ENABLE_PLAYER_TRACKING", True)
        self.enable_court_tracking = self.config.get(
            "ENABLE_COURT_TRACKING", True
        )  # Yet to be implemented

        # Event detection only if all tracking is enabled
        self.enable_event_detection = (
            self.enable_ball_tracking
            and self.enable_player_tracking
            and self.enable_court_tracking
        )

        if self.enable_player_tracking:
            # TODO
            pass

        if self.enable_ball_tracking:
            self.ball_tracker = BallTracker()
            self.ball_tracker.update_config(self.config)
        else:
            self.ball_tracker = None

        if self.enable_court_tracking:
            # TODO
            pass

        if self.enable_event_detection:
            # TODO
            pass

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
                                # Convert numeric strings
                                elif value.replace(".", "").isdigit():
                                    config[key] = (
                                        float(value) if "." in value else int(value)
                                    )
                                else:
                                    config[key] = value

                logger.info(f"Loaded config from {config_path}")
            else:
                logger.warning(f"Config file not found: {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        return config

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

                # Perform analysis
                result_frame = self._analyze_frame(frame)

                # Display results
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

        # 1. Ball Tracking
        if self.enable_ball_tracking and self.ball_tracker:
            self.ball_tracker.update()

        result_frame = self._draw_annotations(result_frame)

        result_frame = self._draw_info_overlay(result_frame)

        return result_frame

    def _draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """Draw all tracking annotations on frame"""
        result_frame = frame.copy()

        # 1. Mark ball position
        if self.enable_ball_tracking and self.ball_tracker:
            result_frame = self.ball_tracker.draw_ball(result_frame)

        return result_frame

    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw information overlay on frame"""
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

        """
        Mark whether each tracking is ON or OFF
        """

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
            pass

        if self.enable_player_tracking and self.player_tracker:
            pass

        if self.enable_ball_tracking and self.ball_tracker:
            print(f"Ball tracking done")

        if self.enable_event_detection and self.event_detector:
            pass

        print("=" * 50)

    def analyze_image_sequence(self, image_dir: str) -> None:
        """
        Analyze sequence of images (e.g., extracted frames)

        Args:
            image_dir: Directory containing image sequence
        """
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

        # Process images sequentially
        for i, image_path in enumerate(image_files):
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            self.frame_count = i + 1
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

    args = parser.parse_args()

    # Initialize
    analyzer = TennisAnalyzer(args.config)
    analyzer.show_display = not args.no_display

    if args.output:
        analyzer.save_output = True
        analyzer.output_path = args.output

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

