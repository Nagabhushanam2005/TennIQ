import argparse
import cv2
import numpy as np
import os
import sys
import glob
from typing import Dict, List, Optional, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
import queue
from collections import deque

# Allow running from project root: `python inference_main.py`
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.src.court_detector import CourtDetector
from inference.src.net_detector import NetDetector
from inference.src.player_tracker import PlayerTracker
from inference.src.ball_tracker import BallTracker
from inference.src.event_detection import EventDetector
from inference.src.scoreboard import Scoreboard
from inference.src.tennis_state_machine import Player


logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# ======================================================================
#  TennisAnalyzer
# ======================================================================

class TennisAnalyzer:
    """End-to-end tennis video / image-sequence analysis pipeline.

    Component wiring
    ----------------
    CourtDetector   → supplies 16 keypoints to NetDetector, EventDetector,
                      and Scoreboard (court_bounds / net_y).
    NetDetector     → initialised from CourtDetector; called each frame
                      by EventDetector._detect_net_hit().
    BallTracker     → feeds ball_positions list to EventDetector.
    PlayerTracker   → feeds active_players boxes to EventDetector
                      (for player-area masking in NetDetector) and
                      player_positions dict to Scoreboard.
    EventDetector   → receives serve_context from Scoreboard.state_machine
                      each frame; emits structured event dicts to Scoreboard.
    Scoreboard      → drives TennisScoringStateMachine; renders overlay.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config: Dict = self._load_config(config_path) if config_path else {}

        # ── Feature flags ──────────────────────────────────────────────
        self.enable_ball_tracking   = self.config.get("ENABLE_BALL_TRACKING",   True)
        self.enable_player_tracking = self.config.get("ENABLE_PLAYER_TRACKING", True)
        self.enable_court_tracking  = self.config.get("ENABLE_COURT_TRACKING",  True)
        self.enable_scoreboard      = self.config.get("ENABLE_SCOREBOARD",      True)

        # Event detection requires all three tracking subsystems
        self.enable_event_detection = (
            self.enable_ball_tracking
            and self.enable_player_tracking
            and self.enable_court_tracking
        )

        # ── Per-component config ───────────────────────────────────────
        self.calib_frames           = int(self.config.get("CALIB_FRAMES",                   10))
        self.player_max_distance    = int(self.config.get("PLAYER_MAX_DISTANCE",            25))
        self.player_max_lost_frames = int(self.config.get("PLAYER_MAX_LOST_FRAMES",         10))
        self.player_exp_pred        = float(self.config.get("PLAYER_EXPONENTIAL_PREDICTION", 1.0))
        self.player_model_path      = self.config.get("PLAYER_MODEL_PATH", "yolo11n.pt")
        self.event_model_path       = self.config.get("BOUNCE_MODEL_PATH")

        # ── Build components ───────────────────────────────────────────
        self.player_tracker  = self._build_player_tracker()
        self.ball_tracker    = self._build_ball_tracker()
        self.court_detector  = CourtDetector(verbose=1) if self.enable_court_tracking else None

        # NetDetector: only needed when event detection is active
        self.net_detector: Optional[NetDetector] = None
        if self.enable_event_detection:
            self.net_detector = NetDetector(
                fps=float(self.config.get("FPS", 30)),
                verbose=0,
            )

        # EventDetector: wires ball_tracker, court_detector, net_detector, player_tracker
        self.event_detector: Optional[EventDetector] = None
        if self.enable_event_detection:
            self.event_detector = EventDetector(
                tracker=self.ball_tracker,
                court_detector=self.court_detector,
                net_detector=self.net_detector,
                player_tracker=self.player_tracker,
                event_model_path=self.event_model_path,
            )

        # Scoreboard: frame dimensions updated after first frame is read
        self.scoreboard: Optional[Scoreboard] = None
        if self.enable_scoreboard:
            self.scoreboard = Scoreboard(
                frame_width=int(self.config.get("FRAME_WIDTH", 1280)),
                frame_height=int(self.config.get("FRAME_HEIGHT", 720)),
                enable_auto_scoring=self.config.get("AUTO_SCORING", True),
                rally_timeout_frames=int(self.config.get("RALLY_TIMEOUT_FRAMES", 50)),
            )

        # ── Runtime state ──────────────────────────────────────────────
        self.frame_count: int = 0
        self.fps: float       = float(self.config.get("FPS", 30))

        # ── Output / display ───────────────────────────────────────────
        self.show_display: bool = self.config.get("SHOW_DISPLAY", True)
        self.save_output: bool  = self.config.get("SAVE_OUTPUT",  False)
        self.output_path: str   = self.config.get("OUTPUT_PATH",  "output.mp4")
        self.output_dir: Optional[str] = None
        self.out_writer: Optional[cv2.VideoWriter] = None

        # Presentation ring-buffer
        self.presentation_fps: int         = 24
        self.presentation_results: deque   = deque()
        self.frame_interval: float         = 1.0 / self.presentation_fps

        # Thread pool – ball tracking runs in a background thread;
        # player tracking (YOLO) must stay on the main thread.
        self.thread_pool = ThreadPoolExecutor(max_workers=3)

        # Internal flag: court + net calibrated at least once
        self._court_calibrated: bool  = False
        # Retry court calibration every N frames until it succeeds
        self._court_calib_retry_every: int  = 5   # retry interval in frames
        self._court_calib_last_attempt: int = 0    # frame_count of last attempt

    # ------------------------------------------------------------------ #
    #  Builder helpers
    # ------------------------------------------------------------------ #

    def _build_player_tracker(self) -> Optional[PlayerTracker]:
        if not self.enable_player_tracking:
            return None
        return PlayerTracker(
            model_path=self.player_model_path,
            max_distance=self.player_max_distance,
            max_lost_frames=self.player_max_lost_frames,
            exponential_prediction=self.player_exp_pred,
        )

    def _build_ball_tracker(self) -> Optional[BallTracker]:
        if not self.enable_ball_tracking:
            return None
        tracker = BallTracker()
        ball_weights = self.config.get("BALL_MODEL_WEIGHTS")
        if ball_weights:
            ball_cfg: Dict = {
                "BALL_MODEL_WEIGHTS": ball_weights,
                "BALL_MODEL_NAME":    self.config.get("BALL_MODEL_NAME", "TrackNetV4_TypeA"),
            }
            catboost_path = self.config.get("BALL_FILL_PATH")
            if catboost_path:
                ball_cfg["CATBOOST_MODEL_PATH"] = catboost_path
            tracker.update_config(ball_cfg)
        return tracker

    # ------------------------------------------------------------------ #
    #  Config loader
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Parse a simple KEY=VALUE config file."""
        config: Dict = {}
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}")
            return config
        try:
            with open(config_path) as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key   = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if value.lower() in ("true", "false"):
                        config[key] = value.lower() == "true"
                    elif value.replace(".", "", 1).lstrip("-").isdigit():
                        config[key] = float(value) if "." in value else int(value)
                    else:
                        config[key] = value
            logger.info(f"Config loaded from {config_path}: {len(config)} keys")
        except Exception:
            logger.exception(f"Error reading config {config_path}")
        return config

    # ------------------------------------------------------------------ #
    #  Presentation / output helpers
    # ------------------------------------------------------------------ #

    def set_presentation_fps(self, fps: int) -> None:
        self.presentation_fps  = max(1, fps)
        self.frame_interval    = 1.0 / self.presentation_fps

    def presentation(self) -> bool:
        """Flush the latest result frame to the VideoWriter. Returns False on error."""
        if not self.presentation_results:
            return True
        frame = self.presentation_results[-1]
        if self.out_writer is not None:
            try:
                self.out_writer.write(frame)
            except Exception:
                logger.exception("VideoWriter.write() failed")
                return False
        # Keep at most ~2 s of frames in the ring-buffer
        max_buf = self.presentation_fps * 2
        while len(self.presentation_results) > max_buf:
            self.presentation_results.popleft()
        return True

    def _init_writer(self, width: int, height: int) -> None:
        if not self.save_output or self.out_writer is not None:
            return
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (width, height)
            )
            if not self.out_writer.isOpened():
                logger.error(f"VideoWriter failed to open: {self.output_path}")
                self.out_writer = None
        except Exception:
            logger.exception("Failed to create VideoWriter")

    def _release_writer(self) -> None:
        if self.out_writer is not None:
            try:
                self.out_writer.release()
            except Exception:
                logger.exception("Failed to release VideoWriter")
            self.out_writer = None

    # ------------------------------------------------------------------ #
    #  Court + net calibration
    # ------------------------------------------------------------------ #

    def _apply_court_calibration(
        self, keypoints: np.ndarray, frame: np.ndarray,
    ) -> None:
        """Propagate validated *keypoints* to all downstream components."""
        # Store on the court_detector so get_keypoints() returns them
        self.court_detector.keypoints = keypoints

        # ── Scoreboard: update frame dimensions + court bounds ─────────
        if self.scoreboard is not None:
            h, w = frame.shape[:2]
            self.scoreboard.frame_width  = w
            self.scoreboard.frame_height = h
            self.scoreboard.set_court_bounds(keypoints)
            logger.info(
                f"Scoreboard: court bounds updated "
                f"(frame {w}×{h}, net_y={self.scoreboard.state_machine.NET_Y})."
            )

        # ── NetDetector: reinitialise from real court keypoints ─────────
        if self.net_detector is not None:
            self.net_detector.initialize_from_frame(frame, self.court_detector)
            logger.info(
                f"NetDetector: initialised from court keypoints "
                f"(net_region={self.net_detector.net_region})."
            )

        # ── EventDetector: invalidate lazy geometry cache ──────────────
        if self.event_detector is not None:
            self.event_detector._court_polygon = None
            self.event_detector._service_boxes = {}
            self.event_detector._net_y         = None
            self.event_detector._ensure_court_geometry()
            logger.info(
                f"EventDetector: court geometry rebuilt "
                f"(court_polygon={'set' if self.event_detector._court_polygon is not None else 'None'}, "
                f"service_boxes={list(self.event_detector._service_boxes.keys())})."
            )

        self._court_calibrated = True

    def _calibrate_court_from_frame(self, frame: np.ndarray) -> bool:
        """
        Run court + net calibration on *frame* and propagate keypoints to
        all downstream components.

        Returns True if calibration succeeded, False otherwise.
        Called once at startup and retried every _court_calib_retry_every
        frames until it succeeds.
        """
        if self.court_detector is None:
            return False

        logger.info("CourtDetector: attempting calibration…")
        keypoints = self.court_detector.detect(frame, use_resized=True, verbose=1)

        if keypoints is None:
            logger.warning(
                "Court calibration FAILED on this frame — will retry in "
                f"{self._court_calib_retry_every} frames."
            )
            if self.net_detector is not None and not self.net_detector._initialized:
                self.net_detector.initialize_from_frame(frame, court_detector=None)
                logger.info("NetDetector: fallback initialisation (no court keypoints).")
            return False

        logger.info(f"Court calibration SUCCESS — {len(keypoints)} keypoints detected.")
        self._apply_court_calibration(keypoints, frame)
        return True

    def _calibrate_court_from_video(self, cap: cv2.VideoCapture) -> None:
        """
        Multi-frame court calibration: detect keypoints on several frames
        spread across the first part of the video, then take the per-keypoint
        median to filter out outliers (e.g. a player occluding a court line).

        Samples 5 frames at 10-frame intervals (frames 0, 10, 20, 30, 40).
        A keypoint is accepted only if detected in at least 3 of the 5 frames.
        Falls back to the per-frame retry mechanism if consensus fails.
        """
        if not self.enable_court_tracking or self.court_detector is None:
            return

        SAMPLE_COUNT = 5
        FRAME_GAP = 10
        MIN_AGREEMENT = 3

        saved_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Collect keypoint detections from multiple frames
        all_detections: List[Optional[np.ndarray]] = []
        sample_frame: Optional[np.ndarray] = None

        for i in range(SAMPLE_COUNT):
            frame_idx = i * FRAME_GAP
            if total_frames > 0 and frame_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(
                    f"CourtDetector: could not read frame {frame_idx} for calibration."
                )
                all_detections.append(None)
                continue

            if sample_frame is None:
                sample_frame = frame

            keypoints = self.court_detector.detect(frame, use_resized=True, verbose=0)
            all_detections.append(keypoints)
            status = f"{len(keypoints)} kp" if keypoints is not None else "FAILED"
            logger.info(f"CourtDetector: frame {frame_idx} detection: {status}")

        # Filter to successful detections
        valid = [kp for kp in all_detections if kp is not None]
        logger.info(
            f"CourtDetector multi-frame: {len(valid)}/{len(all_detections)} "
            f"frames returned keypoints."
        )

        if len(valid) < MIN_AGREEMENT:
            logger.warning(
                f"CourtDetector: fewer than {MIN_AGREEMENT} successful detections "
                "— multi-frame calibration failed, will retry during playback."
            )
            if (
                sample_frame is not None
                and self.net_detector is not None
                and not self.net_detector._initialized
            ):
                self.net_detector.initialize_from_frame(sample_frame, court_detector=None)
            cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
            return

        # Build consensus keypoints via per-index median
        num_keypoints = valid[0].shape[0]  # 16
        consensus = np.zeros((num_keypoints, 2), dtype=np.float32)
        for idx in range(num_keypoints):
            xs = np.array([kp[idx, 0] for kp in valid])
            ys = np.array([kp[idx, 1] for kp in valid])
            consensus[idx, 0] = np.median(xs)
            consensus[idx, 1] = np.median(ys)

        logger.info(
            f"CourtDetector: consensus keypoints computed from "
            f"{len(valid)} frames (median)."
        )

        self._apply_court_calibration(consensus, sample_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)

    # ------------------------------------------------------------------ #
    #  Per-frame analysis
    # ------------------------------------------------------------------ #

    def _analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run all enabled trackers on *frame* and return the annotated frame.

        Execution order matters:
          1. PlayerTracker.update()        – synchronous (YOLO not thread-safe)
          2. BallTracker.update()          – dispatched to thread pool
          3. EventDetector.update()        – needs ball history + player boxes;
                                             runs after ball future resolves
          4. Scoreboard.update()           – consumes event list
          5. _draw_annotations()           – pure rendering
        """
        ball_future: Optional[Future] = None

        # ── Retry court calibration until it succeeds ──────────────────
        # The C++ detector may fail on the first frame (motion blur, bad
        # lighting) but succeed on a later frame.  We keep retrying every
        # _court_calib_retry_every frames so the pipeline self-heals.
        if (
            self.enable_court_tracking
            and not self._court_calibrated
            and self.court_detector is not None
            and (self.frame_count - self._court_calib_last_attempt)
                >= self._court_calib_retry_every
        ):
            self._court_calib_last_attempt = self.frame_count
            self._calibrate_court_from_frame(frame)

        # 1. Player tracking (must be synchronous)
        if self.enable_player_tracking and self.player_tracker is not None:
            if not self.player_tracker.calibration_done:
                logger.debug(f"Player calibration in progress (frame {self.frame_count} of {self.player_tracker.calibration_max_frames})")
            self.player_tracker.update(frame)
            
            # Log calibration completion
            if self.player_tracker.calibration_done and not getattr(self, '_player_calib_logged', False):
                logger.info(f"Player calibration completed at frame {self.frame_count}")
                self._player_calib_logged = True

        # 2. Ball tracking (background)
        if self.enable_ball_tracking and self.ball_tracker is not None:
            ball_future = self.thread_pool.submit(self.ball_tracker.update, frame)

        # 3. Wait for ball tracking before running event detection
        if ball_future is not None:
            try:
                ball_future.result()
            except Exception:
                logger.exception("BallTracker.update() raised an exception")

        # 4. Event detection
        new_events: List[Dict] = []
        if self.enable_event_detection and self.event_detector is not None:
            # Inject current serve context so SERVE_FAULT can be classified
            if self.scoreboard is not None:
                try:
                    serve_ctx = self.scoreboard.state_machine.get_serve_context()
                    self.event_detector.set_serve_context(serve_ctx)
                except Exception:
                    logger.exception("Failed to get serve context from state machine")

            try:
                new_events = self.event_detector.update(frame)
            except Exception:
                logger.exception("EventDetector.update() raised an exception")

        # 5. Scoreboard update
        self._update_scoreboard(new_events, frame)

        # 6. Draw all annotations
        result_frame = self._draw_annotations(frame)

        return result_frame

    def _update_scoreboard(self, new_events: List[Dict], frame: np.ndarray) -> None:
        """Push latest tracking data and events into the scoreboard."""
        if not self.enable_scoreboard or self.scoreboard is None:
            return

        # Ball position: last element of ball_positions list
        ball_pos: Optional[Tuple] = None
        if self.enable_ball_tracking and self.ball_tracker is not None:
            history = self.ball_tracker.get_ball_history()
            if history:
                raw = history[-1]
                # Positions may be 2-tuple, 3-tuple, or None
                if raw is not None and len(raw) >= 2:
                    ball_pos = (int(raw[0]), int(raw[1]))  # Extract just x, y
                    logger.debug(f"Ball position: {ball_pos}, confidence={raw[2] if len(raw) > 2 else 'N/A'}")

        # Player positions: {player_id: (x,y)}
        player_positions: Optional[Dict] = None
        if self.enable_player_tracking and self.player_tracker is not None:
            player_positions = self.player_tracker.get_player_positions()
            if player_positions:
                logger.debug(f"Player positions: {player_positions}")

        # One-shot: push court bounds to scoreboard once court is calibrated
        if (
            self._court_calibrated
            and self.court_detector is not None
            and self.scoreboard.court_bounds is None
        ):
            kp = self.court_detector.get_keypoints()
            if kp is not None:
                self.scoreboard.set_court_bounds(kp)

        try:
            self.scoreboard.update(
                ball_position=ball_pos,
                player_positions=player_positions,
                events=new_events,
            )
        except Exception:
            logger.exception("Scoreboard.update() raised an exception")

    # ------------------------------------------------------------------ #
    #  Annotation rendering
    # ------------------------------------------------------------------ #

    def _draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """Compose all visual overlays onto a copy of *frame*."""
        result = frame.copy()

        # Ball trail
        if self.enable_ball_tracking and self.ball_tracker is not None:
            try:
                result = self.ball_tracker.draw_ball(result)
            except Exception:
                logger.exception("BallTracker.draw_ball() failed")

        # Court lines
        if self.enable_court_tracking and self.court_detector is not None:
            try:
                result = self.court_detector.draw_court_overlay(result)
            except Exception:
                logger.exception("CourtDetector.draw_court_overlay() failed")

        # Player boxes + trails
        if self.enable_player_tracking and self.player_tracker is not None:
            try:
                result = self.player_tracker.draw_players(result)
            except Exception:
                logger.exception("PlayerTracker.draw_players() failed")

        # Event markers (HIT / BOUNCE / OUT / NET / SERVE_FAULT)
        if self.enable_event_detection and self.event_detector is not None:
            try:
                result = self.event_detector.draw_events(result)
            except Exception:
                logger.exception("EventDetector.draw_events() failed")

        # Scoreboard overlay + debug panel
        if self.enable_scoreboard and self.scoreboard is not None:
            try:
                player_pos = (
                    self.player_tracker.get_player_positions()
                    if self.enable_player_tracking and self.player_tracker
                    else None
                )
                result = self.scoreboard.draw_scoreboard(result, player_pos)
                feature_flags = {
                    "Ball Tracking":   self.enable_ball_tracking,
                    "Player Tracking": self.enable_player_tracking,
                    "Court Tracking":  self.enable_court_tracking,
                    "Event Detection": self.enable_event_detection,
                }
                result = self.scoreboard.draw_debug_overlay(
                    result, self.frame_count, feature_flags
                )
            except Exception:
                logger.exception("Scoreboard rendering failed")

        return result

    # ------------------------------------------------------------------ #
    #  Keyboard handler (video mode)
    # ------------------------------------------------------------------ #

    def _handle_key(self, key: int) -> bool:
        """Handle a key press; returns True when user requests quit."""
        if key == ord("q"):
            return True
        if self.scoreboard is None:
            return False
        if key == ord("1"):
            self.scoreboard.manual_award_point(Player.UPPER)
        elif key == ord("2"):
            self.scoreboard.manual_award_point(Player.LOWER)
        elif key == ord("r"):
            self.scoreboard.reset_score()
        elif key == ord("h"):
            print(
                "\nKeyboard controls:\n"
                "  1 – award point to Player 1 (rear/upper)\n"
                "  2 – award point to Player 2 (front/lower)\n"
                "  r – reset score\n"
                "  q – quit\n"
            )
        return False

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #

    def _print_analysis_summary(self) -> None:
        sep = "=" * 52
        lines = [
            "",
            sep,
            "  TENNIS ANALYSIS SUMMARY",
            sep,
            f"  Frames analysed : {self.frame_count}",
            f"  Ball tracking   : {'ON' if self.enable_ball_tracking   else 'OFF'}",
            f"  Player tracking : {'ON' if self.enable_player_tracking else 'OFF'}",
            f"  Court tracking  : {'ON' if self.enable_court_tracking  else 'OFF'}",
            f"  Event detection : {'ON' if self.enable_event_detection else 'OFF'}",
            f"  Scoreboard      : {'ON' if self.enable_scoreboard      else 'OFF'}",
        ]

        if self.enable_court_tracking and self.court_detector:
            status = "calibrated" if self._court_calibrated else "NOT calibrated"
            lines.append(f"  Court detector  : {status}")

        if self.enable_player_tracking and self.player_tracker:
            calib = "done" if self.player_tracker.calibration_done else "incomplete"
            lines.append(f"  Player calib    : {calib}")

        if self.enable_event_detection and self.event_detector:
            total_events = sum(
                len(v) for v in self.event_detector.get_events().values()
            )
            lines.append(f"  Events detected : {total_events}")

        lines.append(sep)
        print("\n".join(lines))

    # ------------------------------------------------------------------ #
    #  Video analysis
    # ------------------------------------------------------------------ #

    def analyze_video(self, video_path: str) -> None:
        """Process a video file frame-by-frame."""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return

        # Read video metadata
        self.fps        = cap.get(cv2.CAP_PROP_FPS) or self.fps
        total_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Video opened: {frame_width}×{frame_height} @ {self.fps:.1f} FPS, "
            f"{total_frames} frames"
        )

        # Set player tracker calibration window
        if self.player_tracker is not None:
            self.player_tracker.calibration_max_frames = max(
                self.calib_frames,
                int(self.fps * 2),   # at least 2 seconds
            )

        # Court calibration from first frame
        self._calibrate_court_from_video(cap)
        self._init_writer(frame_width, frame_height)

        start_time = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                result_frame = self._analyze_frame(frame)
                self.presentation_results.append(result_frame)

                if not self.presentation():
                    break

                if self.show_display:
                    cv2.imshow("TennIQ Analysis", result_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if self._handle_key(key):
                        break

                # Progress log every 30 frames
                if self.frame_count % 30 == 0 and total_frames > 0:
                    pct     = self.frame_count / total_frames * 100
                    elapsed = time.time() - start_time
                    fps_avg = self.frame_count / elapsed if elapsed > 0 else 0
                    print(
                        f"\rProgress: {pct:5.1f}%  "
                        f"({self.frame_count}/{total_frames})  "
                        f"avg {fps_avg:.1f} fps",
                        end="",
                        flush=True,
                    )

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            print()
            cap.release()
            self._release_writer()
            if self.show_display:
                cv2.destroyAllWindows()
            total_time = time.time() - start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            logger.info(
                f"Finished: {self.frame_count} frames in "
                f"{total_time:.1f}s ({avg_fps:.1f} fps avg)"
            )
            self._print_analysis_summary()

    # ------------------------------------------------------------------ #
    #  Image-sequence analysis
    # ------------------------------------------------------------------ #

    def analyze_image_sequence(self, image_dir: str) -> None:
        """Process a directory of images in sorted order."""
        if not os.path.exists(image_dir):
            logger.error(f"Image directory not found: {image_dir}")
            return

        image_files: List[str] = sorted(
            path
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for path in glob.glob(os.path.join(image_dir, ext))
        )
        if not image_files:
            logger.error(f"No images found in {image_dir}")
            return

        total = len(image_files)
        logger.info(f"Image sequence: {total} files in {image_dir}")

        if self.player_tracker is not None:
            self.player_tracker.calibration_max_frames = min(self.calib_frames, total)

        # Limit queue depth to avoid excessive memory use
        QUEUE_DEPTH = 4
        analysis_pool  = ThreadPoolExecutor(max_workers=1)
        frame_queue: queue.Queue = queue.Queue()
        start_time      = time.time()
        processed       = 0

        def _process(idx_frame: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
            idx, frm = idx_frame
            return idx, self._analyze_frame(frm)

        try:
            for i, path in enumerate(image_files):
                frame = cv2.imread(path)
                if frame is None:
                    logger.warning(f"Could not read image: {path}")
                    continue

                # First-frame initialisation
                if i == 0:
                    self._init_writer(frame.shape[1], frame.shape[0])
                    if self.enable_court_tracking:
                        self._calibrate_court_from_frame(frame)

                future = analysis_pool.submit(_process, (i, frame))
                frame_queue.put((i, future))

                # Drain the queue when it reaches the depth limit
                while frame_queue.qsize() >= QUEUE_DEPTH:
                    _, fut = frame_queue.get()
                    try:
                        _, result_frame = fut.result()
                    except Exception:
                        logger.exception("Frame processing failed")
                        continue
                    self.presentation_results.append(result_frame)
                    self.frame_count += 1
                    processed += 1
                    self.presentation()

                    elapsed = time.time() - start_time
                    fps_avg = processed / elapsed if elapsed > 0 else 0
                    print(
                        f"\rProgress: {processed/total*100:5.1f}%  "
                        f"({processed}/{total})  "
                        f"avg {fps_avg:.1f} fps",
                        end="",
                        flush=True,
                    )

            # Drain remaining futures
            while not frame_queue.empty():
                _, fut = frame_queue.get()
                try:
                    _, result_frame = fut.result()
                except Exception:
                    logger.exception("Frame processing failed (drain)")
                    continue
                self.presentation_results.append(result_frame)
                self.frame_count += 1
                processed += 1
                self.presentation()

        finally:
            analysis_pool.shutdown(wait=True)
            self._release_writer()
            if self.show_display:
                cv2.destroyAllWindows()

        print()
        self._print_analysis_summary()


# ======================================================================
#  Logging helper
# ======================================================================

def _setup_file_logger(log_path: str) -> None:
    """Attach a file handler at INFO level to the root logger."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(fh)
    root.setLevel(logging.INFO)


# ======================================================================
#  CLI entry point
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TennIQ – Tennis Video Analysis System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default="inference/data-configs/data_config_alcaraz.txt",
        help="Path to KEY=VALUE configuration file",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input: path to a video file, or a directory containing images",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Override output video path (default: outputs/<timestamp>/inference.mp4)",
    )
    parser.add_argument(
        "--mode",
        choices=["video", "images"],
        default="video",
        help="Input mode: 'video' for a video file, 'images' for an image directory",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the OpenCV preview window",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override presentation FPS (default: from config or 24)",
    )
    args = parser.parse_args()

    # ── Timestamped output directory ──────────────────────────────────
    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", ts)
    os.makedirs(out_dir, exist_ok=True)

    output_video = args.output or os.path.join(out_dir, "inference.mp4")
    log_path     = os.path.join(out_dir, "inference.log")
    _setup_file_logger(log_path)
    logger.info(f"TennIQ session started. Log: {log_path}")

    # ── Build analyser ─────────────────────────────────────────────────
    analyzer = TennisAnalyzer(args.config)
    analyzer.output_dir  = out_dir
    analyzer.output_path = output_video
    analyzer.save_output = True
    analyzer.show_display = not args.no_display

    # Presentation FPS (CLI > config > default)
    pres_fps = args.fps
    if pres_fps is None:
        pres_fps = analyzer.config.get(
            "PRESENTATION_FPS",
            analyzer.config.get("FPS", analyzer.presentation_fps),
        )
    try:
        analyzer.set_presentation_fps(int(pres_fps))
    except (ValueError, TypeError):
        logger.warning("Invalid presentation FPS; using default 24.")

    # ── Run ────────────────────────────────────────────────────────────
    if args.mode == "video":
        analyzer.analyze_video(args.input)
    else:
        analyzer.analyze_image_sequence(args.input)


if __name__ == "__main__":
    main()