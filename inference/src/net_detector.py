import logging
import os
import csv
import numpy as np
import cv2
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Net dimensions ratio: height = 3 feet, width = 36 feet
NET_HEIGHT_TO_WIDTH_RATIO = 3.0 / 36.0  # = 1/12

# Default detector parameters (tunable, but most of these are bought after experimentation and should not be changed just like that)
DEFAULT_WINDOW_SEC = 3.0
DEFAULT_SPIKE_K = 4.5
DEFAULT_EWMA_K = 5.0
DEFAULT_REFRACTORY_SEC = 1.5
DEFAULT_MIN_ABS_COUNT = 100
DEFAULT_EWMA_ALPHA = 2 / 21
DEFAULT_SPIKE_RATIO_THRESHOLD = 4.0
DEFAULT_CONFIRMATION_DELAY = 3

@dataclass
class NetHitEvent:
    """Represents a detected net hit event."""
    frame_idx: int
    timestamp_sec: float
    pixel_count: int
    ewma_value: float
    peak_height: float  # how many baseline-sigma above baseline
    confidence: float   # 0–1
    
    gate_values: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'frame_idx': self.frame_idx,
            'timestamp_sec': self.timestamp_sec,
            'pixel_count': self.pixel_count,
            'ewma_value': self.ewma_value,
            'peak_height': self.peak_height,
            'confidence': self.confidence,
            **self.gate_values
        }


@dataclass
class NetRegion:
    """Represents the net region bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def clip_to_frame(self, frame_width: int, frame_height: int) -> 'NetRegion':
        return NetRegion(
            x1=max(0, self.x1),
            y1=max(0, self.y1),
            x2=min(frame_width, self.x2),
            y2=min(frame_height, self.y2)
        )
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside the net region."""
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


# ─────────────────────────────────────────────────────────────────────────────
# Net Detector (Unified Class)
# ─────────────────────────────────────────────────────────────────────────────

class NetDetector:
    """
    Net hit detection class using temporal frame differencing and statistical spike detection.
    
    Detects when a tennis ball hits the net by:
    1. Computing frame differences in the net region
    2. Removing player areas to avoid false positives
    3. Using EWMA smoothing and multi-gate spike detection
    
    Five gates must all pass for a spike detection:
      1. Raw pixel count is spike_k sigma above the rolling median baseline.
      2. Refractory period: no re-trigger within refractory_sec seconds.
      3. EWMA confirmation: the smoothed signal also rises above its baseline.
      4. Spike ratio: pixel count / mean must be >= spike_ratio_threshold.
      5. EWMA ratio confirmation after delay.
    
    Usage:
        detector = NetDetector(fps=30.0)
        detector.initialize_from_court_detector(court_detector, frame.shape)
        
        # For each frame:
        event = detector.detect(current_frame, player_boxes=[])
        if event:
            print(f"Net hit detected at frame {event.frame_idx}")
    """
    
    def __init__(
        self,
        fps: float = 25.0,
        verbose: int = 0,
        window_sec: float = DEFAULT_WINDOW_SEC,
        spike_k: float = DEFAULT_SPIKE_K,
        ewma_k: float = DEFAULT_EWMA_K,
        refractory_sec: float = DEFAULT_REFRACTORY_SEC,
        min_abs_count: int = DEFAULT_MIN_ABS_COUNT,
        ewma_alpha: float = DEFAULT_EWMA_ALPHA,
        spike_ratio_threshold: float = DEFAULT_SPIKE_RATIO_THRESHOLD,
        confirmation_delay: int = DEFAULT_CONFIRMATION_DELAY,
        player_dilation_factor: int = 10
    ):
        """
        Initialize NetDetector.
        
        Args:
            fps: Video frames per second, used for refactory timing and window sizing
            verbose: Verbosity level (0=quiet, 1=info, 2=debug)
            window_sec: Rolling window duration for baseline calculation
            spike_k: Number of sigma above baseline for raw spike detection
            ewma_k: Number of sigma above baseline for EWMA confirmation
            refractory_sec: Minimum gap between two hit detections
            min_abs_count: Minimum absolute pixel count to consider
            ewma_alpha: EWMA smoothing factor (2/(span+1))
            spike_ratio_threshold: Minimum ratio of signal/mean for confirmation
            confirmation_delay: Frames to wait before confirming spike
            player_dilation_factor: Pixels to dilate player boxes for masking
        """

        self.fps = fps
        self.verbose = verbose
        
        # Spike detection parameters
        self.spike_k = spike_k
        self.ewma_k = ewma_k
        self.refractory_frames = int(refractory_sec * fps)
        self.min_abs_count = min_abs_count
        self.ewma_alpha = ewma_alpha
        self.spike_ratio_threshold = spike_ratio_threshold
        self.confirmation_delay = confirmation_delay
        window_frames = max(10, int(window_sec * fps))
        self._raw_window: deque = deque(maxlen=window_frames)
        self._ewma_window: deque = deque(maxlen=window_frames)
        self._ewma: float = 0.0
        self._ewma_of_ewma: float = 0.0
        self._ewma_initialized: bool = False
        self._last_hit_frame: int = -10_000
        self._pending_hit: Optional[Tuple] = None
        self._pending_hit_countdown: int = 0

        self.player_dilation_factor = player_dilation_factor
        self.net_region: Optional[NetRegion] = None
        self.frame_shape: Optional[Tuple[int, int]] = None
        self._initialized: bool = False
        self._prev_gray_threshold: Optional[np.ndarray] = None
        self._threshold_value: int = 128
        self._frame_idx: int = 0

        # history
        self.signal_values: List[int] = []
        self.ewma_values: List[float] = []
        self.events: List[NetHitEvent] = []
    
    def reset(self):
        """Reset detector state for new video/session."""
        self._raw_window.clear()
        self._ewma_window.clear()
        self._ewma = 0.0
        self._ewma_of_ewma = 0.0
        self._ewma_initialized = False
        self._last_hit_frame = -10_000
        self._pending_hit = None
        self._pending_hit_countdown = 0
        self._prev_gray_threshold = None
        self._threshold_value = 128
        self._frame_idx = 0
        self._initialized = False
        self.signal_values.clear()
        self.ewma_values.clear()
        self.events.clear()
    
    def initialize_from_court_detector(self, court_detector, frame_shape: Tuple[int, ...]):
        """
        Initialize net region from court detector.
        
        Args:
            court_detector: CourtDetector instance with detected keypoints
            frame_shape: Shape of video frames (height, width, ...)
        """
        h, w = frame_shape[:2]
        self.frame_shape = (h, w)
        
        net_points = court_detector.get_net_points()
        
        if net_points is not None and len(net_points) >= 2:
            self.net_region = self._compute_net_region_from_points(net_points)
            if self.verbose > 0:
                logger.info(f"Net region initialized: {self.net_region}")
        else:
            # Fallback: narrow band to reduce false positives without court data
            self.net_region = NetRegion(
                x1=w // 4,
                y1=int(h * 0.35),
                x2=3 * w // 4,
                y2=int(h * 0.45)
            )
            if self.verbose > 0:
                logger.warning("Court detection unavailable, using fallback net region")
    
    def initialize_from_region(
        self,
        x1: int, y1: int, x2: int, y2: int,
        frame_shape: Tuple[int, ...]
    ):
        """
        Initialize net region from explicit coordinates.
        
        Args:
            x1, y1, x2, y2: Net region bounding box
            frame_shape: Shape of video frames
        """
        self.frame_shape = frame_shape[:2]
        self.net_region = NetRegion(x1=x1, y1=y1, x2=x2, y2=y2)
    
    def initialize_from_frame(self, frame: np.ndarray, court_detector=None):
        """
        Full initialization from a frame.
        
        Args:
            frame: Initial BGR frame
            court_detector: Optional CourtDetector instance
        """
        self.frame_shape = frame.shape[:2]
        
        if court_detector is not None:
            self.initialize_from_court_detector(court_detector, frame.shape)
        else:
            # fallback: narrow band to reduce false positives without court data
            h, w = frame.shape[:2]
            self.net_region = NetRegion(x1=w // 4, y1=int(h * 0.35), x2=3 * w // 4, y2=int(h * 0.45))

        self._initialize_threshold(frame)
        self._initialized = True
    
    def _compute_net_region_from_points(self, net_points: np.ndarray) -> NetRegion:
        """
        Compute net bounding box from court detector net points.
        
        Args:
            net_points: Array of shape (2, 2) with left and right net endpoints
            
        Returns:
            NetRegion bounding box
        """
        x1 = int(net_points[0][0])
        x2 = int(net_points[1][0])
        net_width = x2 - x1
        net_height = int(net_width * NET_HEIGHT_TO_WIDTH_RATIO)
        y2 = int((net_points[0][1] + net_points[1][1]) / 2)
        y1 = y2 - net_height
        
        return NetRegion(x1=x1, y1=y1, x2=x2, y2=y2)

    def _initialize_threshold(self, frame: np.ndarray):
        """
        Initialize the adaptive threshold based on frame content.
        
        Args:
            frame: BGR frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._threshold_value = self._compute_adaptive_threshold(gray)
        self._prev_gray_threshold = cv2.threshold(
            gray, self._threshold_value, 255, cv2.THRESH_BINARY
        )[1]
    
    def _compute_adaptive_threshold(self, gray: np.ndarray) -> int:
        """
        Compute adaptive threshold using Otsu's method on relevant image regions.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Computed threshold value
        """
        h, w = gray.shape

        # net_region may be None if court calibration has not run yet;
        # fall back to the middle horizontal band of the frame.
        if self.net_region is None:
            y1, y2 = h // 3, h // 2
        else:
            y1, y2 = max(0, self.net_region.y1), min(h, self.net_region.y2)
        
        if y1 >= y2:
            return 128
        
        lower_half = gray[y1:y2, w // 4: 3 * w // 4]
        lower_half2 = gray[3 * h // 8:, w // 4: 3 * w // 4]
        
        if lower_half.size == 0:
            return 128
        
        val1, _ = cv2.threshold(lower_half, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        val2, _ = cv2.threshold(lower_half2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return int((val1 + val2) // 2)
    
    def _compute_frame_diff(
        self,
        current_frame: np.ndarray,
        player_boxes: List[List[int]]
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute frame difference.
        
        Args:
            current_frame: Current BGR frame
            player_boxes: List of player bounding boxes [[x1,y1,x2,y2], ...]
            
        Returns:
            Tuple of (cleaned symmetric difference image, valid flag)
        """
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_threshold = cv2.threshold(
            current_gray, self._threshold_value, 255, cv2.THRESH_BINARY
        )[1]
        
        if self._prev_gray_threshold is None:
            self._prev_gray_threshold = current_threshold
            return np.zeros_like(current_gray), False
        
        sym_diff = cv2.absdiff(self._prev_gray_threshold, current_threshold)
        cleaned_diff = self._remove_player_areas(sym_diff, player_boxes)
        self._prev_gray_threshold = current_threshold
        
        return cleaned_diff, True
    
    def _remove_player_areas(
        self,
        diff_image: np.ndarray,
        player_boxes: List[List[int]]
    ) -> np.ndarray:
        """
        Zero out regions containing players to avoid false positives.
        
        Args:
            diff_image: Grayscale difference image
            player_boxes: List of player bounding boxes
            
        Returns:
            Cleaned difference image
        """
        if not player_boxes:
            return diff_image
        
        cleaned = diff_image.copy()
        h, w = diff_image.shape
        d = self.player_dilation_factor
        
        for box in player_boxes:
            x1, y1, x2, y2 = box
            # Use symmetric dilation to avoid over-masking net region
            cleaned[
                max(0, y1 - d): min(h, y2 + d),
                max(0, x1 - d): min(w, x2 + d)
            ] = 0
        
        return cleaned
    
    def _extract_net_region_signal(self, diff_image: np.ndarray) -> int:
        """
        Extract signal value (pixel count) from the net region.
        
        Args:
            diff_image: Cleaned difference image
            
        Returns:
            Pixel count in net region
        """
        h, w = self.frame_shape
        clipped = self.net_region.clip_to_frame(w, h)
        
        if clipped.width <= 0 or clipped.height <= 0:
            return 0
        
        net_crop = diff_image[clipped.y1:clipped.y2, clipped.x1:clipped.x2]
        return cv2.countNonZero(net_crop)

    def _update_spike_detection(self, signal_value: int) -> Optional[NetHitEvent]:
        """
        Process one frame's signal value for spike detection.
        
        Args:
            signal_value: The pixel count for this frame
            
        Returns:
            NetHitEvent if a hit is confirmed, else None
        """
        idx = self._frame_idx
        
        # Update EWMA
        if not self._ewma_initialized:
            self._ewma = float(signal_value)
            self._ewma_of_ewma = float(signal_value)
            self._ewma_initialized = True
        else:
            self._ewma = self.ewma_alpha * signal_value + (1 - self.ewma_alpha) * self._ewma
        
        slow_alpha = self.ewma_alpha / 2
        self._ewma_of_ewma = slow_alpha * self._ewma + (1 - slow_alpha) * self._ewma_of_ewma
        
        self._raw_window.append(signal_value)
        self._ewma_window.append(self._ewma)
        self.signal_values.append(signal_value)
        self.ewma_values.append(self._ewma)
        
        # Check pending hit confirmation after delay
        confirmed_event = self._check_pending_confirmation()
        
        # Warm-up period (reduced from 10 to catch early net hits)
        if len(self._raw_window) < 5:
            if self.verbose > 1:
                logger.debug(f"Warm-up: {len(self._raw_window)}/5 frames collected")
            return confirmed_event
        
        # Gate 1: raw spike threshold
        raw_arr = np.array(self._raw_window)
        baseline = np.median(raw_arr)
        std = max(np.std(raw_arr), 5.0)  # floor to avoid near-zero std
        
        gate1_raw_spike = signal_value > baseline + self.spike_k * std
        gate2_refractory = (idx - self._last_hit_frame) >= self.refractory_frames
        
        if self.verbose > 1:
            threshold = baseline + self.spike_k * std
            logger.debug(f"Gate1 check @ frame {idx}: signal={signal_value}, baseline={baseline:.1f}, "
                        f"std={std:.1f}, threshold={threshold:.1f}, pass={gate1_raw_spike}")  # DEBUG: Remove after diagnosis
        
        if signal_value < self.min_abs_count:
            return confirmed_event
        if not gate1_raw_spike:
            return confirmed_event
        if not gate2_refractory:
            return confirmed_event
        
        # Gate 3: EWMA confirmation (initial check)
        ewma_arr = np.array(self._ewma_window)
        ewma_base = np.median(ewma_arr)
        ewma_std = max(np.std(ewma_arr), 3.0)
        gate3_ewma = self._ewma > ewma_base + self.ewma_k * ewma_std
        if not gate3_ewma:
            return confirmed_event
        
        # Gate 4: Raw spike ratio check (immediate)
        raw_mean = np.mean(raw_arr)
        raw_ratio = signal_value / max(raw_mean, 1.0)
        
        if raw_ratio < self.spike_ratio_threshold:
            return confirmed_event
        
        if self.verbose > 0:
            logger.debug(f"[Pending hit] frame={idx} raw_ratio: {raw_ratio:.2f} - "
                        f"waiting {self.confirmation_delay} frames for EWMA confirmation")
        
        # All init gates passed - schedule confirmation
        if self._pending_hit is None:
            pending_gates = {
                'gate1_raw_spike': gate1_raw_spike,
                'gate2_refractory': gate2_refractory,
                'gate3_ewma': gate3_ewma,
                'gate4_raw_ratio': raw_ratio,
            }
            self._pending_hit = (idx, signal_value, baseline, std, pending_gates)
            self._pending_hit_countdown = self.confirmation_delay
        
        return confirmed_event
    
    def _check_pending_confirmation(self) -> Optional[NetHitEvent]:
        """Check and process pending hit confirmation."""
        if self._pending_hit is None:
            return None
        
        self._pending_hit_countdown -= 1
        if self._pending_hit_countdown > 0:
            return None
        
        # Now confirm with EWMA ratio check
        pending_idx, pending_signal, pending_baseline, pending_std, pending_gates = self._pending_hit
        
        ewma_arr = np.array(self._ewma_window)
        ewma_mean = np.mean(ewma_arr)
        ewma_ratio = self._ewma / max(ewma_mean, 1.0)
        
        if self.verbose > 0:
            logger.debug(f"[Confirmation check] frame={pending_idx} ewma_ratio: {ewma_ratio:.2f} "
                        f"(threshold: {self.spike_ratio_threshold})")
        
        confirmed_event = None
        
        if ewma_ratio >= self.spike_ratio_threshold:
            # Confirmed... Create the event
            self._last_hit_frame = pending_idx
            peak_height = (pending_signal - pending_baseline) / pending_std
            confidence = float(np.clip(
                (peak_height - self.spike_k) / (1.5 * self.spike_ratio_threshold) + 0.5, 
                0.0, 1.0
            ))
            
            gate_values = {**pending_gates, 'gate5_ewma_ratio': ewma_ratio}
            
            confirmed_event = NetHitEvent(
                frame_idx=pending_idx,
                timestamp_sec=pending_idx / self.fps,
                pixel_count=pending_signal,
                ewma_value=self._ewma,
                peak_height=peak_height,
                confidence=confidence,
                gate_values=gate_values
            )
            self.events.append(confirmed_event)
            
            if self.verbose > 0:
                logger.info(f"Net hit confirmed at frame={pending_idx} "
                           f"t={confirmed_event.timestamp_sec:.2f}s "
                           f"conf={confidence:.2f}")
        else:
            if self.verbose > 0:
                logger.debug(f"[Rejected] frame={pending_idx} - "
                            f"EWMA ratio {ewma_ratio:.2f} < {self.spike_ratio_threshold}")
        
        self._pending_hit = None
        return confirmed_event

    def detect(
        self,
        frame: np.ndarray,
        player_boxes: Optional[List[List[int]]] = None
    ) -> Optional[NetHitEvent]:
        """
        Process a frame and detect net hits.
        
        Args:
            frame: Current BGR frame
            player_boxes: List of player bounding boxes [[x1,y1,x2,y2], ...]
                         (Future: will be obtained from PlayerTracker singleton)
            
        Returns:
            NetHitEvent if a hit is confirmed, else None
        """
        if player_boxes is None:
            player_boxes = []

        # If net_region is still None (court calibration not yet done),
        # create a sensible fallback from the frame dimensions so the
        # detector can at least initialise without crashing.
        if self.net_region is None:
            h, w = frame.shape[:2]
            self.net_region = NetRegion(x1=w // 4, y1=int(h * 0.35), x2=3 * w // 4, y2=int(h * 0.45))
            logger.debug(
                "NetDetector: net_region was None; using fallback "
                f"({self.net_region.x1},{self.net_region.y1},"
                f"{self.net_region.x2},{self.net_region.y2})"
            )

        # Auto-initialize on first frame
        if not self._initialized:
            self._initialize_threshold(frame)
            self._initialized = True
            self._frame_idx += 1
            return None
        
        # Compute frame difference
        diff_image, valid = self._compute_frame_diff(frame, player_boxes)
        
        if not valid:
            self._frame_idx += 1
            return None
        
        # Extract signal from net region
        signal = self._extract_net_region_signal(diff_image)
        
        # Run spike detection
        event = self._update_spike_detection(signal)
        
        self._frame_idx += 1
        return event

    def get_net_points(self) -> Optional[np.ndarray]:
        """
        Get net region as points array.
        
        Returns:
            Array of shape (2, 2) with [[x1,y1], [x2,y2]] or None
        """
        if self.net_region is None:
            return None
        return np.array([
            [self.net_region.x1, self.net_region.y1],
            [self.net_region.x2, self.net_region.y2]
        ], dtype=np.float32)
    
    def is_ball_in_net(self, point: Tuple[int, int]) -> bool:
        """
        Check if a point is inside the net region.
        
        Args:
            point: (x, y) coordinate
            
        Returns:
            True if point is in net region
        """
        if self.net_region is None:
            return False
        return self.net_region.contains_point(point)
    
    def get_net_hit_events(self) -> List[NetHitEvent]:
        """Get all detected net hit events."""
        return self.events
    
    def get_last_diff_image(self) -> Optional[np.ndarray]:
        """Get the last computed difference image (for visualization)."""
        return self._prev_gray_threshold

    def draw_net_overlay(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw net region on frame.
        
        Args:
            frame: Input BGR frame
            color: Rectangle color (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with net overlay
        """
        if self.net_region is None:
            return frame
        
        result = frame.copy()
        cv2.rectangle(
            result,
            (self.net_region.x1, self.net_region.y1),
            (self.net_region.x2, self.net_region.y2),
            color, thickness
        )
        return result
    
    def draw_net_hit_banner(
        self,
        frame: np.ndarray,
        event: NetHitEvent
    ) -> np.ndarray:
        """
        Draw a 'NET HIT' banner on frame.
        
        Args:
            frame: Input BGR frame
            event: The net hit event
            
        Returns:
            Frame with banner overlay
        """
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Red banner at top
        cv2.rectangle(result, (0, 0), (w, 85), (0, 0, 180), -1)
        cv2.putText(
            result, "NET HIT!", (w // 2 - 170, 68),
            cv2.FONT_HERSHEY_DUPLEX, 2.6, (255, 255, 255), 4
        )
        cv2.putText(
            result,
            f"t={event.timestamp_sec:.2f}s  "
            f"sigma={event.peak_height:.1f}  "
            f"conf={event.confidence:.2f}",
            (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2
        )
        return result
    
    def draw_signal_plot(
        self,
        width: int,
        height: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        show_events: bool = True
    ) -> np.ndarray:
        """
        Draw signal plot with event markers.
        
        Args:
            width: Plot width in pixels
            height: Plot height in pixels
            color: Line color (BGR)
            show_events: Whether to mark detected events
            
        Returns:
            BGR image of the plot
        """
        plot_img = np.zeros((height, width, 3), dtype=np.uint8)
        counts = self.signal_values
        
        if len(counts) < 2:
            return plot_img
        
        max_val = max(counts) if max(counts) > 0 else 1
        y_scale = (height * 0.9) / max_val
        x_step = width / max(len(counts) - 1, 1)
        
        points = []
        for i, val in enumerate(counts):
            x = int(i * x_step)
            y = int(height - val * y_scale)
            points.append((x, y))
        
        # Draw signal line
        for i in range(len(points) - 1):
            cv2.line(plot_img, points[i], points[i + 1], color, 2)
        
        # Draw event markers
        if show_events:
            for ev in self.events:
                if ev.frame_idx < len(points):
                    ex, ey = points[ev.frame_idx]
                    cv2.line(plot_img, (ex, 0), (ex, height), (0, 165, 255), 2)
                    cv2.circle(plot_img, (ex, ey), 6, (0, 165, 255), -1)
        
        # Current count label
        if counts:
            cv2.putText(
                plot_img, f"Count: {counts[-1]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
        
        return plot_img

    def export_events_csv(
        self,
        output_path: str,
        video_name: str = ""
    ):
        """
        Export detected events to CSV file.
        
        Args:
            output_path: Path to output CSV file
            video_name: Optional video name for the CSV
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "video", "frame_idx", "timestamp_sec",
                "pixel_count", "ewma_value", "peak_height_sigma", "confidence",
                "gate1_raw_spike", "gate2_refractory", "gate3_ewma",
                "gate4_raw_ratio", "gate5_ewma_ratio"
            ])
            
            for ev in self.events:
                gate_values = ev.gate_values
                writer.writerow([
                    video_name,
                    ev.frame_idx,
                    f"{ev.timestamp_sec:.4f}",
                    ev.pixel_count,
                    f"{ev.ewma_value:.2f}",
                    f"{ev.peak_height:.2f}",
                    f"{ev.confidence:.3f}",
                    gate_values.get('gate1_raw_spike', ''),
                    gate_values.get('gate2_refractory', ''),
                    gate_values.get('gate3_ewma', ''),
                    gate_values.get('gate4_raw_ratio', ''),
                    gate_values.get('gate5_ewma_ratio', '')
                ])
        
        if self.verbose > 0:
            logger.info(f"Saved {len(self.events)} net hit events to {output_path}")