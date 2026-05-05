import logging
import os
import re
import numpy as np
import cv2
from pathlib import Path
import subprocess
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CourtDetector:
    """
    Detects 16 court keypoints by calling the bundled C++ executable.
    """

    _CPP_DETECTOR_DIR = Path(__file__).parent / "court-detector-cpp"
    _CPP_EXECUTABLE   = _CPP_DETECTOR_DIR / "court16PointsDetector.out"

    def __init__(self, verbose: int = 0):
        self.verbose   = verbose
        self.keypoints = None

        if not self._CPP_EXECUTABLE.exists():
            raise FileNotFoundError(
                f"C++ court detector executable not found: {self._CPP_EXECUTABLE}\n"
                f"Build it by running 'make' inside: {self._CPP_DETECTOR_DIR}"
            )
        logger.info(f"CourtDetector ready. Executable: {self._CPP_EXECUTABLE}")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def detect(self, frame: np.ndarray, use_resized: bool = False,
               verbose: int = None) -> np.ndarray:
        """
        Run the C++ detector on *frame* and return 16 keypoints, or None.

        Args:
            frame:       BGR image.
            use_resized: Resize to 50% before detection (faster).
                         Keypoints are scaled back automatically.
            verbose:     Override instance verbose level for this call.

        Returns:
            np.ndarray of shape (16, 2), or None on any failure.
        """
        v = self.verbose if verbose is None else verbose

        imgs_dir    = self._CPP_DETECTOR_DIR / "imgs"
        img_path    = imgs_dir / "img.png"
        output_path = self._CPP_DETECTOR_DIR / "output.txt"

        try:
            imgs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error(f"CourtDetector: cannot create imgs dir: {exc}")
            return None

        if output_path.exists():
            try:
                output_path.unlink()
            except OSError as exc:
                logger.warning(f"CourtDetector: could not remove old output.txt: {exc}")

        # Write input image
        if use_resized:
            h, w = frame.shape[:2]
            frame_to_save = cv2.resize(frame, (w // 2, h // 2))
        else:
            frame_to_save = frame

        ok = cv2.imwrite(str(img_path), frame_to_save)
        if not ok:
            logger.error(f"CourtDetector: cv2.imwrite failed for {img_path}")
            return None

        # Run C++ subprocess
        cmd = [
            str(self._CPP_EXECUTABLE),
            f"imgs/{img_path.name}",
            output_path.name,
        ]
        if v:
            logger.info(f"CourtDetector cmd: {' '.join(cmd)}  (cwd={self._CPP_DETECTOR_DIR})")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self._CPP_DETECTOR_DIR,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
        except FileNotFoundError:
            logger.error(f"CourtDetector: executable not found at runtime: {self._CPP_EXECUTABLE}")
            return None
        except subprocess.TimeoutExpired:
            logger.error("CourtDetector: subprocess timed out after 30s")
            return None
        except Exception as exc:
            logger.error(f"CourtDetector: unexpected subprocess error: {exc}")
            return None

        elapsed = time.time() - t0

        # ALWAYS show stdout for diagnostic info
        if result.stdout.strip():
            logger.info(f"CourtDetector diagnostic output:\n{result.stdout.strip()}")

        # Always surface stderr — this is the key diagnostic information
        if result.stderr.strip():
            logger.warning(f"CourtDetector stderr:\n{result.stderr.strip()}")

        if result.returncode != 0:
            logger.error(
                f"CourtDetector exited with code {result.returncode} "
                f"after {elapsed:.2f}s.\nstdout: {result.stdout.strip()!r}"
            )
            return None

        if v:
            logger.info(f"CourtDetector subprocess finished in {elapsed:.2f}s")

        # Parse output file
        if not output_path.exists():
            logger.error(
                f"CourtDetector: output.txt not created after successful run. "
                f"stdout: {result.stdout.strip()!r}"
            )
            return None

        keypoints = self._parse_keypoints(output_path)

        if keypoints is None:
            return None

        # Scale back if the image was halved before detection
        if use_resized:
            keypoints = keypoints * 2.0

        self.keypoints = keypoints
        logger.info(f"CourtDetector: calibration OK — 16 keypoints in {elapsed:.2f}s")
        return self.keypoints

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _parse_keypoints(self, filepath: Path) -> np.ndarray:
        """
        Parse output.txt written by the C++ detector.
        Expected: 16 lines each containing a point like (123.45, 678.90).
        Returns np.ndarray (16, 2) or None.
        """
        pattern = re.compile(r'\(\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*\)')
        try:
            text = filepath.read_text(errors="replace")
        except OSError as exc:
            logger.error(f"CourtDetector: cannot read {filepath}: {exc}")
            return None

        keypoints = []
        for line in text.splitlines():
            m = pattern.search(line)
            if m:
                keypoints.append([float(m.group(1)), float(m.group(2))])

        if len(keypoints) != 16:
            logger.error(
                f"CourtDetector: expected 16 keypoints, got {len(keypoints)}.\n"
                f"Raw output.txt:\n{text}"
            )
            return None

        return np.array(keypoints, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Accessors
    # ------------------------------------------------------------------ #

    def get_keypoints(self) -> np.ndarray:
        return self.keypoints

    def get_court_boundaries(self) -> np.ndarray:
        """Doubles court corners: indices 0,1,2,3  [TL, BL, BR, TR]."""
        return self.keypoints[[0, 1, 2, 3]] if self.keypoints is not None else None

    def get_singles_boundaries(self) -> np.ndarray:
        """Singles court corners: indices 4,5,6,7  [TL, BL, BR, TR]."""
        return self.keypoints[[4, 5, 6, 7]] if self.keypoints is not None else None

    def get_net_points(self) -> np.ndarray:
        """Net endpoints: indices 14, 15  [left, right]."""
        return self.keypoints[[14, 15]] if self.keypoints is not None else None

    def is_point_in_court(self, point) -> bool:
        b = self.get_court_boundaries()
        if b is None:
            return False
        contour = b.reshape((-1, 1, 2)).astype(np.float32)
        return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0

    def is_point_in_singles_court(self, point) -> bool:
        b = self.get_singles_boundaries()
        if b is None:
            return False
        contour = b.reshape((-1, 1, 2)).astype(np.float32)
        return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #

    def draw_court_overlay(self, frame: np.ndarray,
                           color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
        """Draw all detected court lines on *frame*. No-op if not calibrated."""
        if self.keypoints is None:
            return frame

        result = frame.copy()
        kp = self.keypoints.astype(np.int32)

        line_pairs = [
            (0,  3),   # top baseline (doubles)
            (1,  2),   # bottom baseline (doubles)
            (0,  1),   # left sideline (doubles)
            (2,  3),   # right sideline (doubles)
            (4,  5),   # left singles sideline
            (6,  7),   # right singles sideline
            (14, 15),  # net
            (12, 13),  # centre service line
            (8,  9),   # top service line
            (10, 11),  # bottom service line
        ]
        for i1, i2 in line_pairs:
            cv2.line(result, tuple(kp[i1]), tuple(kp[i2]), color, thickness)
        return result