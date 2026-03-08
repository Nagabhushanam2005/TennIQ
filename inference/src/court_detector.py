import logging
import os
import numpy as np
import cv2
from pathlib import Path
import shutil
import subprocess
import time



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourtDetector:
    # Path to the C++ executable (relative to this file)
    _CPP_DETECTOR_DIR = Path(__file__).parent / "court-detector-cpp"
    _CPP_EXECUTABLE = _CPP_DETECTOR_DIR / "court16PointsDetector.out"
    
    # Court keypoint indices for line construction
    # Based on C++ CourtModel keypoint ordering:
        # refer to inference/src/court-detector-cpp/Readme.md for keypoint definitions

    
    def __init__(self, verbose=0):
        self.keypoints = None
        
        if not self._CPP_EXECUTABLE.exists():
            raise FileNotFoundError(
                f"C++ court detector executable not found at {self._CPP_EXECUTABLE}. "
                f"Please build it by running 'make' in {self._CPP_DETECTOR_DIR}"
            )
    
    def detect(self, frame, use_resized=False, verbose=0):
        """
        Detect court keypoints from a frame using the C++ detector.
        Args:
            frame (np.ndarray): Input image (BGR)
            use_resized (bool): If True, resize image to 50% before detection (faster, less accurate)
            verbose (int): Verbosity level
        Returns:
            np.ndarray: (16, 2) array of keypoints, or None if detection fails
        """

        imgs_dir = self._CPP_DETECTOR_DIR / "imgs"
        imgs_dir.mkdir(exist_ok=True)
        img_path = imgs_dir / "input_img.png"
        output_path = self._CPP_DETECTOR_DIR / "output.txt"

        if output_path.exists():
            os.remove(output_path)

        if use_resized:
            frame_to_save = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        else:
            frame_to_save = frame
        cv2.imwrite(str(img_path), frame_to_save)

        # Use relative paths from the cwd (court-detector-cpp directory)
        img_path_rel = f"imgs/{img_path.name}"
        output_path_rel = output_path.name
        cmd = [str(self._CPP_EXECUTABLE), img_path_rel, output_path_rel]
        if verbose:
            print(f"Running: {' '.join(cmd)} in {self._CPP_DETECTOR_DIR}")
        t0 = time.time()
        try:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(os.cpu_count())

            subprocess.run(
                cmd,
                cwd=self._CPP_DETECTOR_DIR,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env
            )

        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"Court detector failed: {e.stderr.decode()}")
            return None
        t1 = time.time()
        if verbose:
            print(f"Court detection took {t1-t0:.2f}s")

        self.keypoints = self._parse_keypoints(output_path)
        return self.keypoints
    def _parse_keypoints(self, filepath):
        """
        Parse the output.txt file to extract 16 keypoints.
        Args:
            filepath (str or Path): Path to output.txt
        Returns:
            np.ndarray: (16, 2) array of keypoints, or None if parsing fails
        """
        import re
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            # Expecting 16 lines, each in format (x,y)
            keypoints = []
            pattern = re.compile(r'\(([\d.\-]+), ([\d.\-]+)\)')
            for line in lines:
                match = pattern.search(line.strip())
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    keypoints.append([x, y])
            if len(keypoints) != 16:
                return None
            return np.array(keypoints, dtype=np.float32)
        except Exception as e:
            print(f"Failed to parse keypoints: {e}")
            return None
    
    def get_keypoints(self):
        """
        Get the 16 detected court keypoints.
        
        Returns:
            np.ndarray: Array of shape (16, 2) containing keypoint coordinates
        """
        return self.keypoints
    
    def get_court_boundaries(self):
        """
        Get the doubles court boundary points (indices 0,1,2,3).
        
        Returns:
            np.ndarray: Array of shape (4, 2) containing boundary coordinates
                        [top_left, bottom_left, bottom_right, top_right]
        """
        if self.keypoints is None:
            return None
        return self.keypoints[[0, 1, 2, 3]]
    
    def get_singles_boundaries(self):
        """
        Get the singles court boundary points (indices 4,5,6,7).
        
        Returns:
            np.ndarray: Array of shape (4, 2) containing boundary coordinates
                        [top_left, bottom_left, bottom_right, top_right]
        """
        if self.keypoints is None:
            return None
        return self.keypoints[[4, 5, 6, 7]]
    
    def get_net_points(self):
        """
        Get the net line endpoints (indices 14, 15).
        
        Returns:
            np.ndarray: Array of shape (2, 2) containing [left, right] net points
        """
        if self.keypoints is None:
            return None
        return self.keypoints[[14, 15]]
    
    def is_point_in_court(self, point):
        """
        Check if a point is inside the court boundaries.
        
        Args:
            point: (x, y) coordinate
            
        Returns:
            bool: True if point is inside court
        """
        boundaries = self.get_court_boundaries()
        if boundaries is None:
            return False
        
        # Use cv2.pointPolygonTest
        contour = boundaries.reshape((-1, 1, 2)).astype(np.float32)
        result = cv2.pointPolygonTest(contour, tuple(point), False)
        return result >= 0
    
    def is_point_in_singles_court(self, point):
        """
        Check if a point is inside the singles court boundaries.
        
        Args:
            point: (x, y) coordinate
            
        Returns:
            bool: True if point is inside singles court
        """
        boundaries = self.get_singles_boundaries()
        if boundaries is None:
            return False
        
        contour = boundaries.reshape((-1, 1, 2)).astype(np.float32)
        result = cv2.pointPolygonTest(contour, tuple(point), False)
        return result >= 0
    
    def draw_court_overlay(self, frame, color=(0, 255, 255), thickness=2):
        """
        Draw detected court lines on frame.
        
        Args:
            frame: Input frame (BGR)
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            np.ndarray: Frame with court overlay
        """
        if self.keypoints is None:
            return frame
        
        result = frame.copy()
        kp = self.keypoints.astype(np.int32)
        
        # Draw all court lines (indices from Readme.md)
        line_pairs = [
            (0, 3),    # baseline top (doubles)
            (1, 2),    # baseline bottom (doubles)
            (0, 1),    # left sideline (doubles)
            (2, 3),    # right sideline (doubles)
            (4, 5),    # left singles line
            (6, 7),    # right singles line
            (14, 15),  # net
            (12, 13),  # center service line (service box mid)
            (8, 9),   # top service line
            (10, 11),   # bottom service line
        ]
        for i1, i2 in line_pairs:
            cv2.line(result, tuple(kp[i1]), tuple(kp[i2]), color, thickness)
        return result

