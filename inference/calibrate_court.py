"""
Script for court detection and calibration on video.
It processes the initial frames of the video to detect and finalize the court 
using a homography matrix, assuming a static camera.

Usage:
    python calibrate_court.py --input_video_path <path_to_video> --output_video_path <output_video_path> --calib_frames <num_frames>
"""

import argparse
import cv2
import numpy as np
import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.src.court_detector import CourtDetector
from inference.src.court_reference import CourtReference

# Helper function to get video properties (copied from original context)
def get_video_properties(video):
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, length, v_width, v_height

def main(args):
    """
    Main function to run the court calibration script.
    """
    # Unpack arguments
    input_video_path = args.input_video_path
    output_video_path = args.output_video_path
    calib_frames = args.calib_frames

    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at {input_video_path}")
        sys.exit(1)

    # --- Setup ---
    video_capture = cv2.VideoCapture(input_video_path)
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)

    fps, total_frames, output_width, output_height = get_video_properties(video_capture)
    
    if calib_frames > total_frames:
        calib_frames = total_frames
        print(f"Warning: Calibration frames exceed total frames. Using {total_frames} frames.")

    # Initialize court detector
    court_detector = CourtDetector(verbose=0)

    # Set up video writer for the annotated output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' or 'XVID'
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    # --- Calibration Loop ---
    print(f'Starting court detection and calibration on the first {calib_frames} frames...')
    
    all_lines_locations = []
    
    for frame_i in range(1, calib_frames + 1):
        ret, frame = video_capture.read()
        if not ret:
            break

        print(f'Processing frame {frame_i}/{calib_frames}', end='\r')

        if frame_i == 1:
            # On the first frame, run full detection
            lines = court_detector.detect(frame)
        else: 
            # On subsequent frames, attempt to track (as it's faster), 
            # though detect() will re-run if tracking fails.
            # However, for robust initial calibration assuming a static camera, 
            # it is better to run full detection again for an ensemble of matrices.
            # We'll use the track_court as a robust way to update the matrix but 
            # capture the matrix from the first successful detection.
            
            # Since the goal is initial calibration assuming static camera, 
            # we'll just track to update the matrix until the last frame.
            lines = court_detector.track_court(frame)

        # Collect the final detected line coordinates for averaging/selection later
        if lines is not None:
            all_lines_locations.append(lines)
            
            # Annotate the frame with the detected lines
            annotated_frame = frame.copy()
            for i in range(0, len(lines), 4):
                x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                # Draw lines in a distinct color, e.g., Green (0, 255, 0)
                cv2.line(annotated_frame, (int(x1),int(y1)),(int(x2),int(y2)), (0, 255, 0), 2)
            
            output_video.write(annotated_frame)
        else:
            # If detection/tracking fails, write the un-annotated frame
            output_video.write(frame)

    video_capture.release()
    output_video.release()
    
    print('\nCalibration phase complete.')

    # --- Finalize Calibration Data ---
    if not court_detector.court_warp_matrix:
        print("Error: Court detection failed on all calibration frames. No homography matrix found.")
        return

    # Use the best matrix found during the detection of all frames
    final_warp_matrix = court_detector.court_warp_matrix[-1] 
    final_inv_warp_matrix = court_detector.game_warp_matrix[-1]

    # Save the final homography matrices to a file
    matrices_output_path = os.path.join(os.path.dirname(output_video_path), "court_homography_matrices.npz")
    np.savez(matrices_output_path, 
             court_warp_matrix=final_warp_matrix, 
             game_warp_matrix=final_inv_warp_matrix,
             best_conf=court_detector.best_conf)
    
    print(f"Calibration video saved to: {output_video_path}")
    print(f"Final homography matrices saved to: {matrices_output_path}")
    print("The calibration is finalized and ready for subsequent tracking/processing.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Court detection and calibration on video (static camera).")
    parser.add_argument("--input_video_path", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--output_video_path", type=str, required=True, help="Path to save the annotated calibration video (e.g., 'VideoOutput/calib_video.mp4').")
    parser.add_argument("--calib_frames", type=int, default=30, help="Number of initial frames to use for court calibration.")
    
    args = parser.parse_args()
    main(args)