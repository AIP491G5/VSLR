"""
Extract keypoints from videos using MediaPipe
Simplified version that only extracts keypoints, not labels
"""

import cv2
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.detector import MediaPipeProcessor
from configs.config import Config
from src.utils.interpolate import interpolate_frames

def validate_video_file(video_path: str) -> bool:
    """
    Check if video file exists and can be read
    
    Args:
        video_path: Path to the video file
        
    Returns:
        bool: True if video is valid, False otherwise
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video does not exist: {video_path}")
        return False
    
    if not os.path.isfile(video_path):
        print(f"[ERROR] Path is not a file: {video_path}")
        return False
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            cap.release()
            return False
        
        # Check if video has frames
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"[ERROR] Video has no frames: {video_path}")
            return False
        
        return True
    except Exception as e:
        print(f"[ERROR] Error checking video {video_path}: {e}")
        return False

def extract_keypoints_from_video(video_path: str, config: Config, model: MediaPipeProcessor) -> np.ndarray:
    """
    Extract keypoints from a single video file
    
    Args:
        video_path: Path to the video file
        config: Configuration object
        model: MediaPipe processor
        
    Returns:
        np.ndarray: Keypoints array of shape (frames, vertices, channels) or None if error
    """
    if not validate_video_file(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    
    # Read all frames from video
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    
    cap.release()
    
    if len(all_frames) == 0:
        print(f"[ERROR] Cannot read any frames from {video_path}")
        return None
    
    try:
        # Apply interpolate_frames to normalize frame count
        processed_frames = interpolate_frames(all_frames, target=config.hgc_lstm.sequence_length)
        
        keypoints_list = []
        
        # Process each interpolated frame
        for i, frame in enumerate(processed_frames):
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, res = model.process_frame(image_rgb)
            
            if res.pose_landmarks:
                keypoints = model.extract_keypoints(res)
                keypoints_list.append(keypoints)
            else:
                # If no keypoints detected, use zeros or previous frame keypoints
                if len(keypoints_list) > 0:
                    keypoints_list.append(keypoints_list[-1])
                else:
                    # Initialize with zeros if first frame has no detection
                    zeros_kpts = np.zeros((config.hgc_lstm.num_vertices, config.hgc_lstm.in_channels))
                    keypoints_list.append(zeros_kpts)
        
        keypoints_array = np.array(keypoints_list)
        print(f"[SUCCESS] Extracted {keypoints_array.shape[0]} frames with {keypoints_array.shape[1]} keypoints")
        
        return keypoints_array
    
    except Exception as e:
        print(f"[ERROR] Error processing video {video_path}: {e}")
        return None

def process_videos_from_csv(csv_file=None, video_dir=None, output_dir=None):
    """
    Process all videos listed in CSV file and extract keypoints
    
    Args:
        csv_file: Path to CSV file containing video information
        video_dir: Directory containing video files  
        output_dir: Directory to save extracted keypoints
    """
    # Initialize config
    config = Config()
    
    # Use config values if parameters are None
    if csv_file is None:
        csv_file = config.data.input_csv_file
    if video_dir is None:
        video_dir = config.data.video_input_dir
    if output_dir is None:
        output_dir = config.data.keypoints_output_dir
    
    print(f"[INFO] CSV file: {csv_file}")
    print(f"[INFO] Video directory: {video_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MediaPipe model
    model = MediaPipeProcessor(config)
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"[INFO] Found {len(df)} entries in CSV file")
    except Exception as e:
        print(f"[ERROR] Cannot read CSV file {csv_file}: {e}")
        return
    
    success_count = 0
    error_count = 0
    
    # Process each row in CSV
    for i, row in df.iterrows():
        try:
            # Split videos by comma and strip whitespace
            video_files = [v.strip() for v in row['videos'].split(',')]
            label_id = row['id']
            label_name = row.get('label', 'unknown')
            
            print(f"[INFO] Processing ID {label_id} ({label_name}): {len(video_files)} videos")
            
            for video_file in video_files:
                video_name = video_file.split('.')[0]  # Remove extension
                video_path = os.path.join(video_dir, video_file)
                output_path = os.path.join(output_dir, f"{video_name}.npy")
                
                if not os.path.exists(video_path):
                    print(f"[WARNING] Video not found: {video_path}")
                    error_count += 1
                    continue
                
                print(f"[PROGRESS] Processing video: {video_file}")
                
                # Extract keypoints
                keypoints = extract_keypoints_from_video(video_path, config, model)
                
                if keypoints is None:
                    print(f"[ERROR] Cannot extract keypoints from: {video_path}")
                    error_count += 1
                    continue
                
                # Save keypoints
                try:
                    np.save(output_path, keypoints)
                    print(f"[SUCCESS] Saved keypoints: {output_path} - Shape: {keypoints.shape}")
                    success_count += 1
                except Exception as e:
                    print(f"[ERROR] Cannot save keypoints {output_path}: {e}")
                    error_count += 1
        
        except Exception as e:
            print(f"[ERROR] Error processing row {i}: {e}")
            error_count += 1
    
    print(f"\n[INFO] ========== PROCESSING RESULTS ==========")
    print(f"[INFO] Successfully processed: {success_count} videos")
    print(f"[INFO] Errors: {error_count} videos")
    print(f"[INFO] Keypoints saved to: {output_dir}")

def extract_single_video(video_path, output_path=None):
    """
    Extract keypoints from a single video file
    
    Args:
        video_path: Path to the video file
        output_path: Path to save keypoints (optional)
    """
    config = Config()
    model = MediaPipeProcessor(config)
    
    print(f"[INFO] Processing single video: {video_path}")
    
    keypoints = extract_keypoints_from_video(video_path, config, model)
    
    if keypoints is None:
        print(f"[ERROR] Failed to extract keypoints from: {video_path}")
        return None
    
    if output_path:
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, keypoints)
            print(f"[SUCCESS] Keypoints saved to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Cannot save keypoints: {e}")
            return None
    
    print(f"[SUCCESS] Extracted keypoints shape: {keypoints.shape}")
    return keypoints

if __name__ == "__main__":
    # Example usage
    print("[INFO] ========== KEYPOINTS EXTRACTION ==========")
    
    # Option 1: Process all videos from CSV (recommended)
    print("[INFO] Processing videos from CSV file...")
    process_videos_from_csv()
    
    # Option 2: Process single video
    # single_video_path = "path/to/your/video.mp4"
    # output_path = "path/to/output/keypoints.npy"
    # extract_single_video(single_video_path, output_path)
