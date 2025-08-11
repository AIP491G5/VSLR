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

def adjust_hand_positions(keypoints, has_left_hand, has_right_hand, has_left_hand_history, has_right_hand_history, prev_keypoints=None):
    """
    Adjust hand keypoints positions based on detection status and pose index movement
    
    Args:
        keypoints: Original keypoints array (75, 2)
        has_left_hand: Current frame has left hand detection
        has_right_hand: Current frame has right hand detection  
        has_left_hand_history: Previously detected left hand
        has_right_hand_history: Previously detected right hand
        prev_keypoints: Previous frame keypoints for hand tracking (75, 2)
        
    Returns:
        np.ndarray: Adjusted keypoints array
    """
    # Copy original keypoints
    adjusted = keypoints.copy()
    
    # Indices for different parts
    POSE_START, POSE_END = 0, 33
    LEFT_START, LEFT_END = 33, 54
    RIGHT_START, RIGHT_END = 54, 75
    
    # Pose index landmarks (MediaPipe pose landmarks)
    LEFT_INDEX_POSE = 19   # LEFT_INDEX in pose landmarks
    RIGHT_INDEX_POSE = 20  # RIGHT_INDEX in pose landmarks
    
    # Extract parts
    pose_kpts = keypoints[POSE_START:POSE_END]  # Always keep pose
    left_kpts = keypoints[LEFT_START:LEFT_END]
    right_kpts = keypoints[RIGHT_START:RIGHT_END]
    
    if not has_right_hand:
        # Only left hand detected
        if has_right_hand_history and prev_keypoints is not None:
            # Right hand was detected before, estimate right hand using pose index movement
            # Calculate movement vector from RIGHT_INDEX (20) between frames
            current_right_index = keypoints[RIGHT_INDEX_POSE]  # Current RIGHT_INDEX from pose
            prev_right_index = prev_keypoints[RIGHT_INDEX_POSE]  # Previous RIGHT_INDEX from pose
            movement_vector = current_right_index - prev_right_index
            
            # Apply movement to previous right hand keypoints
            prev_right_hand = prev_keypoints[RIGHT_START:RIGHT_END]
            estimated_right_hand = prev_right_hand + movement_vector
            adjusted[RIGHT_START:RIGHT_END] = estimated_right_hand
        else:
            # No right hand history, zero out right hand
            adjusted[RIGHT_START:RIGHT_END] = 0
            
    if not has_left_hand:
        # Only right hand detected
        if has_left_hand_history and prev_keypoints is not None:
            # Left hand was detected before, estimate left hand using pose index movement
            # Calculate movement vector from LEFT_INDEX (19) between frames
            current_left_index = keypoints[LEFT_INDEX_POSE]  # Current LEFT_INDEX from pose
            prev_left_index = prev_keypoints[LEFT_INDEX_POSE]  # Previous LEFT_INDEX from pose
            movement_vector = current_left_index - prev_left_index
            
            # Apply movement to previous left hand keypoints
            prev_left_hand = prev_keypoints[LEFT_START:LEFT_END]
            estimated_left_hand = prev_left_hand + movement_vector
            adjusted[LEFT_START:LEFT_END] = estimated_left_hand
        else:
            adjusted[LEFT_START:LEFT_END] = 0
    
    return adjusted

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
        
        # Track hand detection status
        has_left_hand_history = False
        has_right_hand_history = False
        
        # Process each interpolated frame
        for i, frame in enumerate(processed_frames):
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, res = model.process_frame(image_rgb)
            
            if res.pose_landmarks:
                # Extract all 75 keypoints (33 pose + 21 left + 21 right)
                keypoints = model.extract_keypoints(res)
                
                # Check current hand detection status
                has_left_hand = res.left_hand_landmarks is not None
                has_right_hand = res.right_hand_landmarks is not None
                
                # Update hand history
                if has_left_hand:
                    has_left_hand_history = True
                if has_right_hand:
                    has_right_hand_history = True
                
                # Adjust keypoints based on hand detection and history
                prev_kpts = keypoints_list[-1] if len(keypoints_list) > 0 else None
                adjusted_keypoints = adjust_hand_positions(
                    keypoints, 
                    has_left_hand, 
                    has_right_hand, 
                    has_left_hand_history, 
                    has_right_hand_history,
                    prev_kpts
                )
                
                keypoints_list.append(adjusted_keypoints)
                
            else:
                # If no pose detected, use zeros or previous frame keypoints
                if len(keypoints_list) > 0:
                    keypoints_list.append(keypoints_list[-1])
                else:
                    # Initialize with zeros if first frame has no detection
                    zeros_kpts = np.zeros((config.hgc_lstm.num_vertices, config.hgc_lstm.in_channels))
                    keypoints_list.append(zeros_kpts)
        
        keypoints_array = np.array(keypoints_list)
        print(f"[SUCCESS] Extracted {keypoints_array.shape[0]} frames with {keypoints_array.shape[1]} keypoints")
        print(f"[SUMMARY] Hand detection history - Left: {has_left_hand_history} | Right: {has_right_hand_history}")
        
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
