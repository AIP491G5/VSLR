import cv2
import pandas as pd
import numpy as np
import os
from detector import MediaPipeProcessor
from config import Config
from cv_to_60 import interpolate_frames

# ========== KHỞI TẠO CONFIG ==========
config = Config()

# ========== KHAI BÁO ĐƯỜNG DẪN ==========
INPUT_CSV_FILE = config.data.input_csv_file
VIDEO_INPUT_DIR = config.data.video_input_dir
VIDEO_OUTPUT_DIR = config.data.video_output_dir
KEYPOINTS_OUTPUT_DIR = config.data.keypoints_output_dir
LABELS_OUTPUT_DIR = config.data.labels_output_dir

# ========== KHỞI TẠO MODEL ==========
model = MediaPipeProcessor(config)
vid_not_kept = []
df_video = pd.DataFrame()

def check_and_create_directory(directory_path: str) -> bool:
    """
    Check and create directory if it doesn't exist
    
    Args:
        directory_path: Path to the directory to check
        
    Returns:
        bool: True if directory exists or created successfully, False if error
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            print(f"[SUCCESS] Directory created: {directory_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Cannot create directory {directory_path}: {e}")
        return False

def validate_input_file(file_path: str) -> bool:
    """
    Check if input file exists and can be read
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if file exists and readable, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] File does not exist: {file_path}")
        return False
    
    if not os.path.isfile(file_path):
        print(f"[ERROR] Path is not a file: {file_path}")
        return False
    
    try:
        # Check if CSV file can be read
        if file_path.endswith('.csv'):
            df_test = pd.read_csv(file_path)
            if df_test.empty:
                print(f"[ERROR] CSV file is empty: {file_path}")
                return False
            
            # Check required columns
            required_columns = ['videos', 'id']
            for col in required_columns:
                if col not in df_test.columns:
                    print(f"[ERROR] CSV file missing column '{col}': {file_path}")
                    return False
        
        print(f"[SUCCESS] File is valid: {file_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Cannot read file {file_path}: {e}")
        return False

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

def setup_directories() -> bool:
    """
    Setup and check all required directories
    
    Returns:
        bool: True if all directories created successfully, False if error
    """
    directories = [
        VIDEO_OUTPUT_DIR,
        KEYPOINTS_OUTPUT_DIR,
        LABELS_OUTPUT_DIR
    ]
    
    success = True
    for directory in directories:
        if not check_and_create_directory(directory):
            success = False
    
    return success

def validate_inputs() -> bool:
    """
    Validate all inputs before starting processing
    
    Returns:
        bool: True if all inputs are valid, False if error
    """
    print("[INFO] ========== CHECKING INPUT PARAMETERS ==========")
    
    # Check CSV file
    if not validate_input_file(INPUT_CSV_FILE):
        return False
    
    # Check video input directory
    if not os.path.exists(VIDEO_INPUT_DIR):
        print(f"[ERROR] Video input directory does not exist: {VIDEO_INPUT_DIR}")
        return False
    
    # Setup output directories
    if not setup_directories():
        return False
    
    print("[SUCCESS] All checks passed!")
    return True

def safe_save_file(file_path: str, data: np.ndarray, file_type: str = "numpy") -> bool:
    """
    Save file safely with error handling
    
    Args:
        file_path: Path to the file to save
        data: Data to save
        file_type: Type of file ('numpy' or 'video')
        
    Returns:
        bool: True if saved successfully, False if error
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if not check_and_create_directory(directory):
            return False
        
        if file_type == "numpy":
            np.save(file_path, data)
        
        return True
    except Exception as e:
        print(f"[ERROR] Error saving file {file_path}: {e}")
        return False

def safe_save_video(video_path: str, frames: list, fps: int = 30) -> bool:
    """
    Save video safely with error handling
    
    Args:
        video_path: Path to the video file to save
        frames: List of frames
        fps: Frames per second
        
    Returns:
        bool: True if saved successfully, False if error
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(video_path)
        if not check_and_create_directory(directory):
            return False
        
        if not frames:
            print(f"[ERROR] No frames to save: {video_path}")
            return False
        
        # Create video writer
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        if not writer.isOpened():
            print(f"[ERROR] Cannot create video writer: {video_path}")
            return False
        
        # Write frames
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        writer.release()
        return True
    except Exception as e:
        print(f"[ERROR] Error saving video {video_path}: {e}")
        return False

def moved(curr_kps: np.ndarray, ref_kps: np.ndarray, threshold=None) -> bool:
    """
    Check if current keypoints have moved significantly from reference keypoints
    
    Args:
        curr_kps: Current keypoints
        ref_kps: Reference keypoints
        threshold: Movement threshold (normalized)
        
    Returns:
        bool: True if movement detected, False otherwise
    """
    if threshold is None:
        threshold = config.data.movement_threshold
    # So sánh độ chênh tuyệt đối từng chiều
    diffs = np.abs(curr_kps - ref_kps)
    return np.any(diffs > threshold)

def filter_video(id, src):
    """
    Process video and extract keypoints and labels
    
    Args:
        id: Video ID for labeling
        src: Source video path
        
    Returns:
        tuple: (keypoints, labels, processed_frames) or (None, None, None) if error
    """
    # Check video before processing
    if not validate_video_file(src):
        return None, None, None
    
    cap = cv2.VideoCapture(src)
    
    # Read all frames from video
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    
    cap.release()
    
    if len(all_frames) == 0:
        print(f"[ERROR] Cannot read any frames from {src}")
        return None, None, None
    
    try:
        # Apply interpolate_frames to normalize frame count
        processed_frames = interpolate_frames(all_frames, target=60)
        
        # Get keypoints from first frame as reference
        image0 = cv2.cvtColor(processed_frames[0], cv2.COLOR_BGR2RGB)
        _, ref_res = model.process_frame(image0)
        ref_kps = model.extract_keypoints(ref_res)
        
        label = [0]
        kpts = [ref_kps]
        vid = []
        
        # Process each interpolated frame
        for i, frame in enumerate(processed_frames):
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb, res = model.process_frame(image_rgb)
            frame_processed = model.draw_landmarks(image_rgb, res)
            vid.append(frame_processed)
            
            # Skip first frame as it's already processed as reference
            if i == 0:
                continue
                
            if res.pose_landmarks:
                kpts_extract = model.extract_keypoints(res)
                if moved(kpts_extract, ref_kps):
                    label.append(-1)
                else:
                    label.append(0)
                kpts.append(kpts_extract)
        
        # Check if label_frames_needed = 0, then no need to process labels
        if config.data.label_frames_needed == 0:
            kpts = np.array(kpts)
            return kpts, None, vid
        
        label = np.array(label)
        indices = np.where(label == -1)[0]
        
        if len(indices) > 0:
            last_idx = indices[-1]
            needed = config.data.label_frames_needed

            # Number of elements remaining from position -1 to end
            after = len(label) - last_idx

            if after >= needed:
                label[last_idx : last_idx + needed] = id
            else:
                extra = needed - after
                start_idx = max(0, last_idx - extra)
                label[start_idx:] = id
        else:
            label[-config.data.label_frames_needed:] = id
        
        # All element = -1 turn to 0
        label[label == -1] = 0
        kpts = np.array(kpts)
        
        return kpts, label, vid
    
    except Exception as e:
        print(f"[ERROR] Error processing video {src}: {e}")
        return None, None, None

def main():
    """
    Main function to process all videos
    """
    # Check all inputs before starting
    if not validate_inputs():
        print("[ERROR] Stopping program due to input validation errors")
        return
    
    # Read CSV file
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
    except Exception as e:
        print(f"[ERROR] Cannot read CSV file {INPUT_CSV_FILE}: {e}")
        return
    
    print("[INFO] ========== STARTING VIDEO PROCESSING ==========")
    
    success_count = 0
    error_count = 0
    
    for i, row in df.iterrows():
        try:
            # Split videos by comma and strip whitespace
            files_name = [v.strip() for v in row['videos'].split(',')]
            id = row['id']
            
            for file_name in files_name:
                file_name_output = file_name.split('.')[0]
                src = os.path.join(VIDEO_INPUT_DIR, file_name)
                dst_vids = os.path.join(VIDEO_OUTPUT_DIR, file_name_output + '.mp4')
                dst_kpts = os.path.join(KEYPOINTS_OUTPUT_DIR, file_name_output)
                dst_label = os.path.join(LABELS_OUTPUT_DIR, file_name_output)
                
                if not os.path.exists(src):
                    print(f"[WARNING] Video not found: {src}")
                    error_count += 1
                    continue
                
                print(f"[PROGRESS] Processing video: {file_name}")
                res_kpts, res_label, res_video = filter_video(id, src)
                
                if res_kpts is None:
                    print(f"[ERROR] Cannot process video: {src}")
                    error_count += 1
                    continue
                
                # Reshape keypoints
                # res_kpts = res_kpts.reshape(res_kpts.shape[0], -1)
                
                # Save video as mp4
                if not safe_save_video(dst_vids, res_video, config.data.video_fps):
                    print(f"[ERROR] Cannot save video: {dst_vids}")
                    error_count += 1
                    continue
                
                # Save keypoints
                if not safe_save_file(dst_kpts, res_kpts, "numpy"):
                    print(f"[ERROR] Cannot save keypoints: {dst_kpts}")
                    error_count += 1
                    continue
                
                # Save labels only when available (not None)
                if res_label is not None:
                    if not safe_save_file(dst_label, res_label, "numpy"):
                        print(f"[ERROR] Cannot save labels: {dst_label}")
                        error_count += 1
                        continue
                    print(f'[SUCCESS] {id} - {file_name} - {res_kpts.shape} frames - {res_label.shape} labels')
                else:
                    print(f'[SUCCESS] {id} - {file_name} - {res_kpts.shape} frames - No labels (label_frames_needed=0)')
                
                success_count += 1
        
        except Exception as e:
            print(f"[ERROR] Error processing row {i}: {e}")
            error_count += 1
    
    print("[INFO] ========== PROCESSING RESULTS ==========")
    print(f"[INFO] Success: {success_count} videos")
    print(f"[INFO] Errors: {error_count} videos")

if __name__ == "__main__":
    main()
