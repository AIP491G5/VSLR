import cv2
import torch
import numpy as np
import sys
import os
import warnings
import logging
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').disabled = True
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.config import Config
from src.utils.detector import MediaPipeProcessor
from src.utils.model_utils import create_model, create_adjacency_matrix
from src.utils.data_utils import load_labels_from_csv
from src.utils.interpolate import interpolate_frames

def predict_from_video(model, processor, id_to_label_mapping, config, device, video_path, thresh_hold=0.8):
    """Perform inference on a video file."""
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    
    print(f"🎬 Processing video: {video_path}")
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return None, None # Sửa đổi: trả về 2 giá trị None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        _, res = processor.process_frame(frame)
        keypoints = processor.extract_keypoints(res)
        if keypoints is not None:
            keypoints_sequence.append(keypoints)
    
    cap.release()
    
    # --- BẮT ĐẦU PHẦN SỬA LỖI ---
    # Kiểm tra nếu chuỗi quá ngắn để thực hiện nội suy
    if len(keypoints_sequence) < 2:
        print(f"❌ Error: Video quá ngắn hoặc không thể phát hiện đủ keypoints. Tìm thấy: {len(keypoints_sequence)} frames.")
        return None, None # Trả về không có nhãn, không có dự đoán
    # --- KẾT THÚC PHẦN SỬA LỖI ---
    
    label = None
    prediction = None # Khởi tạo prediction là None

    # Interpolate to target sequence length
    input_data = interpolate_frames(keypoints_sequence, config.hgc_lstm.sequence_length)
    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_data)
        probs = torch.softmax(output, dim=1)
        max_prob, pred_idx = torch.max(probs, dim=1)
        
        if max_prob.item() > thresh_hold:
            prediction = pred_idx.item()
        else:
            # prediction vẫn là None nếu độ tin cậy thấp
            print(f"Confidence too low: {max_prob.item()}")

    if prediction is not None:
        label = id_to_label_mapping.get(prediction + 1, "Unknown")
        print(f"Predicted class: {prediction + 1}, label: {label}, confidence: {max_prob.item()}")

    prediction_result = int(prediction + 1) if prediction is not None else None
    
    return label, prediction_result

def extract_embedding_from_video(model, processor, video_path, config, device):
    """Extract embeddings from a video file."""
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        _, res = processor.process_frame(frame)
        keypoints = processor.extract_keypoints(res)
        if keypoints is not None:
            keypoints_sequence.append(keypoints)
    
    cap.release()
    
    if len(keypoints_sequence) < 2:
        print(f"❌ Error: Video quá ngắn hoặc không thể phát hiện đủ keypoints. Tìm thấy: {len(keypoints_sequence)} frames.")
        return None
    
    # Interpolate to target sequence length
    input_data = interpolate_frames(keypoints_sequence, config.hgc_lstm.sequence_length)
    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(input_data)
    
    return embedding.cpu().numpy()

# def predict_from_camera():
#     """Perform inference using a webcam."""
#     cap = cv2.VideoCapture(0)
#     keypoints_sequence = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Display the frame
#         cv2.imshow('Webcam', frame)

#         # Process frame to extract keypoints
#         keypoints = processor.process_frame(frame)
#         if keypoints is not None:
#             keypoints_sequence.append(keypoints)

#         # Make prediction if enough frames are collected
#         if len(keypoints_sequence) >= config.hgc_lstm.sequence_length:
#             with torch.no_grad():
#                 output = model(input_data)
#                 prediction = torch.argmax(output, dim=1).item()
#                 print(f"Prediction: {prediction}")

#             keypoints_sequence = []  # Reset sequence

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Load configuration
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = create_adjacency_matrix(config)

    # Load labels
    video_to_label_mapping, label_to_idx, unique_labels, id_to_label_mapping = load_labels_from_csv(None, config)
    num_classes = len(unique_labels)

    # Create model with updated constructor
    model = create_model(config, A, num_classes=num_classes, device=device)

    model_path = 'outputs/models/best_hgc_lstm.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize MediaPipe processor
    processor = MediaPipeProcessor(config)
    
    test_dir = "data/datatest"  # Replace with your video path
    videos = glob.glob(os.path.join(test_dir, "*.mp4"))
    count = 0
    for video_path in videos:
        filename = os.path.basename(video_path)
        number = int(filename.split('_')[1].split('.')[0])
        label, res = predict_from_video(model, processor, id_to_label_mapping, config, device, video_path, thresh_hold=0.6)
        print(f"{number}: {res}")
        if res == number:
            count += 1
    print(f"{count}/{len(videos)} videos matched the expected labels.")

    # Predict from camera
    # predict_from_camera()
