import cv2
import torch
import numpy as np
import sys
import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').disabled = True
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.config import Config
from src.utils.detector import MediaPipeProcessor
from src.models.model_utils import create_model, create_adjacency_matrix
from src.utils.data_utils import load_labels_from_csv
from src.utils.cv_to_60 import interpolate_frames
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

def predict_from_video(video_path, thresh_hold = 0.8):
    """Perform inference on a video file."""
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    
    print(f"🎬 Processing video: {video_path}")
    #check video existence
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Process frame to extract keypoints
        _, res = processor.process_frame(frame)
        keypoints = processor.extract_keypoints(res)
        if keypoints is not None:
            keypoints_sequence.append(keypoints)
    
    cap.release()
    
    # Interpolate to target sequence length
    input_data = interpolate_frames(keypoints_sequence, config.hgc_lstm.sequence_length)
    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(input_data)
        probs = torch.softmax(output, dim=1)
        max_prob, prediction = torch.max(probs, dim=1)
        if max_prob.item() > thresh_hold:
            prediction = prediction.item()
        else:
            print(prediction, max_prob.item())
            prediction = None
    if prediction is not None:
        label = id_to_label_mapping.get(prediction + 1, "Unknown")
        print(f"Predicted class: {prediction + 1}, label: {label}, confidence: {max_prob.item()}")
    cap.release()

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
    video_path = "data/datatest/test_01_contran.mp4"  # Replace with your video path
    predict_from_video(video_path, thresh_hold=0.7)

    # Predict from camera
    # predict_from_camera()
