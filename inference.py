import cv2
import torch
import numpy as np
from config import Config
from detector import MediaPipeProcessor
from model import HGC_LSTM
from training import create_adjacency_matrix, load_labels_from_csv
from cv_to_60 import interpolate_frames
# Load configuration
config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
A = create_adjacency_matrix()
# Load model
video_to_label_mapping, label_to_idx, ids, labels = load_labels_from_csv()
num_classes = len(ids)
model = HGC_LSTM(config, A, num_classes)
model_path = config.training.save_dir + '/' + config.training.model_save_name
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Initialize MediaPipe processor
processor = MediaPipeProcessor(config)

def predict_from_video(video_path, thresh_hold = 0.8):
    """Perform inference on a video file."""
    cap = cv2.VideoCapture(video_path)
    #create a numpy array all 0 have shape (sequence_length, num_joints, 2)
    # keypoints_sequence = np.zeros((config.hgc_lstm.sequence_length, config.hgc_lstm.num_vertices, 2), dtype=np.float32)
    keypoints_sequence = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame to extract keypoints
        _, res = processor.process_frame(frame)
        keypoints = processor.extract_keypoints(res)
        if keypoints is not None:
            keypoints_sequence.append(keypoints)
    
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
        label = labels[prediction + 1] if prediction is not None else "Unknown"
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
    video_path = "data/datatest/test_01.mp4"  # Replace with your video path
    predict_from_video(video_path, thresh_hold=0.7)

    # Predict from camera
    # predict_from_camera()
