import streamlit as st
import os
import torch
import cv2
import numpy as np
import time
from scripts.inference import predict_from_video
from src.utils.model_utils import load_model
from src.utils.detector import MediaPipeProcessor
from src.utils.data_utils import load_labels_from_csv
from src.utils.interpolate import interpolate_frames
from configs.config import Config

@st.cache_resource
def load_inference_model():
    """Load model once and cache it"""
    return load_model()

@st.cache_resource
def load_config_and_labels():
    """Load config and labels once and cache them"""
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video_to_label_mapping, label_to_idx, unique_labels, id_to_label_mapping = load_labels_from_csv(None, config)
    return config, device, id_to_label_mapping

@st.cache_resource
def load_processor():
    """Load MediaPipe processor once and cache it"""
    config = Config()
    return MediaPipeProcessor(config)

# Load model, config and processor (cached)
model = load_inference_model()
config, device, id_to_label_mapping = load_config_and_labels()
processor = load_processor()

# Streamlit app
st.title("Vietnamese Sign Language Recognition")
st.markdown("Upload a video or use your camera to perform inference.")

# Sidebar for options
option = st.sidebar.radio("Choose input method:", ["Use camera", "Insert video"])

if option == "Insert video":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save uploaded video temporarily
        temp_video_path = os.path.join("data/datatest", uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Perform inference
        st.video(temp_video_path)
        st.write("Performing inference...")
        
        # Use the loaded model components
        label, prediction = predict_from_video(model, processor, id_to_label_mapping, 
                                             config, device, temp_video_path, thresh_hold=0.6)
        
        if label:
            st.success(f"Prediction: {label} (Class: {prediction})")
        else:
            st.warning("No prediction made (low confidence)")

elif option == "Use camera":
    st.subheader("📹 Real-time Camera Inference")
    
    # Camera controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎥 Start Camera"):
            st.session_state.camera_active = True
            st.session_state.recording = False
            st.session_state.keypoints_sequence = []
    
    with col2:
        if st.button("⏹️ Stop Camera"):
            st.session_state.camera_active = False
            st.session_state.recording = False
    
    with col3:
        show_skeleton = st.checkbox("Show Skeleton", value=True)
    
    # Recording controls
    col4, col5 = st.columns(2)
    
    with col4:
        if st.button("🔴 Start Recording"):
            if st.session_state.get('camera_active', False):
                st.session_state.recording = True
                st.session_state.keypoints_sequence = []
                st.session_state.recording_start_time = time.time()
                st.success("Recording started!")
            else:
                st.error("Please start camera first!")
    
    with col5:
        if st.button("⏸️ Stop Recording & Predict"):
            if st.session_state.get('recording', False):
                st.session_state.recording = False
                
                # Run prediction on recorded sequence
                if st.session_state.get('keypoints_sequence', []):
                    with st.spinner("Processing and predicting..."):
                        # Interpolate frames to match model input
                        input_data = interpolate_frames(
                            st.session_state.keypoints_sequence, 
                            config.hgc_lstm.sequence_length
                        )
                        input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        # Run inference
                        with torch.no_grad():
                            output = model(input_data)
                            probs = torch.softmax(output, dim=1)
                            max_prob, pred_idx = torch.max(probs, dim=1)
                            
                            confidence = max_prob.item()
                            prediction = pred_idx.item()
                            label = id_to_label_mapping.get(prediction + 1, "Unknown")
                        
                        # Display results
                        if confidence > 0.6:  # threshold
                            st.success(f"🎯 **Prediction: {label.upper()}**")
                            st.info(f"📊 Confidence: {confidence:.2%}")
                            st.info(f"🔢 Class: {prediction + 1}")
                        else:
                            st.warning(f"⚠️ Low confidence prediction: {label} ({confidence:.2%})")
                            
                        st.info(f"📹 Recorded {len(st.session_state.keypoints_sequence)} frames")
                else:
                    st.error("No keypoints recorded!")
            else:
                st.error("Not currently recording!")
    
    # Initialize session state
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'keypoints_sequence' not in st.session_state:
        st.session_state.keypoints_sequence = []
    
    # Camera feed
    if st.session_state.get('camera_active', False):
        # Create placeholder for video feed
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Error: Could not open camera")
            st.session_state.camera_active = False
        else:
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.data.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.data.height)
            cap.set(cv2.CAP_PROP_FPS, config.data.video_fps)
            
            # Camera loop
            frame_count = 0
            while st.session_state.get('camera_active', False):
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read from camera")
                    break
                
                frame_count += 1
                
                # Flip frame horizontally (mirror effect)
                frame = cv2.flip(frame, 1)
                
                # Process frame to get keypoints
                original_frame, res = processor.process_frame(frame)
                
                # Choose display frame based on skeleton setting
                if show_skeleton:
                    display_frame = processor.draw_landmarks(original_frame, res)
                else:
                    display_frame = original_frame
                
                # Record keypoints if recording
                if st.session_state.get('recording', False):
                    keypoints = processor.extract_keypoints(res)
                    if keypoints is not None:
                        st.session_state.keypoints_sequence.append(keypoints)
                    
                    # Show recording status
                    elapsed_time = time.time() - st.session_state.get('recording_start_time', time.time())
                    status_placeholder.info(f"🔴 Recording... {elapsed_time:.1f}s | Frames: {len(st.session_state.keypoints_sequence)}")
                else:
                    status_placeholder.info(f"📹 Camera active | Frame: {frame_count}")
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Small delay to prevent overwhelming the browser
                time.sleep(0.033)  # ~30 FPS
            
            # Release camera when done
            cap.release()
            st.info("📹 Camera stopped")
    else:
        st.info("👆 Click 'Start Camera' to begin real-time inference")
        
        # Show instructions
        st.markdown("""
        ### 📖 Instructions:
        1. **Start Camera**: Begin video feed
        2. **Start Recording**: Begin capturing keypoints for prediction
        3. **Stop Recording & Predict**: End recording and run inference
        4. **Show Skeleton**: Toggle skeleton visualization
        
        ### 💡 Tips:
        - Make sure you're well-lit and clearly visible
        - Perform sign language gestures clearly
        - Record for 2-3 seconds for best results
        """)
        
    # Show current session info
    with st.expander("🔍 Session Info"):
        st.write(f"Camera Active: {st.session_state.get('camera_active', False)}")
        st.write(f"Recording: {st.session_state.get('recording', False)}")
        st.write(f"Keypoints Recorded: {len(st.session_state.get('keypoints_sequence', []))}")
        if st.session_state.get('keypoints_sequence', []):
            st.write(f"Sequence Length: {len(st.session_state.keypoints_sequence)} frames")
            st.write(f"Target Length: {config.hgc_lstm.sequence_length} frames")