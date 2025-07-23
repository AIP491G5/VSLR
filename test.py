import streamlit as st
import os
import torch
from scripts.inference import predict_from_video
from src.utils.data_utils import load_labels_from_csv
from src.models.model_utils import create_model
from configs.config import Config

# Load configuration
config = Config()

# Load labels
video_to_label_mapping, label_to_idx, unique_labels, id_to_label_mapping = load_labels_from_csv(None, config)

# Initialize model


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
        label = predict_from_video(temp_video_path)
        st.write(f"Label: {label}")
elif option == "Use camera":
    st.write("Camera functionality is not implemented yet.")