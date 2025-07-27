import streamlit as st
import os
import torch
import cv2
import numpy as np
import time
from datetime import datetime

# --- IMPORT CÁC THÀNH PHẦN CẦN THIẾT ---
from scripts.inference import predict_from_video 
from src.utils.detector import MediaPipeProcessor
from src.utils.model_utils import create_model, create_adjacency_matrix
from src.utils.data_utils import load_labels_from_csv
from configs.config import Config

# --- CÁC HÀM TẢI DỮ LIỆU VÀ MODEL ---
@st.cache_resource
def load_model_and_config():
    st.info("🔄 Lần chạy đầu tiên, đang tải model và cấu hình...")
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, _, id_to_label_mapping = load_labels_from_csv(None, config)
    num_classes = len(id_to_label_mapping)
    A = create_adjacency_matrix(config)
    model = create_model(config, A, num_classes=num_classes, device=device)
    model_path = 'outputs/models/best_hgc_lstm.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, config, device, id_to_label_mapping

model, config, device, id_to_label_mapping = load_model_and_config()

# --- GIAO DIỆN ỨNG DỤNG STREAMLIT ---
st.set_page_config(page_title="VSLR Inference", layout="wide")
st.title("🤟 Nhận diện Ngôn ngữ Ký hiệu Tiếng Việt (VSLR)")

# Khởi tạo session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'recorded_frames' not in st.session_state:
    st.session_state.recorded_frames = []
if 'recording_start_time' not in st.session_state:
    st.session_state.recording_start_time = 0

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Tùy chọn")
    option = st.radio("Chọn phương thức đầu vào:", ["Sử dụng Camera", "Tải lên Video"])
    st.markdown("---")
    show_skeleton = st.checkbox("Hiển thị khung xương", value=True)
    conf_threshold = st.slider("Ngưỡng tin cậy", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

# --- KHU VỰC HIỂN THỊ CHÍNH ---

# TÙY CHỌN 1: TẢI LÊN VIDEO (Không thay đổi)
if option == "Tải lên Video":
    # (Giữ nguyên phần code xử lý Tải lên Video)
    st.header("📁 Tải lên Video để dự đoán")
    uploaded_file = st.file_uploader("Chọn một file video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        temp_dir = "data/datatest_streamlit"
        os.makedirs(temp_dir, exist_ok=True)
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.video(temp_video_path)
        
        if st.button("Bắt đầu Dự đoán Video", use_container_width=True):
            with st.spinner("Đang xử lý và dự đoán..."):
                processor = MediaPipeProcessor(config)
                label, _ = predict_from_video(model, processor, id_to_label_mapping, config, device, temp_video_path, thresh_hold=conf_threshold)
                
                if label:
                    st.success(f"🎯 **Kết quả dự đoán: {label.upper()}**")
                else:
                    st.warning("⚠️ Không thể đưa ra dự đoán (độ tin cậy thấp).")


# TÙY CHỌN 2: SỬ DỤNG CAMERA
elif option == "Sử dụng Camera":
    st.header("📹 Dự đoán trực tiếp qua Camera")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎥 Bật Camera", use_container_width=True):
            st.session_state.camera_active = True
            st.rerun() # Chạy lại script để kích hoạt camera ngay lập tức
        if st.button("🔴 Bắt đầu Ghi", use_container_width=True):
            if st.session_state.camera_active:
                st.session_state.recording = True
                st.session_state.recorded_frames = []
                st.session_state.recording_start_time = time.time()
            else:
                st.error("Vui lòng bật camera trước!")

    with col2:
        if st.button("⏹️ Tắt Camera", use_container_width=True):
            st.session_state.camera_active = False
            st.session_state.recording = False
            st.session_state.recorded_frames = [] # Xóa frame khi tắt cam
            st.rerun()
        if st.button("⏸️ Dừng Ghi & Dự đoán", use_container_width=True):
            # Chỉ cần set recording = False, logic xử lý sẽ được kích hoạt ở dưới
            if st.session_state.recording:
                st.session_state.recording = False
                st.rerun()

    # --- BẮT ĐẦU LOGIC MỚI ---
    # KIỂM TRA VÀ XỬ LÝ VIDEO VỪA GHI XONG
    # Logic này chạy khi không còn ở trạng thái ghi, nhưng vẫn còn frame đã lưu
    if not st.session_state.recording and st.session_state.get('recorded_frames'):
        frames_to_save = st.session_state.recorded_frames
        with st.spinner("Đang xử lý video và dự đoán..."):
            save_dir = "data/data_test_cam"
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_video_path = os.path.join(save_dir, f"test_cam_{timestamp}.mp4")
            
            stop_time = time.time()
            start_time = st.session_state.get('recording_start_time', stop_time)
            elapsed_time = stop_time - start_time
            actual_fps = len(frames_to_save) / elapsed_time if elapsed_time > 0 else 20.0
            
            height, width, _ = frames_to_save[0].shape
            video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), actual_fps, (width, height))
            for f in frames_to_save:
                video_writer.write(f)
            video_writer.release()
            st.toast(f"✅ Video tạm đã được lưu tại: {temp_video_path}")

            processor = MediaPipeProcessor(config)
            label, _ = predict_from_video(model, processor, id_to_label_mapping, config, device, temp_video_path, thresh_hold=conf_threshold)

            if label:
                st.success(f"🎯 **Kết quả dự đoán: {label.upper()}**")
            else:
                st.warning("⚠️ Không thể đưa ra dự đoán (độ tin cậy thấp).")
        
        # Dọn dẹp state để sẵn sàng cho lần ghi tiếp theo
        st.session_state.recorded_frames = []

    # HIỂN THỊ CAMERA NẾU ĐANG KÍCH HOẠT
    if st.session_state.camera_active:
        video_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        processor = MediaPipeProcessor(config)

        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể đọc frame từ camera.")
                break
            
            frame = cv2.flip(frame, 1)
            original_frame, res = processor.process_frame(frame)
            display_frame = processor.draw_landmarks(original_frame, res) if show_skeleton else original_frame
            
            if st.session_state.recording:
                st.session_state.recorded_frames.append(frame.copy())
                elapsed_time = time.time() - st.session_state.recording_start_time
                cv2.putText(display_frame, f"RECORDING {elapsed_time:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            video_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        
        cap.release()
    else:
        st.info("Nhấn 'Bật Camera' để bắt đầu.")