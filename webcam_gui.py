import cv2
import torch
import numpy as np
import sys
import os
import warnings
import logging
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time

# --- BẮT ĐẦU PHẦN TÍCH HỢP TỪ FILE inference.py ---

# Cấu hình môi trường (giữ nguyên từ file của bạn)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').disabled = True
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import các module của bạn
from configs.config import Config
from src.utils.detector import MediaPipeProcessor
from src.models.model_utils import create_model, create_adjacency_matrix
from src.utils.data_utils import load_labels_from_csv
from src.utils.interpolate import interpolate_frames

class SignLanguageApp:
    def __init__(self, window_title="Sign Language Recognition"):
        """Khởi tạo ứng dụng"""
        self.window = tk.Tk()
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Các biến trạng thái của ứng dụng ---
        self.cap = None
        self.is_camera_on = False
        self.is_recording = False
        self.show_skeleton = True
        self.keypoints_sequence = []
        self.recording_start_time = None
        self.last_prediction = ""
        self.last_confidence = 0.0

        # --- Tải Model và các thành phần xử lý ---
        print("🔄 Đang tải model và các thành phần cần thiết...")
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tải nhãn
        _, _, _, self.id_to_label_mapping = load_labels_from_csv(None, self.config)
        num_classes = len(self.id_to_label_mapping)
        
        # Tạo và tải model
        A = create_adjacency_matrix(self.config)
        self.model = create_model(self.config, A, num_classes=num_classes, device=self.device)
        model_path = 'outputs/models/best_hgc_lstm.pth' # Đường dẫn model của bạn
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Khởi tạo bộ xử lý MediaPipe
        self.processor = MediaPipeProcessor(self.config)
        print("✅ Model đã tải xong!")

        # --- Thiết lập giao diện ---
        self.create_widgets()

        # Bắt đầu vòng lặp cập nhật frame
        self.update_frame()

    def create_widgets(self):
        """Tạo các thành phần trên giao diện"""
        # --- Khung chính ---
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Khung Video ---
        video_frame = ttk.LabelFrame(main_frame, text="Webcam Feed")
        video_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.canvas = tk.Canvas(video_frame, width=self.config.data.width, height=self.config.data.height)
        self.canvas.pack()

        # --- Khung Điều khiển ---
        control_frame = ttk.LabelFrame(main_frame, text="Bảng điều khiển")
        control_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.btn_toggle_cam = ttk.Button(control_frame, text="Bật Camera", command=self.toggle_camera)
        self.btn_toggle_cam.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.btn_toggle_skeleton = ttk.Button(control_frame, text="Tắt Khung xương", command=self.toggle_skeleton)
        self.btn_toggle_skeleton.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.btn_toggle_record = ttk.Button(control_frame, text="Bắt đầu Ghi", command=self.toggle_recording)
        self.btn_toggle_record.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.btn_predict = ttk.Button(control_frame, text="Bắt đầu Dự đoán", command=self.run_prediction)
        self.btn_predict.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # --- Khung Thông tin ---
        info_frame = ttk.LabelFrame(main_frame, text="Thông tin")
        info_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(info_frame, text="Trạng thái:", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=5, pady=(5,0))
        self.lbl_status = ttk.Label(info_frame, text="Sẵn sàng", foreground="blue", font=("Helvetica", 11))
        self.lbl_status.pack(anchor="w", padx=5, pady=(0,10))

        ttk.Label(info_frame, text="Thời gian ghi hình:", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=5, pady=(5,0))
        self.lbl_timer = ttk.Label(info_frame, text="0.0 giây", font=("Helvetica", 11))
        self.lbl_timer.pack(anchor="w", padx=5, pady=(0,10))

        ttk.Label(info_frame, text="KẾT QUẢ DỰ ĐOÁN:", font=("Helvetica", 14, "bold"), foreground="green").pack(anchor="w", padx=5, pady=(15,0))
        self.lbl_prediction_result = ttk.Label(info_frame, text="---", font=("Helvetica", 16, "bold"))
        self.lbl_prediction_result.pack(anchor="w", padx=5, pady=(0,5))
        
        self.lbl_confidence = ttk.Label(info_frame, text="Độ tin cậy: 0.0%", font=("Helvetica", 11))
        self.lbl_confidence.pack(anchor="w", padx=5, pady=(0,10))

    def update_frame(self):
        """Vòng lặp chính để cập nhật frame từ camera và hiển thị"""
        display_frame = None

        if self.is_camera_on and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1) # Lật frame cho giống gương
                
                # 1. Xử lý frame để lấy tọa độ keypoints (không vẽ)
                #    Hàm này trả về (khung hình gốc, kết quả landmarks)
                original_frame, res = self.processor.process_frame(frame)

                # 2. Quyết định xem có vẽ khung xương hay không
                if self.show_skeleton:
                    # Nếu BẬT, gọi hàm draw_landmarks chuyên dụng của bạn
                    display_frame = self.processor.draw_landmarks(original_frame, res)
                else:
                    # Nếu TẮT, chỉ sử dụng khung hình gốc
                    display_frame = original_frame

                # 3. Ghi lại keypoints nếu đang trong chế độ quay (logic này không đổi)
                if self.is_recording:
                    keypoints = self.processor.extract_keypoints(res)
                    if keypoints is not None:
                        self.keypoints_sequence.append(keypoints)
                    
                    elapsed_time = time.time() - self.recording_start_time
                    self.lbl_timer.config(text=f"{elapsed_time:.1f} giây")
        
        # --- Hiển thị frame lên giao diện ---
        # Nếu có frame để hiển thị
        if display_frame is not None:
            # Chuyển màu từ BGR (OpenCV) sang RGB (Tkinter) để hiển thị đúng
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        # Nếu camera tắt, hiển thị khung hình đen
        else:
             blank_frame = np.zeros((self.config.data.height, self.config.data.width, 3), dtype=np.uint8)
             self.photo = ImageTk.PhotoImage(image=Image.fromarray(blank_frame))
             self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Lên lịch cho lần cập nhật tiếp theo
        self.window.after(15, self.update_frame)

    def toggle_camera(self):
        """Bật hoặc tắt camera"""
        if self.is_camera_on:
            self.is_camera_on = False
            self.btn_toggle_cam.config(text="Bật Camera")
            if self.cap:
                self.cap.release()
            self.lbl_status.config(text="Camera đã tắt", foreground="red")
        else:
            self.cap = cv2.VideoCapture(0)
            
            # === BẮT ĐẦU PHẦN THÊM MỚI ===
            # Yêu cầu webcam sử dụng độ phân giải và FPS từ config
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.data.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.data.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.data.video_fps)
            # === KẾT THÚC PHẦN THÊM MỚI ===

            if self.cap.isOpened():
                self.is_camera_on = True
                self.btn_toggle_cam.config(text="Tắt Camera")
                self.lbl_status.config(text="Camera đang bật", foreground="green")
            else:
                self.lbl_status.config(text="Lỗi: Không thể mở camera", foreground="red")

    def toggle_skeleton(self):
        """Bật hoặc tắt hiển thị khung xương"""
        self.show_skeleton = not self.show_skeleton
        text = "Tắt Khung xương" if self.show_skeleton else "Bật Khung xương"
        self.btn_toggle_skeleton.config(text=text)

    def toggle_recording(self):
        """Bắt đầu hoặc kết thúc việc ghi hình"""
        if self.is_recording:
            # Dừng ghi
            self.is_recording = False
            self.btn_toggle_record.config(text="Bắt đầu Ghi")
            self.lbl_status.config(text=f"Đã ghi xong {len(self.keypoints_sequence)} frames.", foreground="blue")
        else:
            # Bắt đầu ghi
            if not self.is_camera_on:
                self.lbl_status.config(text="Lỗi: Vui lòng bật camera trước", foreground="red")
                return
            
            self.is_recording = True
            self.btn_toggle_record.config(text="Kết thúc Ghi")
            self.keypoints_sequence = [] # Xóa sequence cũ
            self.recording_start_time = time.time()
            self.lbl_status.config(text="...Đang ghi...", foreground="orange")
            # Xóa kết quả dự đoán cũ
            self.lbl_prediction_result.config(text="---")
            self.lbl_confidence.config(text="Độ tin cậy: 0.0%")

    def run_prediction(self):
        """Chạy dự đoán trên chuỗi keypoints đã ghi"""
        if self.is_recording:
            self.lbl_status.config(text="Lỗi: Vui lòng dừng ghi hình trước", foreground="red")
            return
            
        if not self.keypoints_sequence:
            self.lbl_status.config(text="Lỗi: Chưa có dữ liệu để dự đoán", foreground="red")
            return
        
        self.lbl_status.config(text="Đang xử lý và dự đoán...", foreground="purple")
        
        # Nội suy frame để khớp với đầu vào của model
        input_data = interpolate_frames(self.keypoints_sequence, self.config.hgc_lstm.sequence_length)
        input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        prediction = None
        label = "Unknown"
        confidence = 0.0

        with torch.no_grad():
            output = self.model(input_data)
            probs = torch.softmax(output, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
            
            confidence = max_prob.item()
            prediction = pred_idx.item()
            label = self.id_to_label_mapping.get(prediction + 1, "Unknown")

        # Cập nhật giao diện với kết quả
        self.lbl_prediction_result.config(text=f"{label.upper()}")
        self.lbl_confidence.config(text=f"Độ tin cậy: {confidence:.2%}")
        self.lbl_status.config(text="Dự đoán hoàn tất!", foreground="green")

    def on_closing(self):
        """Xử lý khi đóng cửa sổ ứng dụng"""
        print("Đang đóng ứng dụng...")
        if self.cap:
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    # Khởi chạy ứng dụng
    app = SignLanguageApp()
    app.window.mainloop()