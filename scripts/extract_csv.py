import os
import shutil
import pandas as pd
from collections import defaultdict
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs.config import Config

def extract_csv(input_folder=None, output_folder=None, csv_file=None):
    """
    Tạo file CSV chứa id, label, videos từ folder chứa video có format <id>_label_...
    và chuyển video sang thư mục mới với tên <id>_<No>.mp4
    
    Args:
        input_folder (str): Thư mục chứa video gốc (nếu None, sẽ dùng từ config)
        output_folder (str): Thư mục đích để lưu video đã đổi tên (nếu None, sẽ dùng từ config)
        csv_file (str): Tên file CSV output (nếu None, sẽ dùng từ config)
    """
    
    # Initialize config
    config = Config()
    
    # Use config values if parameters are None
    if input_folder is None:
        input_folder = getattr(config.data, 'raw_video_input_dir', 'new_dataset')
    if output_folder is None:
        output_folder = config.data.video_input_dir
    if csv_file is None:
        csv_file = config.data.input_csv_file
    
    print(f"[INFO] Input folder: {input_folder}")
    print(f"[INFO] Output folder: {output_folder}")
    print(f"[INFO] CSV file: {csv_file}")
    
    # Tạo thư mục đích nếu chưa có
    os.makedirs(output_folder, exist_ok=True)
    
    # Dictionary để nhóm video theo id và label
    video_groups = defaultdict(lambda: {"label": "", "videos": []})
    
    # Đọc tất cả video trong folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in video_files:
        # Parse tên file: <id>_<label>_...
        parts = video_file.split('_')
        if len(parts) >= 2:
            video_id = parts[0]
            label = parts[1]
            
            # Lấy số thứ tự hiện tại cho id này
            current_count = len(video_groups[video_id]["videos"]) + 1
            
            # Tạo tên file mới
            new_filename = f"{video_id}_{current_count:02d}.mp4"
            
            # Copy video sang thư mục mới với tên mới
            old_path = os.path.join(input_folder, video_file)
            new_path = os.path.join(output_folder, new_filename)
            shutil.copy2(old_path, new_path)
            
            # Cập nhật thông tin
            video_groups[video_id]["label"] = label
            video_groups[video_id]["videos"].append(new_filename)
            
            print(f"Moved: {video_file} -> {new_filename}")
    
    # Tạo DataFrame cho CSV
    csv_data = []
    for video_id, info in video_groups.items():
        csv_data.append({
            "id": video_id,
            "label": info["label"],
            "videos": ", ".join(info["videos"])
        })
    
    # Sắp xếp theo id
    csv_data.sort(key=lambda x: x["id"])
    
    # Tạo CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    
    print(f"\n[SUCCESS] Tạo file CSV: {csv_file}")
    print(f"[SUCCESS] Tổ chức {len(video_files)} video vào thư mục: {output_folder}")
    print(f"[INFO] Tìm thấy {len(video_groups)} ID khác nhau")
    
    return df

# Ví dụ sử dụng
if __name__ == "__main__":
    # Sử dụng config mặc định hoặc có thể override
    config = Config()
    
    # Option 1: Sử dụng config mặc định
    print("[INFO] Sử dụng config mặc định:")
    df = extract_csv()
    
    # Option 2: Override một số tham số
    # df = extract_csv(
    #     input_folder="my_raw_videos",  # Thư mục chứa video gốc
    #     output_folder="processed_videos",  # Thư mục đích
    #     csv_file="my_labels.csv"  # File CSV output
    # )
    
    # Hiển thị kết quả
    print("\n[INFO] Nội dung file CSV:")
    print(df.to_string(index=False))
