import cv2
import pandas as pd
import numpy as np
import os
import sys
import absl.logging
from detector import MediaPipeProcessor
from config import Config
# ---------- Khởi tạo model ----------
model = MediaPipeProcessor(Config())
vid_not_kept = []


def moved(curr_kps: np.ndarray, ref_kps: np.ndarray, threshold=0.07) -> bool:
    """
    Trả về True nếu có ít nhất một phần tử của curr_kps khác ref_kps
    hơn ngưỡng threshold (normalized).
    """
    # So sánh độ chênh tuyệt đối từng chiều
    diffs = np.abs(curr_kps - ref_kps)
    return np.any(diffs > threshold)

# ---------- Lọc & ghi video ----------
def filter_video(src, dst):
    cap = cv2.VideoCapture(src)
    # 1. Đọc khung đầu tiên, lấy keypoints làm reference
    ret, frame0 = cap.read()
    if not ret:
        print(f"✘ Không đọc được khung đầu của {src}")
        cap.release()
        return

    image0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    _, ref_res = model.process_frame(image0)
    ref_kps = model.extract_keypoints(ref_res)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    kept = dropped = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, res = model.process_frame(image_rgb)
        
        if res.pose_landmarks:
            kpts = model.extract_keypoints(res)
            if moved(kpts, ref_kps):
                out.write(frame); kept += 1
                continue

        dropped += 1

    if kept == 0:
        print(f"✘ Không giữ khung nào từ video: {src}")
        vid_not_kept.append(src)
        os.remove(dst)  # xóa file video đã tạo
    cap.release()
    out.release()
    print(f"✔ Giữ {kept} khung, loại {dropped} khung – video lưu: {dst}")

# ---------- Chạy thử ----------
def main():
    files_name = 'top50_daily.csv'
    df = pd.read_csv(files_name)
    for i in df.index:
        files = [x.strip() for x in df.loc[i, "VIDEO"].split(",")]
        print(f"Đang xử lý video gloss {i}: {files}")
        for file in files:
            print(f"Đang xử lý video: {file}")
            src = os.path.join('Dataset', 'Video', file)
            dst = os.path.join('data', 'Video', file)
            if not os.path.exists(src):
                print(f"✘ Không tìm thấy video: {src}")
                continue
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
            filter_video(src, dst)
            print(f"Video đã lưu tại: {dst}")
            print("-" * 50)
            print("\n")
    print(len(vid_not_kept), "video không giữ khung nào:", vid_not_kept)
if __name__ == '__main__':
    main()
    