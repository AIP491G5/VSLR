import os
import cv2
import numpy as np

def interpolate_frames(frames, target=60):
    """
    Nếu len(frames) >= target: sample đều target khung.
    Nếu len(frames) < target: nội suy giữa các khung để đủ target khung.
    """
    L = len(frames)
    if L >= target:
        idx = np.linspace(0, L - 1, target).astype(int)
        return [frames[i] for i in idx]

    # L < target: cần tạo thêm total_missing khung
    total_missing = target - L
    intervals = L - 1  # số khoảng giữa các khung
    base_missing = total_missing // intervals
    extras = total_missing % intervals

    out = []
    for i in range(intervals):
        out.append(frames[i])
        # Với mỗi khoảng [i, i+1], chèn k khung nội suy
        k = base_missing + (1 if i < extras else 0)
        for j in range(1, k+1):
            alpha = j / (k+1)
            # nội suy tuyến tính (blend) giữa frames[i] và frames[i+1]
            f1 = frames[i].astype(np.float32)
            f2 = frames[i+1].astype(np.float32)
            fi = cv2.addWeighted(f2, alpha, f1, 1-alpha, 0)
            out.append(fi.astype(np.uint8))
    out.append(frames[-1])
    return out

def process_video(input_path, output_path, target_frames=60):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"⚠️ Không mở được: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret: break
        frames.append(fr)
    cap.release()

    if len(frames) == 0:
        print(f"⚠️ Video rỗng: {input_path}")
        return

    # interpolate hoặc sample to exactly target_frames
    norm = interpolate_frames(frames, target_frames)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in norm:
        out.write(f)
    out.release()

def main():
    inp_dir  = 'Data_1/Video'  # path lấy video đã cắt
    out_dir  = 'Data_1/Video_60frames'  # path lưu video đủ 60 frames
    os.makedirs(out_dir, exist_ok=True)

    for fn in os.listdir(inp_dir):
        if not fn.lower().endswith(('.mp4','.avi','.mov','.mkv')):
            continue
        src = os.path.join(inp_dir, fn)
        dst = os.path.join(out_dir, fn)
        print(f"→ Processing {fn} …")
        process_video(src, dst)
    print("✅ Xong tất cả.")

if __name__ == '__main__':
    main()
