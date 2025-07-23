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
