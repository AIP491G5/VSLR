from detector import MediaPipeProcessor
from config   import Config
import cv2

# Khởi tạo processor và danh sách lưu keypoints
pose      = MediaPipeProcessor(Config())
all_poses = []  # mỗi phần tử sẽ là list of (x,y,z,visibility) cho 33 điểm

cap = cv2.VideoCapture('Dataset/Video/W03373B.mp4')
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pose Detection', 640, 480)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.flip(image_rgb, 1)  # Lật ảnh ngang nếu cần
    _, results = pose.process_frame(image_rgb)

    # Kiểm tra trước khi truy cập .landmark
    if results.pose_landmarks:
        pts = results.pose_landmarks.landmark
        # Lưu lại keypoints (x,y,z,visibility) nếu cần
        all_poses.append([
            (lm.x, lm.y, lm.z, lm.visibility)
            for lm in pts
        ])

        # Vẽ lên ảnh
        output = pose.draw_landmarks(frame, results)
    else:
        print("No pose detected in this frame.")
        output = frame
    kpts = pose.extract_keypoints(results)
    print(f"Extracted keypoints: {len(kpts)}")
    cv2.moveWindow('Pose Detection',0, 0)
    cv2.imshow('Pose Detection', output)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
