# VSLR (Vietnamese Sign Language Recognition)

Hệ thống nhận dạng ngôn ngữ ký hiệu Việt Nam sử dụng HGC-LSTM(Hierarchical Graph Convolution + LSTM) với MediaPipe để trích xuất keypoints từ video.

## 📋 Tổng quan

Dự án này sử dụng:

- **MediaPipe** để trích xuất keypoints từ video (pose, hands)
- **HGC-LSTM** model để nhận dạng cử chỉ
- **Graph Convolution** để model mối quan hệ giữa các keypoints
- **LSTM** để model temporal dynamics
- **Data Augmentation** để tăng cường dữ liệu (future)

## 🚀 Quy trình thực hiện

### Bước 1: Chuẩn bị dữ liệu video và tạo CSV labels

```bash
python extract_csv.py
```

**Mục đích:** Tổ chức video từ `new_dataset/` theo format chuẩn và tạo file `labels.csv`

**Input:**

- Folder: `new_dataset/` chứa video với format `<id>_<label>_<other>.mp4`
- Ví dụ: `1_xin_chao_001.mp4`, `1_xin_chao_002.mp4`, `2_cam_on_001.mp4`

**Output:**

- Folder: `new_data/` chứa video đã đổi tên: `1_01.mp4`, `1_02.mp4`, `2_01.mp4`, etc.
- File: `labels.csv` chứa mapping giữa ID và label

```csv
id,label,videos
1,xin_chao,"1_01.mp4, 1_02.mp4, 1_03.mp4"
2,cam_on,"2_01.mp4, 2_02.mp4, 2_03.mp4"
```

### Bước 2: Trích xuất keypoints

```bash
python extract_kpts_n_label.py
```

**Mục đích:** Sử dụng MediaPipe để trích xuất keypoints từ video

**Input:** Video từ `data/Videos/`
**Output:**

- `data/Keypoints/` chứa keypoints files (`.npy`)
- `data/Labels/` chứa label files (`.npy`)

**Keypoints format:**

- Shape: `(T, 150)` where T = số frames, 150 = 75 keypoints × 2 coordinates
- 75 keypoints: 33 pose + 21 left hand + 21 right hand

### Bước 3: Training Model

```bash
jupyter notebook train_HGC_LSTM.ipynb
```

hoặc chạy từng cell trong notebook.

## 🔧 Cấu hình

Tất cả parameters được quản lý trong `config.py`:

### Data Configuration

```python
@dataclass
class DataConfig:
    input_csv_file: str = "labels.csv"
    video_input_dir: str = "new_data"
    keypoints_output_dir: str = "data/Keypoints"
    sequence_length: int = 60
    train_split: float = 0.9
```

###HGC-LSTMConfiguration

```python
@dataclass
class HGCLSTMConfig:
    sequence_length: int = 60
    num_vertices: int = 75
    in_channels: int = 2
    hidden_gcn: int = 128
    hidden_lstm: int = 128
    dropout: float = 0.5
    pooling_type: str = "attention"

    # Data Augmentation
    data_augmentation: bool = True
    scale_factors: list = [0.9, 1.0, 1.1]
    translation_x: list = [0.1, 0.15, 0.2, -0.1, -0.15, -0.2]
```

### Training Configuration

```python
@dataclass
class TrainingConfig:
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    scheduler: str = "step"
    early_stopping_patience: int = 50
    model_save_name: str = "best_hgc_lstm.pth"
```

## 📊 Data Augmentation

Hệ thống hỗ trợ data augmentation để tăng cường dataset:

**Mỗi sample gốc sẽ tạo ra 18 augmented samples:**

- 3 scale factors (0.9, 1.0, 1.1)
- 6 translation values (±0.1, ±0.15, ±0.2)
- Total: 3 × 6 = 18 combinations

**Cách bật/tắt:**

```python
# Trong notebook
USE_DATA_AUGMENTATION = True  # hoặc False

# Hoặc trong config.py
data_augmentation: bool = True
```

## 🏗️ Model Architecture

### HGC-LSTMModel

```
Input: (B, T, V, C) = (batch, 60, 75, 2)
↓
GCN Layer 1: (B, T, V, 128)
↓
GCN Layer 2: (B, T, V, 128)
↓
Spatial Pooling: (B, T, 128)
↓
LSTM: (B, T, 128)
↓
Take last timestep: (B, 128)
↓
Dropout + FC: (B, num_classes)
```

### Graph Convolution

- **Adjacency Matrix:** 75×75 modeling skeleton connections
- **Pose connections:** MediaPipe pose landmarks
- **Hand connections:** Left hand (33-53) + Right hand (54-74)
- **Normalization:** Symmetric normalized adjacency matrix

### Spatial Pooling Types

- **adaptive_avg:** Average pooling over joints
- **adaptive_max:** Max pooling over joints
- **attention:** Learnable attention weights

## 📈 Training Process

### 1. Data Loading

```python
# Tự động load từ config
train_dataset = SignLanguageDataset(split_type='train')
val_dataset = SignLanguageDataset(split_type='val')
```

### 2. Model Training

```python
# Với data augmentation: 18x samples
# Với early stopping và learning rate scheduling
history = train_model(model, train_loader, val_loader, config, device)
```

### 3. Evaluation

```python
# Load best model và evaluate
model.load_state_dict(torch.load('best_hgc_lstm.pth'))
accuracy = evaluate_model(model, val_loader, device)
```

## 🎯 Inference

```python
# Predict single sequence
result = predict_sequence(model, keypoints, device, labels)
print(f"Predicted: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## 📁 Cấu trúc thư mục

```
VSLR/
├── config.py                 # Configuration management
├── extract_csv.py            # Video organization & CSV creation
├── cv_to_60.py              # Video FPS conversion
├── extract_kpts_n_label.py  # Keypoints extraction
├── train_HGC_LSTM.ipynb    # Training notebook
├── detector.py              # MediaPipe processing
├── labels.csv               # Labels mapping
├── new_dataset/             # Original videos
├── new_data/                # Organized videos
├── data/
│   ├── Videos/              # 60 FPS videos
│   ├── Keypoints/           # Extracted keypoints (.npy)
│   └── Labels/              # Labels (.npy)
└── models/
    └── best_hgc_lstm.pth   # Trained model
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**

   - Giảm `batch_size` trong config
   - Tắt data augmentation: `data_augmentation = False`

2. **Low Accuracy:**

   - Tăng `num_epochs`
   - Thử `pooling_type = "attention"`
   - Kiểm tra data quality

3. **Training quá chậm:**
   - Giảm `sequence_length`
   - Giảm `hidden_gcn` và `hidden_lstm`
   - Tắt data augmentation

### Performance Tips

1. **Tối ưu Data Augmentation:**

```python
# Giảm augmentation combinations
scale_factors = [0.9, 1.0, 1.1]     # 3 scales
translation_x = [0.1, -0.1]         # 2 translations
# Total: 3 × 2 = 6 combinations thay vì 18
```

2. **Tối ưu Model:**

```python
# Giảm model complexity
hidden_gcn = 64
hidden_lstm = 64
dropout = 0.3
```

3. **Tối ưu Training:**

```python
# Faster convergence
learning_rate = 2e-3
scheduler_step_size = 10
scheduler_gamma = 0.7
```

## 🎊 Kết quả

Model sẽ output:

- **Training history:** Loss và accuracy curves
- **Best model:** Saved as `best_hgc_lstm.pth`
- **Classification report:** Precision, recall, F1-score cho từng class

---

**Ngày cập nhật:** July 2025
