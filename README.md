# Vietnamese Sign Language Recognition (VSLR)

A deep learning project for Vietnamese Sign Language Recognition using HGC-LSTM (Hierarchical Graph Convolution + Long Short-Term Memory) architecture with MediaPipe keypoints extraction.

## 🏗️ Project Structure

```
VSLR/
├── configs/                    # Configuration files
│   └── config.py              # Main configuration settings
├── src/                       # Source code modules
│   ├── models/                # Model architectures
│   │   └── model_utils.py     # HGC-LSTM model, GCN layers, attention pooling
│   ├── training/              # Training utilities
│   │   └── train_utils.py     # Training loops, optimizers, schedulers
│   └── utils/                 # Utility functions
│       ├── data_utils.py      # Dataset classes, data loading, augmentation
│       └── visualization_utils.py  # Plotting and visualization functions
├── scripts/                   # Executable scripts
│   ├── train_hgc_lstm.py      # Main training script
│   ├── detector.py            # Sign language detection
│   ├── inference.py           # Model inference
│   ├── extract_kpts_n_label.py   # Keypoint extraction
│   ├── extract_csv.py         # CSV data processing
│   └── cv_to_60.py           # Video frame conversion
├── outputs/                   # Output files
│   ├── models/               # Trained model checkpoints
│   ├── plots/                # Training curves, confusion matrices
│   └── logs/                 # Training logs
├── data/                     # Processed data
│   ├── videos/               # Video files
│   └── dataset/              # Processed datasets
├── data_original/            # Original raw data
│   ├── Keypoints/           # Extracted keypoints (.npy files)
│   ├── Labels/              # Label files
│   └── Videos/              # Original video files
├── notebooks/               # Jupyter notebooks
└── train_HGC_LSTM.ipynb    # Main training notebook (updated with modular imports)
```

## 🚀 Quick Start

### 1. Training with Script

```bash
python scripts/train_hgc_lstm.py
```

### 2. Training with Notebook

Open `train_HGC_LSTM.ipynb` in Jupyter/VS Code and run the cells.

## 🔧 Configuration

All configuration is centralized in `configs/config.py`:

- **Data Config**: Input/output paths, augmentation settings
- **Model Config**: HGC-LSTM architecture parameters
- **Training Config**: Optimizer, scheduler, training parameters
- **Output Config**: Paths for models, plots, logs

## 📊 Features

### Model Architecture

- **HGC-LSTM**: Hierarchical Graph Convolution + LSTM
- **MediaPipe Integration**: 75 keypoints (33 pose + 21 left hand + 21 right hand)
- **Attention Pooling**: Adaptive attention mechanism
- **Graph Convolution**: Spatial relationship modeling

### Data Augmentation

- **Horizontal Flipping**: Left-right hand/pose swapping
- **Geometric Transforms**: Translation and scaling
- **Stratified Splitting**: Balanced train/validation sets

### Visualization

- **Training Curves**: Loss and accuracy plots
- **Confusion Matrix**: Classification performance analysis
- **Real-time Monitoring**: Progress tracking during training

## 📈 Usage Examples

### Training Configuration

```python
from configs.config import Config

config = Config()
config.training.num_epochs = 300
config.training.batch_size = 8
config.data.use_flip_augmentation = True
```

### Data Loading

```python
from src.utils.data_utils import SignLanguageDataset, load_labels_from_csv

# Load labels and create dataset
video_to_label_mapping, label_to_idx, unique_labels, _ = load_labels_from_csv("labels.csv", config)

train_dataset = SignLanguageDataset(
    keypoints_dir=config.data.input_kp_path,
    video_to_label_mapping=video_to_label_mapping,
    label_to_idx=label_to_idx,
    config=config,
    split_type='train'
)
```

### Model Training

```python
from src.models.model_utils import HGC_LSTM
from src.training.train_utils import train_model

model = HGC_LSTM(
    num_vertices=75,
    in_channels=2,
    hidden_channels=128,
    num_classes=len(unique_labels)
)

history = train_model(model, train_loader, val_loader, optimizer, scheduler, config, device)
```

## 🛠️ Dependencies

- PyTorch >= 1.9.0
- MediaPipe >= 0.8.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Scikit-learn >= 1.0.0
- OpenCV >= 4.5.0

## 📝 File Organization

### Core Modules

- `src/models/model_utils.py`: Model architectures and layers
- `src/training/train_utils.py`: Training pipeline and utilities
- `src/utils/data_utils.py`: Data loading and augmentation
- `src/utils/visualization_utils.py`: Plotting and visualization

### Scripts

- `scripts/train_hgc_lstm.py`: Complete training pipeline
- `scripts/detector.py`: Real-time sign language detection
- `scripts/inference.py`: Model inference on new data

### Configuration

- `configs/config.py`: Centralized configuration management

## 🎯 Model Performance

The HGC-LSTM model achieves state-of-the-art performance on Vietnamese Sign Language recognition tasks with:

- Hierarchical graph convolution for spatial relationship modeling
- LSTM networks for temporal sequence learning
- Attention pooling for adaptive feature aggregation
- Comprehensive data augmentation for improved generalization

## 📄 License

This project is for educational and research purposes.

---

## 🔄 Migration Notes

This project has been reorganized from a monolithic Jupyter notebook structure to a modular Python package:

### What Changed:

- ✅ Functions extracted to separate modules in `src/`
- ✅ Configuration centralized in `configs/config.py`
- ✅ Scripts organized in `scripts/` folder
- ✅ Outputs organized in `outputs/` folder
- ✅ Import statements updated in notebook
- ✅ Proper folder hierarchy established

### Benefits:

- 🔧 **Modularity**: Each component has a specific role
- 🔄 **Reusability**: Functions can be imported and reused
- 🧪 **Testability**: Individual modules can be tested
- 📦 **Maintainability**: Easier to update and debug
- 🚀 **Scalability**: Easy to add new features
