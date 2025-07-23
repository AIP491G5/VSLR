# Vietnamese Sign Language Recognition (VSLR)

A deep learning project for Vietnamese Sign Language Recognition using HGC-LSTM (Hierarchical Graph Convolution + Long Short-Term Memory) architecture with MediaPipe keypoints extraction.

## Project Structure

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

## Quick Start

### Training with Script

```bash
python scripts/train_hgc_lstm.py
```

### Training with Notebook

Open `train_HGC_LSTM.ipynb` in Jupyter/VS Code and run the cells.

## Configuration

All configuration is centralized in `configs/config.py`:

- **Data Config**: Input/output paths, augmentation settings
- **Model Config**: HGC-LSTM architecture parameters
- **Training Config**: Optimizer, scheduler, training parameters
- **Output Config**: Paths for models, plots, logs

## Features

### Model Architecture

- **HGC-LSTM**: Hierarchical Graph Convolution + LSTM
- **MediaPipe Integration**: 75 keypoints (33 pose + 21 left hand + 21 right hand)
- **Attention Pooling**: Adaptive attention mechanism
- **Graph Convolution**: Spatial relationship modeling

### Data Augmentation

The new augmentation system uses a simple array configuration:

```python
# Available augmentations: 'flip', 'translation', 'scaling'
config.data.augmentations = ['flip', 'translation', 'scaling']  # All augmentations
config.data.augmentations = ['flip']                            # Only horizontal flip
config.data.augmentations = ['translation', 'scaling']         # No flip (for sign language)
config.data.augmentations = []                                 # No augmentation
```

**Augmentation Options:**

- **Horizontal Flipping**: Left-right hand/pose swapping
- **Translation**: Random position shifts
- **Scaling**: Random size changes
- **Stratified Splitting**: Balanced train/validation sets

### Visualization

- **Training Curves**: Loss and accuracy plots
- **Confusion Matrix**: Classification performance analysis
- **Real-time Monitoring**: Progress tracking during training

## Usage Examples

### Training Configuration

```python
from configs.config import Config

config = Config()
config.training.num_epochs = 300
config.training.batch_size = 8
config.data.augmentations = ['flip', 'translation']  # Use flip and translation only
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
from src.models.model_utils import create_model, create_adjacency_matrix
from src.training.train_utils import train_model

A = create_adjacency_matrix(config)
model = create_model(config, A, num_classes=10, device='cuda')
history = train_model(model, train_loader, val_loader, config, device)
```

## Dependencies

- PyTorch >= 1.9.0
- MediaPipe >= 0.8.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Scikit-learn >= 1.0.0
- OpenCV >= 4.5.0

## File Organization

### Core Modules

- `src/models/model.py`: Model architectures and layers
- `src/models/model_utils.py`: Model utils, create model
- `src/training/train_utils.py`: Training pipeline and utilities
- `src/utils/data_utils.py`: Data loading and augmentation
- `src/utils/visualization_utils.py`: Plotting and visualization

### Scripts

- `scripts/train_hgc_lstm.py`: Complete training pipeline
- `scripts/detector.py`: Real-time sign language detection
- `scripts/inference.py`: Model inference on new data

### Configuration

- `configs/config.py`: Centralized configuration management

## Model Performance

The HGC-LSTM model achieves state-of-the-art performance on Vietnamese Sign Language recognition tasks with:

- Hierarchical graph convolution for spatial relationship modeling
- LSTM networks for temporal sequence learning
- Attention pooling for adaptive feature aggregation
- Comprehensive data augmentation for improved generalization

## Data Augmentation System

The project includes a flexible augmentation system that supports:

| Configuration                        | Combinations                                        | Multiplier |
| ------------------------------------ | --------------------------------------------------- | ---------- |
| `[]`                                 | original                                            | 1x         |
| `['flip']`                           | original, flip                                      | 2x         |
| `['translation']`                    | original, translation                               | 2x         |
| `['scaling']`                        | original, scaling                                   | 2x         |
| `['flip', 'translation']`            | original, flip, translation, flip+translation       | 4x         |
| `['translation', 'scaling']`         | original, translation, scaling, translation+scaling | 4x         |
| `['flip', 'translation', 'scaling']` | All combinations                                    | 8x         |

## License

This project is for educational and research purposes.
