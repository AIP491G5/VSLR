"""
Configuration management for Vietnamese Sign Language Detection System.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MediaPipeConfig:
    """Configuration for MediaPipe processing."""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False
    model_complexity: int = 1


@dataclass
class DataConfig:
    """Configuration for data handling."""
    # Input data paths
    input_csv_file: str = "labels.csv"
    triplet_csv_file: str = "triplet_dataset.csv"
    input_kp_path: str = "dataset/Keypoints"
    raw_video_input_dir: str = "data/dataset"  # Raw video input directory for processing
    video_input_dir: str = "data/videos"
    
    # Output data paths
    data_dir: str = "data"
    video_output_dir: str = "dataset/Videos"
    keypoints_output_dir: str = "dataset/Keypoints"
    labels_output_dir: str = "dataset/Labels"
    
    # Processing parameters
    movement_threshold: float = 0.36
    video_fps: int = 30
    label_frames_needed: int = 0
    
    # Dataset splitting and augmentation
    use_strategy: bool = True  # True for stratified split, False for random split
    
    # New flexible augmentation system
    # Available options: 'flip', 'translation', 'scaling'
    # Example: ['flip', 'translation'] will use flip and translation augmentations
    # Example: ['scaling'] will use only scaling augmentation
    # Example: [] will use no augmentation
    augmentations: list = field(default_factory=lambda: ['translation', 'scaling'])
    
    # Augmentation parameters
    translation_range: float = 0.1  # Random translation range (-0.1 to +0.1)
    scale_range: float = 0.1  # Random scaling range (0.9 to 1.1)
    
    # File extensions
    video_input_ext: str = ".mp4"
    video_output_ext: str = ".mp4"
    keypoints_ext: str = ".npy"
    labels_ext: str = ".npy"


@dataclass
class HGCLSTMConfig:
    """Configuration for HGC-LSTM model."""
    # Model architecture
    in_channels: int = 2  # x, y coordinates
    hidden_gcn: int = 128
    hidden_lstm: int = 128
    dropout: float = 0.5
    
    # Data parameters
    sequence_length: int = 60
    num_vertices: int = 75  # Total keypoints (33 pose + 21 left hand + 21 right hand)

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    # Training parameters
    num_epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    triplet_margin: float = 2.0  # Margin for triplet loss
    # Optimizer settings
    optimizer: str = "adam"  # "adam" or "sgd"
    momentum: float = 0.9  # For SGD
    
    # Learning rate scheduler
    scheduler: str = "step"  # "step", "cosine", or None
    scheduler_step_size: int = 20  # For HGC-LSTM
    scheduler_gamma: float = 0.5   # For HGC-LSTM
    
    # Training behavior
    early_stopping_patience: int = 50
    gradient_clip_norm: float = 1.0
    
    # Data split
    train_split: float = 0.9
    val_split: float = 0.1
    
    # Saving and logging
    save_dir: str = "outputs/models"
    save_triplet_dir: str = "outputs/models"
    # save_interval: int = 10
    log_interval: int = 1
    model_save_name: str = "best_hgc_lstm.pth"
    model_triplet_save_name: str = "best_hgc_lstm_embedding.pth"
    
    # Device settings
    device: str = "auto"  # "auto", "cpu", "cuda"
    mixed_precision: bool = False
    random_seed: int = 42
    
    # Debugging
    cuda_launch_blocking: bool = True


@dataclass
class ModelConfig:
    """Configuration for model saving and loading."""
    checkpoint_dir: str = "outputs/models"
    save_name: str = "best_hgc_lstm.pth"


@dataclass
class OutputConfig:
    """Configuration for output paths."""
    base_dir: str = "outputs"
    plots_dir: str = "outputs/plots"
    logs_dir: str = "outputs/logs"
    models_dir: str = "outputs/models"


@dataclass
class Config:
    """Main configuration class."""
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hgc_lstm: HGCLSTMConfig = field(default_factory=HGCLSTMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        config = cls()
        
        if 'mediapipe' in data:
            config.mediapipe = MediaPipeConfig(**data['mediapipe'])
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'hgc_lstm' in data:
            config.hgc_lstm = HGCLSTMConfig(**data['hgc_lstm'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'mediapipe': self.mediapipe.__dict__,
            'data': self.data.__dict__,
            'hgc_lstm': self.hgc_lstm.__dict__,
            'training': self.training.__dict__,
        }
    
    def save(self, config_path: str) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
            elif config_path.suffix.lower() == '.json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def get_absolute_path(self, relative_path: str) -> str:
        """Convert relative path to absolute path."""
        if os.path.isabs(relative_path):
            return relative_path
        
        # Get the project root directory
        current_dir = Path(__file__).parent
        return str(current_dir / relative_path)
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.data.video_input_dir,
            self.data.data_dir,
            self.data.video_output_dir,
            self.data.keypoints_output_dir,
            self.data.labels_output_dir,
            self.training.save_dir,
        ]
        
        for directory in directories:
            abs_path = self.get_absolute_path(directory)
            Path(abs_path).mkdir(parents=True, exist_ok=True)
    
    def print_config(self) -> None:
        """Print all configuration parameters."""
        print("\n" + "="*60)
        print(" " * 20 + "CONFIGURATION SUMMARY")
        print("="*60)
        
        print(f"\n[MEDIAPIPE CONFIG]")
        for key, value in self.mediapipe.__dict__.items():
            print(f"  {key}: {value}")
        
        print(f"\n[DATA CONFIG]")
        for key, value in self.data.__dict__.items():
            print(f"  {key}: {value}")
        
        print(f"\n[MODEL CONFIG]")
        for key, value in self.model.__dict__.items():
            print(f"  {key}: {value}")
        
        print(f"\n[HGC-LSTM CONFIG]")
        for key, value in self.hgc_lstm.__dict__.items():
            print(f"  {key}: {value}")
        
        print(f"\n[TRAINING CONFIG]")
        for key, value in self.training.__dict__.items():
            print(f"  {key}: {value}")
        
        print("="*60)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration with fallback to default."""
    if config_path is None:
        # Try to find config file in standard locations
        config_locations = [
            "configs/config.yaml",
            "configs/config.yml", 
            "configs/config.json",
            "config.yaml",
            "config.yml",
            "config.json",
        ]
        
        for location in config_locations:
            if os.path.exists(location):
                config_path = location
                break
    
    if config_path and os.path.exists(config_path):
        try:
            return Config.from_file(config_path)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration.")
    
    # Return default configuration
    config = Config()
    config.ensure_directories()
    return config