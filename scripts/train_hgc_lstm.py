"""
Main training script for VSLR project
Simplified and organized version of the notebook training process
"""

import numpy as np
import torch
import warnings
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import utilities
from configs.config import Config
from src.utils.data_utils import load_labels_from_csv, SignLanguageDataset, create_data_loaders
from src.utils.model_utils import create_adjacency_matrix, create_model
from src.utils.train_utils import train_model
from src.utils.visualization_utils import visualize_training_process, analyze_model_performance

warnings.filterwarnings("ignore")


def setup_logging():
    """Setup logging to both console and file"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(project_root, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # Also print to console
        ]
    )
    
    # Create custom logger
    logger = logging.getLogger('VSLR_Training')
    
    # Also redirect print statements to logger
    class LoggerWriter:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.linebuf = ''

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.level, line.rstrip())

        def flush(self):
            pass
    
    # Redirect stdout to logger
    sys.stdout = LoggerWriter(logger, logging.INFO)
    
    return logger, log_file


def main():
    """Main training function"""
    # Setup logging first
    logger, log_file = setup_logging()
    
    print("="*80)
    print("🚀 VSLR Training Started")
    print("="*80)
    print(f"📝 Log file: {log_file}")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize configuration
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Using device: {device}")
    print(f"Configuration loaded")
    
    # Load labels and create mappings
    print("Loading labels...")
    video_to_label_mapping, label_to_idx, unique_labels, id_to_label_mapping = load_labels_from_csv(
        None, config
    )
    num_classes = len(unique_labels)
    
    # Create datasets
    print("\n Creating datasets...")
    keypoints_dir = config.data.keypoints_output_dir
    
    # Get parameters from config
    use_strategy = config.data.use_strategy
    
    # Get augmentations from config
    train_augmentations = getattr(config.data, 'augmentations', [])
    val_augmentations = []  # Validation uses no augmentation for fair evaluation
    
    print(f"Configuration:")
    print(f"   Split strategy: {'Stratified' if use_strategy else 'Random'}")
    print(f"   Train augmentations: {train_augmentations if train_augmentations else 'None'}")
    print(f"   Val augmentations: {val_augmentations if val_augmentations else 'None (for fair evaluation)'}")
    if 'translation' in train_augmentations:
        print(f"   Translation range: ±{config.data.translation_range}")
    if 'scaling' in train_augmentations:
        print(f"   Scale range: ±{config.data.scale_range}")
    
    train_dataset = SignLanguageDataset(
        keypoints_dir, video_to_label_mapping, label_to_idx, config,
        split_type='train', 
        augmentations=train_augmentations,
        use_strategy=use_strategy
    )
    
    val_dataset = SignLanguageDataset(
        keypoints_dir, video_to_label_mapping, label_to_idx, config,
        split_type='val', 
        augmentations=val_augmentations,
        use_strategy=use_strategy
    )
    
    print(f"\nDataset summary:")
    print(f"  Total classes: {len(unique_labels)}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Strategy: {'Stratified' if use_strategy else 'Random'} split")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)
    
    # Create adjacency matrix
    print("\nCreating adjacency matrix...")
    A = create_adjacency_matrix(config)
    print(f" Adjacency matrix shape: {A.shape}")
    print(f" Number of vertices: {config.hgc_lstm.num_vertices}")
    
    # Create model
    print("\nCreating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config, A, num_classes, device)

    # Start training
    print("\nStarting training...")
    print(f" Training configuration:")
    print(f"  - Epochs: {config.training.num_epochs}")
    print(f"  - Batch size: {config.training.batch_size}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Optimizer: {config.training.optimizer}")
    print(f"  - Scheduler: {config.training.scheduler}")
    print(f"  - Early stopping patience: {config.training.early_stopping_patience}")
    
    history = train_model(model, train_loader, val_loader, config, device)
    
    # Generate visualizations
    print("\nGenerating training visualizations...")
    model_path = Path(config.training.save_dir) / config.training.model_save_name
    print(f"Loading best model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    visualize_training_process(history, config)
    analyze_model_performance(model, val_loader, device, config, unique_labels, id_to_label_mapping)
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"🏆 Best validation accuracy: {max(history.get('val_acc', history.get('val_acc', [0]))):.2f}%")
    print(f"💾 Model saved to: {config.training.save_dir}/{config.training.model_save_name}")
    print(f"📊 Plots saved to: {config.output.plots_dir}")
    print(f"📝 Training log saved to: {log_file}")
    print(f"📅 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
