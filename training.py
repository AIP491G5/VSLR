import pandas as pd
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings

from config import Config
from detector import MediaPipeProcessor
from model import HGC_LSTM

warnings.filterwarnings("ignore")

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

def load_labels_from_csv(csv_file=None):
    """Load labels from CSV file"""
    if csv_file is None:
        csv_file = config.data.input_csv_file
    
    df = pd.read_csv(csv_file)
    video_to_label_mapping = {}
    id_to_label_mapping = {}

    for _, row in df.iterrows():
        label_id = row['id']
        label_name = row['label']
        videos_str = row['videos']
        video_files = [v.strip() for v in videos_str.split(',')]

        id_to_label_mapping[label_id] = label_name
        
        for video_file in video_files:
            video_base = os.path.splitext(video_file)[0]
            video_to_label_mapping[video_base] = label_id
    
    unique_labels = sorted(df['id'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    print(f"[INFO] {len(unique_labels)} classes: {unique_labels}")
    print(f"[INFO] {len(video_to_label_mapping)} videos")
    
    return video_to_label_mapping, label_to_idx, unique_labels, id_to_label_mapping

class SignLanguageDataset(Dataset):
    """Dataset for Sign Language Recognition with GCN format"""
    
    def __init__(self, keypoints_dir, video_to_label_mapping, label_to_idx, sequence_length=None, split_type='train', train_split=None):
        
        self.sequence_length = sequence_length or config.hgc_lstm.sequence_length
        train_split = train_split or config.training.train_split
        
        self.keypoints_dir = keypoints_dir
        self.video_to_label_mapping = video_to_label_mapping
        self.label_to_idx = label_to_idx
        
        available_files = [f for f in os.listdir(keypoints_dir) if f.endswith(config.data.keypoints_ext)]
        valid_files = []
        for file in available_files:
            base_name = os.path.splitext(file)[0]
            if base_name in video_to_label_mapping:
                valid_files.append(base_name)
        
        np.random.shuffle(valid_files)
        split_idx = int(len(valid_files) * train_split)
        
        if split_type == 'train':
            self.files = valid_files[:split_idx]
        else:
            self.files = valid_files[split_idx:]
        
        print(f"[INFO] {split_type.upper()} dataset: {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        base_filename = self.files[idx]
        
        kp_path = os.path.join(self.keypoints_dir, f"{base_filename}{config.data.keypoints_ext}")
        kp_sequence = np.load(kp_path)
        
        if len(kp_sequence.shape) == 2:
            T, features = kp_sequence.shape
            expected_features = config.hgc_lstm.num_vertices * config.hgc_lstm.in_channels
            if features == expected_features:
                kp_sequence = kp_sequence.reshape(T, config.hgc_lstm.num_vertices, config.hgc_lstm.in_channels)
        
        label_id = self.video_to_label_mapping[base_filename]
        label_idx = self.label_to_idx[label_id]
        
        return torch.from_numpy(kp_sequence).float(), torch.tensor(label_idx, dtype=torch.long)

def create_data_loaders(train_dataset, val_dataset, batch_size=None):
    """Create data loaders"""
    batch_size = batch_size or config.training.batch_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Valid batches: {len(val_loader)}")
    print(f"[INFO] Batch size: {batch_size}")
    
    sample_kp, sample_lbl = next(iter(train_loader))
    print(f"[INFO] Sample keypoints shape: {sample_kp.shape}")  # (B, T, V, C)
    print(f"[INFO] Sample labels shape: {sample_lbl.shape}")    # (B,)
    return train_loader, val_loader

def create_adjacency_matrix(num_vertices=None):
    """Create adjacency matrix for skeleton graph"""
    N = num_vertices or config.hgc_lstm.num_vertices
    A = np.zeros((N, N), dtype=np.float32)
    
    # MediaPipe connections for pose (33 points)
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ]
    
    # Hand connections (21 points each)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), 
        (0, 5), (5, 6), (6, 7), (7, 8), 
        (0, 9), (9, 10), (10, 11), (11, 12), 
        (0, 13), (13, 14), (14, 15), (15, 16),  
        (0, 17), (17, 18), (18, 19), (19, 20), 
        (5, 9), (9, 13), (13, 17) 
    ]

    for i, j in POSE_CONNECTIONS:
        if i < 33 and j < 33:
            A[i, j] = 1
            A[j, i] = 1

    for i, j in HAND_CONNECTIONS:
        u, v = i + 33, j + 33
        if u < N and v < N:
            A[u, v] = 1
            A[v, u] = 1

    for i, j in HAND_CONNECTIONS:
        u, v = i + 54, j + 54
        if u < N and v < N:
            A[u, v] = 1
            A[v, u] = 1

    np.fill_diagonal(A, 1)

    D = np.diag(np.sum(A, axis=1) ** -0.5)
    A_norm = D @ A @ D
    A_norm = torch.tensor(A_norm, dtype=torch.float32)

    print(f"[INFO] Adjacency matrix shape: {A_norm.shape}")
    print(f"[INFO] Number of vertices: {config.hgc_lstm.num_vertices}")

    return A_norm

def train_model(model, train_loader, val_loader, config, device):
    """Train the HGC-LSTM model"""
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    if config.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay
        )
    else:  # SGD
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay,
            momentum=config.training.momentum
        )
    
    # Setup scheduler
    if config.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.training.scheduler_step_size, 
            gamma=config.training.scheduler_gamma
        )
    elif config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.training.num_epochs
        )
    else:
        scheduler = None
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    
    # Create save directory
    os.makedirs(config.training.save_dir, exist_ok=True)
    
    for epoch in range(config.training.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (keypoints, labels) in enumerate(train_loader):
            keypoints, labels = keypoints.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            if config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for keypoints, labels in val_loader:
                keypoints, labels = keypoints.to(device), labels.to(device)
                outputs = model(keypoints)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = os.path.join(config.training.save_dir, config.training.model_save_name)
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{config.training.num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            print(f"[INFO] Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"[INFO] Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    return history

def evaluate_model(model, val_loader, device):
    """Evaluate the trained model"""
    # Load the best model weights before final evaluation
    best_model_path = os.path.join(config.training.save_dir, config.training.model_save_name)
    model.load_state_dict(torch.load(best_model_path))
    print(f"[INFO] Loaded best model weights from {best_model_path}")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for keypoints, labels in val_loader:
            keypoints, labels = keypoints.to(device), labels.to(device)
            outputs = model(keypoints)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"[INFO] Final validation accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, zero_division=0))
    
    return accuracy

def save_visualization(history, save_path):
    """Save training history visualization (no display, only save)"""
    plt.figure(figsize=(12, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Training history saved to {save_path}")

if __name__ == "__main__":
    video_to_label_mapping, label_to_idx, id, labels = load_labels_from_csv()
    num_classes = len(id)
    keypoints_dir = config.data.keypoints_output_dir
    train_dataset = SignLanguageDataset(keypoints_dir, video_to_label_mapping, label_to_idx, split_type='train')
    val_dataset = SignLanguageDataset(keypoints_dir, video_to_label_mapping, label_to_idx, split_type='val')
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)
    A = create_adjacency_matrix()
    model = HGC_LSTM(config, A, num_classes)
    model.to(device)
    history = train_model(model, train_loader, val_loader, config, device)
    evaluate_model(model, val_loader, device)
    save_visualization(history, os.path.join(config.training.save_dir, 'training_history.png'))
