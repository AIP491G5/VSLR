"""
Model utilities for VSLR project
Contains GCN layers, HGC-LSTM model, and adjacency matrix creation
"""

import numpy as np
import torch
from ..models.model import HGC_LSTM
from .data_utils import load_labels_from_csv

def create_adjacency_matrix(config, num_vertices=None):
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
    
    return torch.tensor(A_norm, dtype=torch.float32)

def create_model(config, A, num_classes, device):
    """Create and initialize the HGC-LSTM model"""
    model = HGC_LSTM(config, A, num_classes)
    model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model

def load_model(model_path='outputs/models/best_hgc_lstm_13.pth', config=None):
    """
    Load a pre-trained HGC-LSTM model from checkpoint
    
    Args:
        model_path: Path to the model checkpoint
        config: Configuration object (optional, will create new if None)
    
    Returns:
        model: Loaded HGC-LSTM model
    """
    # Import here to avoid circular imports
    from configs.config import Config
    
    # Initialize config if not provided
    if config is None:
        config = Config()
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create adjacency matrix
    A = create_adjacency_matrix(config)
    
    # Load labels
    video_to_label_mapping, label_to_idx, unique_labels, id_to_label_mapping = load_labels_from_csv(None, config)
    num_classes = len(unique_labels)
    
    # Create model
    model = HGC_LSTM(config, A, num_classes)
    model.to(device)
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")
    else:
        print("No checkpoint path provided")
        return None
    
    model.eval()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model