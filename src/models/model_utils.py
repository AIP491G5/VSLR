"""
Model utilities for VSLR project
Contains GCN layers, HGC-LSTM model, and adjacency matrix creation
"""

import numpy as np
import torch
from .model import HGC_LSTM

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
    print(f"Model configuration:")
    print(f"  - GCN layers: {config.hgc_lstm.gcn_layers}")
    print(f"  - GCN hidden dim: {config.hgc_lstm.hidden_gcn}")
    print(f"  - LSTM hidden dim: {config.hgc_lstm.hidden_lstm}")
    print(f"  - LSTM layers: {config.hgc_lstm.lstm_layers}")
    print(f"  - Bidirectional: {config.hgc_lstm.lstm_bidirectional}")
    print(f"  - Dropout: {config.hgc_lstm.dropout}")
    print(f"  - Pooling type: {config.hgc_lstm.pooling_type}")
    print(f"  - Sequence length: {config.hgc_lstm.sequence_length}")
    print(f"  - Number of vertices: {config.hgc_lstm.num_vertices}")

    if config.hgc_lstm.pooling_type == "attention":
        print(f"  - Using Attention Pooling with dropout: {config.hgc_lstm.dropout}")
        print(f"  - Attention weights can be visualized after forward pass")
    
    return model
