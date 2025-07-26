import torch.nn as nn
import torch
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """Graph Convolutional Layer"""
    def __init__(self, in_features, out_features, A, use_batch_norm=True):
        super(GCNLayer, self).__init__()
        self.A = A
        self.fc = nn.Linear(in_features, out_features)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # x: (B, T, V, C)
        B, T, V, C = x.size()
        x = x.reshape(B * T, V, C)  # (B*T, V, C)
        A = self.A.to(x.device)
        x = torch.matmul(A, x)   # (B*T, V, C)
        x = self.fc(x)           # (B*T, V, out_features)
        x = x.reshape(B, T, V, -1)  # (B, T, V, out_features)
        
        if self.use_batch_norm:
            x = x.permute(0, 3, 1, 2)  # (B, out_features, T, V)
            x = self.bn(x.contiguous().view(B, -1, T * V))  # Batch norm
            x = x.view(B, -1, T, V).permute(0, 2, 3, 1)  # Back to (B, T, V, out_features)
        
        # return F.relu(x)
        return F.leaky_relu(x, negative_slope=0.01)

class JointsAttentionPooling(nn.Module):
    """Attention-based pooling over joints dimension"""
    def __init__(self, hidden_dim, dropout=0.1):
        super(JointsAttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, T, V, H)
        B, T, V, H = x.shape
        
        # Compute attention weights
        att_scores = self.attention(x)  # (B, T, V, 1)
        att_weights = F.softmax(att_scores, dim=2)  # Normalize over joints
        
        # Apply dropout to attention weights
        att_weights = self.dropout(att_weights)
        
        # Weighted sum over joints
        output = torch.sum(x * att_weights, dim=2)  # (B, T, H)
        
        return output, att_weights

class TemporalAttentionPooling(nn.Module):
    """Attention-based pooling over temporal dimension"""
    def __init__(self, hidden_dim, dropout=0.1):
        super(TemporalAttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, T, H)
        B, T, H = x.shape
        
        # Compute attention weights
        att_scores = self.attention(x)  # (B, T, 1)
        att_weights = F.softmax(att_scores, dim=1)  # Normalize over time
        
        # Apply dropout to attention weights
        att_weights = self.dropout(att_weights)
        
        # Weighted sum over time
        output = torch.sum(x * att_weights, dim=1)  # (B, H)
        
        return output, att_weights

class HGC_LSTM(nn.Module):
    """Hierarchical Graph Convolution + LSTM with Dual Attention"""
    def __init__(self, config, A, num_classes, in_channels=2, hidden_gcn=64, hidden_lstm=128, dropout=0.1):
        super(HGC_LSTM, self).__init__()
        
        # Fixed architecture: 2 GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(in_channels, hidden_gcn, A, use_batch_norm=True),
            GCNLayer(hidden_gcn, hidden_gcn, A, use_batch_norm=True)
        ])
        
        # First attention pooling (after GCN, over joints dimension)
        self.joint_attention = JointsAttentionPooling(hidden_gcn, dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            hidden_gcn, 
            hidden_lstm, 
            num_layers=1,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout
        )
        
        # Adjust output size for bidirectional LSTM
        lstm_output_size = hidden_lstm * 2
        
        # Second attention pooling (after LSTM, over temporal dimension)
        self.temporal_attention = TemporalAttentionPooling(lstm_output_size, dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        # x: (B, T, V, C)
        
        # GCN layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x)  # (B, T, V, H)
        
        # First attention pooling over joints dimension
        x, joint_att_weights = self.joint_attention(x)  # (B, T, H)
        
        # LSTM processing
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # (B, T, H_lstm)
        
        # Second attention pooling over temporal dimension
        x, temporal_att_weights = self.temporal_attention(x)  # (B, H_lstm)
        
        # Final classification
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)
        
        return x