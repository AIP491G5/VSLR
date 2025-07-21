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
        
        return F.relu(x)

class AttentionPooling(nn.Module):
    """Attention-based pooling over joints dimension"""
    def __init__(self, hidden_dim, dropout=0.1):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
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

class HGC_LSTM(nn.Module):
    """Hierarchical Graph Convolution + LSTM"""
    def __init__(self, config, A, num_classes):
        super(HGC_LSTM, self).__init__()
        self.config = config
        
        # Build GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(config.hgc_lstm.gcn_layers):
            if i == 0:
                in_features = config.hgc_lstm.in_channels
            else:
                in_features = config.hgc_lstm.hidden_gcn
            
            self.gcn_layers.append(
                GCNLayer(in_features, config.hgc_lstm.hidden_gcn, A, config.hgc_lstm.use_batch_norm)
            )
        
        # Pooling layer
        if config.hgc_lstm.pooling_type == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool2d((None, 1))
            self.use_attention = False
        elif config.hgc_lstm.pooling_type == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool2d((None, 1))
            self.use_attention = False
        elif config.hgc_lstm.pooling_type == "attention":
            self.pool = AttentionPooling(config.hgc_lstm.hidden_gcn, config.hgc_lstm.dropout)
            self.use_attention = True
        else:
            self.pool = nn.AdaptiveAvgPool2d((None, 1))  # Default
            self.use_attention = False
        
        # LSTM layer
        self.lstm = nn.LSTM(
            config.hgc_lstm.hidden_gcn, 
            config.hgc_lstm.hidden_lstm, 
            num_layers=config.hgc_lstm.lstm_layers,
            batch_first=True, 
            bidirectional=config.hgc_lstm.lstm_bidirectional
        )
        
        # Adjust output size for bidirectional LSTM
        lstm_output_size = config.hgc_lstm.hidden_lstm * (2 if config.hgc_lstm.lstm_bidirectional else 1)
        
        self.dropout = nn.Dropout(config.hgc_lstm.dropout)
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        # x: (B, T, V, C)
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x)  # (B, T, V, H)
        
        # Pool over joints dimension
        if self.use_attention:
            x, att_weights = self.pool(x)  # (B, T, H)
        else:
            x = x.permute(0, 3, 1, 2)  # (B, H, T, V)
            x = self.pool(x).squeeze(-1)  # (B, H, T)
            x = x.permute(0, 2, 1)  # (B, T, H)
        
        # LSTM processing
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # (B, T, H_lstm)
        x = x[:, -1]  # Take last time step
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)
        return x