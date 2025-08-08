import torch
import torch.nn as nn

class TripletNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, anchor, positive, negative):
        a = self.encoder(anchor)
        p = self.encoder(positive)
        n = self.encoder(negative)
        return a, p, n
