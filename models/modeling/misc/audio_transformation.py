import torch
import torch.nn as nn


class audio_mlp(nn.Module):
    def __init__(self, in_dim=512, middle_dim=4096, out_dim=256): # before: in_dim=128
        """MLP for audio transformation"""
        super(audio_mlp, self).__init__()
        self.embeddings = nn.Sequential(
            nn.Linear(in_dim, middle_dim), nn.ReLU(True), nn.Linear(middle_dim, middle_dim), nn.ReLU(True), nn.Linear(middle_dim, out_dim)
        )

    def forward(self, x):
        B, H, W, C = x.shape # B, 4, 6, 512
        x = x.reshape(B, -1, C) # B, 24, 512
        x = self.embeddings(x) # B, 24, 256
        x = x.reshape(B, H, W, -1) # B, 4, 6, 256
        return x
