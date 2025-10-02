import torch
import torch.nn as nn

class AudioSeparator(nn.Module):
    def __init__(self, in_channels=1, cond_dim=512):
        super(AudioSeparator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, T/2, F/2]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # [B, 64, T/4, F/4]
            nn.ReLU()
        )

        self.condition = nn.Linear(cond_dim, 64)  # Project visual+text to match

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, T/2, F/2]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # [B, 1, T, F]
            nn.Sigmoid()  # Mask output in range [0, 1]
        )

    def forward(self, spec, cond_feat):
        """
        spec: Tensor of shape [B, 1, T, F] (log-mel spectrogram)
        cond_feat: Tensor of shape [B, cond_dim] (fused visual + text features)
        """
        enc = self.encoder(spec)  # [B, 64, T/4, F/4]
        cond = self.condition(cond_feat).unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
        cond = cond.expand_as(enc)  # broadcast to match shape
        fused = enc * cond
        mask = self.decoder(fused)  # [B, 1, T, F]
        return mask
