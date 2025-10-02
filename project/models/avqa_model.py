import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.audio_separator import AudioSeparator

class VisualEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]  # remove avgpool + fc
        self.backbone = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x):  # (B, 32, 3, 224, 224)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)  # (B*T, 512, H', W')
        pooled = self.pool(feats).squeeze(-1).squeeze(-1)  # (B*T, 512)
        pooled = pooled.view(B, T, -1).mean(1)  # (B, 512)
        return self.proj(pooled)  # (B, output_dim)


class AudioEncoder(nn.Module):
    def __init__(self, input_channels=1, output_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(256, output_dim)

    def forward(self, x):  # (B, 1, T, F)
        feat = self.encoder(x).squeeze(-1).squeeze(-1)  # (B, 256)
        return self.proj(feat)  # (B, output_dim)


class TextProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):  # (B, 512)
        return self.proj(x)


class AVQAModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.visual_encoder = VisualEncoder(embed_dim)
        self.audio_encoder = AudioEncoder(output_dim=embed_dim)
        self.text_projector = TextProjector(input_dim=512, output_dim=embed_dim)
        self.separator = AudioSeparator(in_channels=1, cond_dim=embed_dim)

    def forward(self, frames, audio_spec, text):
        """
        frames: (B, 32, 3, 224, 224)
        audio_spec:  (B, 1, T, F)  - spectrogram
        text:   (B, 512)
        """
        v_feat = self.visual_encoder(frames)
        t_feat = self.text_projector(text)
        cond_feat = (v_feat + t_feat) / 2

        # Apply audio separator
        pred_mask = self.separator(audio_spec, cond_feat)
        separated_spec = pred_mask * audio_spec

        a_feat = self.audio_encoder(separated_spec)

        # Normalize features
        v_feat = F.normalize(v_feat, dim=-1)
        a_feat = F.normalize(a_feat, dim=-1)
        t_feat = F.normalize(t_feat, dim=-1)

        return v_feat, a_feat, t_feat
