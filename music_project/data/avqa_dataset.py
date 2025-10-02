import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class AVQADataset(Dataset):
    def __init__(self, dataset_json_path, frame_size=(224, 224)):
        """
        dataset_json_path: path to dataset.json
        frame_size: resolution for frames (width, height)
        """
        with open(dataset_json_path, "r") as f:
            self.dataset = json.load(f)

        self.video_ids = list(self.dataset.keys())
        self.frame_size = frame_size
        self.max_audio_len = 496  # Based on 10s audio, hop_length=320, sample_rate=16000

        # Frame transform pipeline
        self.frame_transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        data = self.dataset[video_id]

        # Load frames
        frame_paths = data["frame_paths"][:32]
        frames = [
            self.frame_transform(Image.open(path).convert("RGB"))
            for path in frame_paths
        ]
        frames = torch.stack(frames)  # (32, 3, H, W)

        # Load precomputed log-mel spectrogram (stored as .npy)
        log_mel_path = data["log_mel_path"]
        log_mel_tensor = torch.from_numpy(np.load(log_mel_path))  # (64, T)
        log_mel_tensor = log_mel_tensor.unsqueeze(0)  # → (1, 64, T)


        # Pad or truncate to fixed time dimension
        T = log_mel_tensor.shape[-1]
        if T < self.max_audio_len:
            pad_amount = self.max_audio_len - T
            log_mel_tensor = F.pad(log_mel_tensor, (0, pad_amount), mode="constant", value=0)
        else:
            log_mel_tensor = log_mel_tensor[:, :, :self.max_audio_len]  # (1, 64, T)

        # Load text feature
        text_vec = np.load(data["text_feature_path"])
        text_tensor = torch.tensor(text_vec).float()  # (512,)

        return {
            "video_id": video_id,
            "frames": frames,
            "audio": log_mel_tensor,
            "text": text_tensor
        }

