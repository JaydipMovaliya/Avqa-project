import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
from PIL import Image
from data.avqa_dataset import AVQADataset
from models.avqa_model import AVQAModel
from tqdm import tqdm 
from torchvision.models import resnet18

# -------------------- Config --------------------
CHECKPOINT_PATH = "checkpoint_epoch_10.pt"
DATASET_JSON = "dataset.json"
VIDEO_INDEX = 62  # change if you want to try a different video
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Load Dataset & Model --------------------
dataset = AVQADataset(DATASET_JSON)
sample = dataset[VIDEO_INDEX]  # Get a single sample without batching
model = AVQAModel().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

for idx in tqdm(range(len(dataset)), desc="Generating heatmaps"):
    sample = dataset[idx]

    frames = sample["frames"].unsqueeze(0).to(device)  # (1, N, 3, 224, 224)
    audio = sample["audio"].unsqueeze(0).to(device)
    text = sample["text"].unsqueeze(0).to(device)

    with torch.no_grad():
        v_feat, a_feat, t_feat = model(frames, audio, text)

    # Get last frame
    last_frame = frames[0, -1]  # (3, 224, 224)

    # Visual feature map using resnet backbone
    
    resnet = resnet18(pretrained=True)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-2]).to(device)
    feat_map = backbone(last_frame.unsqueeze(0))  # (1, 512, H', W')
    spatial_feat = feat_map.squeeze(0).permute(1, 2, 0).view(-1, 512)  # (H*W, 512)

    # Normalize
    spatial_feat = torch.nn.functional.normalize(spatial_feat, dim=1)
    a_feat = torch.nn.functional.normalize(a_feat, dim=1)
    t_feat = torch.nn.functional.normalize(t_feat, dim=1)

    sim_a = torch.matmul(spatial_feat, a_feat.squeeze(0))
    sim_t = torch.matmul(spatial_feat, t_feat.squeeze(0))
    attention = (sim_a + sim_t) / 2.0

    # Attention map → (H, W)
    H, W = feat_map.shape[2:]
    attention_map = attention.view(H, W).detach().cpu().numpy()
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-6)
    attention_map_resized = cv2.resize(attention_map, (224, 224))

    # Frame image
    frame_tensor = last_frame
    frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
    frame_np = (frame_np * 255).astype(np.uint8)

    # Colorize heatmap
    attention_heatmap = cv2.applyColorMap((attention_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    attention_heatmap = cv2.cvtColor(attention_heatmap, cv2.COLOR_BGR2RGB)

    # Blend heatmap with frame
    blended = cv2.addWeighted(frame_np, 0.6, attention_heatmap, 0.4, 0)

    # Save
    video_id = sample["video_id"]
    save_path = os.path.join(SAVE_DIR, f"heatmap_{video_id}.png")
    Image.fromarray(blended).save(save_path)

print("✅ All heatmaps saved to results/")

# # -------------------- Move Inputs --------------------
# frames = sample["frames"].unsqueeze(0).to(device)  # (1, 64, 3, 224, 224)
# audio = sample["audio"].unsqueeze(0).to(device)    # (1, 1, T, F)
# text = sample["text"].unsqueeze(0).to(device)      # (1, 512)

# # -------------------- Forward Pass --------------------
# with torch.no_grad():
#     v_feat, a_feat, t_feat = model(frames, audio, text)  # (1, 512) each

# # -------------------- Generate Attention Map --------------------
# # We'll generate spatial similarity map from last frame features
# from torchvision.models import resnet18

# resnet = resnet18(pretrained=True)
# backbone = torch.nn.Sequential(*list(resnet.children())[:-2]).to(device)
# last_frame = frames[0, -1]  # take last frame of sequence
# last_feat_map = backbone(last_frame.unsqueeze(0))  # (1, 512, H', W')

# # Project feature map to match dim of audio/text embeddings
# proj_layer = torch.nn.Linear(512, 512).to(device)
# spatial_feat = last_feat_map.squeeze(0).permute(1, 2, 0)  # (H, W, 512)
# spatial_feat_flat = spatial_feat.view(-1, 512)  # (H*W, 512)

# # Normalize and compute similarity to audio/text
# spatial_feat_flat = torch.nn.functional.normalize(spatial_feat_flat, dim=1)
# a_feat = torch.nn.functional.normalize(a_feat, dim=1)
# t_feat = torch.nn.functional.normalize(t_feat, dim=1)

# # Compute average similarity to audio and text
# sim_a = torch.matmul(spatial_feat_flat, a_feat.squeeze(0))  # (H*W,)
# sim_t = torch.matmul(spatial_feat_flat, t_feat.squeeze(0))  # (H*W,)
# attention = (sim_a + sim_t) / 2.0  # average of both modalities

# # Reshape and convert to CPU numpy for visualization
# H, W = last_feat_map.shape[2:]
# attention_map = attention.view(H, W).detach().cpu().numpy()
# attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-6)

# # -------------------- Overlay on Frame --------------------
# # -------------------- Overlay on Frame --------------------


# # 1. Convert last frame to numpy image
# frame_tensor = sample["frames"][-1]  # (3, 224, 224)
# frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()  # (224, 224, 3)
# frame_np = (frame_np * 255).astype(np.uint8)

# # 2. Resize attention map to match frame resolution
# attention_map_resized = cv2.resize(attention_map, (224, 224))  # from 7x7 or 14x14 → 224x224
# attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min() + 1e-6)

# # 3. Convert attention to heatmap (color)
# attention_heatmap = cv2.applyColorMap((attention_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
# attention_heatmap = cv2.cvtColor(attention_heatmap, cv2.COLOR_BGR2RGB)  # convert BGR to RGB

# # 4. Blend with original frame
# blended = cv2.addWeighted(frame_np, 0.6, attention_heatmap, 0.4, 0)

# 5. Save output
# save_path = os.path.join(SAVE_DIR, f"heatmap_{sample['video_id']}.png")
# Image.fromarray(blended).save(save_path)
# print(f"✅ Final blended heatmap saved to {save_path}")


