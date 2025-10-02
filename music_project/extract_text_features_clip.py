import os
import json
import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel

# Load dataset
DATASET_JSON = "dataset.json"
TEXT_FEATURE_DIR = "text_features"
os.makedirs(TEXT_FEATURE_DIR, exist_ok=True)

# Load dataset.json
with open(DATASET_JSON, "r") as f:
    dataset = json.load(f)

# Load CLIP model + tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Using device: {device}")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_model.eval()

def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
        text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return text_features.squeeze().cpu().numpy()

# Process all video_ids
for i, (video_id, data) in enumerate(dataset.items(), start=1):
    text_feat_path = os.path.join(TEXT_FEATURE_DIR, f"{video_id}.npy")
    if os.path.exists(text_feat_path):
        continue  # Skip if already extracted

    categories = data.get("categories", [])
    if not categories:
        print(f"⚠️ No category for {video_id}, skipping...")
        continue

    text_input = ", ".join(categories)
    text_embedding = encode_text(text_input)
    np.save(text_feat_path, text_embedding)

    dataset[video_id]["text_feature_path"] = text_feat_path

    if i % 10 == 0 or i == len(dataset):
        print(f"✅ Processed {i}/{len(dataset)} text embeddings")

# Save updated dataset.json
with open(DATASET_JSON, "w") as f:
    json.dump(dataset, f, indent=2)

print("\n🎯 Text feature extraction complete.")
