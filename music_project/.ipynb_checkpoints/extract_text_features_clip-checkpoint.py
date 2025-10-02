import os
import json
import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from multiprocessing import Pool, cpu_count


DATASET_JSON = "dataset.json"
TEXT_FEATURE_DIR = "text_features"
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(TEXT_FEATURE_DIR, exist_ok=True)

# Load CLIP model + tokenizer (once, globally)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_model.eval()

# Load dataset
with open(DATASET_JSON, "r") as f:
    dataset = json.load(f)


def process_single_text(args):
    video_id, categories = args
    text_input = ", ".join(categories)

    inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # CLS token

    out_path = os.path.join(TEXT_FEATURE_DIR, f"{video_id}.npy")
    np.save(out_path, embedding)

    return video_id, out_path



def process_batch(batch):
    args = [(vid, dataset[vid]["categories"]) for vid in batch]
    results = []

    with Pool(processes=min(cpu_count(), len(batch))) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_text, args), start=1):
            print(f"\r🔠 Text features: {i}/{len(batch)}", end="", flush=True)
            results.append(result)

    for video_id, path in results:
        dataset[video_id]["text_feature_path"] = path
    print(" ✅ Done.")


# ========== Main Loop ==========
video_ids = [vid for vid in dataset if not os.path.exists(os.path.join(TEXT_FEATURE_DIR, f"{vid}.npy"))]

for i in range(0, len(video_ids), BATCH_SIZE):
    batch = video_ids[i:i + BATCH_SIZE]
    print(f"\n🔁 Batch {i // BATCH_SIZE + 1} of {(len(video_ids) - 1) // BATCH_SIZE + 1}")
    process_batch(batch)

    # Save dataset.json once per batch
    with open(DATASET_JSON, "w") as f:
        json.dump(dataset, f, indent=2)

print("\n🎯 Text embedding complete for all videos.")
