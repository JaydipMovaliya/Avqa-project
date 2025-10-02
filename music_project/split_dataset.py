import json
import random

def split_dataset(input_path="dataset.json", train_path="train_dataset.json", val_path="val_dataset.json", val_ratio=0.2):
    with open(input_path, "r") as f:
        dataset = json.load(f)

    video_ids = list(dataset.keys())
    random.shuffle(video_ids)

    val_size = int(len(video_ids) * val_ratio)
    val_ids = set(video_ids[:val_size])
    train_ids = set(video_ids[val_size:])

    train_data = {vid: dataset[vid] for vid in train_ids}
    val_data = {vid: dataset[vid] for vid in val_ids}

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"✅ Split completed. Train: {len(train_data)} samples, Val: {len(val_data)} samples")

if __name__ == "__main__":
    split_dataset()
