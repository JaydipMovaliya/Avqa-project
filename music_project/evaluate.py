import os
import json
import numpy as np
import soundfile as sf
import mir_eval
import torch
from tqdm import tqdm
from data.avqa_dataset import AVQADataset
from models.avqa_model import AVQAModel

# Modify as needed
CHECKPOINT = "checkpoint_epoch_10.pt"
DATASET_JSON = "dataset.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_SEPARATED_AUDIO = "separated_audio"

os.makedirs(SAVE_SEPARATED_AUDIO, exist_ok=True)

def separate_sound(audio_waveform):
    # Dummy separation: identity (replace with actual separator model later)
    return audio_waveform, np.zeros_like(audio_waveform)

def compute_sdr(ref, est):
    try:
        sdr, _, _, _ = mir_eval.separation.bss_eval_sources(ref, est, compute_permutation=False)
        return sdr[0]
    except:
        return float('-inf')

def main():
    with open(DATASET_JSON, "r") as f:
        dataset = json.load(f)

    video_ids = list(dataset.keys())
    print(f" Evaluating SDR on {len(video_ids)} samples.")

    all_sdr = []

    for vid in tqdm(video_ids):
        entry = dataset[vid]
        audio_path = entry["audio_path"]

        # Load ground truth waveform
        gt_audio, sr = sf.read(audio_path)

        # Simulate source separation (replace with real model later)
        foreground, _ = separate_sound(gt_audio)

        # Optional: Save separated output
        save_path = os.path.join(SAVE_SEPARATED_AUDIO, f"{vid}_pred.wav")
        sf.write(save_path, foreground, sr)

        # Evaluate SDR
        sdr_score = compute_sdr(gt_audio[np.newaxis], foreground[np.newaxis])
        all_sdr.append(sdr_score)

    avg_sdr = np.mean([s for s in all_sdr if s != float('-inf')])
    print(f"\n Average SDR: {avg_sdr:.4f} dB")

if __name__ == "__main__":
    main()
