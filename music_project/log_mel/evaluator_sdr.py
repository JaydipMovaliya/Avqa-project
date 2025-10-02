import torch
import numpy as np
import json
import librosa
import os
from tqdm import tqdm
from mir_eval.separation import bss_eval_sources

from data.avqa_dataset import AVQADataset

# Dummy separator (just returns the mixture as "separated source")
def dummy_separator(audio):
    return audio  # Simulate "perfect" separation

def calculate_sdr(est_source, true_source):
    """
    est_source: np.array (T,)
    true_source: np.array (T,)
    """
    try:
        sdr, _, _, _ = bss_eval_sources(true_source[np.newaxis, :], est_source[np.newaxis, :])
        return sdr[0]
    except Exception as e:
        print("SDR error:", e)
        return None

def main():
    DATASET_JSON = "dataset.json"
    SAMPLE_RATE = 16000

    # Load dataset
    dataset = AVQADataset(DATASET_JSON)
    print(f" Loaded {len(dataset)} samples for evaluation")

    sdr_scores = []

    for sample in tqdm(dataset, desc="Evaluating SDR"):
        audio_path = sample["audio_path"]

        # Load mono waveform
        waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Apply dummy separator
        separated = dummy_separator(waveform)

        # Use original as pseudo ground-truth (because we don’t have true sources)
        sdr = calculate_sdr(separated, waveform)

        if sdr is not None:
            sdr_scores.append(sdr)

    avg_sdr = np.mean(sdr_scores)
    print(f"\n Average SDR (Dummy Separator): {avg_sdr:.2f} dB over {len(sdr_scores)} samples")

if __name__ == "__main__":
    main()
