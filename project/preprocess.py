import os
import shutil
import cv2
import json
import subprocess
from tqdm import tqdm
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count

ANNOTATION_FILE = r"D:\Study\M.tech\3rd Sem\Project Phase-2\project\Annotations.txt"
VIDEO_DIR = r"D:\Study\M.tech\3rd Sem\Project Phase-2\ave"
AUDIO_DIR = "audio"
FRAME_DIR = "frames"
MEL_DIR = "log_mel"
DATASET_JSON  = "dataset.json"
FRAMES_PER_VIDEO = 32
RESOLUTION = (256, 256)
BATCH_SIZE = 32
SKIP_EXISTING = True


def load_annotations_txt():
    annotations = {}
    with open(ANNOTATION_FILE, "r") as f:
        next(f)  # skip header
        for line in f:
            fields = line.strip().split("&")
            if len(fields) != 5:
                continue
            categories_raw, video_id, quality, start, end = fields
            category_list = [cat.strip() for cat in categories_raw.split(",")]
            annotations[video_id] = {
                "categories": category_list,
                "start": float(start),
                "end": float(end)
            }
    return annotations


def extract_frames_ffmpeg(video_path, output_dir, num_frames=32, resolution=(256, 256), start_time=0, duration=None):
    os.makedirs(output_dir, exist_ok=True)

    if duration is None:
        try:
            result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            duration = float(result.stdout)
        except:
            print("Error finding duration")
            duration = 2

    if duration <= 0:
        print(f"Warning: Duration is non-positive for {video_path}. Cannot extract frames.")
        return []

    fps = num_frames / duration if duration > 0 else num_frames

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
    ]
    if duration is not None:
        cmd.extend(["-t", str(duration)])

    cmd.extend([
        "-vf", f"fps={fps},scale={resolution[0]}:{resolution[1]}",
        "-qscale:v", "2",
        os.path.join(output_dir, "frame_%04d.jpg")
    ])
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_paths = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.endswith(".jpg")]
    return frame_paths


def compute_and_save_logmel(audio_path, video_id, sr=16000, n_mels=64, hop_length=320, win_length=1024):
    os.makedirs(MEL_DIR, exist_ok=True)
    mel_path = os.path.join(MEL_DIR, f"{video_id}.npy")
    if not os.path.exists(mel_path):
        y, _ = librosa.load(audio_path, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
        log_mel = librosa.power_to_db(mel)
        np.save(mel_path, log_mel)
    return mel_path


def process_single_video(args):
    video_id, data = args
    start_time = data["start"]
    end_time = data["end"]
    duration = end_time - start_time

    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    frame_dir = os.path.join(FRAME_DIR, video_id)
    audio_path = os.path.join(AUDIO_DIR, f"{video_id}.wav")

    # Extract frames
    if not os.path.exists(frame_dir) or len(os.listdir(frame_dir)) < FRAMES_PER_VIDEO:
        frame_paths = extract_frames_ffmpeg(video_path, frame_dir, FRAMES_PER_VIDEO, RESOLUTION)
    else:
        frame_paths = [os.path.join(frame_dir, f) for f in sorted(os.listdir(frame_dir))[:FRAMES_PER_VIDEO]]

    # Extract audio
    if not os.path.exists(audio_path):
        os.makedirs(AUDIO_DIR, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-ss", str(start_time), "-t", str(duration),
            "-i", video_path,
            "-ar", "16000", "-ac", "1",
            audio_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Compute and store log-mel spectrogram
    log_mel_path = compute_and_save_logmel(audio_path, video_id)

    dataset_entry = {
        "categories": data["categories"],
        "video_path": video_path,
        "frame_paths": frame_paths,
        "audio_path": audio_path,
        "log_mel_path": log_mel_path,
        "start": start_time,
        "end": end_time,
        "text_feature_path": f"text_features/{video_id}.npy"
    }

    return video_id, dataset_entry


def process_batch(video_ids, annotations, dataset):
    total = len(video_ids)
    print(f"🔄 Processing batch ({total} videos):", end="", flush=True)

    args = [(vid, annotations[vid]) for vid in video_ids]
    results = []
    with Pool(processes=3) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_video, args), start=1):
            results.append(result)
            print(f"\r🔄 Processing batch ({total} videos): {i}/{total}", end="", flush=True)

    for video_id, data in results:
        dataset[video_id] = data

    print(" ✅ Done.")


def main():
    annotations = load_annotations_txt()

    if os.path.exists(DATASET_JSON):
        with open(DATASET_JSON, "r") as f:
            dataset = json.load(f)
        print(f"📄 Loaded existing dataset.json with {len(dataset)} entries.")
    else:
        dataset = {}
        print("🆕 Creating new dataset.json")

    video_ids = list(annotations.keys())

    for i in range(0, len(video_ids), BATCH_SIZE):
        batch = video_ids[i:i + BATCH_SIZE]
        print(f"\n🔁 Batch {i // BATCH_SIZE + 1} of {len(video_ids) // BATCH_SIZE + 1}")
        process_batch(batch, annotations, dataset)

        with open(DATASET_JSON, "w") as f:
            json.dump(dataset, f, indent=2)

    print(f"\n✅ Done. dataset.json saved with {len(dataset)} entries.")


if __name__ == "__main__":
    main()
