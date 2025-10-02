import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data.avqa_dataset import AVQADataset
from models.avqa_model import AVQAModel
from losses.trimodal_loss import TrimodalConsistencyLoss
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


def main():
    # ==================== CONFIG ====================
    DATASET_PATH = "dataset.json"
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 1e-4
    VAL_SPLIT = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== LOAD + SPLIT ====================
    full_dataset = AVQADataset(DATASET_PATH)
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ==================== MODEL + LOSS + OPTIM ====================
    model = AVQAModel().to(device)
    criterion = TrimodalConsistencyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ==================== AUDIO PREPROCESSING ====================
    mel = MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=320, win_length=1024, n_mels=64).to(device)
    db = AmplitudeToDB().to(device)

    # ==================== TRAINING LOOP ====================
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            frames = batch["frames"].to(device)
            waveform = batch["audio"].to(device)
            text = batch["text"].to(device)

            with torch.no_grad():
                audio_spec = db(mel(waveform))

            v_feat, a_feat, t_feat = model(frames, audio_spec, text)
            loss = criterion(v_feat, a_feat, t_feat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                frames = batch["frames"].to(device)
                waveform = batch["audio"].to(device)
                text = batch["text"].to(device)

                audio_spec = db(mel(waveform))
                v_feat, a_feat, t_feat = model(frames, audio_spec, text)
                loss = criterion(v_feat, a_feat, t_feat)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"\n Epoch {epoch+1} complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    main()
