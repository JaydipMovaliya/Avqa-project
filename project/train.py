import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.avqa_dataset import AVQADataset
from models.avqa_model import AVQAModel
from losses.trimodal_loss import TrimodalConsistencyLoss


def main():
    # ==================== CONFIG ====================
    DATASET_PATH = "dataset.json"
    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== LOADERS ====================
    dataset = AVQADataset(DATASET_PATH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # ==================== MODEL + LOSS + OPTIM ====================
    model = AVQAModel().to(device)
    criterion = TrimodalConsistencyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ==================== TRAINING LOOP ====================
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            frames = batch["frames"].to(device)
            audio_spec = batch["audio"].to(device)  # already precomputed (B, 1, 64, T)
            text = batch["text"].to(device)         # shape: (B, 512)

            v_feat, a_feat, t_feat = model(frames, audio_spec, text)
            loss = criterion(v_feat, a_feat, t_feat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"\n✅ Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")

        # Optional: Save checkpoint
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pt")

# 🔧 Required on Windows when using multiprocessing
if __name__ == "__main__":
    main()
