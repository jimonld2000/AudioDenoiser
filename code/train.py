import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_loader_ensemble import SpectrogramDataset  # Your existing dataset class
from model import UNet  # Your UNet

# Paths and Parameters
DATA_DIR = "./data/train_processed"  # Directory that has subfolders: white/, urban/, reverb/, noise_cancellation/
SAVE_DIR = "./saved_models"          # Where we'll save each noise-specific UNet
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 3e-4
VALIDATION_SPLIT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NOISE_TYPES = ["white", "urban", "reverb", "noise_cancellation"]

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        return self.mse(output, target) + self.l1(output, target)


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for noisy, clean in tqdm(dataloader, desc="Training"):
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc="Validation"):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

            outputs = model(noisy)
            loss = criterion(outputs, clean)
            running_loss += loss.item()
    return running_loss / len(dataloader)


def train_for_noise_type(noise_type):
    """
    Trains a UNet specifically for a given noise type.
    Saves the best model to unet_denoiser_{noise_type}.pth.
    """
    print(f"\n=== TRAINING for noise type: {noise_type} ===")

    # Create dataset that only loads data from the subdirectory corresponding to `noise_type`
    noise_subdir = os.path.join(DATA_DIR, noise_type)
    # We'll create a small wrapper or a specialized dataset init that only scans `noise_subdir`.
    
    dataset = SpectrogramDataset(noise_subdir)  # This loads only from that folder
    print(f"Total dataset size ({noise_type}): {len(dataset)}")

    # # Quick check: one sample
    noisy, clean = dataset[0]
    print(f"Noisy shape: {noisy.shape}, Clean shape: {clean.shape}")

    # Split into train/val
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize a fresh UNet
    model = UNet(in_channels=1, num_classes=1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_model_path = os.path.join(SAVE_DIR, f"unet_denoiser_{noise_type}.pth")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS} for noise type: {noise_type}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate_one_epoch(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model for {noise_type} at epoch {epoch+1}")

    print(f"=== Finished training {noise_type}. Best Val Loss: {best_val_loss:.6f} ===")


if __name__ == "__main__":
    # We'll train a separate model for each noise type
    for nt in NOISE_TYPES:
        train_for_noise_type(nt)
    
    print("Done training separate models for each noise type!")
