import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data_loader import SpectrogramDataset
from model import UNet
from loss import CombinedPerceptualLoss

# Paths and Parameters
DATA_DIR = "./data/train_processed"
SAVE_DIR = "./saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_TYPES = ["white", "urban", "reverb", "noise_cancellation"]

# Train & Validate functions


def train_one_epoch(model, dataloader, criterion, optimizer, writer, epoch):
    model.train()
    running_loss = 0.0
    running_stft_loss = 0.0
    running_mel_loss = 0.0
    running_l1_loss = 0.0
    
    for noisy, clean in tqdm(dataloader, desc="Training"):
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(noisy)
        
        # Get all loss components
        loss, stft_loss, mel_loss, l1_loss = criterion(outputs, clean)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        running_loss += loss.item()
        running_stft_loss += stft_loss.item()
        running_mel_loss += mel_loss.item()
        running_l1_loss += l1_loss.item()
    
    # Calculate average losses
    avg_loss = running_loss / len(dataloader)
    avg_stft_loss = running_stft_loss / len(dataloader)
    avg_mel_loss = running_mel_loss / len(dataloader)
    avg_l1_loss = running_l1_loss / len(dataloader)
    
    # Log to tensorboard
    writer.add_scalar('Loss/train/total', avg_loss, epoch)
    writer.add_scalar('Loss/train/stft', avg_stft_loss, epoch)
    writer.add_scalar('Loss/train/mel', avg_mel_loss, epoch)
    writer.add_scalar('Loss/train/l1', avg_l1_loss, epoch)
    
    return avg_loss


def validate_one_epoch(model, dataloader, criterion, writer, epoch):
    model.eval()
    running_loss = 0.0
    running_stft_loss = 0.0
    running_mel_loss = 0.0
    running_l1_loss = 0.0
    
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc="Validation"):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            outputs = model(noisy)
            
            # Get all loss components
            loss, stft_loss, mel_loss, l1_loss = criterion(outputs, clean)
            
            # Accumulate losses
            running_loss += loss.item()
            running_stft_loss += stft_loss.item()
            running_mel_loss += mel_loss.item()
            running_l1_loss += l1_loss.item()
    
    # Calculate average losses
    avg_loss = running_loss / len(dataloader)
    avg_stft_loss = running_stft_loss / len(dataloader)
    avg_mel_loss = running_mel_loss / len(dataloader)
    avg_l1_loss = running_l1_loss / len(dataloader)
    
    # Log to tensorboard
    writer.add_scalar('Loss/val/total', avg_loss, epoch)
    writer.add_scalar('Loss/val/stft', avg_stft_loss, epoch)
    writer.add_scalar('Loss/val/mel', avg_mel_loss, epoch)
    writer.add_scalar('Loss/val/l1', avg_l1_loss, epoch)
    
    return avg_loss

# Training for each noise type


def train_for_noise_type(noise_type):
    print(f"\n=== TRAINING for noise type: {noise_type} ===")
    noise_subdir = os.path.join(DATA_DIR, noise_type)
    dataset = SpectrogramDataset(noise_subdir)

    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    writer = SummaryWriter(f"torchlogs/model_{noise_type}/")
    model = UNet(in_channels=1, num_classes=1).to(DEVICE)
    criterion = CombinedPerceptualLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_model_path = os.path.join(SAVE_DIR, f"unet_denoiser_{noise_type}.pth")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS} for noise type: {noise_type}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, writer, epoch)
        val_loss = validate_one_epoch(model, val_loader, criterion, writer, epoch)

        print(
            f"Train Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model for {noise_type} at epoch {epoch+1}")
            dummy_input = torch.randn(1, 1, 256, 64).to(DEVICE)
            writer.add_graph(model, dummy_input, verbose=True)
            writer.close()

    print(
        f"=== Finished training {noise_type}. Best Val Loss: {best_val_loss:.6f} ===")


if __name__ == "__main__":
    for nt in NOISE_TYPES:
        train_for_noise_type(nt)
    print("Done training separate models for each noise type!")
