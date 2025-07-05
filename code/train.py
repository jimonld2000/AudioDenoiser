# new_unet_training.py (Final Version with Argument Parsing and Logging)

import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
import time
import argparse
import random

# Import the necessary components
from new_unet_data_loader import WavToSpecDataset
from model_unet import UNet
from loss_unet import CombinedPerceptualLoss

def setup_logger(log_path):
    """Sets up a logger that outputs to a file and the console."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("unet_training_logger")
    if logger.hasHandlers(): logger.handlers.clear() # Prevent duplicate logs
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)
    
    return logger

def parse_args():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="U-NET Audio Denoising Training Script")
    
    # Group for run and path configurations
    run_group = parser.add_argument_group("Run & Path Configuration")
    run_group.add_argument("--run_name", type=str, default=f"UNET_Run_{int(time.time())}", help="A unique name for the training run.")
    run_group.add_argument("--base_dataset_path", type=str, required=True, help="Path to the root directory of the dataset (containing train/val/test folders).")
    run_group.add_argument("--output_path", type=str, default="./training_outputs_unet", help="Directory to save models and logs.")
    
    # Group for training hyperparameters
    hyper_group = parser.add_argument_group("Training Hyperparameters")
    hyper_group.add_argument("--epochs", type=int, default=50)
    hyper_group.add_argument("--batch_size", type=int, default=16)
    hyper_group.add_argument("--learning_rate", type=float, default=1e-4)
    hyper_group.add_argument("--num_workers", type=int, default=4, help="Number of CPU cores for data loading.")
    hyper_group.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of the dataset to use (e.g., 0.1 for 10%).")
    
    return parser.parse_args()

def train_one_epoch(model, dataloader, criterion, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0.0
    for noisy, clean in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        outputs = model(noisy)
        loss, _, _, _ = criterion(outputs, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    return avg_loss

def validate_one_epoch(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc=f"Validation Epoch {epoch+1}"):
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy)
            loss, _, _, _ = criterion(outputs, clean)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/validation', avg_loss, epoch)
    return avg_loss

def main(args):
    run_output_dir = os.path.join(args.output_path, args.run_name)
    os.makedirs(os.path.join(run_output_dir, "checkpoints"), exist_ok=True)
    
    logger = setup_logger(os.path.join(run_output_dir, "training.log"))
    logger.info(f"--- Starting U-NET Run: {args.run_name} ---")
    logger.info(f"Full configuration: \n{json.dumps(vars(args), indent=2)}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # --- Dataset Loading ---
    train_data_dir = os.path.join(args.base_dataset_path, "train")
    logger.info(f"Loading on-the-fly dataset from: {train_data_dir}")
    full_dataset = WavToSpecDataset(
        data_dir=train_data_dir,
        subset_fraction=args.subset_fraction
    )

    val_split_ratio = 0.1
    val_size = int(len(full_dataset) * val_split_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    # --- DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- Model, Loss, Optimizer ---
    model = UNet(in_channels=1, num_classes=1).to(DEVICE)
    criterion = CombinedPerceptualLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(log_dir=os.path.join(run_output_dir, "tensorboard_logs"))
    
    logger.info(f"U-NET Model initialized. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # --- Training Loop ---
    best_val_loss = float("inf")
    model_save_path = os.path.join(run_output_dir, "checkpoints", "best_model.pth")

    logger.info("--- Starting Training Loop ---")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, writer, epoch)
        val_loss = validate_one_epoch(model, val_loader, criterion, DEVICE, writer, epoch)

        logger.info(f"Epoch {epoch + 1}/{args.epochs} -> Train Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"✅ New best model saved to {model_save_path} (Val Loss: {best_val_loss:.6f})")

    writer.close()
    logger.info("--- Training Finished ---")
    logger.info(f"Final best model saved at: {model_save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
