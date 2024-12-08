import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PreprocessedAudioDenoisingDataset
from model import AudioDenoiserUNet
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    # Paths
    clean_dir = './data/preprocessed/clean'
    noisy_dir = './data/preprocessed/noisy'
    log_dir = './logs'

    # Hyperparameters
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001

    # Dataset and DataLoader
    max_length = 94  # bazat pe try and error
    dataset = PreprocessedAudioDenoisingDataset(clean_dir, noisy_dir)

    # Splitting dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=3, pin_memory=True)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # Model, Loss, Optimizer
    model = AudioDenoiserUNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop with Early Stopping
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement
    early_stop_counter = 0  # Counter to track lack of improvement
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (clean_spec, noisy_spec) in enumerate(train_loader):
            clean_spec, noisy_spec = clean_spec.to(
                device), noisy_spec.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Use mixed precision
                output = model(noisy_spec, target=clean_spec)
                loss = criterion(output, clean_spec)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            if i % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clean_spec, noisy_spec in val_loader:
                clean_spec, noisy_spec = clean_spec.to(
                    device), noisy_spec.to(device)
                # Pass target to model for alignment
                output = model(noisy_spec, target=clean_spec)

                val_loss += criterion(output, clean_spec).item()

        val_loss /= len(val_loader)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(
                log_dir, "best_model.pth"))
            print(f"Saved Best Model at Epoch {epoch+1}")
        else:
            early_stop_counter += 1
            print(
                f"No improvement in validation loss for {early_stop_counter} epochs.")

        # Early stopping condition
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    writer.close()
