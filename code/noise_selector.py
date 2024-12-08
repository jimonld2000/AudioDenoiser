import os
import shutil
import random

# Define paths
urban_sound_path = "..\\UrbanSound8K\\UrbanSound8K\\audio\\"
train_noise_path = ".\\data\\urban_noise"
os.makedirs(train_noise_path, exist_ok=True)

# Loop through each fold, select 10 random files, and copy them
for fold_num in range(1, 11):  # Assuming folds are named Fold1, Fold2, ..., Fold10
    fold_path = os.path.join(urban_sound_path, f"fold{fold_num}")
    all_files = [f for f in os.listdir(fold_path) if f.endswith(".wav")]

    # Randomly select 10 files
    selected_files = random.sample(all_files, 10)

    for file in selected_files:
        source_path = os.path.join(fold_path, file)
        dest_path = os.path.join(train_noise_path, f"{fold_num}_{file}")  # Prefix with fold number
        shutil.copy(source_path, dest_path)

print("Selected files have been copied to train/noise.")
