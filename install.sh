#!/bin/bash

# ---
# Bash script to download and prepare the IRMAS dataset.
# This is a conversion of the original PowerShell script.
# ---

# Enable strict error handling
# set -e: Exit immediately if a command exits with a non-zero status.
# set -o pipefail: The return value of a pipeline is the status of the last command to exit with a non-zero status.
set -e
set -o pipefail

# --- Step 1: Download the zip file ---
zip_url="https://zenodo.org/records/1290750/files/IRMAS-TrainingData.zip?download=1"
zip_file="./irmas.zip"
echo "Step 1: Downloading the ZIP file from $zip_url"
# Use wget to download the file. -O specifies the output filename.
wget -O "$zip_file" "$zip_url"
echo "Download complete: $zip_file"
echo ""

# --- Step 2: Unzip the file ---
unzip_destination="./irmas"
echo "Step 2: Unzipping $zip_file to $unzip_destination"
# Use unzip. -o overwrites existing files without prompting. -d specifies the destination directory.
unzip -o "$zip_file" -d "$unzip_destination"
echo "Unzip complete."
echo ""

# --- Step 3: Move all .wav files to the training directory ---
train_clean_dir="./data/train/clean"
# Ensure the destination directory exists. `mkdir -p` creates parent directories if they don't exist
# and doesn't throw an error if the directory already exists.
echo "Creating destination directory if it doesn't exist: $train_clean_dir"
mkdir -p "$train_clean_dir"

echo "Step 3: Moving all .wav files from $unzip_destination to $train_clean_dir"
# Use `find` to locate all files (-type f) ending with .wav (-name "*.wav")
# and execute `mv` on each file found. `{}` is a placeholder for the found file path. `\;` terminates the -exec command.
find "$unzip_destination" -type f -name "*.wav" -exec mv {} "$train_clean_dir" \;
echo "All .wav files have been moved to $train_clean_dir."
echo ""

# --- Step 4: Clean up the downloaded ZIP and the unzipped folder ---
echo "Step 4: Deleting the unzipped directory and the ZIP file."
# Check if the directory exists and then remove it recursively and forcefully.
if [ -d "$unzip_destination" ]; then
    rm -rf "$unzip_destination"
    echo "Deleted folder: $unzip_destination"
fi
# Check if the file exists and then remove it forcefully.
if [ -f "$zip_file" ]; then
    rm -f "$zip_file"
    echo "Deleted file: $zip_file"
fi
echo ""

# --- Step 5: Move some random files into the test folder ---
source_dir="./data/train/clean"
dest_dir="./data/test/clean"

echo "Step 5: Preparing test dataset by selecting random files from $source_dir"

# Check if the source directory exists
if [ ! -d "$source_dir" ]; then
    # Print error to stderr and exit with an error code
    echo "Error: Source directory does not exist: $source_dir" >&2
    exit 1
fi

# Create the destination directory if it doesn't exist
if [ ! -d "$dest_dir" ]; then
    mkdir -p "$dest_dir"
    echo "Created test destination directory: $dest_dir"
fi

# Get the count of files in the source directory
# `ls -1` lists one file per line, `wc -l` counts the lines.
file_count=$(ls -1 "$source_dir" | wc -l)
if [ "$file_count" -lt 5 ]; then
    echo "Error: Not enough files in $source_dir (found $file_count file(s)). Exiting." >&2
    exit 1
fi

echo "Selecting 5 random files to move..."
# `ls` lists files, `shuf -n 5` shuffles them and picks 5 random lines (filenames).
# The loop reads each filename and moves it.
ls "$source_dir" | shuf -n 5 | while IFS= read -r file; do
    # Construct full source and destination paths
    src_path="$source_dir/$file"
    dest_path="$dest_dir/$file"
    
    # Move the file
    mv "$src_path" "$dest_path"
    echo "Moved file to test dataset: $src_path -> $dest_path"
done

# Verify moved files
echo "Verifying moved files in $dest_dir:"
# List the files in the destination directory for verification.
ls -l "$dest_dir"

echo "Step 5 complete: Test dataset created in $dest_dir."
echo ""

# --- Step 6: Create a virtual environment and activate it ---
echo "Step 6: Creating a virtual environment."
venv_dir="./venv"
if [ ! -d "$venv_dir" ]; then
    # Use python3 to create the virtual environment
    python3 -m venv "$venv_dir"
    echo "Virtual environment created at $venv_dir"
else
    echo "Virtual environment already exists at $venv_dir"
fi

echo "Activating the virtual environment."
# In Bash, you `source` the activation script.
# The script is located in `bin/` on Linux, not `Scripts\`.
source "$venv_dir/bin/activate"
echo "Virtual environment activated."
echo ""

# --- Step 7: Install pip requirements ---
echo "Step 7: Installing pip requirements."
requirements_file="./requirements.txt"
if [ ! -f "$requirements_file" ]; then
    echo "Error: Requirements file not found: $requirements_file" >&2
    # Deactivate venv before exiting
    deactivate
    exit 1
fi

# This command will use the pip from the activated virtual environment
pip install -r "$requirements_file"
echo "Pip requirements installed successfully."
echo ""

echo "--------------------------------------------------------"
echo "All steps completed successfully."
echo "You can now test the model or retrain it as per choice!"
echo "The virtual environment is active. To exit it, run 'deactivate'."
echo "--------------------------------------------------------"
