# AudioDenoiser

Audio Denoiser project made in Python using an ML approach


## Setup
The first thing would be to be sure the CUDA setup is ready for the local usage of the training script (with GPU, otherwise CPU will be used).
The steps can be found here: https://github.com/clementw168/install-and-test-gpu?tab=readme-ov-file

After setup and testing, the compatible cuda runtime and pytorch versions must be set in the requirements.txt:
```
nvidia-cuda-runtime-cu12 # change the version if needed
```
and 
```
--extra-index-url https://download.pytorch.org/whl/cu124 # change the version if needed
```

After this the setup of the project just needs to be started by running the *install.ps1* script. This will automatically do the following things:
1. Download, extract and place the clean audio data from the IRMAS dataset
2. Create necessary folders for test dataset and move files there (split)
3. Delete the zip and other remaining content
4. Create a venv and activate it
5. Install modules from requirements.txt

After this, the scripts can be used!

## Usage
The first step would be to activate the venv which has the modules installed:
```
.\venv\Scripts\activate
```

After this, the datasets need to be created (which means converting the .wav audio files into the processsed .npy spectrogram form we use in the project).
For this the next scripts must be ran:
```
python -u .\code\create_train_dataset.py
```
```
python -u .\code\create_test_dataset.py
```
After these, you can train/test the models
### Training
You can run a new training by just running the training script:
```
python -u .\code\train.py
```
### Testing the models
You can run a new test by just running the testing script:
```
python -u .\code\test.py
```

