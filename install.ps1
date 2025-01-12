# Enable error handling
$ErrorActionPreference = "Stop"

try {
    #---------------------------------------------------
    # Step 1: Download the zip file
    $zipUrl = "https://zenodo.org/records/1290750/files/IRMAS-TrainingData.zip?download=1"
    $zipFile = ".\irmas.zip"
    Write-Output "Step 1: Downloading the ZIP file from $zipUrl"
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipFile
    Write-Output "Download complete: $zipFile"
    
    #---------------------------------------------------
    # Step 2: Unzip the file
    $unzipDestination = ".\irmas"
    Write-Output "Step 2: Unzipping $zipFile to $unzipDestination"
    Expand-Archive -Path $zipFile -DestinationPath $unzipDestination -Force
    Write-Output "Unzip complete."
    
    #---------------------------------------------------
    # Step 3: Move all files from the unzipped folder to data\train\clean
    $trainCleanDir = ".\data\train\clean"
    # Ensure destination directory exists
    if (-not (Test-Path -Path $trainCleanDir)) {
        Write-Output "Creating destination directory: $trainCleanDir"
        New-Item -ItemType Directory -Path $trainCleanDir | Out-Null
    }
    
    Write-Output "Step 3: Moving all files from $unzipDestination to $trainCleanDir"
    Get-ChildItem -Path $unzipDestination -Recurse -File | Where-Object { $_.Extension -eq ".wav" }| Copy-Item -Destination $trainCleanDir
    Write-Output "All files have been moved to $trainCleanDir."
    
    #---------------------------------------------------
    # Step 4: Clean up the downloaded ZIP and the unzipped folder
    Write-Output "Step 4: Deleting the unzipped directory and the ZIP file."
    if (Test-Path -Path $unzipDestination) {
        Remove-Item -Path $unzipDestination -Recurse -Force
        Write-Output "Deleted folder: $unzipDestination"
    }
    if (Test-Path -Path $zipFile) {
        Remove-Item -Path $zipFile -Force
        Write-Output "Deleted file: $zipFile"
    }
    
    #---------------------------------------------------
    # Step 5: Move some random files into the test folder
    # Absolute paths to source and destination directories
    $sourceDir = ".\data\train\clean"
    $destDir   = ".\data\test\clean"

    Write-Output "Step 5: Preparing test dataset by selecting random files from $sourceDir"

    # Check if the source directory exists
    if (-not (Test-Path -Path $sourceDir)) {
        Write-Error "Source directory does not exist: $sourceDir"
        exit
    }

    # Create the destination directory if it doesn't exist
    if (-not (Test-Path -Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        Write-Output "Created test destination directory: $destDir"
    }

    # Get all files from the source directory
    $allFiles = Get-ChildItem -Path $sourceDir -File
    if ($allFiles.Count -lt 5) {
        Write-Error "Not enough files in $sourceDir (found $($allFiles.Count) file(s)). Exiting."
        exit
    }

    # Shuffle and select 5 random files
    $randomFiles = $allFiles | Get-Random -Count 5
    Write-Output "Selected random files: $($randomFiles.Name -join ', ')"

    # Move selected files to the destination directory
    foreach ($file in $randomFiles) {
        $destPath = Join-Path -Path $destDir -ChildPath $file.Name
        Move-Item -literalpath $file.FullName -Destination $destPath -Force
        Write-Output "Moved file to test dataset: $($file.FullName) -> $destPath"
    }

    # Verify moved files
    Write-Output "Verifying moved files in $($destDir):"
    Get-ChildItem -Path $destDir -File | ForEach-Object {
        Write-Output "File found: $($_.FullName)"
    }

    Write-Output "Step 5 complete: Test dataset created in $destDir."

    # Step 6: Create a virtual environment and activate it
    Write-Output "Step 6: Creating a virtual environment."
    $venvDir = ".\venv"
    if (-not (Test-Path -Path $venvDir)) {
        python -m venv $venvDir
        Write-Output "Virtual environment created at $venvDir"
    } else {
        Write-Output "Virtual environment already exists at $venvDir"
    }

    Write-Output "Activating the virtual environment."
    $activateScript = Join-Path -Path $venvDir -ChildPath "Scripts\Activate.ps1"
    if (-not (Test-Path -Path $activateScript)) {
        Write-Error "Activation script not found: $activateScript"
        exit
    }
    & $activateScript
    Write-Output "Virtual environment activated."

    # Step 7: Install pip requirements
    Write-Output "Step 7: Installing pip requirements."
    $requirementsFile = ".\requirements.txt"
    if (-not (Test-Path -Path $requirementsFile)) {
        Write-Error "Requirements file not found: $requirementsFile"
        exit
    }
    pip install -r $requirementsFile
    Write-Output "Pip requirements installed successfully."

    Write-Output "--------------\nAll steps completed successfully. You can test the model now or retrain it as per choice!"

}
catch {
    Write-Error "An error occurred: $_"
}
