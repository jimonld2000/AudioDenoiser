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
    Get-ChildItem -Path $unzipDestination -Recurse -File | ForEach-Object {
        Move-Item -Path $_.FullName -Destination $trainCleanDir
        Write-Output "Moved file: $($_.Name)"
    }
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
    # Step 5: Randomly copy 5 files to create a test dataset
    $sourceDir = ".\data\train\clean"
    $destDir   = ".\data\test\clean"
    
    Write-Output "Step 5: Preparing test dataset by selecting random files from $sourceDir"
    
    # Create the destination directory if it doesn't exist
    if (-not (Test-Path -Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir | Out-Null
        Write-Output "Created test destination directory: $destDir"
    }
    
    # Get all files from the source directory
    $allFiles = Get-ChildItem -Path $sourceDir -File
    if ($allFiles.Count -lt 5) {
        Write-Error "Not enough files in $sourceDir (found $($allFiles.Count) files). Exiting."
        exit
    }
    
    # Shuffle and select 5 random files
    $randomFiles = $allFiles | Get-Random -Count 5
    foreach ($file in $randomFiles) {
        $destPath = Join-Path -Path $destDir -ChildPath $file.Name
        Move-Item -Path $file.FullName -Destination $destPath
        Write-Output "Moved file to test dataset: $($file.Name)"
    }
    Write-Output "Step 5 complete: Test dataset created in $destDir."
    
    Write-Output "All steps completed successfully."
}
catch {
    Write-Error "An error occurred: $_"
}
