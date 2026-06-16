# Adopted from GCAD Repo
# Edited by Phoebe Kuang

import os
import datetime

def make_main_directory(settings, custom_folder_path=None):
    """
    Creates a timestamped directory to save results.
    """
    date = datetime.datetime.now()
    
    # Use the custom name from settings if provided
    if custom_folder_path:
        base_name = custom_folder_path
    else:
        base_name = settings.get("folder_name", "GA_Run")

    # Format: ./results/2026-02-06_PulseGenerator/
    folder_name = f"{base_name}_{date.strftime('%Y-%m-%d')}"
    
    # Create directory if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created directory: {folder_name}")
    else:
        print(f"Directory exists: {folder_name}")
        
    return folder_name