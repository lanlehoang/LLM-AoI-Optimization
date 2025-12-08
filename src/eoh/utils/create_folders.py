import os


def create_folders(results_path):
    # Specify the path where you want to create the folder
    folder_path = results_path

    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Create the main folder "results"
        os.makedirs(folder_path)

    # Create subfolders inside "results"
    subfolders = ["history", "pops", "pops_best"]
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
