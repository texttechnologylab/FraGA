import os
import zipfile
from tqdm import tqdm
from pathlib import Path

# Variables to configure
input_folder = "tracking_json"  # Change this to your input folder path
output_folder = "tracking_json_zip"  # Change this to your output folder path


def zip_folder_exclude_audio_json(folder_path, zip_path):
    """Zip a folder excluding audio.json files"""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file != 'audio.json':
                    file_path = os.path.join(root, file)
                    # Calculate the archive name (relative path from the folder being zipped)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)


def main():
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return

    ids = set()
    # Iterate through each item in the input folder
    for item in tqdm(os.listdir(input_folder)):
        item_path = os.path.join(input_folder, item)

        # Only process directories
        if os.path.isdir(item_path):
            # Create zip filename
            item_split = item.split("_")
            item_split[0] = "fraga"

            item = "_".join(item_split[:4])

            if item_split[3] in ids:
                zip_filename = f"{item}_1.zip"
                zip_path = os.path.join(output_folder, zip_filename)
            else:
                ids.add(item_split[3])
                zip_filename = f"{item}_0.zip"
                zip_path = os.path.join(output_folder, zip_filename)

            try:
                print(f"Zipping folder: {item}")
                zip_folder_exclude_audio_json(item_path, zip_path)
                print(f"Successfully created: {zip_filename}")
            except Exception as e:
                print(f"Error zipping folder '{item}': {str(e)}")

    print("Zipping process completed!")


if __name__ == "__main__":
    main()