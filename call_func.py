import os
import gdown
import requests


def download_files_from_folder(folder_id, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get file list from the folder
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    # response = requests.get(folder_url)
    # print(response)
    # file_ids = []
    gdown.download_folder(folder_url, quiet=False)
    # for line in response.text.split("\n"):
    #     if "data-id" in line:
    #         file_id = line.split('data-id="')[1].split('"')[0]
    #         file_ids.append(file_id)

    # Download each file in the folder
    # for file_id in file_ids:
    #     file_url = f"https://drive.google.com/uc?id={file_id}"
    #     gdown.download(file_url, os.path.join(output_directory, file_id), quiet=False)


# Example usage
folder_id = "1W53g9bR-vwcqBF73tug96x_qqdIkc-hh"  # Replace with the actual folder ID
output_directory = "datasets/electricity_data"  # Replace with the desired output directory

download_files_from_folder(folder_id, output_directory)

#https://drive.google.com/drive/folders/1W53g9bR-vwcqBF73tug96x_qqdIkc-hh?usp=share_link