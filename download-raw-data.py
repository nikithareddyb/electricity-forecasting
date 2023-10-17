# Two types of datasets are being downloaded here
# 1. Electricity Data: 40 zip files
# 2. Electricity rate hike each year: 1 .csv file

import gdown


class DatasetDownloader:
    def __init__(self, drive_links_inp):
        self.drive_links = drive_links_inp

    def get_fileid_from_link(self, drive_link):
        # Extract file ID from the Google Drive link
        file_id = drive_link.split("/")[-2]
        # Construct the download URL
        download_url = f"https://drive.google.com/uc?id={file_id}"
        return download_url

    def get_folderid_from_link(self, drive_link):
        # Extract file ID from the Google Drive link
        folder_id = drive_link.split("?")[0].split("/")[-1]
        # Construct the download URL
        download_url = f"https://drive.google.com/drive/folders/{folder_id}"
        return download_url

    def download_datasets(self, output_directory):
        for filename, path in self.drive_links.items():
            if 'folder' in path:
                gdown.download_folder(self.get_folderid_from_link(path), output=output_directory+filename, quiet=False)
            else:
                gdown.download(self.get_fileid_from_link(path), output_directory+filename, quiet=False)


# dataset links
data_download_links = {
    "electricity_data": "https://drive.google.com/drive/folders/1W53g9bR-vwcqBF73tug96x_qqdIkc-hh?usp=share_link",
    "PG&E Rate Hikes vs Inflation.csv": "https://drive.google.com/file/d/1wp55C5TA26mGWCTcU64YFpQ1h9MHKLzx/view?usp=share_link"
}


downloader = DatasetDownloader(data_download_links)
downloader.download_datasets("datasets/")
