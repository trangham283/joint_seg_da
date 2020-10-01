import os
import argparse
from urllib.request import urlretrieve
from zipfile import ZipFile

def download_data(args):
    """Download and unpack dialogs"""
    if not os.path.exists(args.raw_data_dir):
        os.makedirs(args.raw_data_dir)
    zipfile_path = f"{args.raw_data_dir}/swda.zip"
    extracted_dir = f"{args.raw_data_dir}/swda"

    if not os.path.exists(args.zipfile_path):
        print(f'Downloading {args.download_url} to {args.zipfile_path}')
        urlretrieve(args.download_url, args.zipfile_path)
        print(f'Successfully downloaded {args.zipfile_path}')

    zip_ref = ZipFile(args.zipfile_path, 'r')
    zip_ref.extractall(args.raw_data_dir)
    zip_ref.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_url", 
            default="http://compprag.christopherpotts.net/code-data/swda.zip")
    parser.add_argument("--raw_data_dir", default="/s0/ttmt001")
    args = parser.parse_args()

    download_data(args)

