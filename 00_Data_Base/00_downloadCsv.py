#!/usr/bin/env python3.9

# Import libraries
import json
import os
import zipfile
from loguru import logger
 
def connectKaggleApi():
    
    # Open JSON file
    f = open('kaggle_credentials.json')

    data = json.load(f)

    # Declare credentials for connecting to the kaggle API
    os.environ['KAGGLE_USERNAME'] = data['username']
    os.environ['KAGGLE_KEY'] = data['key']

    from kaggle.api.kaggle_api_extended import KaggleApi

    # Start connection to the Kaggle API"
    api = KaggleApi()
    api.authenticate()

    logger.success(f"Connection to Kaggle API started successfully.") 

    return api

def main():

    # Start connection to the Kaggle API" 
    api = connectKaggleApi()

    logger.debug(f"Downloading the Dataset Genius Song Lyrics...") 

    dataset_name = "genius-song-lyrics-with-language-information"
    user_owner = "carlosgdcj"
    
    # Download the dataset
    api.dataset_download_files(f'{user_owner}/{dataset_name}', path=".")

    logger.success(f"File downloaded successfully.") 

    logger.debug(f"Extracting file from zip compression...") 
    # Extract the csv file from the zip compression
    with zipfile.ZipFile(f'{dataset_name}.zip', 'r') as zip_ref:

        file_name = zip_ref.filelist[0].filename
        zip_ref.extractall(".")
    logger.success(f"File {file_name} extracted successfully.") 
    
    # Remove zip file
    os.remove(f'{dataset_name}.zip')
    logger.success(f"{dataset_name}.zip file deleted successfully") 

    logger.debug(f"Downloading the Dataset Country Codes Iso-3166...") 

    dataset_name = "country-codes-iso-3166"
    user_owner = "wbdill"

    # Download the dataset
    api.dataset_download_files(f'{user_owner}/{dataset_name}', path=".")

    logger.success(f"File downloaded successfully.") 

    logger.debug(f"Extracting file from zip compression...") 
    # Extract the csv file from the zip compression
    with zipfile.ZipFile(f'{dataset_name}.zip', 'r') as zip_ref:

        file_name = zip_ref.filelist[0].filename
        zip_ref.extractall(".")
    logger.success(f"File {file_name} extracted successfully.") 

    # Remove zip file
    os.remove(f'{dataset_name}.zip')
    logger.success(f"{dataset_name}.zip file deleted successfully") 

if __name__ == "__main__":

    main()

    # Exit the application
    exit()