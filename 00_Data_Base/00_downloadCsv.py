#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-05
Description: 
    This script downloads datasets "Genius Song Lyrics" and "Country Codes (ISO 3166)" from Kaggle using the Kaggle API. 
    It connects to the Kaggle API, retrieves the specified datasets, and saves them to the local filesystem. 
    The datasets are downloaded in ZIP format and then extracted. 
    The script also logs the progress and status of the download operations.

Usage:
    1. Ensure all dependencies are installed and accessible.
    2. Configure Kaggle API credentials in the file Data_Input/kaggle_credentials.json.
    3. Run the script: python 00_downloadCsv.py
"""

# Standard library imports
import os

# Third-party library imports
import requests
from loguru import logger

# Local application/library specific imports
import functions


def downloadDataset(api, user_owner, dataset_name, file_name):

    # Get dataset metadata
    # Construct URL for dataset download
    url = f'https://www.kaggle.com/api/v1/datasets/download/{user_owner}/{dataset_name}'

    # Make request and stream content
    response = requests.get(url, stream=True, headers={
        'Authorization': f'Bearer {os.environ["KAGGLE_KEY"]}'
    })

    # Get total file size from headers
    total_size = int(response.headers.get('content-length', 0))

    if total_size >= 1024 ** 3:
        total_size = f"{total_size/(1024 ** 3):.2f} GB"
    elif total_size >= 1024 ** 2:
        total_size = f"{total_size/(1024 ** 2):.2f} MB"
    elif total_size >= 1024:
        total_size = f"{total_size/(1024):.2f} KB"

    # Download dataset
    logger.info(f"Downloading file: {dataset_name}.zip ({total_size})")
    api.dataset_download_files(f'{user_owner}/{dataset_name}', path=".", quiet=True, unzip=True)
    logger.success(f"{file_name} file extracted successfully.") 
    logger.success(f"{dataset_name}.zip file deleted successfully") 


def main():

    # Connecting to Kaggle files
    api = functions.connectKaggleApi()
    logger.success("Connection to Kaggle API started successfully.") 

    datasets = [
        ("carlosgdcj", "genius-song-lyrics-with-language-information", "song_lyrics.csv"),
        ("wbdill", "country-codes-iso-3166", "countries_iso3166b.csv")
    ]

    # Donwloading files
    for user_owner, dataset_name, file in datasets:
        downloadDataset(api, user_owner, dataset_name, file)


if __name__ == "__main__":

    # Start the application
    main()

    # Exit the application
    exit()
