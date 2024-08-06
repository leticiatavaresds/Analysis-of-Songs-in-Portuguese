#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-05
Description:
    This script processes music data from spotify, lastFm, Genius and Lima sources and merges them into a single SQLite table. 
    It performs the following tasks:
    1. Connects to the SQLite database.
    2. Reads music data from a CSV file and two SQL tables.
    3. Cleans and normalizes the data by removing special characters and converting text to lowercase.
    4. Merges the data from different sources based on matching titles and artists.
    5. Filters and cleans the merged data to remove duplicates and retain relevant records.
    6. Creates a new table in the database and inserts the cleaned data.

Usage:
    1. Ensure all dependencies are installed and accessible.
    2. Have run scripts 02, 03 and 05 before this.
    3. Run the script: 09_joinLimaData.py

Note:
    - The script uses logging to track progress and errors.
    - The CSV and SQL data are merged based on normalized and cleaned title and artist names.
    - Special characters are removed from text fields before merging.
    - The final table `TblSongsLima` will contain unique records with cleaned data.

"""


# Standard library imports
import re
import sqlite3

# Third-party library imports
import pandas as pd
from loguru import logger
import zipfile

# Local application/library specific imports
from vars import table_lastfm, table_spotify, file_db, folder_input, csv_lima


def removeSpecialChars(text):

    # Set of characters to remove
    chars_to_remove = {'!','"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.', '/', ':',
    ';', '=', '?', '@', '[', ']', '_', '`', '|', '}', '\x90', '¡', '´', 'ñ', '\u200b', '’'}

    # Remove specified special characters from the text
    regex_pattern = '[' + re.escape(''.join(chars_to_remove)) + ']'
    return re.sub(regex_pattern, '', text)

def createTables(sqlite_connection, cursor):
    # Drop existing table if it exists and create a new table
    sqlite_connection.execute(f'DROP TABLE IF EXISTS TblSongsLima') 
    
    cursor.execute(f'''CREATE TABLE TblSongsLima(
        id_genius INTEGER PRIMARY KEY,
        title TEXT,
        artist TEXT,
        genre TEXT             
    )''')

    logger.success(f"Table TblSongsLima created successfully.")

def main():
    # Initiate connection to the database
    logger.info(f"Connecting to the SQLite database: {file_db}")
    sqlite_connection = sqlite3.connect(file_db)
    cursor = sqlite_connection.cursor()

    logger.success(f"Connection to the Database {file_db} initiated successfully.") 

    # Read data from CSV and SQL tables
    logger.info(f"Reading data from {csv_lima}.zip")
    with zipfile.ZipFile(f"{folder_input}/{csv_lima}.zip", 'r') as z:

        with z.open(f"{csv_lima}.csv") as f:
            df = pd.read_csv(f)

    logger.info(f"Reading data from {table_spotify} where songFound is True")
    df_sp = pd.read_sql_query(f"SELECT * FROM {table_spotify} WHERE songFound = 'True'", sqlite_connection)
    logger.info(f"Reading data from {table_lastfm} where songFound is True")
    df_fm = pd.read_sql_query(f"SELECT * FROM {table_lastfm} WHERE songFound = 'True'", sqlite_connection)

    logger.info(f"Merging Data")

    # Convert columns to lowercase
    df.artist_name = df['artist_name'].str.lower()
    df.music_title = df['music_title'].str.lower()

    df_sp.titleGenius = df_sp['titleGenius'].str.lower()
    df_sp.artistGenius = df_sp['artistGenius'].str.lower()

    df_fm.title = df_fm['title'].str.lower()
    df_fm.artist = df_fm['artist'].str.lower()

    # Remove special characters from the text
    df.artist_name = df['artist_name'].apply(removeSpecialChars)
    df.music_title = df['music_title'].apply(removeSpecialChars)

    df_sp.titleGenius = df_sp['titleGenius'].apply(removeSpecialChars)
    df_sp.artistGenius = df_sp['artistGenius'].apply(removeSpecialChars)

    # Normalize text to ASCII
    df_sp.titleGenius = df_sp['titleGenius'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df_sp.artistGenius = df_sp['artistGenius'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    # Merge datasets
    df_merge_genius = df_sp.merge(df ,left_on=["titleGenius", "artistGenius"], right_on=["music_title", "artist_name"])

    df.music_title = df['music_title'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df.artist_name = df['artist_name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    df_merge_sp = df_sp.merge(df ,left_on=["title", "artist"], right_on=["music_title", "artist_name"])
    df_merge_fm = df_fm.merge(df ,left_on=["title", "artist"], right_on=["music_title", "artist_name"])

    # Concatenate merged dataframes
    df_merge = pd.concat([df_merge_genius, df_merge_sp, df_merge_fm])

    # Filter and clean the merged dataframe
    logger.info(f"Filtering and cleaning the merged dataframe")
    df_merge = df_merge[((df_merge.similarityTitle == 1) & (df_merge.similarityArtist == 1)) | df_merge.manualMatch == True].reset_index(drop=True)
    df_merge = df_merge.drop_duplicates(subset=['idGenius'])
    df_merge = df_merge[['idGenius','music_title', 'artist_name', 'genre']]
    df_merge = df_merge.rename({"idGenius": "id_genius", 'music_title':'title', 'artist_name':'artist'}, axis=1)

    # Create tables and insert data
    logger.info(f"Creating tables and inserting data into TblSongsLima")
    createTables(sqlite_connection, cursor)
    df_merge.to_sql('TblSongsLima', sqlite_connection, if_exists='append', index=False)

    # Close the SQLite connection
    sqlite_connection.close()
    logger.success(f"Connection to Database {file_db} closed.")

if __name__ == "__main__":
    
    # Start the application
    main()

    # Exit the application
    exit()