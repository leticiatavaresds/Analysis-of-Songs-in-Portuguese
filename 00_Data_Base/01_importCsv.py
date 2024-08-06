#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-05
Description:
    This script reads data from file "song_lyrics.csv" containing song lyrics and metadata, 
    and imports this data into a SQLite database. It performs the following tasks:
    1. Connects to a SQLite database.
    2. Deletes any existing table with the name specified.
    3. Reads the data from the CSV file.
    4. Creates a new table in the database.
    5. Imports the data from the CSV file into the newly created table.
    6. Logs the progress and status of these operations.

Usage:
    1. Ensure all dependencies are installed and accessible.    
    2. Ensure that the file is downloaded (run script 00).
    3. Run the script to start the search and update process: python 01_importCsv.py

"""

# Standard library imports
import csv
import sqlite3

# Third-party library imports
from tqdm import tqdm
from loguru import logger

# Local application/library specific imports
from vars import csv_genius, file_db, table_genius


def main():

    # Initiate connection to the database of the file "geniusSongsLyrics.db"
    sqlite_connection = sqlite3.connect(file_db)
    cursor = sqlite_connection.cursor()

    # Delete the table "geniusSongsLyrics if it already exists in the database
    cursor.execute(f'''DROP TABLE IF EXISTS {table_genius}''')

    logger.success(f"Connection to the Database {file_db} initiated successfully.") 

    logger.debug(f"Reading File {csv_genius}...")

    # Read the csv file with the data
    with open(f'{csv_genius}', encoding="utf8") as f:
        reader = csv.reader(f)
        data = list(reader)

    len_df = len(data)

    # Delete the first line that contains the column names
    del data[0] 
    logger.success("File read successfully.")
        
    # Create the table "geniusSongsLyrics" as an empty tabled
    logger.debug("Creating empty table...")
    cursor.execute(f'''CREATE TABLE {table_genius} (
        id INTEGER,
        title TEXT,
        tag TEXT,
        artist TEXT,
        year INTEGER,
        views INTEGER,
        features TEXT,
        lyric TEXT,
        languageCld3 TEXT,
        languageFt TEXT,
        language TEXT
    )''')

    logger.success(f"Table {table_genius} created successfully.")

    logger.debug("Importing data into the table...")

    # For each line of data read from the csv, import the data into the table
    with tqdm(total = len_df) as progress_bar:

        for row in data:
            cursor.execute(f"""INSERT INTO {table_genius} (title, tag, artist, year, views, features, lyric, id, languageCld3, languageFt, language) 
                           values (?,?,?,?,?,?,?,?,?,?,?)""", row)

            progress_bar.update(1)
            
    logger.success(f"Data successfully imported into the table {table_genius} save to file {file_db}")

    # Commit the changes in the database
    sqlite_connection.commit()

    # Close the connection
    sqlite_connection.close()

    logger.success(f"Connection to Database {file_db} closed.")


if __name__ == "__main__":
    
    # Start the application
    main()

    # Exit the application
    exit()