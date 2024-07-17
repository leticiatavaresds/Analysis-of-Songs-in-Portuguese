#!/usr/bin/env python3.9
# Import libraries
import csv
import sqlite3
from tqdm import tqdm
from loguru import logger

print("\n")

def main():

    table_name = "tblGeniusSongsLyrics"
    file_db = "geniusSongsLyrics.db"
    csv_name = "song_lyrics.csv"

    # Initiate connection to the database of the file "geniusSongsLyrics.db"
    sqlite_connection = sqlite3.connect(file_db)
    cursor = sqlite_connection.cursor()

    # Delete the table "geniusSongsLyrics if it already exists in the database
    cursor.execute(f'''DROP TABLE IF EXISTS {table_name}''')

    logger.success(f"Connection to the Database {file_db} initiated successfully.") 

    logger.debug(f"Reading File {csv_name}...")

    # Read the csv file with the data
    with open(f'C://Users//letic//Desktop//Spotify//{csv_name}', encoding="utf8") as f:
        reader = csv.reader(f)
        data = list(reader)

    len_df = len(data)

    # Delete the first line that contains the column names
    del data[0] 
    logger.success("File read successfully.")
        
    # Create the table "geniusSongsLyrics" as an empty tabled
    logger.debug("Creating empty table...")
    cursor.execute(f'''CREATE TABLE {table_name} (
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

    logger.success(f"Table {table_name} created successfully.")

    logger.debug("Importing data into the table...")

    # For each line of data read from the csv, import the data into the table
    with tqdm(total = len_df) as progress_bar:

        for row in data:
            cursor.execute(f"""INSERT INTO {table_name} (title, tag, artist, year, views, features, lyric, id, languageCld3, languageFt, language) 
                           values (?,?,?,?,?,?,?,?,?,?,?)""", row)

            progress_bar.update(1)
            
    logger.success(f"Data successfully imported into the table {table_name} save to file {file_db}")

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