
#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-05
Description:
    This script recalculates similarity scores between song titles and artists across the Spotify and Genius tables.
    API. It performs the following tasks:
    1. Establishes a connection to the SQLite database.
    2. Reads song data from a specified table into a DataFrame.
    3. Cleans and preprocesses song titles and artists.
    4. Updates the song data with improved similarity scores.
    5. Identifies and corrects mismatches in artist names.
    6. Recalculates similarity scores based on updated data.
    7. Inserts the updated data back into the SQLite database.
    8. Logs progress and status updates throughout the process.


Usage:
    1. Ensure all dependencies are installed and accessible.
    2. Have run scripts 02 and 03 before this.
    3. Run the script: python 04_improveSimilaritiesSpotify.py

Note:
    - The script uses logging to track progress and errors.
    - Temporary tables are created and dropped during execution.
    - Database connections are opened and closed within the script.

"""


# Standard library imports
import sqlite3

# Third-party library imports
import numpy as np
import pandas as pd
from loguru import logger

# Local application/library specific imports
import functions
from vars import table_spotify, file_db


def dfIntoTable(table_name, df, sqlite_connection):
    temporary_table = f"temporary_table_{table_name}"

    # Drop temporary table if exists
    sqlite_connection.execute(f'DROP TABLE IF EXISTS {temporary_table}')

    # Convert NaN to empty strings
    df = df.astype(object).replace(np.nan, "")

    # Create temporary table containing the dataframe data
    df.to_sql(temporary_table, sqlite_connection, if_exists='replace', index=False)

    # Update main table with data from the temporary table
    logger.info(f"Updating main table: {table_name}")
    sqlite_connection.execute(f"""
        UPDATE {table_name} 
        SET id = temp.id,
            title = temp.title,
            artist = temp.artist,
            artistId = temp.artistId,
            explicit = temp.explicit,
            releaseDate = temp.releaseDate,
            isrc = temp.isrc,
            similarityTitle = temp.similarityTitle,
            similarityArtist = temp.similarityArtist,
            diffYears = temp.diffYears,
            songFound = temp.songFound
        FROM {temporary_table} AS temp 
        WHERE {table_name}.idGenius = temp.idGenius;
    """)

    # Drop temporary table
    sqlite_connection.execute(f'DROP TABLE {temporary_table}')
    sqlite_connection.commit()

def main():
    
    # Establish SQLite connection
    sqlite_connection = sqlite3.connect(file_db)
    logger.success(f"Connection to the Database {file_db} initiated successfully.") 
    encoding = "utf8"
    sqlite_connection.text_factory = lambda x: str(x, encoding)

    cursor = sqlite_connection.cursor()

    # Read data from SQLite table into DataFrame
    logger.info(f"Reading data from table: {table_spotify}")
    df = pd.read_sql_query(f"SELECT * FROM {table_spotify}", sqlite_connection)
    df_found = df[df.songFound == "True"].reset_index(drop=True)

    # DataFrame with exact title and artist match
    df_1_1 = df[(df.similarityTitle == 1) & (df.similarityArtist == 1)].reset_index(drop=True)

    terms = [" ao vivo", " live", " feat", " acustico", " playback", " bonus track", " bonus", 
             " remasterizado", " versao brasileira", " original album", " remastered", 
             " radio edit", " remaster", " - 2004 Digital Remaster", " (Deluxe Edition)",
             " - Instrumental", " - Faixa Bônus", " - Trilha Sonora Do Filme"]

    # DataFrame with non-exact matches
    df_not_equal = df[(df.similarityTitle < 1) | (df.similarityArtist < 1)].reset_index(drop=True)

    # Clean the title and artist strings
    logger.info("Cleaning title and artist strings")
    df_not_equal['titleGenius'] = df_not_equal.apply(lambda row: functions.cleanString(row["titleGenius"]), axis=1)
    df_not_equal['artistGenius'] = df_not_equal.apply(lambda row: functions.cleanString(row["artistGenius"]), axis=1)

    # Remove terms from the titles
    logger.info("Removing specific terms from titles")
    for term in terms:
        df_not_equal['title'] = df_not_equal.apply(lambda row: functions.getStringBefore(row["titleGenius"], term), axis=1)
        df_not_equal['titleGenius'] = df_not_equal.apply(lambda row: functions.getStringBefore(row["titleGenius"], term), axis=1)

    # Recalculate title similarity
    logger.info("Recalculating title similarity")
    df_not_equal['similarityTitle'] = df_not_equal.apply(lambda row: functions.getMatchSimilarity(row["title"], row["titleGenius"])["score"], axis=1)

    # Identify artists with mismatches
    logger.info("Identifying artists with mismatches")
    dict_artists = df_not_equal[(df_not_equal.similarityTitle < 1) | (df_not_equal.similarityArtist < 1)].artist.value_counts().to_dict()
    dict_replace = {}

    # Find corresponding Genius artists for mismatched artists
    logger.info("Finding corresponding Genius artists for mismatched artists")
    for artist in dict_artists.keys():
        count_songs = dict_artists[artist]
        genius_artist = df_not_equal[(df_not_equal.artist == artist) & 
                                     (df_not_equal.similarityArtist < 1)].reset_index(drop=True).artistGenius[0]
        dict_replace[genius_artist] = artist

        if count_songs < 7:
            break

    # Update artistGenius field for mismatched artists
    logger.info("Updating artistGenius field for mismatched artists")
    for artist in dict_replace.keys():
        df_not_equal.loc[(df_not_equal.similarityArtist < 1) & 
                         (df_not_equal.artist == artist), 'artistGenius'] = dict_replace[artist]

    # Recalculate artist similarity
    logger.info("Recalculating artist similarity")
    df_not_equal['similarityArtist'] = df_not_equal.apply(lambda row: functions.getMatchSimilarity(row["title"], row["titleGenius"])["score"] 
                       if row['artistGenius'] in dict_replace.keys() and row['similarityArtist'] < 1 
                       else row["similarityArtist"], axis=1)

    # DataFrame with updated exact matches
    df_new_1_1 = df_not_equal[(df_not_equal.similarityTitle == 1) & (df_not_equal.similarityArtist == 1)].reset_index(drop=True)

    # Insert updated DataFrame back into the table
    logger.info(f"Inserting updated DataFrame back into table: {table_spotify}")
    dfIntoTable(table_spotify, df_new_1_1, sqlite_connection)
    logger.success(f"Table {table_spotify} updated with new similarities.")

    # Close the SQLite connection
    sqlite_connection.close()
    logger.success(f"Connection to Database {file_db} closed.")

if __name__ == "__main__":
    
    # Start the application
    main()

    # Exit the application
    exit()
