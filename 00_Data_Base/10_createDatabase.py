#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-05
Description:
    This script initializes and populates a SQLite database with music data from various sources. 
    It performs the following tasks:
    1. Opens and loads genre data from a JSON file.
    2. Creates or drops existing tables and defines their schema in the SQLite database.
    3. Reads and processes data from multiple sources, including JSON, CSV, and SQL tables.
    4. Cleans and prepares the data for insertion into the database, including renaming columns, handling missing values, and transforming tags.
    5. Inserts the cleaned and processed data into the appropriate tables in the database.
    6. Updates genre information based on predefined rules and merges data from different sources.

Usage:
    1. Ensure all dependencies are installed and accessible.
    2. Have run all the scripts before this.
    3. Run the script: 10_createDatabase.py

Note:
    - The script uses logging to track progress and errors.
    - Data from various sources is merged and cleaned before being inserted into the database.
    - The script updates the genre information and handles the conversion of tags to lower case.
    - The final database will include tables for songs, genres, countries, lyrics, and tags.

"""


# Standard library imports
import json
import sqlite3

# Third-party library imports
import pandas as pd
from loguru import logger

# Local application/library specific imports
from vars import (
    csv_countries,
    file_db, 
    file_db_final, 
    folder_input,
    json_genres, 
    table_genius, 
    table_lastfm, 
    table_lima, 
    table_spotify, 
    table_spotify_audio_feats
)
# Opening JSON file
f = open(f"{folder_input}/{json_genres}")

# returns JSON object as 
# a dictionary
genres = json.load(f)

# Closing file
f.close()

def createTables(sqlite_connection, cursor):

    # TABLE GENIUS SONG
    sqlite_connection.execute(f'DROP TABLE IF EXISTS Tbl_Songs_Genius')  
    cursor.execute(f'''CREATE TABLE Tbl_Songs_Genius(
        id INTEGER PRIMARY KEY,
        title TEXT,
        artist TEXT,
        year INTEGER,
        features TEXT,
        tag TEXT,
        id_spotify TEXT,
        similarity_title_sp REAL,
        similarity_artist_sp REAL,
        manual_match_sp TEXT,
        id_lastfm TEXT,
        similarity_title_fm REAL,
        similarity_artist_fm REAL,
        manual_match_fm TEXT,
        id_lima TEXT,
        similarity_title_lm REAL,
        similarity_artist_lm REAL,
        FOREIGN KEY(id_spotify) REFERENCES Tbl_Songs_Spotify(id),    
        FOREIGN KEY(id_lastfm) REFERENCES Tbl_Songs_Lastfm(id),  
        FOREIGN KEY(id_lima) REFERENCES Tbl_Songs_Lima(id)
    )''')

    logger.success(f"Table Tbl_Songs_Genius created successfully.")

    # TABLE SPOTIFY
    sqlite_connection.execute(f'DROP TABLE IF EXISTS Tbl_Songs_Spotify') 
    cursor.execute(f'''CREATE TABLE Tbl_Songs_Spotify(
        id TEXT PRIMARY KEY,
        title TEXT,
        artist TEXT,
        isrc TEXT,
        explicit INTEGER,
        release_date TEXT,
        danceability REAL,
        energy REAL,
        key INTEGER,
        loudness TEXT,
        mode INTEGER,
        speechiness REAL,
        acousticness REAL,
        instrumentalness REAL,
        liveness REAL,
        valence REAL,
        tempo REAL,
        duration_ms INTEGER,
        time_signature INTEGER      
    )''')

    logger.success(f"Table Tbl_Songs_Spotify created successfully.")

    # TABLE LAST FM
    sqlite_connection.execute(f'DROP TABLE IF EXISTS Tbl_Songs_Lastfm') 
    cursor.execute(f'''CREATE TABLE Tbl_Songs_Lastfm(
        id INTEGER PRIMARY KEY,
        title TEXT,
        artist TEXT,
        mbid TEXT             
    )''')

    logger.success(f"Table Tbl_Songs_Lastfm created successfully.")


    # TABLE Vagalume
    sqlite_connection.execute(f'DROP TABLE IF EXISTS Tbl_Songs_Lima') 
    cursor.execute(f'''CREATE TABLE Tbl_Songs_Lima(
        id INTEGER PRIMARY KEY,
        title TEXT,
        artist TEXT,
        genre TEXT             
    )''')

    logger.success(f"Table Tbl_Songs_Lima created successfully.")

    # TABLE Countries
    sqlite_connection.execute(f'DROP TABLE IF EXISTS Tbl_Countries') 
    cursor.execute(f'''CREATE TABLE Tbl_Countries(
        iso2 TEXT PRIMARY KEY,
        country TEXT          
    )''')

    logger.success(f"Table Tbl_Countries created successfully.")


    # TABLE Genres
    sqlite_connection.execute(f'DROP TABLE IF EXISTS Tbl_Tags_Genres')
    cursor.execute(f'''CREATE TABLE Tbl_Tags_Genres(
        tag TEXT PRIMARY KEY,
        genre TEXT          
    )''')

    logger.success(f"Table Tbl_Genres created successfully.")



    # TABLE LYRIC
    sqlite_connection.execute(f'DROP TABLE IF EXISTS Tbl_Lyric')
    cursor.execute(f'''CREATE TABLE Tbl_Lyric(
        id_genius INTEGER,
        lyric TEXT,    
        FOREIGN KEY(id_genius) REFERENCES Tbl_Songs_Genius(id)  
    )''')

    logger.success(f"Table Tbl_Lyric created successfully.")



    # TABLE Song Genres
    sqlite_connection.execute(f'DROP TABLE IF EXISTS Tbl_Songs_Tags')
    cursor.execute(f'''CREATE TABLE Tbl_Songs_Tags(
        id_lastfm INTEGER,
        tag TEXT,
        FOREIGN KEY(id_lastfm) REFERENCES Tbl_Songs_Lastfm(id),    
        FOREIGN KEY(tag) REFERENCES Tbl_Tags_Genres(tag)          
    )''')

    logger.success(f"Table Tbl_Songs_Genres created successfully.")


def main():
    #  Initiate connection to the database of the file "geniusSongsLyrics.db"
    sqlite_connection = sqlite3.connect(file_db)
    cursor = sqlite_connection.cursor()
    logger.success(f"Connection to the Database {file_db} initiated successfully.") 

    df_genius = pd.read_sql_query(f"""SELECT {table_genius}.* 
                             FROM {table_genius}
                            INNER JOIN {table_spotify} ON {table_spotify}.idGenius = {table_genius}.id
                            WHERE {table_spotify}.songFound = 'True'""", sqlite_connection)

    df_spotify = pd.read_sql_query(f"SELECT * FROM {table_spotify} WHERE songFound = 'True'", sqlite_connection)

    df_audio_feats = pd.read_sql_query(f"SELECT * FROM {table_spotify_audio_feats} WHERE songFound = 'True'", sqlite_connection)

    df_lastfm = pd.read_sql_query(f"SELECT * FROM {table_lastfm} WHERE songFound = 'True'", sqlite_connection)

    df_Tbl_Vagalumes = pd.read_sql_query(f"SELECT * FROM {table_lima}", sqlite_connection)
    df_Tbl_Vagalumes = df_Tbl_Vagalumes.reset_index()
    df_Tbl_Vagalumes = df_Tbl_Vagalumes.rename({"id_genius":"id", "index":"id_lima"}, axis=1)

    # Close the connection
    sqlite_connection.close()

    logger.success(f"Connection to Database {file_db} closed.")


    df_lastfm = df_lastfm.reset_index()
    df_lastfm = df_lastfm.rename(columns={'index': 'id'})

    df_iso = pd.read_csv(csv_countries, encoding = "latin1", delimiter = ",")

    df_Tbl_Songs_Genius = df_genius[['id', 'title', 'artist', 'features', 'year', 'tag']]

    df_Tbl_Songs_Spotify = df_spotify[['id', 'title', 'artist', 'isrc', 'explicit', 'releaseDate']]
    df_Tbl_Songs_Spotify = df_Tbl_Songs_Spotify.rename(columns={'releaseDate': 'release_date'})
    df_Tbl_Songs_Spotify = df_Tbl_Songs_Spotify.drop_duplicates(subset=['id'])

    df_audio_feats_true = df_audio_feats.drop(['songFound'], axis=1)
    df_Tbl_Songs_Spotify = df_Tbl_Songs_Spotify.merge(df_audio_feats_true, on = "id")
    df_Tbl_Songs_Spotify = df_Tbl_Songs_Spotify.rename(columns={"durationMs":"duration_ms", 
                                                                "timeSignature":"time_signature"})
    df_Tbl_Songs_Spotify = df_Tbl_Songs_Spotify.drop_duplicates("id")

    df_Tbl_Lyric = df_genius[['id', 'lyric']]
    df_Tbl_Lyric = df_Tbl_Lyric.rename(columns={'id': 'id_genius'})

    df_Tbl_Songs_Lastfm = df_lastfm[['id','title', 'artist', 'mbid']]

    df_genius_spotify = df_spotify[['idGenius','id', 'similarityTitle', 
                                    'similarityArtist', 'manualMatch']]

    df_genius_spotify = df_genius_spotify.rename(columns={'id': 'id_spotify',
                                                                'idGenius': 'id',
                                                                'similarityTitle': 'similarity_title_sp',
                                                                'similarityArtist': 'similarity_artist_sp',
                                                                'manualMatch': 'manual_match_sp'})

    df_genius_lastFm = df_lastfm[['idGenius','id', 'similarityTitle', 
                                    'similarityArtist', 'manualMatch']]

    df_genius_lastFm = df_genius_lastFm.rename(columns={'id': 'id_lastfm',
                                                                'idGenius': 'id',
                                                                'similarityTitle': 'similarity_title_fm',
                                                                'similarityArtist': 'similarity_artist_fm',
                                                                'manualMatch': 'manual_match_fm'})

    df_Tbl_Songs_Genius = df_Tbl_Songs_Genius.merge(df_genius_spotify, on = "id")
    df_Tbl_Songs_Genius = df_Tbl_Songs_Genius.merge(df_genius_lastFm, on = "id")
    df_Tbl_Songs_Genius = df_Tbl_Songs_Genius.merge(df_Tbl_Vagalumes[["id","id_lima"]], on = "id")

    df_Tbl_Songs_Genius["similarity_title_lm"] = 1.0
    df_Tbl_Songs_Genius["similarity_artist_lm"] = 1.0

    df_Tbl_Vagalumes = df_Tbl_Vagalumes.drop(["id"], axis=1)
    df_Tbl_Vagalumes = df_Tbl_Vagalumes.rename({"id_lima": "id"}, axis = 1)

    df_Tbl_Countries = df_iso[['iso2', 'country_common']]
    df_Tbl_Countries = df_Tbl_Countries.rename(columns={'country_common': 'country'})

    df_Songs_Tags = df_lastfm[['idGenius','id', 'tags']].copy()

    # Convert from string to list
    df_Songs_Tags['tags'] = df_Songs_Tags.apply(lambda row:  row['tags'].strip('[]').replace("'", '').replace(' ', '').split(',')   , axis=1)

    # Convert all tags to lower case.
    df_Songs_Tags["tags"] = df_Songs_Tags["tags"].apply(
            lambda song_tags: [t.lower() for t in song_tags])

    # Usando explode para transformar listas em linhas separadas
    df_Songs_Tags = df_Songs_Tags.explode('tags')

    # Drop rows that do not have a valid tag.
    df_Tbl_Songs_Tags = df_Songs_Tags[df_Songs_Tags.tags != ""].reset_index(drop = True)
    df_Tbl_Songs_Tags = df_Tbl_Songs_Tags.rename(columns={'id': 'id_lastfm',
                                                        'tags': 'tag'})
    df_Tbl_Songs_Tags = df_Tbl_Songs_Tags.drop(['idGenius'], axis=1)


    # Obtendo os valores únicos
    unique_tags = df_Songs_Tags['tags'].unique()

    unique_tags = set(unique_tags.tolist())

    unique_tags = [v for v in unique_tags if v not in genres.keys() and v != ""]


    for tag in unique_tags:
        genres[tag] = "outros"


    df_Tbl_Tags_Genres = pd.DataFrame.from_dict(genres.items())
    df_Tbl_Tags_Genres = df_Tbl_Tags_Genres.rename(columns={0: 'tag', 1: 'genre'})

    df_Tbl_Tags_Genres['genre'] = df_Tbl_Tags_Genres['genre'].str.lower()

    #  Initiate connection to the database of the file "geniusSongsLyrics.db"
    sqlite_connection = sqlite3.connect(file_db_final)
    cursor = sqlite_connection.cursor()

    createTables(sqlite_connection, cursor)

    logger.success(f"Connection to the Database {file_db_final} initiated successfully.") 

    df_Tbl_Songs_Genius.to_sql('Tbl_Songs_Genius', sqlite_connection, if_exists='append', index=False)     
    df_Tbl_Songs_Spotify.to_sql('Tbl_Songs_Spotify', sqlite_connection, if_exists='append', index=False) #release_date
    df_Tbl_Songs_Lastfm.to_sql('Tbl_Songs_Lastfm', sqlite_connection, if_exists='append', index=False)
    df_Tbl_Vagalumes.to_sql('Tbl_Songs_Lima', sqlite_connection, if_exists='append', index=False)
    df_Tbl_Countries.to_sql('Tbl_Countries', sqlite_connection, if_exists='append', index=False) 
    df_Tbl_Lyric.to_sql('Tbl_Lyric', sqlite_connection, if_exists='append', index=False) 
    df_Tbl_Songs_Tags.to_sql('Tbl_Songs_Tags', sqlite_connection, if_exists='append', index=False) 
    df_Tbl_Tags_Genres.to_sql('Tbl_Tags_Genres', sqlite_connection, if_exists='append', index=False) 

    logger.success(f"Base populated successfully.") 

    # Close the connection
    sqlite_connection.close()

    logger.success(f"Connection to Database {file_db_final} closed.")


if __name__ == "__main__":
    
    # Start the application
    main()

    # Exit the application
    exit()