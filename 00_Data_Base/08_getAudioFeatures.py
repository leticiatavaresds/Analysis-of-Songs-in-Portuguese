#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-05
Description:
    This script retrieves audio features for tracks from Spotify and updates the local SQLite database with the fetched data.
    It performs the following tasks:
    1. Connects to the SQLite database.
    2. Creates a table for storing Spotify audio features if it does not already exist.
    3. Retrieves an access token for the Spotify API.
    4. Queries the database for tracks that do not have audio features and retrieves their audio features from Spotify.
    5. Updates the database with the fetched audio features.
    6. Logs the number of tracks successfully found and the number of tracks not found, along with the execution time.


Usage:
    1. Ensure all dependencies are installed and accessible.
    2. Configure Spotify API credentials in the file Data_Input/spotify_genius.json.
    3. Have run script 03 before this.
    4. Run the script: python 08_getAudioFeatures.py

Note:
    - The script uses logging to track progress and errors.
    - A progress bar is used to show the status of the fetching process.
    - Temporary tables are used during execution to update the main table.

"""


# Standard library imports
import sqlite3
import time

# Third-party library imports
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

# Local application/library specific imports
import functions
from vars import table_spotify, file_db, table_spotify_audio_feats


def spotifyGetTracksResponse(endpoint, listIds):

    global access_token, dict_credenctials_sp, credential

    header = {
            'Authorization': f'Bearer {access_token}'}

    ids = ",".join(listIds)

    url = f"https://api.spotify.com/v1/{endpoint}?ids=" + ids

    response = requests.get(url, headers=header, timeout = 15)
    
    # Check if response contains no errors
    response = functions.processResponse(response, url, None)

    # Return response as a dict
    return response



def dfIntoTable(table_name, df, sqlite_connection):

    temporary_table = 'temporary_table_' + table_name

    # Drop temporary table if exists
    sqlite_connection.execute(f'DROP TABLE IF EXISTS {temporary_table}')  

    # Create temporary table containing the dataframe data
    df.to_sql(temporary_table, sqlite_connection) 
    
    # Join the temporary table to the the table using the idGenius as key
    sqlite_connection.execute(f"""
        UPDATE {table_name} 
        SET danceability = temp.danceability,
            energy = temp.energy,
            key = temp.key,
            loudness = temp.loudness,
            mode = temp.mode,
            speechiness = temp.speechiness,
            acousticness = temp.acousticness,
            instrumentalness = temp.instrumentalness,
            liveness = temp.liveness,
            valence = temp.valence,
            tempo = temp.tempo,
            durationMs = temp.duration_ms,
            timeSignature = temp.time_signature,
            songFound = temp.songFound
        FROM {temporary_table} AS temp 
        WHERE {table_name}.id = temp.id;
        """)   

       
    # Drop temporary table if exists
    sqlite_connection.execute(f'DROP TABLE {temporary_table}')  
    sqlite_connection.commit()


def createTable(table_spotify_audio_feat, table_spotify, sqlite_connection, cursor):

    # Create spotify table from Genius table selecting only some columns
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_spotify_audio_feat} AS
            SELECT id
            FROM {table_spotify}
            WHERE songFound = 'True';""")

    sqlite_connection.commit()

    # Dictionary containing new columns names for the spotify table and their respective types
    dict_columns = {
        "danceability": "REAL",
        "energy": "REAL",
        "key": "INTEGER",
        "loudness": "TEXT",
        "mode": "INTEGER",
        "speechiness": "REAL",
        "acousticness": "REAL",
        "instrumentalness": "REAL",
        "liveness": "REAL",
        "valence": "REAL",
        "tempo": "REAL",
        "durationMs": "INTEGER",
        "timeSignature": "INTEGER",
        "songFound":"TEXT"
        }
    

    # For each column in dictionary, create the column in the table
    for column, type in dict_columns.items():
        cursor.execute(f"ALTER TABLE {table_spotify_audio_feat} ADD COLUMN {column} '{type}'")

    sqlite_connection.commit()
    

def Get_Musics(tracks_id):
    
    # Get Track Json  
    tracks_features = spotifyGetTracksResponse("audio-features", tracks_id)['audio_features']

  
    df_tracks = pd.DataFrame()
    
    for i in range(len(tracks_features)):

        # try:

        #     print(id_track)
        #     print([track_ft for track_ft in tracks_features if id_track in track_ft.values()])
        #     track_features = [track_ft for track_ft in tracks_features if id_track in track_ft.values()][0]  
        #     print(id_track)
            
        #     # Delete some data
        #     del track_features['analysis_url']
        #     del track_features['uri']
        #     del track_features['track_href']
        #     del track_features['type']

        #     dict_track_features = {key: val for key, val in track_features.items()}

        # except:

        dict_track_features = tracks_features[i]

        if dict_track_features is not None:

            del dict_track_features['analysis_url']
            del dict_track_features['uri']
            del dict_track_features['track_href']
            del dict_track_features['type']

            dict_track_features['songFound'] = "True"

        else:
            keyList = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo', 'id', 'duration_ms', 'time_signature', 'songFound']
            dict_track_features = dict(zip(keyList, [None]*len(keyList)))
            dict_track_features['id'] = tracks_id[i]
            dict_track_features['songFound'] = "False"
            
            
        df_dict_track = pd.DataFrame([dict_track_features], columns= dict_track_features.keys())

        df_tracks = pd.concat([df_tracks, df_dict_track])
        
    df_tracks.reset_index(drop = True, inplace = True)
    df_tracks.insert(0, 'id', df_tracks.pop('id'))
      
    return df_tracks
    

def main():

    global access_token, credential

    # Start connection to the database
    logger.success(f"Connection to the Database {file_db} initiated successfully.") 
    sqlite_connection = sqlite3.connect(file_db)
    cursor = sqlite_connection.cursor()

    query_response = cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_spotify_audio_feats}'")

    # Check if table exists, if not, create it
    if query_response.fetchone() == None:

        logger.debug(f"Creating table {table_spotify_audio_feats} in the database...")  

        createTable(table_spotify_audio_feats, table_spotify, sqlite_connection, cursor)

        logger.success(f"Table {table_spotify_audio_feats} created successfully.")

    else:
        logger.info(f"Table {table_spotify_audio_feats} not created as it already exists in the database.") 


    # Get Spotify API access token
    access_token = functions.spotifyAccessToken(dict_credenctials_sp[credential]["client_id"], 
                                    dict_credenctials_sp[credential]["client_secret"]) 


    # Get database rows that have not been searched
    df = pd.read_sql_query(f"SELECT * FROM {table_spotify_audio_feats}", sqlite_connection)
    df = df[df.danceability.isna()].reset_index(drop=True)
    len_df = len(df)
    len_group = len(df.index)//50
    logger.info(f"Table {table_spotify_audio_feats} read successfully, there are {len_df} songs to search.")  

    logger.debug(f"Searching for songs on the Spotify database...")
    start = time.time()

    # Update progress bar at each iteration
    with tqdm(total = len_group) as progress_bar:

        for idx, dfRows in df.groupby(df.index//50):

            # try:
            tracks_id=list(dfRows.id)

            df_tracks = Get_Musics(tracks_id)

            dfIntoTable(table_spotify_audio_feats, df_tracks, sqlite_connection)

            # Update the printed progress bar
            progress_bar.update(1)
        

    # Get time of search execution
    time_exec = time.time() - start
    time_exec = functions.secondsToStr(time_exec)

    logger.success(f"Search completed successfully.")   

    # Get the number of songs that were found
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_spotify_audio_feats} WHERE songFound == 'True'")
    num_founds = query_response.fetchone()[0]

    # Get the number of songs that were NOT found
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_spotify_audio_feats} WHERE songFound == 'False'")
    num_not_founds = query_response.fetchone()[0]

    space = 45 * " "
    logger.info(f"""Search Performance: 
                    {space} Returned data for {num_founds} songs
                    {space} Did not return data for {num_not_founds} songs.
                    {space} Execution time: {time_exec}""")  

    #Closing the connection
    sqlite_connection.close()

    logger.success(f"Connection to Database {file_db} closed.")

if __name__ == "__main__":

    genius = functions.getGenius()
    dict_credenctials_sp = functions.getSpotifyDict()
    credential = 0
    
    # Start the application
    main()

    # Exit the application
    exit()