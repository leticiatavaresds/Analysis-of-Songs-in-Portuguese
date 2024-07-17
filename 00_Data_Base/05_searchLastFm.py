#!/usr/bin/env python3.9
# Import libraries
import pandas as pd
import requests
import sqlite3
import time
import json

from datetime import timedelta
from difflib import SequenceMatcher
from loguru import logger
from lyricsgenius import Genius
from time import localtime, sleep, strftime
from tqdm import tqdm

import re
from unidecode import unidecode
from multiset import Multiset
from textdistance import damerau_levenshtein

f = open('genius_credentials.json')
data = json.load(f)
genius_key = data['key']
genius = Genius(genius_key)

f = open('lastfm_credentials.json')
data = json.load(f)
user_agent = data['user_agent']
api_key = data['api_key']

table_genius = "tblGeniusSongsLyrics"
table_spotify =  "tblSongsSpotify"
table_lastfm =  "tblSongsLastFm"
file_db = "geniusSongsLyrics.db"


def secondsToStr(elapsed=None):

    # Convert seconds to string in the format "days hours:minutes:seconds"
    if elapsed is None:
        return strftime("%d %H:%M:%s", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def createTable(table_lastfm, table_spotify, sqlite_connection, cursor):
    
    # Create lastFm table from Spotify table selecting only some columns
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_lastfm} AS
            SELECT idGenius, 
                   titleGenius, 
                   artistGenius,
                   artist AS artistSpotify,
                   featuresGenius
            FROM {table_spotify}""")

    sqlite_connection.commit()

    # Dictionary containing new columns names for the lastfm table and their respective types
    dict_columns = {
        "title": "TEXT",
        "artist": "TEXT",
        "mbid": "TEXT",
        "similarityTitle": "REAL",
        "similarityArtist": "REAL",
        "songFound": "TEXT",
        "tags": "TEXT"
        }
    

    # For each column in dictionary, create the column in the table
    for column, type in dict_columns.items():
        cursor.execute(f"ALTER TABLE {table_lastfm} ADD COLUMN {column} '{type}'")

    sqlite_connection.commit()


def get_song_lastfm(name_track, name_singer = None): 

    # Get API response.   

    headers = {
    'user-agent': user_agent
    }

    payload = {
    'api_key': api_key,
    'method': 'track.getInfo',
    'format': 'json',
    'artist': name_singer,
    'track': name_track
    }

    response_song = requests.get('https://ws.audioscrobbler.com/2.0/', headers=headers, params=payload)

    return response_song

def processResponse(response):

    status_code = response.status_code

    # If response has no errors, return the response json
    if status_code == 200 or status_code == "200":

        try:
            response = response.json()
        except:
            response = {"error": response.status_code}
         
    # Else return a dict containing the error
    else:
        response = {"error": response.status_code}

    return response

def updateSongFound(table_name, song_id, status, sqlite_connection):
    
    # Set column 'song_Found' as status at the row of the song, locating using the song_id as key
    sqlite_connection.execute(f"""
        UPDATE {table_name} 
        SET songFound = '{status}'
        WHERE idGenius = {song_id};
        """)                                      
    sqlite_connection.commit()


def dfIntoTable(table_name, df, sqlite_connection):

    # Drop temporary table if exists
    sqlite_connection.execute('DROP TABLE IF EXISTS temporary_table_genius')  

    # Create temporary table containing the dataframe data
    df.to_sql('temporary_table_genius', sqlite_connection) 
    
    # Join the temporary table to the the table using the idGenius as key
    sqlite_connection.execute(f"""
        UPDATE {table_name} 
        SET title = temp.title,
            artist = temp.artist,
            mbid = temp.mbid,
            similarityTitle = temp.similarityTitle,
            similarityArtist = temp.similarityArtist,
            tags = temp.tags,
            songFound = temp.songFound
        FROM temporary_table_genius AS temp 
        WHERE {table_name}.idGenius = temp.idGenius;
        """)   
       
    # Drop temporary table if exists
    sqlite_connection.execute('DROP TABLE temporary_table_genius')  
    sqlite_connection.commit()

def tokenizeText(txt):
    
    # Convert a phrase into a count of bigram tokens of its words
    arr = []
    for wrd in txt.lower().split('  '):
        arr += ([wrd] if len(wrd) == 1 else [wrd[i:i+2]
                for i in range(len(wrd)-1)])
        
    return Multiset(arr)

def sorensonDice(text1, text2):
    
    # Sorenson-Dice similarity of Multisets
    text1, text2 = tokenizeText(text1), tokenizeText(text2)
    dice = 2 * len(text1 & text2) / (len(text1) + len(text2))

    return dice

def cleanString(text):

    # Remove the accents
    text = unidecode(text)

    text = text.replace(" & ", " e ")

    # Remove special characters
    text = re.sub('[^a-zA-Z0-9]', ' ', text)

    # Remove space at the beginning of the string and at the end
    text = text.strip()

    # Remove multiple spaces
    text = re.sub(' +', ' ', text)

    # Convert all uppercase characters in a string into lowercase characters 
    text = text.lower()
    
    return text

def getMatchSimilarity(str1, str2):

    # Clean the Strings
    # str1, str2 = cleanString(str1), cleanString(str2)

    # Calculate the Soreson Dice 
    dice_sim = sorensonDice(str1, str2)
    
    # Calculate the Damerau Lavenshtein distance
    lv_sim = damerau_levenshtein.normalized_similarity(str1, str2)

    # Set the weights
    dice_weight = 0.8
    lv_weight = 1 - dice_weight

    # Calculate the Score
    score =  (lv_sim * lv_weight) + (dice_sim * dice_weight)

    return {
        'dice': dice_sim,
        'lv': lv_sim,
        'score': score
    }

def searchSongsGenius(df, len_df, sqlite_connection):

    # Update progress bar at each iteration
    with tqdm(total = len_df) as progress_bar:

        # For each row of the dataframe, performs the search for the song
        for index_row in range(0, len_df):

            # Get track's Genius ID
            song_id = df["idGenius"][index_row]

            # Get track's name and artist to search
            song = df["titleGenius"][index_row]
            # artist = df["artist"][index_row]
            artist_genius = df["artistGenius"][index_row]
            artist_spotify = df["artistSpotify"][index_row]       
            feats = df['featuresGenius'][index_row].strip('{}')
            feats = feats.split(', ')


            # Get API response
            try:
                response = get_song_lastfm(song, artist_spotify)

            except:
                # Skip if an error occured.
                print(f"    !!! Error for song {song} - {artist_spotify}")

            time.sleep(0.25)

            # Check if response contains no errors
            lfm_data = processResponse(response)


            # Skip if response is an error
            if "error" in lfm_data.keys():

                if lfm_data["error"] == 6:

                    # Set marker indicating that the track was NOT FOUND in the database
                    updateSongFound(table_lastfm, song_id, "False", sqlite_connection)
                    f"{index_row+1} of {len(df)}: Not Found data for {artist_spotify} - {song}"
                else:
                    # Set marker indicating that the search returned error in the database
                    updateSongFound(table_lastfm, song_id, "Error", sqlite_connection)


                # Update the printed progress bar
                progress_bar.update(1)
                continue

            # If no error, get the data
            song_lastfm = lfm_data["track"]["name"]
            artist_lastfm = lfm_data["track"]["artist"]["name"]

            artist_lastfm = cleanString(artist_lastfm)
            artist_spotify = cleanString(artist_spotify)
            song_lastfm = cleanString(song_lastfm)
            song = cleanString(song)

            # Get the distance between the artist name from lastFm and Genius 
            similarity_artist = getMatchSimilarity(artist_lastfm, artist_spotify)["score"]

            # Get the distance between the song name from lastFm and Genius 
            similarity_title = getMatchSimilarity(song_lastfm, song)["score"]

            if similarity_artist < 1:
                similarity_artist_2 = getMatchSimilarity(artist_lastfm, cleanString(artist_genius))["score"]

                if similarity_artist_2 > similarity_artist:
                    similarity_artist = similarity_artist_2

        

            if similarity_artist < 1 or similarity_title < 1:

                for feat in feats:

                    try:
                        response_2 = get_song_lastfm(song, feat)

                    except:
                        # Skip if an error occured.
                        print(f"    !!! Error for song {song} - {artist_spotify}")

                    lfm_data_2 = processResponse(response)

                    if "error" in lfm_data.keys():
                        continue

                    song_lastfm_2 = cleanString(lfm_data_2["track"]["name"])
                    artist_lastfm_2 = cleanString(lfm_data_2["track"]["artist"]["name"])

                    # Get the distance between the artist name from lastFm and Genius 
                    similarity_artist_2 = getMatchSimilarity(artist_lastfm_2, artist_spotify)["score"]

                    # Get the distance between the song name from lastFm and Genius 
                    similarity_title_2 = getMatchSimilarity(song_lastfm_2, song)["score"]

                    if (similarity_title_2 >= similarity_title) and (similarity_artist_2 >= similarity_artist):
                        lfm_data = lfm_data_2

                        if (similarity_title_2 == 1) and (similarity_artist_2 == 1):
                            break


            # If tags was returned, get tags
            if "toptags" not in lfm_data["track"]:
                tags = None

            else:
                tags = [x["name"] for x in lfm_data["track"]["toptags"]["tag"]] 

            # If mbid was returned, get mbid
            if "mbid" not in lfm_data["track"]:
                mbid = None

            else:
                mbid = lfm_data["track"]["mbid"]

            # Save Row data and API data to new dataframe.
            dfSearch = df.iloc[[index_row]].copy()        

            dfSearch["title"] = [song_lastfm]
            dfSearch["artist"] = [artist_lastfm]
            dfSearch["mbid"] = [mbid]
            dfSearch["similarityTitle"] = similarity_title
            dfSearch["similarityArtist"] = similarity_artist
            dfSearch["tags"] = [str(tags)]
            dfSearch["songFound"] = ['True']

            # Update SQliteTable with new row using Genius ID as identifier
            dfIntoTable(table_lastfm, dfSearch, sqlite_connection)

            # Update the printed progress bar
            progress_bar.update(1)


def main():

    # Start connection to the database
    logger.success(f"Connection to the Database {file_db} initiated successfully.") 
    sqlite_connection = sqlite3.connect(file_db)
    cursor = sqlite_connection.cursor()

    query_response = cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_lastfm}'")

    # Check if table exists, if not, create it
    if query_response.fetchone() == None:

        logger.debug(f"Creating table {table_lastfm} in the database...")  

        createTable(table_lastfm, table_spotify, sqlite_connection, cursor)

        logger.success(f"Table {table_lastfm} created successfully.")

    else:
        logger.info(f"Table {table_lastfm} not created as it already exists in the database.") 


    df = pd.read_sql_query(f"SELECT * FROM {table_lastfm} WHERE songFound IS NULL OR songFound = 'False'", sqlite_connection)
    len_df = len(df) 

    logger.debug(f"Searching for songs on the LastFm database...")
    start = time.time()

    # perform search for songs
    searchSongsGenius(df, len_df, sqlite_connection)

    # Get time of search execution
    time_exec = time.time() - start
    time_exec = secondsToStr(time_exec)

    logger.success(f"Search completed successfully.")   

    # Get the number of songs that were found
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_lastfm} WHERE songFound == 'True'")
    num_founds = query_response.fetchone()[0]

    # Get the number of songs that were NOT found
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_lastfm} WHERE songFound == 'False'")
    num_not_founds = query_response.fetchone()[0]

    # Get the number of songs that returned error
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_lastfm} WHERE songFound LIKE 'Error%'")
    num_errors = query_response.fetchone()[0]

    # Get the number of songs that was not searched
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_lastfm} WHERE songFound IS NULL")
    num_unsearched = query_response.fetchone()[0]

    space = 45 * " "
    logger.info(f"""Search Performance: 
                    {space} Returned data for {num_founds} songs
                    {space} Did not return data for {num_not_founds} songs.
                    {space} Showed errors for {num_errors} songs.
                    {space} Not performed for {num_unsearched} songs.
                    {space} Execution time: {time_exec}""") 

    #Closing the connection
    sqlite_connection.close()

    logger.success(f"Connection to Database {file_db} closed.")


if __name__ == "__main__":
    
    # Start the application
    main()

    # Exit the application
    exit()