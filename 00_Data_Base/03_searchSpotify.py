#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-05
Description:
    This script interacts with the Spotify API and Genius API to search for song information based on
    song titles and artist names. It performs two types of searches:
    1. By track ID: Retrieves detailed information about a specific track using its Spotify ID.
    2. By name: Searches for tracks by name and artist, returning relevant information such as track ID,
       artist ID and release date.
    
    The script also updates the local SQLite database with the search results. It handles various scenarios
    including errors and low similarity matches, and performs database operations to ensure that song
    information is correctly recorded and updated.

Usage:
    1. Ensure all dependencies are installed.
    2. Configure Genius API credentials in the file Data_Input/kaggle_genius.json.
    3. Configure Spotify API credentials in the file Data_Input/spotify_genius.json.
    4. Ensure that the database exists (created by script 01 and filtered by 02), If it does not exist, run script 02. 
        This script will search for all 5 million songs in the Genius database.
    5. Run the script to start the search and update process: python 03_searchSpotify.py

Note:
    - The script uses global variables for API credentials and configuration.
    - Error handling is included to manage interruptions and API errors.

"""

# Standard library imports
import datetime as dt
import sqlite3
import time
from difflib import SequenceMatcher

# Third-party library imports
import numpy as np
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm
from unidecode import unidecode

# Local application/library specific imports
import functions
from vars import file_db, table_genius, table_spotify



def spotifySearch(name, artist):

    global access_token, dict_credenctials_sp, credential

    header = {
            'Authorization': f'Bearer {access_token}'}

    query = f"track%3A{name.replace(' ', '%20')}%20artist:{artist.replace(' ', '%20')}"
    'https://api.spotify.com/v1/search?q=track%3ABotas+artist%3AXutos+%26+Pontap%C3%A9s&type=track'

    params = {"q": query,
                "type": "track"}

    url = "https://api.spotify.com/v1/search?q=" + query

    # Get response from 'search' endpoint using song name and artist name as parameters
    response = requests.get(url,  params=params, headers=header, timeout=15)

    # Check if response contains no errors
    response = functions.processResponse(response, url, params)

    # Return response as a dict
    return response


def spotifyGetTrack(id):

    global access_token, dict_credenctials_sp, credential

    header = {
            'Authorization': f'Bearer {access_token}'}

    url = "https://api.spotify.com/v1/tracks/" + id

    # Get response from 'tracks' endpoint using song id as parameter
    response = requests.get(url, headers = header, timeout = 15)

    # Check if response contains no errors
    response = functions.processResponse(response, url, None)

    # Return response as a dict
    return response

def searchSongSpotifyByName(song, artist, feats, year_genius):

    global access_token

    dict_track = {} 
    
    try:
        tracks = spotifySearch(song, artist)

    except KeyboardInterrupt:
        logger.error("\nProgram terminated by user.")
        exit()
        
    except:
        dict_track['error'] = True
        return dict_track
    
    if "error" in list(tracks.keys()):
        dict_track['error'] = tracks['error']
    
    # If the search returned results, get the data
    elif len(tracks['tracks']['items']) > 0:
    
        track = tracks['tracks']['items'][0]
        artist_name = track['artists'][0]['name'].replace('"', "'")
        song_name = track["name"].replace('"', "'")
        id_spotify = track["id"]
        isrc_code = track['external_ids']['isrc']
        artist_id =  track['artists'][0]['id']
        explicit = track['explicit']
        date = track['album']['release_date']
        date_precision = track['album']['release_date_precision']
        all_artists = [artist['name'].replace('"', "'").upper() for artist in track['artists']]
        all_artistsStr = unidecode(", ".join(sorted(all_artists)))

        # Set the string format of the data based on the date precision
        if date_precision == "day":
            date_format = "%Y-%m-%d"

        elif date_precision == "month":
            date_format = "%Y-%m"

        elif date_precision == "year":
            date_format = "%Y"
        
        # Get the year from the date release
        ano = dt.datetime.strptime(date, date_format).year

        # If the release year of the Spotify and Genius songs are the same, 
        #   assign the difference between the years as 0
        if ano == year_genius:
            diff_years = 0
        
        # Else, calculate the distance between the year from spotify and Genius 
        else:
            # If the two years are after 1970, calculate the difference in float
            if ano >= 1970 and year_genius >= 1970:
                data_stamp= time.mktime(dt.datetime.strptime(date, date_format).timetuple())
                data_stamp_genius = time.mktime(dt.datetime.strptime(str(year_genius), "%Y").timetuple())
                diff_years = round((abs(data_stamp - data_stamp_genius)/2592000)/12, 2)

            # If any of the years is after 1970, calculate the difference in integer
            else:
                diff_years = abs(ano - year_genius)


        artist_name = functions.cleanString(artist_name)
        artistClean = functions.cleanString(artist)
        song_name = functions.cleanString(song_name)
        song = functions.cleanString(song)    

     
        # Get the distance between the artist name from spotify and Genius 
        similarity_artist = functions.getMatchSimilarity(artist_name, artistClean)["score"]

    
        if similarity_artist < 1 and " & " in artist:
            artists = artist.upper().split(" & ")
            artistsStr = unidecode(", ".join(sorted(artists)))            
            new_similarity = functions.getMatchSimilarity(artistsStr, all_artistsStr)["score"]
            
            

            if new_similarity > similarity_artist:
                similarity_artist = new_similarity

            if new_similarity < 1:
                firstArtist = artist.upper().split(" & ")[0].split(",")[0]
                new_similarity = functions.getMatchSimilarity(functions.cleanString(firstArtist), artist_name)["score"]

               
                if new_similarity > similarity_artist:
                    similarity_artist = new_similarity

        # Get the distance between the song name from spotify and Genius  
        similarity_title = functions.getMatchSimilarity(song_name, song)["score"]


        similarity_allArtist = SequenceMatcher(None, feats.upper(), all_artistsStr.upper()).ratio()
    
        
        dict_track = {  "title": song_name,
                    'artist': artist_name,
                    "id": id_spotify,
                    "artistId": artist_id,
                    "explicit": explicit,
                    "releaseDate": date,
                    "similarityTitle": similarity_title,
                    "similarityArtist": similarity_artist,
                    "similarityAllArtist": similarity_allArtist,
                    "diffYears": diff_years,
                    "songFound": "True",
                    'isrc': isrc_code,
                    'all_artists': all_artistsStr}
        
    return dict_track

def searchSongSpotifyById(id_spotify, song, artist,feats, year_genius):

    global access_token

    try:    
        # Connect to Spotify API to get the song data by the id code
        track = spotifyGetTrack(id_spotify)
        artist_name = track['artists'][0]['name'].replace('"', "'")
        song_name = track["name"].replace('"', "'")

    except KeyboardInterrupt:
        logger.error("\nProgram terminated by user.")
        exit()

    except:
        # If the request got an error, return en empty dictionary
        dict_track = {'error': True}
        return dict_track

    # Get the song data from the response
    isrc_code = track['external_ids']['isrc'] # the ISRC code
    artist_id =  track['artists'][0]['id'] # the artist spotify id
    explicit = track['explicit'] # a boolean indicator if the song has explicit content
    date = track['album']['release_date'] # the album release date
    date_precision = track['album']['release_date_precision'] # the precision with which release_date value is known
    all_artists = [artist['name'].replace('"', "'") for artist in track['artists']]
    all_artistsStr = unidecode(", ".join(sorted(all_artists)))

    # Set the string format of the data based on the date precision
    if date_precision == "day":
        date_format = "%Y-%m-%d"

    elif date_precision == "month":
        date_format = "%Y-%m"

    elif date_precision == "year":
        date_format = "%Y"
    
    # Get the year from the date release
    ano = dt.datetime.strptime(date, date_format).year

    # If the release year of the Spotify and Genius songs are the same, 
    #   assign the difference between the years as 0
    if ano == year_genius:
        diff_years = 0
    
    # Else, calculate the distance between the year from spotify and Genius 
    else:
        # If the two years are after 1970, calculate the difference in float
        if ano >= 1970 and year_genius >= 1970:
            data_stamp= time.mktime(dt.datetime.strptime(date, date_format).timetuple())
            data_stamp_genius = time.mktime(dt.datetime.strptime(str(year_genius), "%Y").timetuple())
            diff_years = round((abs(data_stamp - data_stamp_genius)/2592000)/12, 2)

        # If any of the years is after 1970, calculate the difference in integer
        else:
            diff_years = abs(ano - year_genius)

    # Clean the strings 
    artist_name = functions.cleanString(artist_name)
    artist = functions.cleanString(artist)
    song_name = functions.cleanString(song_name)
    song = functions.cleanString(song)    

    # Get the distance between the artist name from spotify and Genius 
    similarity_artist = functions.getMatchSimilarity(artist_name, artist)["score"]

    if similarity_artist < 1 and " & " in artist:
        artistsStr = artist.upper().split(" & ")        
        all_artistsStr = unidecode(", ".join(sorted(all_artists)))
        new_similarity = functions.getMatchSimilarity(artistsStr, all_artistsStr)["score"]
        

        if new_similarity > similarity_artist:
            similarity_artist = new_similarity

        if new_similarity < 1:
            firstArtist = artist.upper().split(" & ")[0].split(",")[0]
            new_similarity = functions.getMatchSimilarity(functions.cleanString(firstArtist), artist_name)["score"]
            

            if new_similarity > similarity_artist:
                similarity_artist = new_similarity

    # Get the distance between the song name from spotify and Genius  
    similarity_title = functions.getMatchSimilarity(song_name, song)["score"]

    similarity_allArtist = SequenceMatcher(None, feats.upper(), all_artistsStr.upper()).ratio()

    dict_track = {  "title": song_name,
                    'artist': artist_name,
                    "id": id_spotify,
                    "artistId": artist_id,
                    "explicit": explicit,
                    "releaseDate": date,
                    "similarityTitle": similarity_title,
                    "similarityArtist": similarity_artist,
                    "similarityAllArtist": similarity_allArtist,
                    "diffYears": diff_years,
                    "songFound": "True",
                    'isrc': isrc_code,
                    'all_artists': all_artistsStr}
    
    # Return the song data
    return dict_track

def createTable(table_spotify, table_genius, sqlite_connection, cursor, dict_columns):

    # Create spotify table from Genius table selecting only some columns
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_spotify} AS
            SELECT id       AS idGenius, 
                   title    AS titleGenius, 
                   artist   AS artistGenius, 
                   year     AS yearGenius,  
                   features AS featuresGenius  
            FROM {table_genius};""")

    sqlite_connection.commit()

    # Dictionary containing new columns names for the spotify table and their respective types
    dict_columns = {
        "title": "TEXT",
        "artist": "TEXT",
        "id": "TEXT",
        "artistId": "TEXT",
        "isrc": "TEXT",
        "explicit": "INTEGER",
        "similarityTitle": "REAL",
        "similarityArtist": "REAL",
        "diffYears": "REAL",
        "songFound": "TEXT",
        "releaseDate": "TEXT"
        }
    

    # For each column in dictionary, create the column in the table
    for column, type in dict_columns.items():
        cursor.execute(f"ALTER TABLE {table_spotify} ADD COLUMN {column} '{type}'")

    sqlite_connection.commit()

def dfIntoTable(table_name, df, sqlite_connection):

    temporary_table = f"temporary_table_{table_name}"

    # Drop temporary table if exists
    sqlite_connection.execute(f'DROP TABLE IF EXISTS {temporary_table}')  

    df = df.astype(object).replace(np.nan, "")

    df_found = df[df.songFound == "True"]

    # Create temporary table containing the dataframe data
    df_found.to_sql(temporary_table, sqlite_connection) 
    
    # Join the temporary table to the the table using the idGenius as key
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
       
    # Drop temporary table if exists
    sqlite_connection.execute(f'DROP TABLE {temporary_table}')  
    sqlite_connection.commit()

    df_not_found = df[df.songFound != "True"]

    # Create temporary table containing the dataframe data
    df_not_found.to_sql(temporary_table, sqlite_connection) 
    
    # Join the temporary table to the the table using the idGenius as key
    sqlite_connection.execute(f"""
        UPDATE {table_name} 
        SET songFound = temp.songFound
        FROM {temporary_table} AS temp 
        WHERE {table_name}.idGenius = temp.idGenius;
        """)   
       
    # Drop temporary table if exists
    sqlite_connection.execute(f'DROP TABLE {temporary_table}')  
    sqlite_connection.commit()


def songNotFound(table_name, song_id, sqlite_connection):
    
    # Set column 'song_Found' as 'False' at the row of the song, locating using the song_id as key
    sqlite_connection.execute(f"""
        UPDATE {table_name} 
        SET songFound = 'False'
        WHERE idGenius = {song_id};
        """)                                      
    sqlite_connection.commit()

def createTable(table_spotify, table_genius, sqlite_connection, cursor):

    # Create spotify table from Genius table selecting only some columns
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_spotify} AS
            SELECT id       AS idGenius, 
                   title    AS titleGenius, 
                   artist   AS artistGenius, 
                   year     AS yearGenius,  
                   features AS featuresGenius  
            FROM {table_genius};""")

    sqlite_connection.commit()

    # Dictionary containing new columns names for the spotify table and their respective types
    dict_columns = {
        "title": "TEXT",
        "artist": "TEXT",
        "id": "TEXT",
        "artistId": "TEXT",
        "isrc": "TEXT",
        "explicit": "INTEGER",
        "similarityTitle": "REAL",
        "similarityArtist": "REAL",
        "diffYears": "REAL",
        "songFound": "TEXT",
        "releaseDate": "TEXT"
        }
    

    # For each column in dictionary, create the column in the table
    for column, type in dict_columns.items():
        cursor.execute(f"ALTER TABLE {table_spotify} ADD COLUMN {column} '{type}'")

    sqlite_connection.commit()
    

def searchSongsSpotify(df, len_df, sqlite_connection):


    df = df.reset_index(drop = True)
    max = 50
    count_loop = 0
    columns = ['idGenius', 'title', 'artist', 'id', 'artistId', 'explicit', 
                'releaseDate', 'similarityTitle', 'similarityArtist', "similarityAllArtist",
                'diffYears', 'songFound', 'isrc', 'all_artists']
    
    # Create an empty dataFrame to save the searchs data
    df_searchs = pd.DataFrame(columns = columns)

    # Update progress bar at each iteration
    with tqdm(total = len_df) as progress_bar:

        # For each row of the dataframe, performs the search for the song
        for index_row in range(0, len_df):           

            count_loop += 1

            song_found = False
            low_similarity = False
            error = False

            # Get the song data to search
            song = df["titleGenius"][index_row]
            artist = df["artistGenius"][index_row]
            song_id = df["idGenius"][index_row]
            year_genius = df["yearGenius"][index_row]
            
            song_search = f'{song} {artist}'
            song_search = ' '.join(song_search.split())

            # Create an empty dataFrame to save the search data
            df_search = pd.DataFrame(columns = columns)

            # Get more Genius Data about the song
            try:
                song_genius = genius.song(song_id)
                artists = [artist['name'] for artist in song_genius['song']['primary_artists']]
                feats = [artist['name'] for artist in song_genius['song']['featured_artists']]
                artists = artists + feats
                feats = unidecode(", ".join(sorted(artists)))
            except:
                
                feats = df['featuresGenius'][index_row]
                artists = artist + feats
            
            feats = unidecode(", ".join(sorted(artists)))

            try:
                # check if there is the spotify ID
                media = song_genius["song"]['media']
                spotify = [a for a in media if a['provider'] == 'spotify']

            # Exit the application if user terminate by pressing ctrl + c
            except KeyboardInterrupt:
                logger.error("\nProgram terminated by user.")
                exit()
            
            except:
                spotify = []

            # If there is the spotify id, search the song by id
            if len(spotify) > 0:

                # Try to get the Spotify Song data by the Spotify id
                id_spotify = spotify[0]['url'].split("track/")[1]
                dict_track = searchSongSpotifyById(id_spotify, song, artist, feats, year_genius)
                

                # If the search returned data without errors, save it in the dataframe
                if len(dict_track) > 0 and "error" not in dict_track.keys():                

                    new_row = [song_id] + list(dict_track.values())
                    df_search.loc[len(df_search)] = new_row
                    song_found = True


                    if (dict_track["similarityArtist"] < 1 or dict_track['similarityTitle'] < 1):
                        low_similarity = True
                        

            # If the song was not found or has a low similarity, 
            #   perform a new search for the name of the song and artist
            if not song_found or low_similarity:      

                for artist in artists:

                    try:  
                            
                        dict_track = searchSongSpotifyByName(song, artist, feats, year_genius)

                        # If the search returned data, save it in the dataframe 
                        if len(dict_track) > 0 and "error" not in dict_track.keys():
                            song_found = True
                            
                            new_row = [song_id] + list(dict_track.values())
                            df_search.loc[len(df_search)] = new_row


                            if dict_track["similarityArtist"] >= 1.0:
                                break

                        elif "error" in dict_track.keys():
                            df_search = pd.DataFrame(columns = columns)
                            df_search.at[0,"idGenius"] = song_id
                            df_search.at[0,"songFound"] = f'Error  {dict_track["error"]}'
                            error = True
                        
                    except:

                        df_search = pd.DataFrame(columns = columns)
                        df_search.at[0,"idGenius"] = song_id
                        df_search.at[0,"songFound"] = 'Error'
                        error = True

            # If both searchs returned data, keep the data with more similarity
            if len(df_search) > 1:
                df_search['similarityFinal'] = df_search['similarityArtist'] + df_search['similarityTitle']
                df_search = df_search.sort_values('similarityFinal', 
                                                ascending=False).drop_duplicates("idGenius", 
                                                                                keep="first").reset_index(drop=True)
                df_search.drop(['similarityFinal'], axis=1, inplace=True)
            
            
            # If the song was not found by any search, 
            #   update the table by setting False at the column 'musicFound'
            elif (not song_found and not error):  

                df_search = pd.DataFrame(columns = columns)
                df_search.at[0,"idGenius"] = song_id
                df_search.at[0,"songFound"] = 'False'

            # Concat the searchs dataframe and the current search dataframe
            df_searchs = pd.concat([df_searchs, df_search]).reset_index(drop = True)
                

            # Update the table with the searchs data every 50 iterations
            if(count_loop == max or index_row == (len_df - 1)):

                count_loop = 0
                # Update the table with the searchs data
                dfIntoTable(table_spotify, df_searchs, sqlite_connection)
                df_searchs = pd.DataFrame(columns = columns)


            # Update the printed progress bar
            progress_bar.update(1)

       
def main():

    global access_token, credential

    # Start connection to the database
    logger.success(f"Connection to the Database {file_db} initiated successfully.") 
    sqlite_connection = sqlite3.connect(file_db)
    cursor = sqlite_connection.cursor()

    query_response = cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_spotify}'")

    # Check if table exists, if not, create it
    if query_response.fetchone() == None:

        logger.debug(f"Creating table {table_spotify} in the database...")  

        createTable(table_spotify, table_genius, sqlite_connection, cursor)

        logger.success(f"Table {table_spotify} created successfully.")

    else:
        logger.info(f"Table {table_spotify} not created as it already exists in the database.") 


    # Get Spotify API access token
    access_token = functions.spotifyAccessToken(dict_credenctials_sp[credential]["client_id"], 
                                    dict_credenctials_sp[credential]["client_secret"]) 


    # Get database rows that have not been searched
    df = pd.read_sql_query(f"SELECT * FROM {table_spotify} WHERE (songFound != 'False' AND songFound != 'True') OR songFound IS NULL", 
                        sqlite_connection)
    
    # len_df = len(df) 
    len_df = 100
    logger.info(f"Table {table_spotify} read successfully, there are {len_df} songs to search.")  

    logger.debug(f"Searching for songs on the Spotify database...")
    start = time.time()

    # perform search for songs
    searchSongsSpotify(df, len_df, sqlite_connection)

    # Get time of search execution
    time_exec = time.time() - start
    time_exec = functions.secondsToStr(time_exec)

    logger.success(f"Search completed successfully.")   

    # Get the number of songs that were found
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_spotify} WHERE songFound == 'True'")
    num_founds = query_response.fetchone()[0]

    # Get the number of songs that were NOT found
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_spotify} WHERE songFound == 'False'")
    num_not_founds = query_response.fetchone()[0]

    # Get the number of songs that returned error
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_spotify} WHERE songFound LIKE 'Error%'")
    num_errors = query_response.fetchone()[0]

    # Get the number of songs that was not searched
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_spotify} WHERE songFound IS NULL")
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

    genius = functions.getGenius()
    dict_credenctials_sp = functions.getSpotifyDict()
    credential = 0
    
    # Start the application
    main()

    # Exit the application
    exit()
