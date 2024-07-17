#!/usr/bin/env python3.9
# Import libraries
import pandas as pd
from unidecode import unidecode
import re
from multiset import Multiset
from textdistance import damerau_levenshtein
import sqlite3
from loguru import logger
import numpy as np

# File and table names
file_db = "geniusSongsLyrics.db"
table_spotify =  "tblSongsSpotify"
table_lastfm =  "tblSongsLastFm"

# Establish a connection to the SQLite database
sqlite_connection = sqlite3.connect(file_db)
encoding = "utf8"
sqlite_connection.text_factory = lambda x: str(x, encoding)
logger.success(f"Connection to the Database {file_db} initiated successfully.") 

# Create a cursor object to execute SQL commands
cursor = sqlite_connection.cursor()

# Function to get substring before a specified substring in a string
def getStringBefore(name, substring):
    return name.split(substring)[0]

# Function to clean a string
def cleanString(text):
    text = unidecode(text)  # Remove accents
    text = text.replace(" & ", " e ")  # Replace "&" with "e"
    text = re.sub('[^a-zA-Z0-9]', ' ', text)  # Remove special characters
    text = text.strip()  # Remove spaces at the beginning and end
    text = re.sub(' +', ' ', text)  # Remove multiple spaces
    text = text.lower()  # Convert to lowercase
    return text

# Function to tokenize text into bigrams
def tokenizeText(txt):
    arr = []
    for wrd in txt.lower().split('  '):
        arr += ([wrd] if len(wrd) == 1 else [wrd[i:i+2] for i in range(len(wrd)-1)])
    return Multiset(arr)

# Function to calculate Sorenson-Dice similarity of multisets
def sorensonDice(text1, text2):
    text1, text2 = tokenizeText(text1), tokenizeText(text2)
    dice = 2 * len(text1 & text2) / (len(text1) + len(text2))
    return dice

# Function to get match similarity score using Sorenson-Dice and Damerau-Levenshtein
def getMatchSimilarity(str1, str2):
    dice_sim = sorensonDice(str1, str2)
    lv_sim = damerau_levenshtein.normalized_similarity(str1, str2)
    dice_weight = 0.8
    lv_weight = 1 - dice_weight
    score =  (lv_sim * lv_weight) + (dice_sim * dice_weight)
    return {
        'dice': dice_sim,
        'lv': lv_sim,
        'score': score
    }

# Function to update a table in the SQLite database with a DataFrame
def dfIntoTable(table_name, df, sqlite_connection):
    sqlite_connection.execute('DROP TABLE IF EXISTS temporary_table_genius')  # Drop temporary table if it exists
    df.to_sql('temporary_table_genius', sqlite_connection)  # Create temporary table with DataFrame data
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
    sqlite_connection.execute('DROP TABLE temporary_table_genius')  # Drop temporary table
    sqlite_connection.commit()  # Commit changes

# Load data from the tables into DataFrames
df_lastfm = pd.read_sql_query(f"SELECT * FROM {table_lastfm}", sqlite_connection)
df_spotify = pd.read_sql_query(f"SELECT * FROM {table_spotify}", sqlite_connection)

# Load data from the tables into DataFrames
try:
    df_lastfm = pd.read_sql_query(f"SELECT * FROM {table_lastfm}", sqlite_connection)
    df_spotify = pd.read_sql_query(f"SELECT * FROM {table_spotify}", sqlite_connection)
    logger.success("Data loaded successfully from the database tables.")
except Exception as e:
    logger.error(f"Failed to load data from database tables: {e}")
    exit()


# Filter Spotify data for entries where similarity is not perfect
df_sp_not_1_1 = df_spotify[(df_spotify.similarityTitle < 1) | (df_spotify.similarityArtist < 1)].reset_index(drop=True)
logger.info(f"Filtered Spotify data: {len(df_sp_not_1_1)} entries with non-perfect similarity.")

# Filter LastFM data for non-empty tags and non-perfect similarity
df_not_equal = df_lastfm[(df_lastfm.tags != "[]") & (~df_lastfm.tags.isna()) & 
                              ((df_lastfm.similarityTitle < 1) | (df_lastfm.similarityArtist < 1))].reset_index(drop=True)
logger.info(f"Filtered LastFM data: {len(df_not_equal)} entries with non-empty tags and non-perfect similarity.")

# Create dictionaries for artists and replacements
dict_artists = df_not_equal[(df_not_equal.similarityTitle < 1) | (df_not_equal.similarityArtist < 1)].artist.value_counts().to_dict()
dict_replace = {}

# Determine replacements for artists based on similarity
for artist in dict_artists.keys():
    count_songs = dict_artists[artist]
    genius_artist = df_not_equal[(df_not_equal.artist == artist) & 
             (df_not_equal.similarityArtist < 1)].reset_index(drop=True).artistGenius[0]
    dict_replace[genius_artist] = artist
    if count_songs < 4:
        break
logger.info(f"Artist replacements determined: {len(dict_replace)} replacements identified.")

# Update artistGenius in the DataFrame based on replacements
for artist in dict_replace.keys():
    df_not_equal.loc[(df_not_equal.similarityArtist < 1) & 
                 (df_not_equal.artist == artist), 'artistGenius'] = dict_replace[artist]

logger.info("ArtistGenius values updated in the DataFrame based on replacements.")

# Recalculate similarity for updated artists
df_not_equal['similarityArtist'] = df_not_equal.apply(lambda row: getMatchSimilarity(row["title"], row["titleGenius"])["score"] 
                   if row['artistGenius'] in dict_replace.keys() and row['similarityArtist'] < 1 
                   else row["similarityArtist"], axis=1)

logger.info("Similarity recalculated for updated artists.")

# Filter updated DataFrame for perfect matches
df_new_1_1 = df_not_equal[(df_not_equal.similarityTitle == 1) & (df_not_equal.similarityArtist == 1)].reset_index(drop=True)
logger.info(f"Filtered for perfect matches: {len(df_new_1_1)} entries found.")

# Update LastFM table with new perfect matches
dfIntoTable(table_lastfm, df_new_1_1, sqlite_connection)

# Reload LastFM data
df_lastfm = pd.read_sql_query(f"SELECT * FROM {table_lastfm}", sqlite_connection)
logger.info("LastFM data reloaded after updates.")

# Merge Spotify and LastFM DataFrames
df_merge = pd.merge(df_spotify, df_lastfm, how="inner", on="idGenius").reset_index(drop=True)
logger.info(f"Merged Spotify and LastFM DataFrames: {len(df_merge)} entries merged.")

# Filter merged data for non-perfect matches
df_not_1 = df_merge[(df_merge.similarityTitle_y < 1) | (df_merge.similarityArtist_y < 1)
         | (df_merge.similarityTitle_x < 1) | (df_merge.similarityArtist_x < 1)].reset_index(drop=True)
df_not_1 = df_not_1[df_not_1.songFound_y == "True"]
df_not_1 = df_not_1[df_not_1.tags != "[]"].reset_index(drop=True)
logger.info(f"Filtered merged data for non-perfect matches: {len(df_not_1)} entries remain.")

# Prepare DataFrame for manual analysis
df_manual_analysis = df_not_1[['idGenius', 'titleGenius_x', 'artistGenius_x',
       'featuresGenius_x', 'id', 'title_x', 'artist_x', 'similarityTitle_x', 'similarityArtist_x',
         'title_y', 'artist_y', 'similarityTitle_y', 'similarityArtist_y', 'tags']]

# Rename columns for clarity
df_manual_analysis = df_manual_analysis.rename(columns={'titleGenius_x': 'titleGenius',
                                                        'artistGenius_x': 'artistGenius',
                                                        'featuresGenius_x': 'featuresGenius',
                                                        'title_x': 'titleSP',
                                                        'artist_x': 'artistSP',
                                                        'similarityTitle_x': 'similarityTitleSP',
                                                        'similarityArtist_x': 'similarityArtistSP',
                                                        'title_y': 'titleFM',
                                                        'artist_y': 'artistFM',
                                                        'similarityTitle_y': 'similarityTitleFM',
                                                        'similarityArtist_y': 'similarityArtistFM'})

# Add columns for manual analysis
df_manual_analysis["perfectMatchSP"] = [""] * len(df_manual_analysis)
df_manual_analysis["perfectMatchFM"] = [""] * len(df_manual_analysis)
logger.info("Prepared DataFrame for manual analysis with new columns added.")

# Save DataFrame for manual analysis to a CSV file
df_manual_analysis.to_csv("df_manual_analysis.csv", index=False, encoding="utf-8-sig", sep = ";")
logger.success(f"DataFrame for manual analysis 'df_manual_analysis.csv' exported successfully.")

# Close the SQLite connection
sqlite_connection.close()
logger.success(f"Connection to Database {file_db} closed.")