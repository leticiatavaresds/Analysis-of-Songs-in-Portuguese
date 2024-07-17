#!/usr/bin/env python3.9
#  Import libraries
import pandas as pd
from unidecode import unidecode
import re
from multiset import Multiset
from textdistance import damerau_levenshtein
import sqlite3
from loguru import logger
import numpy as np

file_db = "geniusSongsLyrics.db"
table_spotify =  "tblSongsSpotify"

def getStringBefore(name, substring):
    return name.split(substring)[0]

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

def dfIntoTable(table_name, df, sqlite_connection):

    temporary_table = f"temporary_table_{table_name}"

    # Drop temporary table if exists
    sqlite_connection.execute(f'DROP TABLE IF EXISTS {temporary_table}')  

    df = df.astype(object).replace(np.nan, "")

    # Create temporary table containing the dataframe data
    df.to_sql(temporary_table, sqlite_connection) 
    
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



sqlite_connection = sqlite3.connect(file_db)
encoding = "utf8"
sqlite_connection.text_factory = lambda x: str(x, encoding)

cursor = sqlite_connection.cursor()

df = pd.read_sql_query(f"SELECT * FROM {table_spotify}", sqlite_connection)
df_found = df[df.songFound == "True"].reset_index(drop = True)

df_1_1 = df[(df.similarityTitle == 1) & (df.similarityArtist == 1)].reset_index(drop = True)

terms = [" ao vivo", " live", " feat", " acustico", " playback", " bonus track", " bonus", 
         " remasterizado", " versao brasileira", " original album", " remastered", 
         " radio edit", " remaster", " - 2004 Digital Remaster", " (Deluxe Edition)",
         " - Instrumental", " - Faixa Bônus", " - Trilha Sonora Do Filme"]

df_not_equal = df[(df.similarityTitle < 1) | (df.similarityArtist < 1)].reset_index(drop = True)

df_not_equal['titleGenius'] = df_not_equal.apply(lambda row: cleanString(row["titleGenius"]), axis=1)
df_not_equal['artistGenius'] = df_not_equal.apply(lambda row: cleanString(row["artistGenius"]), axis=1)

for term in terms:

    df_not_equal['title'] = df_not_equal.apply(lambda row: getStringBefore(row["titleGenius"], term), axis=1)
    df_not_equal['titleGenius'] = df_not_equal.apply(lambda row: getStringBefore(row["titleGenius"], term), axis=1)


df_not_equal['similarityTitle'] = df_not_equal.apply(lambda row: getMatchSimilarity(row["title"], row["titleGenius"])["score"], axis=1)


dict_artists = df_not_equal[(df_not_equal.similarityTitle < 1) | (df_not_equal.similarityArtist < 1)].artist.value_counts().to_dict()
dict_replace = {}

for artist in dict_artists.keys():

    count_songs = dict_artists[artist]

    genius_artist = df_not_equal[(df_not_equal.artist == artist) & 
             (df_not_equal.similarityArtist < 1)].reset_index(drop = True).artistGenius[0]
    dict_replace[genius_artist] = artist

    if count_songs < 7:
        break



for artist in dict_replace.keys():

    df_not_equal.loc[(df_not_equal.similarityArtist < 1) & 
                 (df_not_equal.artist == artist), 'artistGenius'] = dict_replace[artist]
    

df_not_equal['similarityArtist'] = df_not_equal.apply(lambda row: getMatchSimilarity(row["title"], row["titleGenius"])["score"] 
                   if row['artistGenius'] in dict_replace.keys() and row['similarityArtist'] < 1 
                   else row["similarityArtist"], axis=1)

df_new_1_1 = df_not_equal[(df_not_equal.similarityTitle == 1) & (df_not_equal.similarityArtist == 1)].reset_index(drop = True)

dfIntoTable(table_spotify, df_new_1_1, sqlite_connection)

sqlite_connection.close()