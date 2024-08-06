#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-06
Version: 1.0.0
Description: This script processes CSV files containing NLP data and merges them with song data from a SQLite database. 
    It performs several operations such as loading CSV files, merging DataFrames, executing SQL queries, and preparing the final dataset for analysis. 
    The processed data is then saved to the CSV file "data_analysis.csv". 


Usage:
    1. Run the script: python 01_preparing_data.py.

"""


# Standard library imports
import os  # Operating system interface
import sqlite3  # SQLite database interface
from functools import reduce  # Functional programming utilities

# Third-party library imports
import pandas as pd  # Data manipulation and analysis
from loguru import logger  # Advanced logging functionality

# Path to the folder containing CSV files
nlp_folder = '../01_NLP/Data_Output/'
db_folder = '../00_Data_Base/'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(nlp_folder) if f.endswith('.csv')]

# Dictionary to store DataFrames
dfs = {}

# Read each CSV file and create a dynamic variable with the file name
for file in csv_files:
    # Generate variable name based on file name
    var_name = f"df_{os.path.splitext(file)[0]}"
    if var_name != "df_00_clean_lyric" and "tfidf" not in var_name and "lda" not in var_name:
        # Read CSV into a DataFrame
        file_path = os.path.join(nlp_folder, file)
        dfs[var_name] = pd.read_csv(file_path)
        
        # Use globals() to dynamically create the variable in the global scope
        globals()[var_name] = dfs[var_name]
        
        # Log loading of each DataFrame
        logger.info(f"Loaded DataFrame {var_name} from {file_path}")



# Merge all DataFrames on 'id'
dataframes = list(dfs.values())
df_merged = reduce(lambda left, right: pd.merge(left, right, on='id'), dataframes)


# Drop unnecessary columns from merged DataFrame
# drop_columns = ['lyric', 'clean_lyric_x','clean_lyric_y', 'tokens', 'stop_words', 
#                 'bigrams', 'trigrams', 'slangs', 'slurs', 'lemmas']
drop_columns = ['clean_lyric_x','clean_lyric_y', 'tokens', 'stop_words', 
                'bigrams', 'trigrams']
# df_merged = df_merged.drop(drop_columns, axis=1)

# Define database file path
output_folder = "Data_Output"
db_file = "SongsDatabase.db"

# Establish connection to the database
sqlite_connection = sqlite3.connect(db_folder + db_file)
cursor = sqlite_connection.cursor()
logger.success(f"Connected to the database {db_file} successfully.")

# Execute SQL query to fetch data
query = """
WITH genres_cte AS (
    SELECT
        g.id,
        GROUP_CONCAT(DISTINCT ge.genre) AS genres
    FROM
        Tbl_Songs_Genius g
    INNER JOIN Tbl_Songs_Tags tg ON tg.id_lastfm = g.id_lastfm
    INNER JOIN Tbl_Tags_Genres ge ON tg.tag = ge.tag
    WHERE
        ((similarity_title_sp = 1 AND similarity_artist_sp = 1) OR manual_match_sp = 'True')
        AND ((similarity_title_fm = 1 AND similarity_artist_fm = 1) OR manual_match_fm = 'True')
    GROUP BY
        g.id
)
SELECT DISTINCT
    g.id AS id_genius,
    l.lyric,
    s.*,
    GROUP_CONCAT(DISTINCT tg.tag) AS tags,
    CASE
        WHEN genres_cte.genres = 'outros' THEN 'outros'
        WHEN INSTR(genres_cte.genres, 'outros') > 0 THEN
            TRIM(REPLACE(REPLACE(genres_cte.genres, 'outros,', ''), ',outros', ''), ',')
        ELSE
            genres_cte.genres
    END AS genres
FROM
    Tbl_Songs_Genius g
INNER JOIN Tbl_Lyric l ON g.id = l.id_genius
INNER JOIN Tbl_Songs_Spotify s ON g.id_spotify = s.id
INNER JOIN Tbl_Songs_Tags tg ON tg.id_lastfm = g.id_lastfm
INNER JOIN Tbl_Tags_Genres ge ON tg.tag = ge.tag
INNER JOIN genres_cte ON g.id = genres_cte.id
WHERE
    ((similarity_title_sp = 1 AND similarity_artist_sp = 1) OR manual_match_sp = 'True')
    AND ((similarity_title_fm = 1 AND similarity_artist_fm = 1) OR manual_match_fm = 'True')
GROUP BY
    tg.id_lastfm
HAVING genres != 'outros' 
"""

# Execute query and fetch data into DataFrame
df_songs_fm = pd.read_sql_query(query, sqlite_connection)
logger.success("Executed SQL query and fetched data successfully.")

# Execute SQL query to fetch data
query = """
SELECT DISTINCT
    g.id AS id_genius,
    l.lyric,
    s.*,
    v.genre
FROM
    Tbl_Songs_Genius g
INNER JOIN Tbl_Lyric l ON g.id = l.id_genius
INNER JOIN Tbl_Songs_Spotify s ON g.id_spotify = s.id
INNER JOIN Tbl_Songs_Lima v ON g.id = v.id_genius
WHERE
    ((similarity_title_sp = 1 AND similarity_artist_sp = 1) OR manual_match_sp = 'True')
"""

# Execute query and fetch data into DataFrame
df_songs_vg= pd.read_sql_query(query, sqlite_connection)
logger.success("Executed SQL query and fetched data successfully.")

df_songs_vg = df_songs_vg.rename({"genre": "genres"}, axis = 1)
df_songs_vg.genres = df_songs_vg.genres.str.replace("funk-carioca", "funk").str.replace("bossa-nova", "bossanova")

df_vg_filter = df_songs_vg[~df_songs_vg['id_genius'].isin(df_songs_fm['id_genius'])]

df_songs = pd.concat([df_songs_fm, df_vg_filter], ignore_index=True)
# df_songs = df_songs_fm.copy()

# Drop unnecessary columns from df_songs DataFrame
drop_columns = ['lyric', 'title', 'artist', 'isrc', 'tags', 'release_date']
df_songs = df_songs.drop(drop_columns, axis=1)

# Merge df_songs with df_merged on 'id_genius'
df_final = df_songs.merge(df_merged, left_on="id_genius", right_on="id")

# Drop duplicate 'id' column and rename 'id_x' to 'id_spotify'
df_final = df_final.drop('id_y', axis=1)
df_final = df_final.rename(columns={"id_x": "id_spotify"})

# Split 'genres' column into multiple binary columns
df_final['genres'] = df_final['genres'].str.split(',')
tags_dummies = df_final['genres'].apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0).infer_objects()

# Concatenate binary genre columns to df_final DataFrame
df = pd.concat([df_final, tags_dummies], axis=1)

# Log successful completion of the process
logger.success("Data processing completed successfully.")

# Close SQLite connection
sqlite_connection.close()
logger.success("Connection to the database closed.")


# Save the DataFrame to a new CSV file
output_file = "data_analysis.csv"
df.to_csv(output_file, index=False, quoting=1, encoding="utf-8")
logger.success(f"Saved DataFrame for analysis to {output_file}.")