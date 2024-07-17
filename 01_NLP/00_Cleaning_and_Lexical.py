#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import sqlite3
from loguru import logger

# Define the database file path
folder_output = "Data_Output"
file_db = "..\Songs_Database.db"

# Initiate connection to the database
sqlite_connection = sqlite3.connect(file_db)
cursor = sqlite_connection.cursor()
logger.success(f"Connection to the Database {file_db} initiated successfully.")

# Execute SQL query to fetch distinct song data
# This SQL query selects distinct song IDs, lyrics, and song durations tables:
# - Tbl_Songs_Genius (aliased as 'g')
# - Tbl_Lyric (aliased as 'l')
# - Tbl_Songs_Spotify (aliased as 's')
# Where:
# - (similarity_title_sp = 1 AND similarity_artist_sp = 1) OR manual_match_sp = 'True' for Spotify
# - (similarity_title_fm = 1 AND similarity_artist_fm = 1)s OR manual_match_fm = 'True' for Last.fm
query = """
SELECT DISTINCT g.id, l.lyric, duration_ms
FROM Tbl_Songs_Genius g
INNER JOIN Tbl_Songs_Tags t ON g.id_lastfm = t.id_lastfm
INNER JOIN Tbl_Lyric l ON g.id = l.id_genius
INNER JOIN Tbl_Songs_Spotify s ON g.id_spotify = s.id
WHERE ((similarity_title_sp = 1 AND similarity_artist_sp = 1) OR manual_match_sp = 'True')
AND ((similarity_title_fm = 1 AND similarity_artist_fm = 1) OR manual_match_fm = 'True')
"""

df = pd.read_sql_query(query, sqlite_connection)
logger.success("SQL query executed and data fetched successfully.")

# Clean the lyrics by removing unnecessary characters and formatting issues
df["clean_lyric"] = df["lyric"].copy()
df["clean_lyric"] = df.apply(lambda x: re.sub('\[[^]]+\]', '', x["clean_lyric"]), axis=1) # Remove text within brackets
df["clean_lyric"] = df.apply(lambda x: re.sub(r'\n\s*\n', '\n\n', x["clean_lyric"]), axis=1) # Remove extra newline characters
df["clean_lyric"] = df.apply(lambda x: re.sub(r'^{re.escape("\n")}', '', x["clean_lyric"]), axis=1)
df["clean_lyric"] = df.apply(lambda x: re.sub(r'^{re.escape("\n")}', '', x["clean_lyric"]), axis=1)



logger.info("Lyrics cleaned and formatted.")

# Save DataFrame to CSV with appropriate quoting
df_clean_lyric = df[["id", "clean_lyric"]].copy()
df_clean_lyric.to_csv(f"{folder_output}/00_clean_lyric.csv", index=False, quoting=1, encoding="utf-8") # quoting=1 ensures that text with \n is properly quoted
logger.success(f"Cleaned lyrics saved to {folder_output}/00_clean_lyric.csv.")

# Count the number of strophes (verses) in the lyrics
df["count_strophes"] = df.apply(lambda x: x["clean_lyric"].count("\n\n") + 1, axis=1)
logger.info("Strophe count calculated.")

# Count the number of lines in the lyrics
df["line_count"] = df.apply(lambda x: x["clean_lyric"].count("\n") + 1, axis=1)
df["lines"] = df.apply(lambda x: x["clean_lyric"].split("\n"), axis=1)

# Identify unique lines and their count
df["unique_lines"] = df.apply(lambda x: list(set(x["lines"])), axis=1)
df["unique_lines"] = df.apply(lambda x: [t for t in x["unique_lines"] if t != ''], axis=1)

# Count blank lines and unique lines
df["blank_line_count"] = df.apply(lambda x: x["lines"].count(''), axis=1)
df["unique_line_count"] = df.apply(lambda x: len(x["unique_lines"]), axis=1)
df["blank_line_ratio"] = df["blank_line_count"]/df["line_count"]
df["repeat_line_ratio"] = (df["line_count"] - df["unique_line_count"])/df["line_count"]

# Drop unnecessary columns
df = df.drop(['lines', 'unique_lines'], axis=1)
logger.info("Line and strophe metrics calculated.")

# Further clean the lyrics and count words and characters
df["clean_lyric"] = df["clean_lyric"].str.replace("\n", " ")
df["count_words"] = df.apply(lambda x: len(re.findall(r'\w+', x["clean_lyric"])), axis=1)

df["count_char"] = df.apply(lambda x: len(x["clean_lyric"]), axis=1)
df["count_char_no_space"] = df.apply(lambda x: len(x["clean_lyric"].replace(" ", "")), axis=1)

# Count punctuation marks and other characters
df["exclamation_marks"] = df.apply(lambda x: x["clean_lyric"].count("!"), axis=1)
df["question_marks"]    = df.apply(lambda x: x["clean_lyric"].count("?"), axis=1)
df["colons"]            = df.apply(lambda x: x["clean_lyric"].count(":"), axis=1)
df["semicolons"]        = df.apply(lambda x: x["clean_lyric"].count(";"), axis=1)
df["commas"]            = df.apply(lambda x: x["clean_lyric"].count(","), axis=1)
df["dots"]              = df.apply(lambda x: x["clean_lyric"].count("."), axis=1)
df["hyphens"]           = df.apply(lambda x: x["clean_lyric"].count("-"), axis=1)
df["single_quotes"]     = df.apply(lambda x: x["clean_lyric"].count("'"), axis=1)
df["double_quotes"]     = df.apply(lambda x: x["clean_lyric"].count('"'), axis=1)
df["digits"]            = df['clean_lyric'].str.count(r'\d')
df["quotes"]            = df["single_quotes"] + df["double_quotes"]
df = df.drop(['single_quotes', 'double_quotes'], axis=1)
logger.info("Character and punctuation metrics calculated.")

# Remove non-alphabetic characters and convert to lowercase
df["clean_lyric"] = df.apply(lambda x: re.sub(r'[^ \nA-Za-zÀ-ÖØ-öø-ÿЀ-ӿ-]+', '', x["clean_lyric"]), axis=1)
df["clean_lyric"] = df["clean_lyric"].str.replace("-", " ")
df["clean_lyric"] = df["clean_lyric"].str.lower()
logger.info("Lyrics cleaned of non-alphabetic characters and converted to lowercase.")


df = df.drop(["duration_ms"], axis = 1)

# Save the cleaned data to a CSV file
df.to_csv(f"{folder_output}/00_nlp_cleanText.csv", index=False, quoting=1, encoding="utf-8")
logger.success("Cleaned data saved to 00_nlp_cleanText.csv.")

# Close the connection
sqlite_connection.close()
logger.success(f"Connection to Database {file_db} closed.")