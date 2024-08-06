#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-05
Description:
    This script updates song databases (Spotify and Last.fm) with manually verified matches.
    It performs the following tasks:
    1. Connects to the SQLite database.
    2. Creates a new column 'manualMatch' in both Spotify and Last.fm tables if it does not already exist.
    3. Reads the manual analysis CSV file containing manually verified perfect matches.
    4. Filters the data to identify new perfect matches and updates the corresponding tables in the database.
    5. Logs the number of manually marked perfect matches for both Spotify and Last.fm databases.
    6. Closes the database connection.


Usage:
    1. Ensure all dependencies are installed and accessible.
    2. Place the manual analysis CSV file ('df_manual_analysis.csv') created in script 06 in the same directory as this script.
    3. Run the script: python 07_UpdateManualMatches.py

Note:
    - The script uses logging to track progress and errors.
    - Temporary tables are created and dropped during execution.
    - The code only updates the table with the csv file generated in script 06, 
        if no manual changes were made to the file, no updates will be made.

"""


# Standard library imports
import sqlite3

# Third-party library imports
import pandas as pd
from loguru import logger

# Local application/library specific imports
from vars import table_spotify, table_lastfm, file_db


# Function to create a new column if it doesn't exist
def createColumn(sqlite_connection, cursor, table_name):
    
    # Get list of columns in the table
    list_columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
    list_columns = [column[1] for column in list_columns]

    # Add 'manualMatch' column if it doesn't exist
    if "manualMatch" not in list_columns:
        cursor.execute(f'ALTER TABLE {table_name} ADD COLUMN manualMatch "TEXT"')
        sqlite_connection.commit()
        logger.success(f"New column 'manualMatch' created in table {table_name} successfully.") 
    else:
        logger.info(f"Column 'manualMatch' not created in table {table_name} because it already exists.") 

# Function to update the 'manualMatch' column based on a DataFrame
def updateColumn(table_name, df, sqlite_connection):
    # Drop temporary table if it exists
    sqlite_connection.execute('DROP TABLE IF EXISTS temporary_table')  

    # Create temporary table containing the DataFrame data
    df.to_sql('temporary_table', sqlite_connection) 
    
    # Update the original table by setting 'manualMatch' to "True" where idGenius matches
    sqlite_connection.execute(f"""
        UPDATE {table_name} 
        SET manualMatch = "True"
        FROM temporary_table AS temp 
        WHERE {table_name}.idGenius = temp.idGenius;
        """)   
       
    # Drop the temporary table
    sqlite_connection.execute('DROP TABLE temporary_table')  
    sqlite_connection.commit()

def main():

    # Establish a connection to the SQLite database
    sqlite_connection = sqlite3.connect(file_db)
    encoding = "utf8"
    sqlite_connection.text_factory = lambda x: str(x, encoding)
    cursor = sqlite_connection.cursor()
    logger.success(f"Connection to the Database {file_db} initiated successfully.") 

    # Create 'manualMatch' column in both tables if it doesn't exist
    createColumn(sqlite_connection, cursor, table_spotify)
    createColumn(sqlite_connection, cursor, table_lastfm)

    # Attempt to read the manual analysis CSV file
    try:
        df_manual_analysis = pd.read_csv("df_manual_analysis.csv", sep=";")
        logger.success(f"Dataframe 'df_manual_analysis.csv' read successfully.") 
    except:
        logger.error(f"Dataframe 'df_manual_analysis.csv' not found in directory.")
        sqlite_connection.close()
        exit()

    # Filter DataFrame for new perfect matches
    df_new_matchs = df_manual_analysis[(df_manual_analysis.perfectMatchSP == True) & (df_manual_analysis.perfectMatchFM == True)]
    df_new_matchs_sp = df_manual_analysis[df_manual_analysis.perfectMatchSP == True].reset_index(drop=True)
    df_new_matchs_fm = df_manual_analysis[df_manual_analysis.perfectMatchFM == True].reset_index(drop=True)

    # Update the Spotify table with new perfect matches
    if len(df_new_matchs_sp) > 0:
        updateColumn(table_spotify, df_new_matchs_sp, sqlite_connection)
        logger.success(f"Table {table_spotify} updated successfully.") 
    else:
        logger.info(f"There are no new rows to update the table {table_spotify}.") 

    # Update the LastFM table with new perfect matches
    if len(df_new_matchs_fm) > 0:
        updateColumn(table_lastfm, df_new_matchs_fm, sqlite_connection)
        logger.success(f"Table {table_lastfm} updated successfully.") 
    else:
        logger.info(f"There are no new rows to update the table {table_lastfm}.") 

    # Get the number of songs manually marked as a perfect match in Spotify
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_spotify} WHERE manualMatch == 'True'")
    num_sp = query_response.fetchone()[0]

    # Get the number of songs manually marked as a perfect match in LastFM
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_lastfm} WHERE manualMatch == 'True'")
    num_fm = query_response.fetchone()[0]

    # Log the results
    space = 45 * " "
    logger.info(f"""Manual Matching: 
                    {space} {num_sp} songs manually marked as a perfect match in the Spotify database.
                    {space} {num_fm} songs manually marked as a perfect match in the LastFm database.""") 
        
    # Close the SQLite connection
    sqlite_connection.close()
    logger.success(f"Connection to Database {file_db} closed.")

if __name__ == "__main__":
    
    # Start the application
    main()

    # Exit the application
    exit()