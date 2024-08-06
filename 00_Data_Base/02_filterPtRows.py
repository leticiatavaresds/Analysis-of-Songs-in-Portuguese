#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-05
Description:
    This script connects to the SQLite database and performs data cleanup on the Genius table. Specifically, it:
    1. Connects to the SQLite database.
    2. Deletes rows from the Genius table where the 'language' column is not equal to 'pt'.
    3. Logs the number of rows deleted and the number of rows remaining in the table.
    4. Commits the changes to the database.
    5. Closes the database connection.
    
Usage:
    1. Ensure all dependencies are installed and accessible.
    2. Ensure that the database exists (created by script 01).
    3. Run the script: python 02_filterPtRows.py

"""

# Standard library imports
import sqlite3

# Third-party library imports
from loguru import logger

# Local application/library specific imports
from vars import file_db, table_genius


def main():

    print("\n")

    # Initiate connection to the database of the file "geniusSongsLyrics.db"
    sqlite_connection = sqlite3.connect(file_db)
    cursor = sqlite_connection.cursor()
    logger.success(f"Connection to Database {file_db} started successfully.")     
    
    logger.debug(f"Deleting rows from table {table_genius} whose 'language' column is different from 'pt'...")
    # Delete rows from table whose 'language' column is different from 'pt'
    cursor.execute(f""" DELETE FROM {table_genius} 
                        WHERE language != 'pt'""")
    
    # Get number of deleted rows
    result = cursor.rowcount
    logger.success(f"{result} lines were successfully deleted.")

    # Get number of rows from the table after removal
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_genius}")
    num_rows = query_response.fetchone()[0]

    # Commit the changes in the database
    sqlite_connection.commit()


    logger.info(f"The table {table_genius} now has {num_rows} rows.")

    # Close the connection
    sqlite_connection.close()

    logger.success(f"Connection to Database {file_db} closed.")

if __name__ == "__main__":

    # Start the application
    main()

    # Exit the application
    exit()