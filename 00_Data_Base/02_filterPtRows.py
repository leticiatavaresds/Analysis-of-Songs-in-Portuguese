#!/usr/bin/env python3.9
# Import libraries
from loguru import logger
import sqlite3

def main():

    print("\n")

    table_name = "tblGeniusSongsLyrics"
    file_db = "geniusSongsLyrics.db"

    # Initiate connection to the database of the file "geniusSongsLyrics.db"
    sqlite_connection = sqlite3.connect(file_db)
    cursor = sqlite_connection.cursor()
    logger.success(f"Connection to Database {file_db} started successfully.")     
    
    logger.debug(f"Deleting rows from table {table_name} whose 'language' column is different from 'pt'...")
    # Delete rows from table whose 'language' column is different from 'pt'
    cursor.execute(f""" DELETE FROM {table_name} 
                        WHERE language != 'pt'""")
    
    # Get number of deleted rows
    result = cursor.rowcount
    logger.success(f"{result} lines were successfully deleted.")

    # Get number of rows from the table after removal
    query_response = cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    num_rows = query_response.fetchone()[0]

    # Commit the changes in the database
    sqlite_connection.commit()


    logger.info(f"The table {table_name} now has {num_rows} rows.")

    # Close the connection
    sqlite_connection.close()

    logger.success(f"Connection to Database {file_db} closed.")

if __name__ == "__main__":

    # Start the application
    main()

    # Exit the application
    exit()