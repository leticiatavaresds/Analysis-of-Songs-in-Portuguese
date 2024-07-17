#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm
import sys
import os
from tqdm.auto import tqdm

# Adicionar o diretório "utils" ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'Data_Input', 'LeIA-master')))

import leia 
from leia import SentimentIntensityAnalyzer


# Define the data output folder and load the cleaned text data
folder_data = 'Data_Output'

# Initialize tqdm integration with pandas
tqdm.pandas()


# Initialize the Sentiment Intensity Analyzer
analyzer = SentimentIntensityAnalyzer()

logger.info("Sentiment Intensity Analyzer initialized.")


df = pd.read_csv(f"{folder_data}/00_clean_lyric.csv")
logger.info(f"Songs successfully obtained, there are {len(df)} songs to analyze.") 

def get_sentiment_scores(row):

    lyric = row["clean_lyric"]

    scores = analyzer.polarity_scores(lyric)
    return scores


logger.info("Running Vader model to calculate the Mood Scores...")

# Apply the evaluation function to each row with tqdm progress bar
result = df.progress_apply(get_sentiment_scores, axis=1)
columns = ["id"] + list(result[0].keys()) 

# Convert the result to a DataFrame and concatenate it with original DataFrame
df_result = pd.DataFrame(result.tolist())
df = pd.concat([df, df_result], axis=1)
df = df[columns]

# Log a success message indicating the completion of the process
logger.success(" Vader Model Evaluation process completed successfully.")

# Save the evaluated DataFrame to CSV
output_csv_path = f"{folder_data}/02_SE_Vader.csv"
df.to_csv(output_csv_path, index=False)
logger.success(f"Saved evaluated Regressive Imagery to {output_csv_path}.")