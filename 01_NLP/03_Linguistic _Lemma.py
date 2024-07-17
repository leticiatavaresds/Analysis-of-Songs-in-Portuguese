#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import ast
import spacy
from loguru import logger
from tqdm.auto import tqdm

# Initialize tqdm integration with pandas
tqdm.pandas()

# Define the name of the model
model_name = "pt_core_news_sm"

# Check if the model is already installed
if not spacy.util.is_package(model_name):
    logger.info(f"{model_name} not found. Downloading...")
    spacy.cli.download(model_name)
    logger.info("Downloaded Portuguese language model for spaCy.")

# Load the model
nlp = spacy.load(model_name)

# Define the data output folder and load the cleaned text data
folder_data = 'Data_Output'
folder_input = 'Data_Input'

df = pd.read_csv(f"{folder_data}/01_tokenization.csv")
logger.info(f"Loaded cleaned text data from {folder_data}/01_tokenization.csv.")

# Load the spaCy model
nlp = spacy.load('pt_core_news_sm')
logger.info("Loaded spaCy Portuguese language model.")

# Select relevant columns from the DataFrame
df = df[['id', 'tokens']]
logger.info("Selected relevant columns from the DataFrame.")

# Convert string representations of lists into actual lists
df["tokens"] = df.apply(lambda x: ast.literal_eval(x["tokens"]), axis=1)
logger.info("Converted token strings into lists.")

df['clean_lyric'] = df['tokens'].apply(lambda x: ' '.join(x))

# Load slangs and slurs from text files
df_slangs = pd.read_csv(f"{folder_input}/girias.txt")
slangs = list(df_slangs.giria)
logger.info("Loaded slang words from girias.txt.")

df_slurs = pd.read_csv(f"{folder_input}/insultos.txt")
slurs = list(df_slurs["aidético"])
logger.info("Loaded slur words from insultos.txt.")


# Identify and calculate ratios for slang words
logger.info("Identifying slang words and calculating their ratios.")
df["slangs"] = df.progress_apply(lambda x: [t for t in x["tokens"] if t in slangs], axis=1)
df["slang_words_ratio"] = df.apply(lambda x: len(x["slangs"])/len(x["tokens"]), axis=1)

# Identify and calculate ratios for slur words
logger.info("Identifying slurs words and calculating their ratios.")
df["slurs"] = df.progress_apply(lambda x: [t for t in x["tokens"] if t in slurs], axis=1)
df["uncommon_words_ratio"] = df.apply(lambda x: len(x["slurs"])/len(x["tokens"]), axis=1)
df["unique_uncommon_words_ratio"] = df.apply(lambda x: len(set(x["slurs"]))/len(x["tokens"]), axis=1)

# Lemmatize the clean lyrics and calculate the lemma ratio
logger.info("Applying Lemmatization and Calculating Lemma Ratio")
df["lemmas"] = df.progress_apply(lambda x: [token.lemma_ for token in nlp(x["clean_lyric"])], axis=1)
df["lemma_ratio"] = df.apply(lambda x: len(set(x["lemmas"]))/len(x["tokens"]), axis=1)


df["lemmas"] = df.apply(lambda x: [token.lemma_ for token in nlp(x["clean_lyric"])], axis=1)
df["lemma_ratio"] = df.apply(lambda x: len(set(x["lemmas"]))/len(x["tokens"]), axis=1)
logger.info("Lemmatized clean lyrics and calculated lemma ratios.")

df = df.drop(["tokens"], axis = 1)


# Save the DataFrame to a new CSV file
output_file = f"{folder_data}/03_LI_Lemma.csv"
df.to_csv(output_file, index=False)
logger.success(f"Saved AFINN scores to {output_file}.")