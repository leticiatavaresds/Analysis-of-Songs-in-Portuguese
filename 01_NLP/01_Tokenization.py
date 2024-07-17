#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from loguru import logger

from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

from tqdm.auto import tqdm
import os

# Initialize tqdm integration with pandas
tqdm.pandas()

# Check if the directory where NLTK data is stored already exists
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# List of packages to be checked and downloaded if necessary
nltk_packages = [
    ('tokenizers/punkt', 'punkt'),
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet')
]

# Function to check and download packages
def download_nltk_package(package_path, package_name):
    try:
        nltk.data.find(package_path)
        logger.info(f"Package '{package_name}' already downloaded.")
    except LookupError:
        nltk.download(package_name, quiet=True)
        logger.success(f"Package '{package_name}' downloaded successfully.")

# Check and download packages as needed
for package_path, package_name in nltk_packages:
    download_nltk_package(package_path, package_name)

logger.info("Downloaded necessary NLTK data files.")

# Define the data output folder and load the cleaned text data
folder_data = 'Data_Output'
df = pd.read_csv(f"{folder_data}/00_nlp_cleanText.csv")

logger.info("Cleaned text data loaded successfully from 00_nlp_cleanText.csv.")

# Function to tokenize the lyrics column
def tokenization(df, column):
    df["tokens"] = df.apply(lambda x: word_tokenize(x[column]), axis=1)
    logger.info(f"Tokenized text in column '{column}'.")
    return df

# Define stopwords in Portuguese and add custom stopwords
stop_words = stopwords.words('portuguese')
additional_stopwords = ['já', 'viu', 'vai', 'né', 'aí', 'ai', 'tá', 'ta', 'gente', 'não', 'aqui', 
                        'também', 'vc', 'você', 'então', 'até', 'agora', 'ser', 'sempre', 'ter', 
                        'só', 'porque', 'sobre', 'ainda', 'lá', 'tudo', 'nada', 'ninguém', 'de', 
                        'pra', 'uns', 'tô']
stop_words.extend(additional_stopwords)

logger.info("Stopwords defined and additional stopwords added.")

# Function to generate bigrams from a list of words
def bigramas(words):
    bigrams = []
    for i in range(0, len(words) - 1):
        bigrams.append(words[i] + ' ' + words[i + 1])
    return bigrams

# Function to generate trigrams from a list of words
def trigramas(words):
    trigrams = []
    for i in range(0, len(words) - 2):
        trigrams.append(words[i] + ' ' + words[i + 1] + ' ' + words[i + 2])
    return trigrams

# Select necessary columns from the dataframe
df = df[['id', 'clean_lyric', 'line_count', 'blank_line_count', 'count_char']]
logger.info("Selected necessary columns from the dataframe.")

# Tokenize the clean lyrics
logger.info("Tokenizing the clean lyrics...")
df["tokens"] = df.progress_apply(lambda x: word_tokenize(x["clean_lyric"]), axis=1)


# Identify stopwords in the tokenized lyrics
df["stop_words"] = df.apply(lambda x: [t for t in x["tokens"] if t in stop_words], axis=1)
df["count_stop_words"] = df.apply(lambda x: len(x["stop_words"]), axis=1)
df["tokens_no_stopWords"] = df.apply(lambda x: [t for t in x["tokens"] if t not in stop_words], axis=1)
logger.info("Identified stopwords in the tokenized lyrics.")

logger.info("Generating bigrams and trigrams from the tokenized lyrics...")
# Generate bigrams and trigrams from the tokenized lyrics
df["bigrams"] = df.progress_apply(lambda x: bigramas(x["tokens"]), axis=1)
df["trigrams"] = df.progress_apply(lambda x: trigramas(x["tokens"]), axis=1)


# Calculate various metrics for the tokenized lyrics
df["token_count"] = df.apply(lambda x: len(x["tokens"]), axis=1)
df["unique_token_ratio"] = df.apply(lambda x: len(set(x["tokens"])) / len(x["tokens"]), axis=1)
df["unique_bigram_ratio"] = df.apply(lambda x: len(set(x["bigrams"])) / len(x["bigrams"]), axis=1)
df["unique_trigram_ratio"] = df.apply(lambda x: len(set(x["trigrams"])) / len(x["trigrams"]), axis=1)
df["repeat_word_ratio"] = df.apply(lambda x: (len(x["tokens"]) - len(set(x["tokens"]))) / len(x["tokens"]), axis=1)
df["avg_token_length"] = df.apply(lambda x: pd.Series(x["tokens"]).apply(len).mean(), axis=1)
logger.info("Calculated various metrics for the tokenized lyrics.")

logger.info("Calculating hapax legomenon, dis legomenon, and tris legomenon ratios...")
# Calculate hapax legomenon (words that appear only once), dis legomenon (words that appear twice), and tris legomenon (words that appear three times) ratios
df["hapax_legomenon_ratio"] = df.progress_apply(lambda x: len(set([t for t in x["tokens"] if x["tokens"].count(t) == 1])) / len(x["tokens"]), axis=1)
df["dis_legomenon_ratio"] = df.progress_apply(lambda x: len(set([t for t in x["tokens"] if x["tokens"].count(t) == 2])) / len(x["tokens"]), axis=1)
df["tris_legomenon_ratio"] = df.progress_apply(lambda x: len(set([t for t in x["tokens"] if x["tokens"].count(t) == 3])) / len(x["tokens"]), axis=1)


# Calculate unique tokens per line and average tokens per line
df["unique_tokens_per_line"] = df.apply(lambda x: len(set(x["tokens"])) / (x["line_count"] - x["blank_line_count"]), axis=1)
df["average_tokens_per_line"] = df.apply(lambda x: len(x["tokens"]) / (x["line_count"] - x["blank_line_count"]), axis=1)
logger.info("Calculated unique tokens per line and average tokens per line.")

# Calculate stopwords ratio and stopwords per line
df["stopwords_ratio"] = df.apply(lambda x: len(x["stop_words"]) / len(x["tokens"]), axis=1)
df["stopwords_per_line"] = df.apply(lambda x: len(x["stop_words"]) / (x["line_count"] - x["blank_line_count"]), axis=1)
logger.info("Calculated stopwords ratio and stopwords per line.")

# Drop unnecessary columns
df = df.drop(['line_count', 'blank_line_count', 'count_char', 'count_stop_words', 'tokens_no_stopWords', 'clean_lyric'], axis=1)
logger.info("Dropped unnecessary columns from the dataframe.")


# Save the DataFrame to a new CSV file
output_file = f"{folder_data}/01_tokenization.csv"
df.to_csv(output_file, index=False)
logger.success(f"Saved Variables to {output_file}.")