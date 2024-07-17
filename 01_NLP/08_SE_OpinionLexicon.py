#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm


# Initialize tqdm integration with pandas
tqdm.pandas()

folder_data = 'Data_Output'
folder_input = 'Data_Input'

# Read the CSV file containing lyrics data
df = pd.read_csv(f"{folder_data}/01_tokenization.csv")
logger.info(f"Loaded 01_tokenization.csv from {folder_data}.")

# Select only the 'id' and 'clean_lyric' columns from the DataFrame
df = df[['id', 'clean_lyric']]

# Path to the Sentilex-PT lexicon file
sentilex_pt_path = f'{folder_input}/SentiLex-flex-PT02.txt'

class Opinion:
    def __init__(self):
        self.lyric_id = None
        self.opinion = 0.0

class OpinionLexicon:
    def __init__(self, lexicon_file: str):
        self.lexicon = self.load_sentilex_pt(lexicon_file)

    def load_sentilex_pt(self, lexicon_file: str) -> dict:
        lexicon = {}
        with open(lexicon_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  # Ignore blank lines
                    parts = line.strip().split(';')
                    word_info = parts[0].split(',')
                    word = word_info[0]
                    for part in parts[1:]:
                        if part.startswith('POL:N0='):
                            score_info = part.split('=')
                            try:
                                score = float(score_info[-1])
                                lexicon[word] = score
                            except ValueError:
                                continue  # Ignore non-numeric values
        return lexicon

    def get(self, word: str) -> float:
        return self.lexicon.get(word, 0.0)

class OpinionEvaluator:
    def __init__(self, lexicon: OpinionLexicon):
        self.lexicon = lexicon
    
    def evaluate(self, text: str) -> float:
        lyric_tokens = text.lower().split()
        opinion_value = sum(self.lexicon.get(token) for token in lyric_tokens)
        sentiment_words = sum(1 for token in lyric_tokens if self.lexicon.get(token) != 0)
        opinion_score = opinion_value / sentiment_words if sentiment_words > 0 else 0.0
        return opinion_score

# Create the opinion lexicon using the SentiLex file
lexicon = OpinionLexicon(sentilex_pt_path)
logger.info(f"Loaded opinion lexicon from {sentilex_pt_path}.")

# Create the opinion evaluator
evaluator = OpinionEvaluator(lexicon)

# Apply opinion evaluation to the 'clean_lyric' column of the DataFrame
# Log an initial message indicating the start of the process
logger.debug("Starting Opinion Lexicon evaluation process...")

df['opinion_score'] = df['clean_lyric'].progress_apply(lambda x: evaluator.evaluate(x))
logger.info("Applied opinion evaluation to 'clean_lyric' column.")

# Log a success message indicating the completion of the process
logger.success("Opinion Lexicon Evaluation process completed successfully.")

# Select columns for final output
df = df[["id", "opinion_score"]]

# Save the DataFrame to a new CSV file
output_file = f"{folder_data}/08_SE_OpinionLexicon.csv"
df.to_csv(output_file, index=False)
logger.success(f"Saved evaluated opinions to {output_file}.")

