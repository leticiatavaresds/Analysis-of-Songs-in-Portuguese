#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import joblib
import re
import spacy
from loguru import logger
from tqdm.auto import tqdm
import ast

# Initialize tqdm integration with pandas
tqdm.pandas()

folder_data = 'Data_Output'
folder_input = 'Data_Input'

# Load the Portuguese language model from spaCy
nlp = spacy.load('pt_core_news_sm')

# Load the POS tagger using joblib
pos_tagger = joblib.load(f'{folder_input}/POS_tagger_brill.pkl')
logger.info("Loaded POS tagger from {folder_input}/POS_tagger_brill.pkl.")

# Read the CSV file and convert the 'tokens' column to lists
df = pd.read_csv(f"{folder_data}/01_tokenization.csv")
df["tokens"] = df["tokens"].apply(ast.literal_eval)
logger.info(f"Loaded 01_tokenization.csv from {folder_data}.")

# Dictionary of past tense endings for different persons
past_tense_dict = {
    '1': ['amos', 'ara', 'ava', 'ei', 'emos', 'era', 'i', 'ia', 'imos', 'ira', 'áram', 
          'ávamos', 'êram', 'íamos', 'íram', 'iz','uei','oei'], 
    '2': ['aras', 'aste', 'astes', 'avas', 'eras', 'este', 'estes', 'ias', 
          'iras', 'iste', 'istes', 'árei', 'áveis', 'êrei', 'íeis', 'írei'], 
    '3': ['ara', 'aram', 'ava', 'avam', 'era', 'eram', 'eu', 'ia', 'iam', 'ira', 
          'iram', 'iu', 'ou', 'i', 'oi','ez','aiu']
}
not_past = {"sei"}

# Compile regular expressions for past tense endings
past_tense_patterns = {
    person: [re.compile(rf'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{ending}$', re.IGNORECASE) 
             for ending in endings]
    for person, endings in past_tense_dict.items()
}

def is_past_tense_verb(token):
    if token.text in not_past:
        return False
    
    try:
        person = token.morph.get('Person')
        tense = token.morph.get('Tense')
        
        if tense:
            if tense[0] in ["Past", "Pqp"]:
                return True
            elif tense[0] == "Pres":
                return False

        if person:
            person = person[0]
            for pattern in past_tense_patterns[person]:
                if pattern.search(token.text):
                    return True

        return False

    except:
        return False

def calculate_past_tense_ratio(tokens):
    verb_count = 0
    past_tense_verb_count = 0

    pos_tagged_tokens = pos_tagger.tag(tokens)
    spacy_tokens = nlp(" ".join(tokens))

    for token, tag in zip(spacy_tokens, pos_tagged_tokens):
        if tag[1].startswith('V'):  # Assuming verbs are tagged with 'V'
            verb_count += 1
            is_past_tense = is_past_tense_verb(token)
            if is_past_tense:
                past_tense_verb_count += 1

    if verb_count == 0:
        return 0.0
    else:
        return past_tense_verb_count / verb_count

def apply_evaluate(row):
    tokens = row['tokens']
    past_tense_ratio = calculate_past_tense_ratio(tokens)
    return {"past_tense_ratio": past_tense_ratio}

logger.debug("Calculating Past Tense Ratio...")

# Apply the evaluation function to each row with tqdm progress bar
result = df.progress_apply(apply_evaluate, axis=1)
columns = ["id"] + list(result[0].keys())

# Convert the result to a DataFrame and concatenate it with original DataFrame
df_result = pd.DataFrame(result.tolist())
df = pd.concat([df, df_result], axis=1)
df = df[columns]

# Log a success message indicating the completion of the process
logger.success("Past Tense Ratio Calculation completed successfully.")

# Save the evaluated DataFrame to CSV
output_file = f"{folder_data}/10_SY_PastTense.csv"
df.to_csv(output_file, index=False)
logger.success(f"Saved past tense ratio to {output_file}.")
