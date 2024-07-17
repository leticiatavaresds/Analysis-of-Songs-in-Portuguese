#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import ast
from loguru import logger
from tqdm.auto import tqdm

# Initialize tqdm integration with pandas
tqdm.pandas()

folder_data = 'Data_Output'

# Load the CSV file into a DataFrame
df = pd.read_csv(f"{folder_data}/01_tokenization.csv")
logger.info("Loaded 01_tokenization.csv into a DataFrame.")

# Convert the string representation of lists into actual lists
df["tokens"] = df.apply(lambda x: ast.literal_eval(x["tokens"]), axis=1)
logger.info("Converted token strings to lists of tokens.")

class PronounsEvaluator:
    def __init__(self, tokens):
        self.tokens = tokens

    def evaluate(self):
        pronouns = Pronouns()
        total_tokens = len(self.tokens)

        # Initialize counters for different pronouns
        i = you = it = we = they = non_pronouns = 0

        # Count the occurrences of each pronoun
        for token in self.tokens:
            word = token.lower()
            if word in group_i:
                i += 1
            elif word in group_you:
                you += 1
            elif word in group_it:
                it += 1
            elif word in group_we:
                we += 1
            elif word in group_they:
                they += 1
            else:
                non_pronouns += 1

        # Calculate combined counts and ratios
        i_and_we = i + we
        others = you + it + they
        i_vs_you = self.get_i_vs_you(i, you)
        excentricity = self.get_excentricity(i_and_we, others)

        # Store the calculated ratios in the Pronouns object
        pronouns.i = i / total_tokens
        pronouns.you = you / total_tokens
        pronouns.it = it / total_tokens
        pronouns.we = we / total_tokens
        pronouns.they = they / total_tokens
        pronouns.i_vs_you = i_vs_you
        pronouns.excentricity = excentricity


        return pronouns

    def get_i_vs_you(self, i, you):
        if i > 2 * you:
            return 1.0
        elif you > 2 * i:
            return 0.0
        else:
            return 0.5

    def get_excentricity(self, i_and_we, others):
        if i_and_we > 2 * others:
            return 1.0
        elif others > 2 * i_and_we:
            return 0.0
        else:
            return 0.5

class Pronouns:
    def __init__(self):
        self.lyric_id = 0
        self.i = 0
        self.you = 0
        self.it = 0
        self.we = 0
        self.they = 0
        self.i_vs_you = 0
        self.excentricity = 0

# Define pronoun groups
group_i = ["eu", "me", "meu", "minha", "meus", "minhas", "mim", "comigo"]
group_you = ["você", "teu", "tua", "teus", "tuas", "seu", "sua", "seus", "suas", "te"]
group_it = ["ele", "ela", "dele", "dela", "deles", "delas", "si", "consigo",
            "isso", "isto", "esse", "essa", "este", "esta", "disso", "desse", "dessa", "desta", "deste", "disto"]
group_we = ["nós", "a gente", "nossa", "nosso", "nossos", "nossas", "conosco"]
group_they = ["eles", "elas", "deles", "delas"]

def evaluatePronouns(row):
    evaluator = PronounsEvaluator(row["tokens"])
    pronouns = evaluator.evaluate()

    dict_pronouns = {"i": pronouns.i,
    "you": pronouns.you,
    "it": pronouns.it,
    "we":  pronouns.we,
    "they": pronouns.they,
    "i_vs_you":pronouns.i_vs_you,
    "excentricity": pronouns.excentricity

    }

    return dict_pronouns


logger.debug("Evaluating pronoun ratios for the given tokens.")
# Apply the evaluation function to each row with tqdm progress bar
result = df.progress_apply(evaluatePronouns, axis=1)
columns = ["id"] + list(result[0].keys())

# Convert the result to a DataFrame and concatenate it with original DataFrame
df_result = pd.DataFrame(result.tolist())
df = pd.concat([df, df_result], axis=1)
df = df[columns]

# Log a success message indicating the completion of the process
logger.success("Pronoun Ratios Evaluation process completed successfully.")

# Save the evaluated DataFrame to CSV
output_file = f"{folder_data}/06_Sy_Pronouns.csv"
df.to_csv(output_file, index=False)
logger.success(f"Saved evaluated Repetitive Structures to {output_file}.")
