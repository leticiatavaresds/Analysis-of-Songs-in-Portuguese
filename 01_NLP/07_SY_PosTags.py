#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from collections import defaultdict
import joblib
from loguru import logger
from tqdm.auto import tqdm
from nltk import word_tokenize
from typing import List

# Initialize tqdm integration with pandas
tqdm.pandas()

folder_data = 'Data_Output'
folder_input = 'Data_Input'

# Load the POS tagger model
tagger = joblib.load(f"{folder_input}/POS_tagger_brill.pkl")
logger.info("Loaded POS tagger model from POS_tagger_brill.pkl.")

df = pd.read_csv(f"{folder_data}/00_clean_lyric.csv")
len_df = len(df)

logger.info(f"Songs successfully obtained, there are {len_df} songs to analyze.") 

class SuperPOSTagsEvaluator:
    def __init__(self, lyric):
        self.lyric_tokens = lyric.get_tokens()
        self.lyric = lyric.text

    def evaluate(self):
        pos_tags = SuperPOSTags()
        total_tokens = len(self.lyric_tokens)

        wh_pronouns = ["quem", "qual", "onde", "quando", "como", "aonde"]
        wh_pronous_2 = ["por que", "por quê", "o que", "o quê"]

        
        special_characters_tags = ['$', '.', ',', ':', "'",'"',"$","@","!","?",";","(",")","-"]

        tag_counts = defaultdict(int)
        verbs = participles = nouns = adjectives = adverbs = pdens = pronouns = conjus = interjs = preps = foreigns = wh_questions = spChars = 0

        # Tag the tokens
        tagged_tokens = tagger.tag(self.lyric_tokens)

        # Count occurrences of each POS tag
        for word, tag in tagged_tokens:
            tag = tag.upper()
            if tag in {"V", "VAUX"}:
                verbs += 1
            elif tag in {"PCP"}:
                participles += 1
            elif tag in {"N", "NPROP", "N|EST"}:
                nouns += 1
            elif tag in {"ADJ"}:
                adjectives += 1
            elif tag in {"ADV", "ADV-KS", "ADV-KS-REL"}:
                adverbs += 1
            elif tag in {"PDEN"}:
                pdens += 1
            elif tag in {"PROADJ", "PRO-KS", "PRO-KS-REL", "PROSUB"}:
                pronouns += 1
            elif tag in {"KC", "KS"}:
                conjus += 1
            elif tag in {"IN"}:
                interjs += 1
            elif tag in {"PREP"}:
                preps += 1
            if "|EST" in tag:
                foreigns += 1

            if tag in special_characters_tags:
                spChars += 1

            if word.lower() in wh_pronouns:
                wh_questions += 1

            tag_counts[tag] += 1

        for pronoun in wh_pronous_2:
            wh_questions += self.lyric.lower().count(pronoun)

        # Calculate ratios
        pos_tags.verbs = verbs / total_tokens
        pos_tags.participles = participles / total_tokens
        pos_tags.nouns = nouns / total_tokens
        pos_tags.adjectives = adjectives / total_tokens
        pos_tags.adverbs = adverbs / total_tokens
        pos_tags.denotatives_particle = pdens / total_tokens
        pos_tags.pronouns = pronouns / total_tokens
        pos_tags.conjunctions = conjus / total_tokens
        pos_tags.interjectios = interjs / total_tokens
        pos_tags.prepositions = preps / total_tokens
        pos_tags.foreignisms = foreigns / total_tokens
        pos_tags.spChars = spChars / total_tokens
        pos_tags.wh_questions = wh_questions / total_tokens



        # Store tag counts and ratios
        for tag, count in tag_counts.items():
            pos_tag_C = {"tag": tag,
                     "amount": count,
                     "ratio": count / total_tokens}
            pos_tags.tags.append(pos_tag_C)

        return pos_tags

class SuperPOSTags:
    def __init__(self):
        self.lyric_id = 0
        self.verbs = 0
        self.participles = 0
        self.nouns = 0
        self.adjectives = 0
        self.adverbs = 0
        self.denotatives_particle = 0
        self.pronouns = 0
        self.conjunctions = 0
        self.interjectios = 0
        self.prepositions = 0
        self.foreignisms = 0
        self.tags = []
        self.wh_questions = 0
        self.spChars = 0

class POSTag:
    def __init__(self):
        self.tag = 0
        self.amount = 0
        self.ratio = 0

class PreparedLyric:
    def __init__(self, text: str):

        self.text = text
        self.tokens = word_tokenize(text, language='portuguese')

    def get_tokens(self) -> List[str]:
        return self.tokens



def apply_evaluate(row):
    text = row['clean_lyric']

    prepared_lyric = PreparedLyric(text)
    
    evaluator = SuperPOSTagsEvaluator(prepared_lyric)
    pos_tags = evaluator.evaluate()
    

    # Prepare the result dictionary
    result = {
        'verbs': pos_tags.verbs,
        'participles': pos_tags.participles,
        'nouns': pos_tags.nouns,
        'adjectives': pos_tags.adjectives,
        'adverbs': pos_tags.adverbs,
        'denotatives_particle': pos_tags.denotatives_particle,
        'pronouns': pos_tags.pronouns,
        'conjunctions': pos_tags.conjunctions,
        'interjectios': pos_tags.interjectios,
        'prepositions': pos_tags.prepositions,
        'foreignisms': pos_tags.foreignisms,
        'wh_questions': pos_tags.wh_questions,
        'special_characters': pos_tags.spChars,
        "tags_pos": pos_tags.tags
    }
    
    return result


# Log an initial message indicating the start of the process
logger.debug("Starting Pos Tagging evaluation process...")

# Apply the evaluation function to each row with tqdm progress bar
result = df.progress_apply(apply_evaluate, axis=1)
columns = ["id"] + list(result[0].keys())

# Convert the result to a DataFrame and concatenate it with original DataFrame
df_result = pd.DataFrame(result.tolist())
df = pd.concat([df, df_result], axis=1)
df = df[columns]

# Log a success message indicating the completion of the process
logger.success("Pos Tagging Evaluation process completed successfully.")


# Save the evaluated DataFrame to CSV
output_file = f"{folder_data}/07_SY_PosTags.csv"
df.to_csv(output_file, index=False)
logger.success(f"Saved evaluated Repetitive Structures to {output_file}.")
