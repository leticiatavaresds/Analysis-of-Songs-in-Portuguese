import pandas as pd
import spacy
from collections import defaultdict
from typing import List
from tqdm.auto import tqdm
from textdistance import damerau_levenshtein
from loguru import logger
import math

# Define the database file path
folder_input = 'Data_Input'
folder_data = 'Data_Output'

# Define the name of the model
model_name = "pt_core_news_sm"

# Check if the model is already installed
if not spacy.util.is_package(model_name):
    logger.info(f"{model_name} not found. Downloading...")
    spacy.cli.download(model_name)
    logger.info("Downloaded Portuguese language model for spaCy.")

# Load the model
nlp = spacy.load(model_name)

# Load dictionary and data
with open(f'{folder_input}/palavras.txt', 'r', encoding='utf-8') as file:
    dict_pt = [w.strip() for w in file.readlines()]

df = pd.read_csv(f"{folder_data}/00_clean_lyric.csv")
logger.info(f"Songs successfully obtained, there are {len(df)} songs to analyze.") 

# Clean lyrics
df["clean_lyric"] = df["clean_lyric"].str.replace(r'[^ \nA-Za-zÀ-ÖØ-öø-ÿЀ-ӿ-]+', '', regex=True)
df["clean_lyric"] = df["clean_lyric"].str.replace("-", " ")
df["clean_lyric"] = df["clean_lyric"].str.replace("\n\n", "\n")

# Token class to store information about tokens
class Token:
    def __init__(self, token):
        self.token = nlp(token)[0]
        self.text = self.token.text
        self.lemma_ = self.token.lemma_
        self.tag_ = self.token.pos_

# Line class to represent a line of text with tokens
class Line:
    def __init__(self):
        self.tokens = []
        self.text = ""

    def add_tokens(self, tokens: List[str]):
        self.tokens.extend(tokens)

    def add_token(self, token: str):
        self.tokens.append(token)

    def get_tokens(self) -> List[str]:
        return self.tokens

    def get_text(self) -> str:
        return self.text

    def set_text(self, text: str):
        self.text = text

    def __str__(self):
        return f"Line{{text='{self.text}'}}"

    def remove_last(self):
        self.tokens.pop()
        self.text = self.text[:-1]

    def insert(self):
        self.tokens.insert(0, Token('"'))
        self.text = '"' + self.text

# Tokenizer class to tokenize the text
class Tokenizer:
    def __init__(self):
        self.nlp = nlp

    def tokenize(self, text: str) -> List[Line]:
        text = text.replace(",", "").replace('"', "")
        lines_text = [line for line in text.split("\n") if line != '']
        doc = [self.nlp(line) for line in lines_text]
        lines = []

        add_a = False
        for sent in doc:
            line = Line()
            line.set_text(sent.text.strip())
            line.add_tokens([Token(token.text.lower()) for token in sent if token.text != "\n"])
            if add_a:
                line.insert()
                add_a = False
            if len(sent.text.strip()) and sent.text.strip()[-1] == '"':
                line.remove_last()
                add_a = True
            lines.append(line)
        return lines

    @staticmethod
    def get_instance():
        return Tokenizer()

# PreparedLyric class to prepare and process song lyrics
class PreparedLyric:
    def __init__(self, lyric):
        self.tokenizer = Tokenizer.get_instance()
        self.lyric = lyric
        self.tokens = None
        self.lines = []

    def get_tokens(self) -> List[str]:
        if self.tokens is None:
            self.tokens = []
            for line in self.get_lines():
                self.tokens.extend(line.get_tokens())
        return self.tokens

    def get_lines(self) -> List[Line]:
        if not self.lines:
            self.lines = self.tokenizer.tokenize(self.lyric)
        return self.lines

    def get_lines_verse(self, verse_text) -> List[Line]:
        return self.tokenizer.tokenize(verse_text)

    def get_text(self) -> str:
        return self.lyric

class LineEchoism:
    def __init__(self):
        self.musical_words = 0
        self.reduplication = 0
        self.rhyme_alike = 0

class EchoismEvaluator:
    def __init__(self, lyric):
        self.lyric = lyric
        self.tokens = lyric.get_tokens()
        self.line_echoism = {
            'length1': LineEchoism(),
            'length2': LineEchoism(),
            'min_length3': LineEchoism()
        }
        self.word_echoism = {
            'musical_words': 0
        }
    
    def evaluate(self):
        echoism = defaultdict(float)
        self.compute_musical_words()
        self.compute_multi_word_echoisms()
        total_tokens = len(self.tokens)
        echoism['word_echoism_musical_words_ratio'] = self.word_echoism['musical_words'] / total_tokens 
        echoism['multi_echoism_musical_words_ratio_length1'] = self.line_echoism['length1'].musical_words / total_tokens
        echoism['multi_echoism_reduplication_ratio_length1'] = self.line_echoism['length1'].reduplication / total_tokens
        echoism['multi_echoism_rhyme_alike_ratio_length1'] = self.line_echoism['length1'].rhyme_alike / total_tokens
        echoism['multi_echoism_musical_words_ratio_length2'] = self.line_echoism['length2'].musical_words / total_tokens
        echoism['multi_echoism_reduplication_ratio_length2'] = self.line_echoism['length2'].reduplication / total_tokens
        echoism['multi_echoism_rhyme_alike_ratio_length2'] = self.line_echoism['length2'].rhyme_alike / total_tokens
        echoism['multi_echoism_musical_words_min_length_3'] = self.line_echoism['min_length3'].musical_words / total_tokens
        echoism['multi_echoism_reduplication_min_length_3'] = self.line_echoism['min_length3'].reduplication / total_tokens
        echoism['multi_echoism_rhyme_alike_min_length_3'] = self.line_echoism['min_length3'].rhyme_alike / total_tokens
        return dict(echoism)
    
    def edit(self, token1, token2):
        lv_sim = damerau_levenshtein.distance(token1, token2)
        distance = (1 / math.sqrt(len(token1) * len(token2))) * lv_sim
        return distance

    def compute_musical_words(self):
        tokens = self.lyric.get_tokens()
        for word in tokens:
            if self.is_musical_word(word.text):
                self.word_echoism['musical_words'] += 1

    def is_musical_word(self, word):
        unique_chars = set(word)
        letter_innovation = len(unique_chars) / len(word)
        if letter_innovation < 0.4 or (letter_innovation < 0.5 and word not in dict_pt):
            return True
        return False

    def compute_multi_word_echoisms(self):
        lines = self.lyric.get_lines()
        for line in lines:
            tokens = line.get_tokens()
            if len(tokens) > 1:
                tokens_list = []
                for i in range(len(tokens) - 1):
                    if tokens[i].text == '"':
                        continue
                    distance = self.edit(tokens[i].text, tokens[i + 1].text)
                    if distance < 0.5:
                        if not tokens_list:
                            tokens_list.append(tokens[i])
                        tokens_list.append(tokens[i + 1])
                    last_element = (i + 1) >= (len(tokens) - 1)
                    if distance >= 0.5 or last_element:
                        if tokens_list:
                            length = len(tokens_list)
                            echoism = self.getLineEchoism(length - 1)
                            self.classifyEchoism(echoism, tokens_list)
                            tokens_list.clear()

    def getLineEchoism(self, length):
        if length <= 1:
            return 'length1'
        elif length <= 2:
            return 'length2'
        else:
            return 'min_length3'
    
    def classifyEchoism(self, echoism, tokens):
        lemmas = [token.lemma_.lower() for token in tokens]
        lemmas = list(set(lemmas))
        echoism = self.line_echoism[echoism]
        if len(lemmas) == 1:
            for lemma in lemmas:
                if lemma and (lemma in dict_pt):
                    echoism.reduplication += 1
                else:                                
                    echoism.musical_words += 1      
        elif len(lemmas) > 1:
            all_in_wiktionary = True
            nothing_in_wiktionary = True
            for lemma in lemmas:
                if lemma and (lemma in dict_pt):
                    nothing_in_wiktionary = False
                else:
                    all_in_wiktionary = False
            if all_in_wiktionary:
                echoism.rhyme_alike += 1
            elif nothing_in_wiktionary:
                echoism.musical_words += 1

def apply_evaluate(row):
    text = row['clean_lyric']
    prepared = PreparedLyric(text)
    evaluator = EchoismEvaluator(prepared)
    result = evaluator.evaluate()
    return result


# Initialize tqdm integration with pandas
tqdm.pandas()

# Log an initial message indicating the start of the process
logger.debug("Starting Repetitive Structures evaluation process...")

# Apply the evaluation function to each row with tqdm progress bar
result = df.progress_apply(apply_evaluate, axis=1)
columns = ["id"] + list(result[0].keys()) 

# Convert the result to a DataFrame and concatenate it with original DataFrame
df_result = pd.DataFrame(result.tolist())
df = pd.concat([df, df_result], axis=1)
df = df[columns]

# Log a success message indicating the completion of the process
logger.success("Text Chunks Evaluation process completed successfully.")

# Save the evaluated DataFrame to CSV
output_csv_path = f"{folder_data}/05_LI_Echoism.csv"
df.to_csv(output_csv_path, index=False)
logger.success(f"Saved evaluated Regressive Imagery to {output_csv_path}.")
