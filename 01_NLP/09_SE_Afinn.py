import json
import pandas as pd
import ast
from nltk import ngrams
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

folder_data = 'Data_Output'
folder_input = 'Data_Input'


afinn_file = f"{folder_input}/AFINN-pt.json"

df = pd.read_csv(f"{folder_data}/00_nlp_cleanText.csv")
# df["tokens"] = df.apply(lambda x: ast.literal_eval(x["tokens"]), axis=1)

# Função para corrigir expressões com um dicionário customizado
def correct_expressions_spacy(text, custom_dict):
    doc = nlp(text)
    corrected_tokens = []
    for token in doc:
        corrected_token = custom_dict.get(token.text.lower(), token.text)
        corrected_tokens.append(corrected_token)
    corrected_text = ' '.join(corrected_tokens)
    return corrected_text


def load_dict_girias(file_name):

    dict_file = {}
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:

                if "," in line:
                    parts = line.strip().split(',')

                    if len(parts) == 2:
                        dict_file[parts[0].lower()] = parts[1]

    except Exception as e:
        dict_file = {}  # Set an empty dictionary in case of error

    return dict_file


class AFINNDictionary:
    def __init__(self, afinn_file):
        self.dictionary = {}
        self.load_afinn(afinn_file)
        self.dictionary = self.normalize_keys(self.dictionary)
        self.valence = 0

    def load_afinn(self, afinn_file):
        try:
            with open(afinn_file, 'r', encoding='utf-8') as file:
                self.dictionary = json.load(file)
        except Exception as e:
            print(f"Error loading AFINN dictionary: {e}")

    def normalize_keys(self, input_dict):
        normalized_dict = {}
        
        for key, value in input_dict.items():
            key = key.replace("-", " ")
            lower_key = key.lower()
            
            if lower_key in normalized_dict:
                current_value = normalized_dict[lower_key]
                if abs(value) > abs(current_value):
                    normalized_dict[lower_key] = value
            else:
                normalized_dict[lower_key] = value
                
        return normalized_dict

    def get(self, word: str) -> int:
        return self.dictionary.get(word.lower().strip(), 0)

class AFINN:
    def __init__(self):
        self.lyric_id = None
        self.valence = 0.0
        self.all_ngrams = []
        self.used_ngrams = []
        self.used_indices = []
        self.score = 0.0

class AFINNEvaluator:
    def __init__(self, lyric, afinn_dict: AFINNDictionary):
        self.lyric = lyric
        self.afinn_dict = afinn_dict
        

    def evaluate(self) -> AFINN:

        afinn = AFINN()
        corrected_text = correct_expressions_spacy(self.lyric.lower(), dict_girias)
        text_tokens = corrected_text.split()
        all_ngrams = []
        used_ngrams = []
        
        for n in range(3, 0, -1):  # Trigramas, Bigramas, Unigramas
            n_grams = [' '.join(ngram) for ngram in ngrams(text_tokens, n)]
            all_ngrams.extend(n_grams)

        score = 0
        used_indices = set()
        sentiment_words = 0

        for ngram in all_ngrams:

            value = self.afinn_dict.get(ngram)
            if value != 0: 
                ngram_tokens = ngram.split()
                ngram_indices = [
                    index for index in range(len(text_tokens) - len(ngram_tokens) + 1)
                    if text_tokens[index:index + len(ngram_tokens)] == ngram_tokens
                ]
                
                for start_index in ngram_indices:
                    indices_range = set(range(start_index, start_index + len(ngram_tokens)))

                    if not indices_range & used_indices:  # Se não houver interseção
                        score += value
                        sentiment_words += 1
                        used_ngrams.append((ngram, value))
                        used_indices.update(indices_range)
                        break

        afinn.valence = score / sentiment_words if sentiment_words > 0 else 0.0
        afinn.all_ngrams = all_ngrams
        afinn.used_indices = used_indices
        afinn.score = score
        afinn.used_ngrams = used_ngrams

        return afinn
    
dict_girias = load_dict_girias(f"{folder_input}/girias.txt")

afinn_dict = AFINNDictionary(afinn_file)

# Log an initial message indicating the start of the process
logger.debug("Starting AFFIN Evaluation process...")

# Apply sentiment evaluation to the 'tokens' column of the dataframe
df['afinn_score'] = df['clean_lyric'].progress_apply(lambda x: AFINNEvaluator(x, afinn_dict).evaluate().valence)

# Log a success message indicating the completion of the process
logger.success("AFFIN Evaluation process completed successfully.")

# Select columns for final output
df = df[["id", "afinn_score"]]

# Save the DataFrame to a new CSV file
output_file = f"{folder_data}/09_SE_Afinn.csv"
df.to_csv(output_file, index=False)
logger.success(f"Saved AFINN scores to {output_file}.")