import pandas as pd
from loguru import logger
import spacy
from typing import List
from tqdm.auto import tqdm
import os 

# Initialize tqdm integration with pandas
tqdm.pandas()

import nltk
from nltk.tag import UnigramTagger, BigramTagger
from nltk.corpus import mac_morpho
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

# Check if the directory where NLTK data is stored already exists
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# List of packages to be checked and downloaded if necessary
nltk_packages = [
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('corpora/mac_morpho', 'mac_morpho'),
    ('stemmers/rslp', 'rslp'),
    ('corpora/stopwords', 'stopwords')
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


folder_data = 'Data_Output'
folder_input = 'Data_Input'

# Define the name of the model
model_name = "pt_core_news_sm"

# Check if the model is already installed
if not spacy.util.is_package(model_name):
    logger.info(f"{model_name} not found. Downloading...")
    spacy.cli.download(model_name)
    logger.info("Downloaded Portuguese language model for spaCy.")

# Load the model
nlp = spacy.load(model_name)

# Define the database file path
folder_output = "Data_Output"

# Load data
df = pd.read_csv(f"{folder_data}/00_clean_lyric.csv")
len_df = len(df)

logger.info(f"Songs successfully obtained, there are {len_df} songs to analyze.") 

# Load the tagged sentences from the mac_morpho corpus
train_sents = mac_morpho.tagged_sents()
logger.info("Loaded tagged sentences from the mac_morpho corpus.")

# Create a UnigramTagger using the mac_morpho tagged sentences as training data
# A UnigramTagger assigns the most frequent tag to each word based on the training data
unigram_tagger = UnigramTagger(train_sents)
logger.info("Created UnigramTagger using mac_morpho tagged sentences as training data.")

# Create a BigramTagger using the mac_morpho tagged sentences as training data
# The BigramTagger attempts to tag words based on the previous word's tag. If it fails, it backs off to the UnigramTagger
bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)
logger.info("Created BigramTagger with backoff to UnigramTagger using mac_morpho tagged sentences as training data.")

# Initialize the RSLP stemmer for Portuguese
# The RSLP stemmer is used to reduce words to their root forms
stemmer = RSLPStemmer()
logger.info("Initialized RSLP stemmer for Portuguese.")

stop_words = set(stopwords.words('portuguese'))
additional_stopwords = ['já', 'ta', 'também', 'vc', 'até', 'agora', 
                        'sempre', 'ter', 'porque', 'sobre', 'ainda', 'lá', 'tudo', 'de', 
                        'pra', 'uns', 'tô']



class PreparedLyric:
    def __init__(self, text: str):

        self.text = text
        self.tokens = word_tokenize(text, language='portuguese')
        self.pos_tags = pos_tag(self.tokens)
        self.lemmatizer = WordNetLemmatizer()
        self.tokens_sem_stopwords = [token for token in self.tokens if token.lower() not in stop_words]

    def get_tokens(self) -> List[str]:
        return self.tokens

    def get_pos_tags(self) -> List[str]:
        return bigram_tagger.tag(self.tokens)

    def get_lemmas(self) -> List[str]:
        return [stemmer.stem(token) for token in self.tokens_sem_stopwords]

class BagOfWords:
    def __init__(self):

        self.lemmatized_content_words = []
        self.pos_tags = []

    def set_lemmatized_content_words(self, lemmatized_content_words: List[str]):
        self.lemmatized_content_words = lemmatized_content_words

    def set_pos_tags(self, pos_tags: List[str]):
        self.pos_tags = pos_tags

class BagOfWordsEvaluator:
    def __init__(self, lyric: PreparedLyric):
        self.lyric = lyric

    def evaluate(self) -> BagOfWords:
        bag_of_words = BagOfWords()

        bag_of_words.set_lemmatized_content_words(self.get_lemmatized_content_words())
        bag_of_words.set_pos_tags(self.get_pos_tags())

        return bag_of_words

    def get_lemmatized_content_words(self) -> List[str]:
        return [lemma for lemma in self.lyric.get_lemmas() if lemma]

    def get_pos_tags(self) -> List[str]:
        return self.lyric.get_pos_tags()

def apply_evaluate(row):
    text = row['clean_lyric']

    prepared_lyric = PreparedLyric(text)
    evaluator = BagOfWordsEvaluator(prepared_lyric)
    bag_of_words = evaluator.evaluate()
    
    pos_tags = bag_of_words.pos_tags
    pos_tags = [pair[1] for pair in pos_tags if pair[1] is not None]
    # Prepare the result dictionary
    result = {
        "bag_pos_tag": pos_tags
    }
    
    return result


# Log an initial message indicating the start of the process
logger.debug("Starting Pos Tagging Bag of Word evaluation process...")

# Apply the evaluation function to each row with tqdm progress bar
result = df.progress_apply(apply_evaluate, axis=1)
columns = ["id"] + list(result[0].keys())

# Convert the result to a DataFrame and concatenate it with original DataFrame
df_result = pd.DataFrame(result.tolist())
df = pd.concat([df, df_result], axis=1)
df = df[columns]

# Log a success message indicating the completion of the process
logger.success("Pos Tagging Bag of Word Evaluation process completed successfully.")

# Save the evaluated DataFrame to CSV
df.to_csv(f"{folder_data}/13_SY_Pos_Bag.csv", index=False)
logger.success(f"Saved evaluated Repetitive Structures to {folder_data}/13_SY_Pos_Bag.csv")