import nltk
from collections import defaultdict
import spacy

import pandas as pd
from loguru import logger
import spacy
from typing import List
from tqdm.auto import tqdm

# Importe as classes necessárias
from collections import defaultdict
from typing import List

from nltk import pos_tag, word_tokenize

# Initialize tqdm integration with pandas
tqdm.pandas()


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

# Load data
df = pd.read_csv(f"{folder_data}/00_clean_lyric.csv")
len_df = len(df)

logger.info(f"Songs successfully obtained, there are {len_df} songs to analyze.") 



class PreparedLyric:
    def __init__(self, text: str):

        self.text = text
        self.tokens = word_tokenize(text, language='portuguese')
        self.pos_tags = pos_tag(self.tokens)

    def get_tokens(self) -> List[str]:
        return self.tokens


class ChunkTag:
    def __init__(self):
      self.tag = 0.0
      self.amount = 0.0
      self.average_length = 0.0
      self.ratio = 0.0

class SuperChunkTags:
    def __init__(self):
        self.noun_phrases_ratio = 0.0
        self.adjective_and_adverb_phrases_ratio = 0.0
        self.prepositional_phrases_ratio = 0.0
        self.special_phrases_ratio = 0.0
        self.verb_phrases_ratio = 0.0
        self.tags = []

    def add_tag(self, chunk_tag):
        self.tags.append(chunk_tag)
        
class SuperChunkTagsEvaluator:
    def __init__(self, lyric: PreparedLyric):
        self.lyric = lyric

    def evaluate(self) -> SuperChunkTags:
        chunk_tags = SuperChunkTags()

        chunk_counts = defaultdict(int)
        noun_phrases = 0
        verb_phrases = 0
        prepositional_phrases = 0
        adjective_and_adverb_phrases = 0
        special_phrases = 0

        tokens = self.lyric.get_tokens()
        total_tokens = len(tokens)
        text = ' '.join([token for token in tokens])

        doc = nlp(text)
        pos_tags = [(token.text, token.pos_) for token in doc]

        # Define chunk grammar
        grammar = r"""
          NP: {<DET>?<ADJ>*<NOUN>+}          # Frase nominal básica
              {<DET>?<NUM>*<ADJ>*<NOUN>+}    # Frase nominal com números
              {<PRON>}                       # Frase nominal como pronome
          VP: {<VERB.*>}             # Chunk verbs
          PP: {<ADP><NP>}            # Chunk prepositions followed by NP
          ADJP: {<ADJ>}              # Chunk adjectives
          ADVP: {<ADV>}              # Chunk adverbs
          CONJP: {<CONJ>}                    # Chunk conjunctions
          INTJ: {<INTJ>}                     # Chunk interjections
          LST: {<NUM>}                       # Chunk lists
          PRT: {<PART>}                      # Chunk particles
          UCP: {<SYM>}                       # Chunk uncategorized parts
        """

        # Create a chunk parser
        chunk_parser = nltk.RegexpParser(grammar)

        # Parse the POS tagged sentence
        tree = chunk_parser.parse(pos_tags)

        tags = []

        def extract_chunks(tree):
            for subtree in tree:
                if type(subtree) == nltk.Tree:
                    if subtree.label() in ["NP", "VP", "PP", "ADJP", "ADVP", "CONJP", "INTJ", "LST", "PRT", "UCP"]:
                        tags.append(subtree.label())
                    extract_chunks(subtree)

        extract_chunks(tree)

        for tag in tags:

            if tag == "NP":
                noun_phrases += 1
            elif tag == "VP":
                verb_phrases += 1
            elif tag == "PP":
                prepositional_phrases += 1
            elif tag in ["ADJP", "ADVP"]:
                adjective_and_adverb_phrases += 1
            elif tag in ["CONJP", "INTJ", "LST", "PRT", "UCP"]:
                special_phrases += 1

            chunk_counts[tag] += 1

  

        chunk_tags.noun_phrases_ratio = noun_phrases / total_tokens
        chunk_tags.adjective_and_adverb_phrases_ratio = adjective_and_adverb_phrases / total_tokens
        chunk_tags.prepositional_phrases_ratio = prepositional_phrases / total_tokens
        chunk_tags.verb_phrases_ratio = verb_phrases / total_tokens
        chunk_tags.special_phrases_ratio = special_phrases / total_tokens

        for k, v in chunk_counts.items():
            chunk_tag = ChunkTag()
            tag = k
            amount = v
            average_length = sum(1 for t in tags if t == k or t == f"B-{k}" or t == f"I-{k}") / v
            ratio = v / total_tokens


            chunk_tag = {"tag": tag,
                        "amount": amount,
                        "average_length": average_length,
                        "ratio": ratio}
            
            chunk_tags.add_tag(chunk_tag)

        return chunk_tags
    
def apply_evaluate(row):
    text = row['clean_lyric']

    prepared_lyric = PreparedLyric(text)
    evaluator = SuperChunkTagsEvaluator(prepared_lyric)
    result = evaluator.evaluate()

    dict_result = {"noun_phrases_ratio": result.noun_phrases_ratio,
                   "adjective_and_adverb_phrases_ratio": result.adjective_and_adverb_phrases_ratio,
                   "prepositional_phrases_ratio": result.prepositional_phrases_ratio,
                   "special_phrases_ratio": result.special_phrases_ratio,
                   "verb_phrases_ratio": result.verb_phrases_ratio,
                   "phrases_tags": result.tags
                   }
    
    return dict_result

# Log an initial message indicating the start of the process
logger.debug("Starting Text Chunks evaluation process...")

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
df.to_csv(f"{folder_data}/14_Text_Chunks.csv", index=False)
logger.success(f"Saved evaluated Repetitive Structures to {folder_data}/14_Text_Chunks.csv")