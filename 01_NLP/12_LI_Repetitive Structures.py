import pandas as pd
from loguru import logger
import spacy
from typing import List, Tuple, Set
from tqdm.auto import tqdm
from typing import List, Tuple

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

# Load tokenized data
df = pd.read_csv(f"{folder_data}/00_clean_lyric.csv")
len_df = len(df)

logger.info(f"Songs successfully obtained, there are {len_df} songs to analyze.") 



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
        lines = []
        doc = self.nlp(text)
        add_a = False

        for sent in doc.sents:
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

# Verse class to represent a verse containing multiple lines
class Verse:
    def __init__(self, index):
        self.lines = []
        self.index = index

    def add_line(self, line: Line):
        self.lines.append(line)

    def add_lines(self, lines: List[Line]):
        self.lines.extend(lines)

    def get_lines(self) -> List[Line]:
        return self.lines

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

    def get_verses(self) -> List['Verse']:
        verses = []
        i = 0
        verse = Verse(i)
        verses_text = self.lyric.split("\n\n")

        for verse_text in verses_text:
            get_lines = self.get_lines_verse(verse_text)

            for line in get_lines:
                if line.get_text().strip():
                    verse.add_line(line)
                    self.lines.append(line)

            if verse.lines:
                i += 1
                verses.append(verse)
                verse = Verse(i)

        return verses

# SimilarLine class to represent similar lines
class SimilarLine:
    def __init__(self, line, verse_index, line_index, similarity):
        self.line = line
        self.verse_index = verse_index
        self.line_index = line_index
        self.similarity = similarity

# Block class to represent a block of lines
class Block:
    def __init__(self):
        self.lines = []

# RepetitiveStructure class to store repetitiveness metrics
class RepetitiveStructure:
    def __init__(self):
        self.lyric_id = None
        self.block_count = None
        self.average_block_size = None
        self.blocks_per_line = None
        self.average_alignment_score = None
        self.repetitivity = None
        self.block_reduplication = None
        self.type_token_ratio_lines = None
        self.type_token_ratio_inlines = None

    def set_lyric_id(self, lyric_id):
        self.lyric_id = lyric_id

    def set_block_count(self, block_count):
        self.block_count = block_count

    def set_average_block_size(self, average_block_size):
        self.average_block_size = average_block_size

    def set_blocks_per_line(self, blocks_per_line):
        self.blocks_per_line = blocks_per_line

    def set_average_alignment_score(self, average_alignment_score):
        self.average_alignment_score = average_alignment_score

    def set_repetitivity(self, repetitivity):
        self.repetitivity = repetitivity

    def set_block_reduplication(self, block_reduplication):
        self.block_reduplication = block_reduplication

    def set_type_token_ratio_lines(self, type_token_ratio_lines):
        self.type_token_ratio_lines = type_token_ratio_lines

    def set_type_token_ratio_inlines(self, type_token_ratio_inlines):
        self.type_token_ratio_inlines = type_token_ratio_inlines

# Function to create shingles from tokens
def create_shingles(tokens: List[str], n: int) -> List[List[str]]:
    return [tokens[i:i+n] for i in range(len(tokens)-n+1)]

# Function to calculate word similarity
def word_similarity(x: List[str], y: List[str]) -> float:
    ngrams_x = set(" ".join(token.lemma_ for token in shingle) for shingle in create_shingles(x, 2))
    ngrams_y = set(" ".join(token.lemma_ for token in shingle) for shingle in create_shingles(y, 2))

    intersection_size = len(ngrams_x & ngrams_y)
    max_size = max(len(ngrams_x), len(ngrams_y))

    return intersection_size / max_size if max_size > 0 else 0.0

# Function to calculate structural similarity
def struct_similarity(x: List[str], y: List[str]) -> float:

    ngrams_x = create_shingles(x, 2)
    ngrams_y = create_shingles(y, 2)

    x_prime = []
    y_prime = []

    for t1 in ngrams_x:
        if not any(" ".join(token.lemma_ for token in t1) == " ".join(token.lemma_ for token in t2) for t2 in ngrams_y):
            x_prime.append(t1)

    for t1 in ngrams_y:
        if not any(" ".join(token.lemma_ for token in t1) == " ".join(token.lemma_ for token in t2) for t2 in ngrams_x):
            y_prime.append(t1)

    ngrams_x_prime = set(" ".join(token.tag_ for token in shingle) for shingle in x_prime)
    ngrams_y_prime = set(" ".join(token.tag_ for token in shingle) for shingle in y_prime)

    if not ngrams_x_prime and not ngrams_y_prime:
        return 1.0

    intersection_size = len(ngrams_x_prime & ngrams_y_prime)
    max_size = max(len(ngrams_x_prime), len(ngrams_y_prime))

    return (intersection_size / max_size) ** 2 if max_size > 0 else 0.0

# Function to calculate total similarity
def similarity(x: List[str], y: List[str]) -> float:

    word_sim = word_similarity(x, y)
    struct_sim = struct_similarity(x, y)
    alpha = word_sim
    return alpha * word_sim + (1 - alpha) * struct_sim

# Function to get similar lines
def get_similar_lines(verses):

    similar_lines = []
    seen_pairs = set()
    for i, verse1 in enumerate(verses):
        for j, verse2 in enumerate(verses[i + 1:], start=i + 1):
            for k, l1 in enumerate(verse1.lines):
                for l, l2 in enumerate(verse2.lines):
                    sim = similarity(l1.tokens, l2.tokens)
                    if sim >= 0.25:
                        pair = (i, k, j, l)
                        reverse_pair = (j, l, i, k)
                        if pair not in seen_pairs and reverse_pair not in seen_pairs:
                            similar_lines.append((SimilarLine(l1, i, k, sim), SimilarLine(l2, j, l, sim)))
                            seen_pairs.add(pair)
    return similar_lines

# Function to sort blocks by size in descending order
def sort_blocks_by_size_descending(blocks):
    return sorted(blocks, key=lambda block: len(block

.lines), reverse=True)

# Function to get blocks from similar lines
def get_blocks(similar_lines: List[Tuple[SimilarLine, SimilarLine]]) -> Set[Block]:
    blocks = set()
    
    similar_lines_dict = {}
    for i, pair in enumerate(similar_lines):
        pair_key =  str([(pair[0].verse_index, pair[0].line_index), (pair[1].verse_index, pair[1].line_index)])
        similar_lines_dict[pair_key] = i

    for pair in similar_lines:
        block = Block()
        block.lines.append(pair[0])

        line1 = pair[0].line_index
        line2 = pair[1].line_index
        verse1 = pair[0].verse_index
        verse2 = pair[1].verse_index

        while True:
            next_pair = str([(verse1, line1 + 1), (verse2, line2 + 1)])

            try:
                index = similar_lines_dict[next_pair]
                block.lines.append(similar_lines[index][0])
                line1 += 1
                line2 += 1

            except:
                break

        line1 = pair[0].line_index
        line2 = pair[1].line_index

        while True:
            next_pair = str([(verse1, line1 - 1), (verse2, line2 - 1)])

            try:
                index = similar_lines_dict[next_pair]
                block.lines.append(similar_lines[index][0])
                line1 -= 1
                line2 -= 1

            except:
                break

        blocks.add(block)

    return blocks

# Evaluate class to evaluate the repetitiveness of lyrics
class evaluate:
    def __init__(self, text, lyric_id):
        prepared = PreparedLyric(text)
        self.verses = prepared.get_verses()
        self.lines = prepared.lines
        self.lyric_id = lyric_id

    def similiar(self):
        similar_lines = get_similar_lines(self.verses)
        self.blocks = get_blocks(similar_lines)
        repetitive_structure = RepetitiveStructure()

        lines = [line for verse in self.verses for line in verse.lines]
        total_lines = len(lines)
        repetitive_lines = sum(len(b.lines) for b in self.blocks)
        alignment_scores = sum(sum(l.similarity for l in block.lines) for block in self.blocks)

        repetitive_structure.lyric_id = self.lyric_id
        repetitive_structure.block_count = len(self.blocks)
        repetitive_structure.average_block_size = repetitive_lines / repetitive_structure.block_count if self.blocks else 0.0
        repetitive_structure.blocks_per_line = repetitive_structure.block_count / total_lines if total_lines else 0.0
        repetitive_structure.average_alignment_score = alignment_scores / repetitive_lines if repetitive_lines else 0.0
        repetitive_structure.repetitivity = repetitive_lines / total_lines if total_lines else 0.0

        repetitive_blocks = {"\n".join(line.line.text.lower() for line in block.lines).strip() for block in self.blocks}
        repetitive_structure.block_reduplication = len(repetitive_blocks) / repetitive_structure.block_count if self.blocks else 0.0
        repetitive_structure.type_token_ratio_lines = len(set(l.text.lower() for l in lines)) / total_lines if total_lines else 0.0
        repetitive_structure.type_token_ratio_inlines = sum(len(set(t.lemma_.lower() for t in l.tokens)) / len(l.tokens) for l in lines) / total_lines if lines else 0.0

        return repetitive_structure

# Function to evaluate each row of the DataFrame
def apply_evaluate(row):
    text = row['clean_lyric']
    lyric_id = row['id']
    
    # Perform evaluation and similarity calculation
    eval = evaluate(text, lyric_id).similiar()

    # Prepare the result dictionary
    result = {
        "block_count": eval.block_count,
        "average_block_size": eval.average_block_size,
        "blocks_per_line": eval.blocks_per_line,
        "average_alignment_score": eval.average_alignment_score,
        "repetitivity": eval.repetitivity,
        "block_reduplication": eval.block_reduplication,
        "type_token_ratio_lines": eval.type_token_ratio_lines,
        "type_token_ratio_inlines": eval.type_token_ratio_inlines
    }
    
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
logger.success("Repetitive Structures Evaluation process completed successfully.")

# Save the evaluated DataFrame to CSV
output_file = f"{folder_data}/12_LI_RepetitiveStructures.csv"
df.to_csv(output_file, index=False)
logger.success(f"Saved evaluated Repetitive Structures to {output_file}.")