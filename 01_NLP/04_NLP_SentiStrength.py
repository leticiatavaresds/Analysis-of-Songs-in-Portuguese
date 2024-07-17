import pandas as pd
import nltk
from sentistrength import PySentiStr
from loguru import logger
from collections import Counter
from tqdm.auto import tqdm
import os

# Initialize tqdm integration with pandas
tqdm.pandas()

# Check if the directory where NLTK data is stored already exists
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download NLTK tokenizer data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # Configurar contexto SSL para download (necessário em alguns sistemas)
    nltk.download('punkt')

# Define data folders
folder_data = 'Data_Output'
folder_input = 'Data_Input'

# Define the database file path
folder_output = "Data_Output"

# Initialize PySentiStr instance
pySentiStr = PySentiStr()

# Set paths for SentiStrength JAR and language data
pySentiStr.setSentiStrengthPath(f"{folder_input}/SentiStrength.jar")
pySentiStr.setSentiStrengthLanguageFolderPath(f"{folder_input}/SentStrength_Data/")


# Load tokenized data
df = pd.read_csv(f"{folder_data}/00_clean_lyric.csv")
len_df = len(df)

logger.info(f"Songs successfully obtained, there are {len_df} songs to analyze.") 

df["lines"] = df.apply(lambda x: x["clean_lyric"].split("\n"), axis=1)

class PySentiStrength:
    """
    User interface for interacting with SentiStrength tool
    """

    def get_scores_batch(self, lines) -> dict:
        """
        Calculate sentiment scores for a batch of lines

        Args:
            lines (list): List of text lines

        Returns:
            dict: Dictionary containing average sentiment scores
        """
        lines = [line for line in lines if line.strip() != '']
        num_lines = len(lines)
        dict_lines = dict(Counter(lines))
   

        # Initialize mood counts
        positive_mood = 0
        negative_mood = 0
        neutral_mood = 0
        total = 0

        for line in dict_lines.keys():

            result = pySentiStr.getSentiment(line, score='trinary')
            positive, negative, neutral = list(result[0])

            total += dict_lines[line]
            positive_mood += (dict_lines[line] * positive)
            negative_mood += (dict_lines[line] * negative)
            neutral_mood += (dict_lines[line] * neutral)

        
        return {
            'positive_mood': positive_mood / num_lines,
            'negative_mood': negative_mood / num_lines,
            'neutral_mood': neutral_mood / num_lines,
        }

    def score_classifier(self, positive: int, negative: int, length: int, count_neutral_words: int) -> str:
        """
        Determine sentiment balance based on provided scores

        Args:
            positive (int): Positive sentiment score
            negative (int): Negative sentiment score
            length (int): Length of the document
            count_neutral_words (int): Count of neutral words in the document

        Returns:
            str: Balance result ('impartial' or 'partial')
        """
        porcNeutral = count_neutral_words / length
        if porcNeutral > 0.6:
            return "impartial"
        elif positive < 3 and negative > -3:
            return "impartial"
        else:
            return "partial"

def evaluate_sentiment(row):
    """
    Apply sentiment analysis to each row in DataFrame using PySentiStrength instance

    Args:
        row (pd.Series): DataFrame row containing 'lines' text
        sentiment_analyzer (PySentiStrength): Instance of PySentiStrength

    Returns:
        pd.Series: Updated row with sentiment scores
    """

    score = pySentiStrength.get_scores_batch(row["lines"])

    dict_score = {"positive_mood": score["positive_mood"],
            "negative_mood": score["negative_mood"],
            "neutral_mood": score["neutral_mood"]}



    return dict_score

# Initialize PySentiStrength instance
pySentiStrength = PySentiStrength()


# Apply sentiment analysis to each row with tqdm progress bar
result = df.progress_apply(evaluate_sentiment, axis=1)
df_result = pd.DataFrame(result.tolist())

# Concatenate the result with the original DataFrame
df = pd.concat([df, df_result], axis=1)

# Define the columns to keep
columns = ["id"] + list(result[0].keys())
df = df[columns]

# Log a success message indicating the completion of the process
logger.success("Sentiment Analysis completed successfully.")


# Save the evaluated DataFrame to CSV
output_file = f"{folder_data}/04_SE_sentiStrength.csv"
df.to_csv(output_file, index=False)
logger.success(f"Saved sentiment analysis results to {output_file}.")
