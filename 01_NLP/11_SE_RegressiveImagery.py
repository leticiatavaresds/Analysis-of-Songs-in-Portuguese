import re
import pandas as pd
import ast
from collections import defaultdict
from loguru import logger
from tqdm.auto import tqdm

# Initialize tqdm integration with pandas
tqdm.pandas()

folder_data = 'Data_Output'
folder_input = 'Data_Input'

# Load the tokenized data
df = pd.read_csv(f"{folder_data}/01_tokenization.csv")
logger.info("Loaded tokenized data from 01_tokenization.csv.")

# Define features mapping for RID categories to human-readable names
features = {
    "ORALITE": "primary_need_orality",
    "ANALITE": "primary_need_anality",
    "SEXE": "primary_need_sex",
    "TOUCHER": "primary_sensation_touch",
    "GOUT": "primary_sensation_taste",
    "ODORAT": "primary_sensation_odor",
    "SENSATION GENERALE": "primary_sensation_general_sensation",
    "OUIE": "primary_sensation_sound",
    "VUE": "primary_sensation_vision",
    "FROID": "primary_sensation_cold",
    "DUR": "primary_sensation_hard",
    "DOUX": "primary_sensation_soft",
    "PASSIVITE": "primary_defensive_symbol_passivity",
    "VOYAGE": "primary_defensive_voyage",
    "MOUVEMENT NON ORIENTE": "primary_defensive_symbol_random_movement",
    "DIFFUS": "primary_defensive_symbol_diffusion",
    "CHAOS": "primary_defensive_symbol_chaos",
    "INCONNU": "primary_regressive_cognition_unknown",
    "INTEMPOREL": "primary_regressive_cognition_timelessness",
    "ALTERATION DE LA CONSCIENCE": "primary_regressive_cognition_consciousness_alternation",
    "FRANCHISSEMENT & PASSAGE": "primary_regressive_cognition_brink_passage",
    "NARCISSISME": "primary_regressive_cognition_narcissism",
    "CONCRET": "primary_regressive_cognition_concreteness",
    "MONTER": "primary_icarian_imagery_ascend",
    "HAUT": "primary_icarian_imagery_height",
    "DESCENDRE": "primary_icarian_imagery_descent",
    "PROFONDEUR": "primary_icarian_imagery_depth",
    "FEU": "primary_icarian_imagery_fire",
    "EAU": "primary_icarian_imagery_water",
    "PENSEE ABSTRAITE": "secondary_abstraction",
    "COMPORTEMENT SOCIAL": "secondary_social_behavior",
    "COMPORTEMENT INSTRUMENTAL": "secondary_instrumental_behavior",
    "LOI & RESTRICTION": "secondary_restraint",
    "ORDRE": "secondary_order",
    "REFERENCE TEMPORELLE": "secondary_temporal_references",
    "IMPERATIF MORAL": "secondary_moral_imperative",
    "AFFECT POSITIF": "emotions_positive_affect",
    "ANXIÉTÉ": "emotions_anxiety",
    "TRISTESSE": "emotions_sadness",
    "AMOUR": "emotions_affection",
    "AGRESSION": "emotions_aggression",
    "COMPORTEMENT EXPRESSIF": "emotions_expressive_behavior",
    "TRIOMPHE": "emotions_glory",
    "COUNT": "word_count"
}

# Initialize a dictionary for counting initial occurrences
dict_count_initial = {v: 0 for v in features.values()}

# Define category lists for grouping counts
primary = [k for k in dict_count_initial.keys() if 'primary_' in k]
primary_need = [k for k in dict_count_initial.keys() if 'primary_need_' in k]
primary_sensation = [k for k in dict_count_initial.keys() if 'primary_sensation_' in k]
primary_defensive = [k for k in dict_count_initial.keys() if 'primary_defensive_' in k]
primary_regressive_cognition = [k for k in dict_count_initial.keys() if 'primary_regressive_cognition_' in k]
primary_icarian_imagery = [k for k in dict_count_initial.keys() if 'primary_icarian_imagery_' in k]
secondary = [k for k in dict_count_initial.keys() if 'secondary_' in k]
emotions = [k for k in dict_count_initial.keys() if 'emotions_' in k]

# Function to read RID.CAT dictionary file
# Function to read RID.CAT dictionary file
def ler_dicionario_cat(file_path):
    lines = []
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            lines.append(line)
    dict_rid = {}
    
    for i in range(len(lines)):
        line = lines[i]
        nivel_indentacao = len(re.match(r"\s*", line).group())
        
        if nivel_indentacao == 1:
            key_2 = line.strip()
            
        elif nivel_indentacao == 2:
            conteudo = line.strip()

            if "(1)" in conteudo:
                conteudo = conteudo.replace(" (1)", "")
                dict_rid[conteudo] = key_2
            else:
                key_3 = conteudo
                
        elif nivel_indentacao == 3:
            conteudo = line.strip().replace(" (1)", "")
            dict_rid[conteudo] = key_3
     
    return dict_rid

# Function to read RID.EXC dictionary file
def ler_dicionario_exc(file_path):
    excecoes = set()
    
    with open(file_path, 'r', encoding='iso-8859-1') as file:
        for line in file:
            line = line.strip()
            if line:
                excecoes.add(line)
    
    return excecoes

# Function to find matches in text based on list_rid
def find_match(text):

    initial_letter = text[0].upper()
    texto_upper = text.upper()

    compiled_patterns = dict_patterns[initial_letter]

    patter_keys = []
    
    for pattern in compiled_patterns:
        if pattern.match(texto_upper):
            patter_key = pattern.pattern.replace(".", "")
            if patter_key[-1] == "$":
                patter_key = patter_key[1:-1]

            patter_keys.append(patter_key)
       
    if len(patter_keys):
        return patter_keys
        
    return []



# Function to evaluate RID categories in tokens
def RidEvaluator(row):
    tokens = ast.literal_eval(row['tokens'])
    dict_count = dict_count_initial.copy()
    word_count = 0
    
    for token in tokens:

        word_match = find_match(token.upper())
        # print(word_match)
  
        if len(word_match) == 1:
            word_count+=1
            categorie = rid_cat[word_match[0]]
            dict_count[features[categorie]] += 1
            

        elif len(word_match) > 1:

            word_count+=1
            categories = [rid_cat[word] for word in word_match]
            categories = list(set(categories))

            for categorie in categories:
                dict_count[features[categorie]] += 1

    # dict_count["word_count"] = word_count
    dict_sum = {
        'primary': sum(dict_count[k] for k in primary),
        'primary_need': sum(dict_count[k] for k in primary_need),
        'primary_sensation': sum(dict_count[k] for k in primary_sensation),
        'primary_defensive_symbol': sum(dict_count[k] for k in primary_defensive),
        'primary_regressive_cognition': sum(dict_count[k] for k in primary_regressive_cognition),
        'primary_icarian_imagery': sum(dict_count[k] for k in primary_icarian_imagery),
        'secondary': sum(dict_count[k] for k in secondary),
        'emotions': sum(dict_count[k] for k in emotions)
    }

    result = {**dict_count, **dict_sum}

    if word_count != 0:
        result = {k:v/word_count for k,v in result.items()}
    
    result["word_count"] = word_count

    return result

def create_dict_patterns(list_rid):
    dict_patterns = defaultdict(list)
    
    for padrao in list_rid:
        initial_letter = padrao[0].upper()
        pattern_regex = padrao.replace('*', '.*')

        if "*" not in pattern_regex:
            pattern_regex = f'^{pattern_regex}$'
        dict_patterns[initial_letter].append((re.compile(pattern_regex, re.IGNORECASE)))
    
    return dict_patterns


# File paths for .CAT and .EXC files
cat_file_path = f"{folder_input}/Portuguese RID.CAT"
exc_file_path = f"{folder_input}/Portuguese RID.EXC"

# Read dictionaries from files
rid_cat = ler_dicionario_cat(cat_file_path)
rid_exc = ler_dicionario_exc(exc_file_path)

# Create list of RID keys for matching
list_rid = list(rid_cat.keys())

dict_patterns = create_dict_patterns(list_rid)

logger.debug("Applying Regressive Imagery...")
# Apply the evaluation function to each row with tqdm progress bar
result = df.progress_apply(RidEvaluator, axis=1)
columns = ["id"] + list(result[0].keys()) 

# Convert the result to a DataFrame and concatenate it with original DataFrame
df_result = pd.DataFrame(result.tolist())
df = pd.concat([df, df_result], axis=1)
df = df[columns]
# Log a success message indicating the completion of the process
logger.success("Regressive Imagery Evaluation process completed successfully.")

# Save the evaluated DataFrame to CSV
output_csv_path = f"{folder_data}/11_SE_RegressiveImagery.csv"
df.to_csv(output_csv_path, index=False)
logger.success(f"Saved evaluated Regressive Imagery to {output_csv_path}.")