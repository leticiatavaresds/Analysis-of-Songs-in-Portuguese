#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-06
Version: 1.0.0

Description: 
    This script contains functions and variables for the classification models execution.
"""

# Third-party library imports
import pandas as pd # Data manipulation and analysis,
from sklearn.preprocessing import LabelEncoder

folder_output = "../02_Output_Kfold_Models"

def get_data(folder = "../"):
    """
    Loads the genre classification data from a CSV file and defines genre lists.

    Parameters:
        folder (str): Path to the directory containing the data file.

    Returns:
        tuple: A tuple containing the dataframe, all genres list, and BR genres list.
    """
    data_file = f"{folder}data_analysis.csv"
    df = pd.read_csv(data_file)

    all_genres = ['pop', 'sertanejo', 'rock', 'alternativo', 'mpb', 'samba',
    'bossanova', 'indie', 'pagode', 'gospel']

    br_genres = ['mpb', 'sertanejo', 'gospel', 'samba', 'pagode']

    return df, all_genres, br_genres

def get_tdfdf_lda_data(file):
    """
    Loads TF-IDF and LDA feature data from CSV files and removes the 'id' column.

    Parameters:
        file (str): Identifier for the data files to be loaded.

    Returns:
        tuple: A tuple containing the TF-IDF dataframe and the LDA dataframe.
    """
    folder_data = "../01_NLP/Data_Output"

    df_tfidf = pd.read_csv(f"../{folder_data}/16_tfidf_{file}.csv")
    df_tfidf = df_tfidf.drop(["id"], axis=1)

    df_lda = pd.read_csv(f"../{folder_data}/17_lda_{file}.csv")
    df_lda = df_lda.drop(["id"], axis=1)

    return df_tfidf, df_lda

def dict_feature_group():
    """
    Defines the feature groups for the analysis.

    Returns:
        dict: A dictionary containing different feature groups and their associated features.
    """
    feature_group = {
        'statistical': [
            'token_count',
            'unique_token_ratio',
            'unique_bigram_ratio',
            'unique_trigram_ratio',
            'avg_token_length',
            'unique_tokens_per_line',
            'average_tokens_per_line',
            'repeat_word_ratio',
            'line_count',
            'unique_line_count',
            'blank_line_count',
            'blank_line_ratio',
            'repeat_line_ratio',
            'digits',
            'exclamation_marks',
            'question_marks',
            'colons',
            'semicolons',
            'quotes',
            'commas',
            'dots',
            'hyphens',
            'stopwords_ratio',
            'stopwords_per_line',
            'hapax_legomenon_ratio',
            'dis_legomenon_ratio',
            'tris_legomenon_ratio',
            'syllables_per_line',
            'syllables_per_word',
            'syllable_variation',
        ],
        'statistical_time': [
            'words_per_minute',
            'chars_per_minute',
            'lines_per_minute',
        ],
        'explicitness': ['explicit'],
        'pronouns': [
            'i',
            'you',
            'it',
            'we',
            'they',
            'i_vs_you',
            'excentricity'
        ],
        'postags': [
            'verbs', 'participles', 'nouns', 'adjectives', 'adverbs',
            'denotatives_particle', 'pronouns', 'conjunctions', 'interjectios',
            'prepositions', 'foreignisms', 'wh_questions', 'special_characters'
        ],
        'lemma': [
            'lemmas_ratio', 'uncommon_words_ratio'
        ],
        'afinn': [
            'afinn_score'
        ],
        'vader': [
            'compound'
        ],
        'rid': [
            'primary_sensation',
            'primary_defensive_voyage',
            'primary_defensive_symbol_random_movement',
            'primary_regressive_cognition',
            'primary_icarian_imagery',
            'secondary_instrumental_behavior',
            'secondary_restraint',
            'secondary_temporal_references',
            'emotions_positive_affect',
            'emotions_anxiety',
            'emotions_sadness',
            'emotions_affection',
            'emotions_aggression',
            'emotions_expressive_behavior',
            'emotions_glory'
        ],
        'audio': [
            'tempo',
            'energy',
            'liveness',
            'speechiness',
            'acousticness',
            'danceability',
            'loudness',
            'valence',
            'instrumentalness',
            'duration_ms',
        ],
    }

    return feature_group

def dict_feature_group_art():
    """
    Defines the feature groups specific to the article's feature model.

    Returns:
        dict: A dictionary containing the feature groups and their features.
    """
    feature_group_model_article = {
        'statistical': [
            'token_count',
            'unique_token_ratio',
            'unique_bigram_ratio',
            'unique_trigram_ratio',
            'avg_token_length',
            'unique_tokens_per_line',
            'average_tokens_per_line',
            'repeat_word_ratio',
            'line_count',
            'unique_line_count',
            'blank_line_count',
            'blank_line_ratio',
            'repeat_line_ratio',
            'digits',
            'exclamation_marks',
            'question_marks',
            'colons',
            'semicolons',
            'quotes',
            'commas',
            'dots',
            'hyphens',
            'stopwords_ratio',
            'stopwords_per_line',
            'hapax_legomenon_ratio',
            'dis_legomenon_ratio',
            'tris_legomenon_ratio',
            'syllables_per_line',
            'syllables_per_word',
            'syllable_variation',
        ],
        'statistical_time': [
            'words_per_minute',
            'chars_per_minute',
            'lines_per_minute',
        ],
        'explicitness': ['explicit'],
        'audio': [
            'tempo',
            'energy',
            'liveness',
            'speechiness',
            'acousticness',
            'danceability',
            'loudness',
            'valence',
            'instrumentalness',
            'duration_ms',
        ]
    }

    return feature_group_model_article

def make_df_genres(df, genres):
    """
    Filters the dataframe to include only the specified genres and ensures that each row has exactly one genre.

    Parameters:
        df (DataFrame): The input dataframe.
        genres (list): List of genres to include in the filtered dataframe.

    Returns:
        DataFrame: The filtered dataframe with the specified genres.
    """
    df = df[genres] 
    df = df[df.sum(axis=1) == 1]  

    return df

def exec_model(function_model, df, genres, feat_group_model, model, with_vect = True):
    """
    Executes the specified model function for different feature groups and combines the results.

    Parameters:
        function_model (function): The function to execute the model training and evaluation.
        df (DataFrame): The dataframe containing the data.
        genres (list): List of genres to be used in the analysis.
        feat_group_model (dict): Dictionary of feature groups.
        model (str): The type of model to be used.
        with_vect (bool): Whether to include TF-IDF and LDA features.

    Returns:
        DataFrame: The dataframe containing the results of the model evaluations.
    """

    # Filtrar e preparar os dados
    filtered_df = df.loc[df[genres].sum(axis=1) == 1].reset_index(drop=True)

    if model == "Neural Network":
        print("OIEE")
        import numpy as np
        y = filtered_df[genres].values
        y_encoded = np.argmax(y, axis=1)

    else:
        label_encoder = LabelEncoder()
        y = filtered_df[genres]
        y = y.idxmax(axis=1)
        y_encoded = label_encoder.fit_transform(y)



    # Prepare result DataFrame
    results_df = pd.DataFrame()

    # Evaluate different feature groups
    for group_feat in feat_group_model.keys():
        
        feats = feat_group_model[group_feat]
        X = filtered_df[feats].values
        result_group = function_model(X, y_encoded, group_feat, genres)
        results_df = pd.concat([results_df, result_group], ignore_index=True)

    if(with_vect):

        if len(genres) > 6:
            df_tfidf, df_lda = get_tdfdf_lda_data("all_genres")
        else:
            df_tfidf, df_lda = get_tdfdf_lda_data("br_genres")

        #tfidf
        X = df_tfidf.values
        result_group = function_model(X, y_encoded, "tf-idf", genres)
        results_df = pd.concat([results_df, result_group], ignore_index=True)

        #lda
        X = df_lda.values
        result_group = function_model(X, y_encoded, "lda", genres)
        results_df = pd.concat([results_df, result_group], ignore_index=True)
        

    # Combined features excluding 'audio'
    combined = [feats for k, v in feat_group_model.items() for feats in v if k != "audio"]
    X = filtered_df[combined].values
    result_group = function_model(X, y_encoded, "combined", genres)
    results_df = pd.concat([results_df, result_group], ignore_index=True)

    if(with_vect):

        # Combined features excluding 'audio'
        combined = [feats for k, v in feat_group_model.items() for feats in v if k != "audio"]
        df_X = filtered_df[combined]
        df_X = pd.concat([df_X, df_tfidf, df_lda], axis=1)
        X = df_X.values
        result_group = function_model(X, y_encoded, "combined (tf-idf + lda)", genres)
        results_df = pd.concat([results_df, result_group], ignore_index=True)

    # Combined features including 'audio'
    combined_audio = [feats for k, v in feat_group_model.items() for feats in v]
    X = filtered_df[combined_audio].values
    result_group = function_model(X, y_encoded, "combined + audio", genres)
    results_df = pd.concat([results_df, result_group], ignore_index=True)

    if(with_vect):
        # Combined features including 'audio'
        combined_audio = [feats for k, v in feat_group_model.items() for feats in v]
        df_X = filtered_df[combined_audio]
        df_X = pd.concat([df_X, df_tfidf, df_lda], axis=1)
        X = df_X.values
        result_group = function_model(X, y_encoded, "combined (tf-idf + lda) + audio", genres)
        results_df = pd.concat([results_df, result_group], ignore_index=True)

    # Order the DataFrame by results
    if "mean_test_f1_micro" in results_df.columns:
        results_df = results_df.sort_values(by='mean_test_f1_micro', ascending=False).reset_index(drop=True)
        # results_df = results_df[(results_df['rank_test_f1_macro'] == 1) | (results_df['rank_test_f1_micro'] == 1)].reset_index(drop=True)

    if "mean_test_score" in results_df.columns:
        results_df = results_df.sort_values(by='mean_test_score', ascending=False).reset_index(drop=True)
        results_df = results_df.drop_duplicates(subset="feature_group", keep="first").reset_index(drop=True)

    elif "mean_accuracy" in results_df.columns:
        results_df = results_df.sort_values(by='mean_accuracy', ascending=False).reset_index(drop=True)
        results_df = results_df.drop_duplicates(subset="feature_group", keep="first").reset_index(drop=True)
    

    return results_df

    