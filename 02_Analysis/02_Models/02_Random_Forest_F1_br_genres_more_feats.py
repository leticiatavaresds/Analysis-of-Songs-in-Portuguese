#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-

"""
Author: Letícia Tavares
Date: 2024-08-06
Version: 1.0.0

Description:
    This script trains and evaluates a Random Forest classifier on genre classification data.
    It uses the `analysis_functions` module to load and prepare the data, perform hyperparameter tuning
    with Grid Search, and evaluate the model's performance. The results are saved in a CSV file.
    
    The script performs the following steps:
    1. Loads genre classification data using a custom function.
    2. Prepares dataframes for Brazilian genres.
    3. Defines and trains an Random Forest classifier using a pipeline with a StandardScaler.
    4. Performs Grid Search with cross-validation to tune hyperparameters.
       The parameter grid includes:
       - `n_estimators`: [10, 100, 300]
    5. Evaluates the model using F1 scores (micro and macro).
    6. Saves the bets results to a CSV file, including the combination of parameters that generates the best result.

Usage:
    1. Ensure all dependencies are installed and accessible.
    2. Ensure the `functions` directory is in the correct path and contains `analysis_functions.py`.
    3. Run the script: python 02_Random_Forest_F1_br_genres_more_feats.py

Notes:
    - Adjust paths and filenames as needed.
    - Results are saved to 'random_forest_results_F1_br_genres_more_feats.csv' in the specified output directory.
"""

# Standard library imports
import os  # Operating system interface
import sys  # System-specific parameters and functions

# Third-party library imports
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis
import random # Random number generation and related operations 
from loguru import logger  # Logging
from sklearn.model_selection import train_test_split, GridSearchCV, KFold  # Model selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline  # Pipeline for combining multiple steps
from sklearn.ensemble import RandomForestClassifier  # Random forest classifier

# Local application/library specific imports
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../functions')))
import analysis_functions
from analysis_functions import folder_output

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# Load data
logger.info("Loading data...")
df, all_genres, br_genres = analysis_functions.get_data()

# Prepare dataframes for all genres and BR genres
logger.info("Preparing dataframes for all genres and BR genres...")
df_all_genres = analysis_functions.make_df_genres(df, all_genres)
df_br_genres = analysis_functions.make_df_genres(df, br_genres)

# Load feature group model for artist
logger.info("Loading feature group model for artist...")
feat_group_model = analysis_functions.dict_feature_group()

def train_and_evaluate_rf(X, y, feature_group_name, genres):
    logger.info("Training and evaluating Random Forest for feature group: {}".format(feature_group_name))

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_jobs=-1, random_state=42)),
    ])

    # Define the hyperparameter grid
    param_grid = {
        'model__n_estimators': [10,100,300]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, 
                                scoring=['f1_micro', 'f1_macro'], refit='f1_macro',
                                cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                n_jobs=-1, verbose=2, return_train_score=True)

    grid_search.fit(X_train, y_train)

    # Get the best model
    logger.info("Grid search complete. Compiling results...")
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['feature_group'] = feature_group_name


    # Join genres into a string
    genres = ",".join(str(element) for element in genres)
    results_df['genre_labels'] = genres

    logger.success("Results compiled into dataframe.")
    return results_df

# Execute model training and evaluation for BR genres and all genres
logger.info("Executing model training and evaluation for BR genres...")
df_results_RF = analysis_functions.exec_model(train_and_evaluate_rf, df, br_genres, feat_group_model, "Random Forest", False)

# Combine results and save to CSV
df_results_RF.to_csv(f'{folder_output}/random_forest_results_F1_br_genres_more_feats.csv', index=False)
logger.success(f"Results saved to '{folder_output}/random_forest_results_F1_br_genres_more_feats.csv'")