#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-06
Version: 1.0.0

Description:
    This script trains and evaluates a Linear Support Vector Classification (SVC) model using multi-output classification on genre classification data.
    It utilizes the `analysis_functions` module to load and prepare the data, perform hyperparameter tuning with Grid Search,
    and evaluate the model's performance. The results are saved in a CSV file.

    The script performs the following steps:
    1. Loads genre classification data using a custom function.
    2. Prepares dataframes for all genres and Brazilian genres.
    3. Defines and trains a LinearSVC model using multi-output classification with a pipeline that includes scaling.
    4. Performs Grid Search with cross-validation to tune hyperparameters.
       The parameter grid includes:
       - `C`: [0.1, 0.5, 1.0, 2.0, 5.0]
       - `loss`: ['squared_hinge']
    5. Evaluates the model using F1 scores (micro and macro).
    6. Saves the results to a CSV file, including the combination of parameters that generates the best result.


Usage:
    1. Ensure all dependencies are installed and accessible.
    2. Ensure the `functions` directory is in the correct path and contains `analysis_functions.py`.
    3. Run the script: python 02_SVC_F1_all_genres.py

Notes:
    - Adjust paths and filenames as needed.
    - Results are saved to 'svc_results_F1_all_genres.csv' in the specified output directory.
"""

# Standard library imports
import os  # Operating system interface
import sys  # System-specific parameters and functions

# Third-party library imports
import pandas as pd  # Data manipulation and analysis
from loguru import logger  # Logging

from sklearn.model_selection import train_test_split, GridSearchCV, KFold  # Model selection
from sklearn.preprocessing import StandardScaler  # Data preprocessing
from sklearn.pipeline import Pipeline  # Pipeline for combining multiple steps
from sklearn.svm import LinearSVC  # Support Vector Classification
from sklearn.multioutput import MultiOutputClassifier  # Multi-output classification

# Local application/library specific imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../functions')))
import analysis_functions
from analysis_functions import folder_output


# Load data
logger.info("Loading data...")
df, all_genres, br_genres = analysis_functions.get_data()

# Prepare dataframes for all genres and BR genres
logger.info("Preparing dataframes for all genres and BR genres...")
df_all_genres = analysis_functions.make_df_genres(df, all_genres)
df_br_genres = analysis_functions.make_df_genres(df, br_genres)

# Load feature group model for artist
logger.info("Loading feature group model for artist...")
feat_group_model = analysis_functions.dict_feature_group_art()

def train_and_evaluate_linear_svc(X, y, feature_group_name, genres):
    logger.info(f"Training and evaluating LinearSVC for feature group: {feature_group_name}")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MultiOutputClassifier(LinearSVC())),
    ])

    # Define the hyperparameter grid
    param_grid = {
        'model__estimator__C': [0.1, 0.5, 1.0, 2.0, 5.0],
        'model__estimator__loss': ['squared_hinge'],
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=-1,
                                scoring=['f1_micro', 'f1_macro'], refit=False,
                                verbose=3, return_train_score=True, 
                                cv=KFold(n_splits=5, shuffle=True, random_state=42))
                                
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['feature_group'] = feature_group_name

    # Join genres into a string
    genres = ",".join(str(element) for element in genres)
    results_df['genre_labels'] = genres

    logger.success("Results compiled into dataframe.")
    return results_df

# Execute model training and evaluation for BR genres and all genres
logger.info("Executing model training and evaluation for All genres...")
df_results_SVC = analysis_functions.exec_model(train_and_evaluate_linear_svc, df, all_genres, feat_group_model, "SVC")

# Combine results and save to CSV
df_results_SVC.to_csv(f'{folder_output}/svc_results_F1_all_genres.csv', index=False)
logger.success(f"Results saved to '{folder_output}/svc_results_F1_all_genres.csv'")
