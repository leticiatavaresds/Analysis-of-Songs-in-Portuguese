#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-06
Version: 1.0.0

Description:
    This script trains and evaluates a Neural Network model for genre classification.
    The model is built using TensorFlow and Keras, with multiple dense layers and dropout for regularization.
    It utilizes the `analysis_functions` module to load and prepare the data, performs K-Fold cross-validation,
    and evaluates the model's performance. The results are saved in a CSV file.

    The script performs the following steps:
    1. Loads genre classification data using a custom function.
    2. Prepares dataframes for all genres and Brazilian genres.
    3. Defines and trains a Neural Network model using K-Fold cross-validation with different configurations.
    4. Evaluates the model's performance using accuracy and F1 scores.
    5. Saves the best results to a CSV file, including the combination of parameters that generates the best result.

    Parameters and Combinations:
    - Dense Sizes: (32, 32), (64, 64)
    - Dropout Rates: 0.1
    - Epochs: 50
    - Batch Sizes: 2

Usage:
    1. Ensure all dependencies are installed and accessible.
    2. Ensure the `functions` directory is in the correct path and contains `analysis_functions.py`.
    3. Run the script: python 02_Neural_Network_F1_all_genres_more_feats.py

Notes:
    - Adjust paths and filenames as needed.
    - Results are saved to 'neural_network_results_F1_all_genres_more_feats.csv' in the specified output directory.
"""

# Standard library imports
import os  # Operating system interface
import sys  # System-specific parameters and functions

# Third-party library imports
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis
import random # Random number generation and related operations 
import tensorflow as tf # Deep learning library
from loguru import logger  # Logging
from sklearn.preprocessing import StandardScaler  # Data preprocessing
from sklearn.model_selection import KFold  # Model selection
from sklearn.metrics import accuracy_score, f1_score  # Performance metrics


# Local application/library specific imports
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../functions')))
import analysis_functions
from analysis_functions import folder_output

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

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

def create_nn_model(dense_sizes=(32, 32), dropout_rate=0.1, input_shape=None, output_shape=None):
    inp = tf.keras.layers.Input(shape=(input_shape,))
    
    # Dense layers
    layer = inp
    for size in dense_sizes:
        layer = tf.keras.layers.Dense(size, activation="selu", kernel_initializer="lecun_normal")(layer)
        layer = tf.keras.layers.Dropout(dropout_rate)(layer)

    # Output layer
    out = tf.keras.layers.Dense(output_shape, activation="softmax")(layer)
    
    # Create the model
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_nn(X, y, feature_group_name, genres):
    logger.info(f"Training and evaluating Neural Network for feature group: {feature_group_name}")

    results = []
    genres = ",".join(str(element) for element in genres)

    # Parameters
    dense_sizes_list = [(32, 32),(64, 64)]
    dropout_rates = [0.1]
    epochs = 50
    batch_size = 32

    # Perform K-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Determine the number of classes
    output_shape = np.max(y) + 1
    
    for dense_sizes in dense_sizes_list:
        for dropout_rate in dropout_rates:
            fold_accuracies = []
            fold_f1_micro_scores = []
            fold_f1_macro_scores = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = create_nn_model(dense_sizes=dense_sizes, dropout_rate=dropout_rate, input_shape=X.shape[1], output_shape = output_shape)
                
                # Normalize data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train the model
                model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                # Evaluate the model
                y_pred = model.predict(X_test_scaled)
                y_pred = np.argmax(y_pred, axis=1)

                accuracy = accuracy_score(y_test, y_pred)
                f1_micro = f1_score(y_test, y_pred, average='micro')
                f1_macro = f1_score(y_test, y_pred, average='macro')

                fold_accuracies.append(accuracy)
                fold_f1_micro_scores.append(f1_micro)
                fold_f1_macro_scores.append(f1_macro)

            mean_accuracy = np.mean(fold_accuracies)
            mean_f1_micro = np.mean(fold_f1_micro_scores)
            mean_f1_macro = np.mean(fold_f1_macro_scores)
    
            results.append({
                'feature_group': feature_group_name,
                'params': {'dense_sizes': dense_sizes, 'dropout_rate': dropout_rate, 'epochs': epochs, 'batch_size': batch_size},
                'dense_sizes': dense_sizes,
                'dropout_rate': dropout_rate,
                'mean_accuracy': mean_accuracy,
                'mean_test_f1_micro': mean_f1_micro,
                'mean_test_f1_macro': mean_f1_macro,
                'epochs': epochs,
                'batch_size': batch_size,
                'genre_labels': genres,
            })

    results_df = pd.DataFrame(results)
    results_df['genre_labels'] = genres
    
    logger.success("Results compiled into dataframe.")
    
    return results_df

# Execute model training and evaluation for All genres
logger.info("Executing model training and evaluation for All genres...")
df_results_NN = analysis_functions.exec_model(train_and_evaluate_nn, df, all_genres, feat_group_model, "Neural Network", False)

# Combine results and save to CSV
df_results_NN.to_csv(f'{folder_output}/neural_network_results_F1_all_genres_more_feats.csv', index=False)
logger.success("Results saved to f'{folder_output}/neural_network_results_F1_all_genres_more_feats.csv'")
