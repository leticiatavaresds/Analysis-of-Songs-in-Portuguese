#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Author: Letícia Tavares
Date: 2024-08-06
Version: 1.0.0

Description: 
    This script contains functions and variables for the execution of the best models.
"""

# Standard library imports
import os  # Operating system interface

# Third-party library imports
import pandas as pd  # Data manipulation and analysis
from sklearn.model_selection import train_test_split  # Splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Feature scaling and label encoding
from sklearn.ensemble import RandomForestClassifier  # Random Forest model for classification
from sklearn.metrics import confusion_matrix, classification_report  # Performance metrics
from loguru import logger  # Advanced logging functionality
import tensorflow as tf  # Deep learning framework

# Local application/library specific imports
import functions.analysis_functions as analysis_functions  # Custom analysis functions

folder_output = "03_Output_Best_Models"

if not os.path.exists(folder_output):
    os.makedirs(folder_output)

def model_random_forest(params, df, genres, feats, type, group):

    df_genres = analysis_functions.make_df_genres(df, genres)
    
    best_params = {
        'random_state': 42,
        'n_jobs': -1
    }

    best_params = {**best_params, **params}

    filtered_df = df.loc[df[genres].sum(axis=1) == 1].reset_index(drop=True)

    # Split the data into training and test sets
    X = filtered_df[feats]

    label_encoder = LabelEncoder()
    y = df_genres
    y = y.idxmax(axis=1)
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model with the best parameters
    logger.info("Training the model with the best parameters...")
    model = RandomForestClassifier(**best_params)
    model.fit(X_train_scaled, y_train)

    # Predict on the test set
    logger.info("Predicting on the test set...")
    y_pred = model.predict(X_test_scaled)

    # Generate the classification report
    report = classification_report(y_test, y_pred, zero_division=1)
    # logger.info(f"Classification Report:\n{report}")

    # Save the classification report to a CSV file
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    report_df = pd.DataFrame(report_dict).transpose()
    file_name = f"{folder_output}/random_forest_{type}_{group}_classification_report.csv"
    report_df.to_csv(file_name)
    logger.success(f"Classification report saved to '{file_name}'")

    # Generate the confusion matrix
    logger.info("Generating the confusion matrix...")
    conf_matrix = confusion_matrix(y_test, y_pred)
    df_conf_matrix = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    file_name = f"{folder_output}/random_forest_{type}_{group}_conf_matrix.csv"
    df_conf_matrix.to_csv(file_name)
    logger.success(f"Confusion Matrix saved to {file_name}")

    # Calculate the correlation matrix
    logger.info("Calculating the correlation matrix...")
    corr_matrix = X.corr()
    file_name = f"{folder_output}/random_forest_{type}_{group}_corr_matrix.csv"
    corr_matrix.to_csv(file_name)
    logger.success(f"Correlation Matrix saved to {file_name}")

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances = feature_importances.to_frame().rename({0: "importance"}, axis = 1)
    file_name = f"{folder_output}/random_forest_{type}_{group}_feature_importances.csv"
    feature_importances.to_csv(file_name)
    logger.success(f"Feature Importances saved to {file_name}")

def create_nn_model(dense_sizes=(32, 32), dropout_rate=0.1, input_shape=None, output_shape=None):
    
    inp = tensorflow.keras.layers.Input(shape=(input_shape,))

    # Dense layers
    layer = inp
    for size in dense_sizes:
        layer = tensorflow.keras.layers.Dense(size, activation="selu", kernel_initializer="lecun_normal")(layer)
        layer = tensorflow.keras.layers.AlphaDropout(dropout_rate)(layer)

    # Output layer
    out = tensorflow.keras.layers.Dense(output_shape, activation="sigmoid")(layer)

    # Create the model
    model = tensorflow.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def model_neural_network(params, df, genres, feats, type, group):   

    filtered_df = df.loc[df[genres].sum(axis=1) == 1].reset_index(drop=True)

    # Split the data into training and test sets
    X = filtered_df[feats]
    y = filtered_df[genres].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model with the best parameters
    logger.info("Training the model with the best parameters...")

    model = create_nn_model(dense_sizes = params["dense_sizes"], dropout_rate = params["dropout_rate"], 
                            input_shape=X.shape[1], output_shape=y.shape[1])
    
    model.fit(X_train_scaled, y_train, epochs = params["epochs"], batch_size = params["batch_size"], verbose=1)

    # Predict on the test set
    logger.info("Predicting on the test set...")
    y_pred = model.predict(X_test_scaled)

    # Convert predictions to binary labels
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Save the classification report to a CSV file
    report_dict = classification_report(y_test, y_pred_binary, target_names=genres, output_dict=True, zero_division=1)
    report_df = pd.DataFrame(report_dict).transpose()
    file_name = f"{folder_output}/neural_network_{type}_{group}_classification_report.csv"
    report_df.to_csv(file_name)
    logger.success(f"Classification Report saved to {file_name}")

    label_encoder = LabelEncoder()
    y = filtered_df[genres]
    y = y.idxmax(axis=1)
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    # Generate the confusion matrix
    logger.info("Generating the confusion matrix...")
    conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_binary.argmax(axis=1))
    df_conf_matrix = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    file_name = f"{folder_output}/neural_network_{type}_{group}_conf_matrix.csv"
    df_conf_matrix.to_csv(file_name)
    logger.success(f"Confusion Matrix saved to {file_name}")

    # Calculate the correlation matrix
    logger.info("Calculating the correlation matrix...")
    corr_matrix = X.corr()
    file_name = f"{folder_output}/neural_network_{type}_{group}_corr_matrix.csv"
    corr_matrix.to_csv(file_name)
    logger.success(f"Correlation Matrix saved to {file_name}")
