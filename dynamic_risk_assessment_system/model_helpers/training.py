# /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the ModelTrainer class which is responsible for training a machine learning model.
It reads data from a CSV file, splits it into training and testing sets, fits a Logistic Regression model,
and saves the trained model to a specified path.
"""

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class ModelTrainer():
    """    A class to train a Logistic Regression model on the provided dataset.
    Attributes:
        output_model_path (str): Path to save the trained model.
        output_folder_path (str): Path to the folder containing the dataset.
        model (LogisticRegression): The Logistic Regression model instance.
    """
    def __init__(
            self,
            output_model_path,
            output_folder_path
    ) -> None:
        """
        Initialize the ModelTrainer with paths for output model and output folder.
        """
        self.output_folder_path = output_folder_path
        self.output_model_path = output_model_path
        self.model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)

    
    def train(
            self
    ) -> None:
        """
        Train the model using the provided data.
        """
        data = pd.read_csv(os.path.join(self.output_folder_path, 'finaldata.csv'))
        # Split the data into features and target variable
        x = data.drop(columns=['corporation', 'exited'], axis=1)
        y = data['exited']
        
        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Fit the model
        self.model.fit(x_train, y_train)
        
        #write the trained model to your workspace in a file called trainedmodel.pkl
        with open(os.path.join(self.output_model_path, 'trainedmodel.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
