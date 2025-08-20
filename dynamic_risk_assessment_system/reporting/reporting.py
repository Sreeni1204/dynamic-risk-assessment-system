# /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides diagnostic functions for the dynamic risk assessment system.
It includes methods for model predictions, dataframe summary statistics,
missing data analysis, execution time measurement, and package version checks.
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import metrics
import seaborn as sns

from dynamic_risk_assessment_system.diagnostics.diagnostics import Diagnostics


class ModelReport:
    def __init__(
            self,
            dataset_csv_path,
            test_data_path,
            prod_deployment_path,
            input_folder_path
    ) -> None:
        """
        Initializes the ModelReport class with paths for dataset, test data,
        production deployment, and input folder.
        Args:
            dataset_csv_path: Path to the directory containing the dataset CSV file.
            test_data_path: Path to the directory containing the test data.
            prod_deployment_path: Path to the directory containing the production model.
            input_folder_path: Path to the directory containing input data for ingestion.
        """
        self.dataset_csv_path = dataset_csv_path
        self.test_data_path = test_data_path
        self.prod_deployment_path = prod_deployment_path
        self.input_folder_path = input_folder_path
        self.data_diagnostics = Diagnostics(
            self.dataset_csv_path,
            self.test_data_path,
            self.prod_deployment_path,
            self.input_folder_path
        )

    def generate_report(
            self
    ) -> None:
        """
        Generates a report containing model predictions, dataframe summary statistics,
        missing data analysis, execution time measurement, and package version checks.
        """
        # Model Predictions
        test_data = pd.read_csv(os.path.join(self.test_data_path, "testdata.csv"))
        predictions = self.data_diagnostics.model_predictions(test_data)
        
        actual_values = test_data['exited']
        confusion_matrix = metrics.confusion_matrix(actual_values, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(self.dataset_csv_path, 'confusionmatrix.png'))
        plt.close()
