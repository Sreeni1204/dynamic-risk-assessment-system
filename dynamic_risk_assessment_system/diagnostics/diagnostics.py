# /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides diagnostic functions for the dynamic risk assessment system.
It includes methods for model predictions, dataframe summary statistics,
missing data analysis, execution time measurement, and package version checks.
"""

import pandas as pd
import numpy as np
import timeit
import os
import pickle
import re
import requests
import toml

from dynamic_risk_assessment_system.model_helpers.training import ModelTrainer
from dynamic_risk_assessment_system.data_ingestion.ingestion import DataIngestion


class Diagnostics:
    """
    A class to perform diagnostics on the dynamic risk assessment system.
    It includes methods for model predictions, dataframe summary statistics,
    missing data analysis, execution time measurement, and package version checks.
    """
    def __init__(
            self,
            dataset_csv_path,
            test_data_path,
            prod_deployment_path,
            input_folder_path
    ) -> None:
        """
        Initializes the Diagnostics class with paths for dataset, test data,
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
        self.model = None
        self.dataset = None

    
    def model_predictions(
            self,
            test_data: pd.DataFrame
    ) -> list:
        """
        This function should load the model from the production deployment path,
        and then use it to make predictions on the test data provided as a DataFrame.
        It should return a list of predictions.
        Args:
            test_data: A pandas DataFrame containing the test data for predictions.
        Returns:
            A list of predictions made by the model.
        """
        if not os.path.exists(self.prod_deployment_path):
            raise FileNotFoundError(f"Model not found at {self.prod_deployment_path}")
        # Load the model
        with open(os.path.join(self.prod_deployment_path, "trainedmodel.pkl"), 'rb') as f:
            self.model = pickle.load(f)
        
        # Load the test data
        if test_data is None:
            raise ValueError("Test data not loaded.")
        
        predictions_list = []
        for index, row in test_data.iterrows():
            # drop the target column if it exists
            row = row.drop('exited', errors='ignore')
            row = row.drop('corporation', errors='ignore')
            # convert row to a list and make prediction
            row = row.values.tolist()
            prediction = self.model.predict([row])[0]
            predictions_list.append(prediction)

        predictions_list = [int(p) for p in predictions_list]
        return predictions_list

    def dataframe_summary(
            self
    ) -> list:
        """
        This function should calculate summary statistics for each numerical column in the dataset stored in the directory specified
        by dataset_csv_path in config.json. It should output a Python list containing the mean, median, and standard deviation for each numerical column.
        """
        if self.dataset is None:
            self.dataset = pd.read_csv(os.path.join(self.dataset_csv_path, "finaldata.csv"))

        summary_stats = []
        for column in self.dataset.select_dtypes(include=[np.number]).columns:
            mean = self.dataset[column].mean()
            median = self.dataset[column].median()
            std_dev = self.dataset[column].std()
            summary_stats.append({
                'column': column,
                'mean': mean,
                'median': median,
                'std_dev': std_dev
            })

        return summary_stats
    

    def missing_data(
            self
    ) -> list:
        """
        This function should calculate the number and percentage of missing data for each column in the dataset stored
        in the directory specified by dataset_csv_path in config.json. It should output a Python list containing the column name,
        the number of missing values, and the percentage of missing values for each column.
        """
        if self.dataset is None:
            self.dataset = pd.read_csv(os.path.join(self.dataset_csv_path, "finaldata.csv"))

        missing_data = []
        for column in self.dataset.columns:
            
            if self.dataset[column].dtype == 'object':
                # For categorical columns, we can skip missing data calculation
                continue
            missing_data_count = float(self.dataset[column].isnull().sum())
            missing_percentage = float(self.dataset[column].isnull().mean() * 100)
            missing_data.append({
                'column': column,
                'missing_count': missing_data_count,
                'missing_percentage': missing_percentage
            })

        return missing_data
    

    def execution_time(
            self
    ) -> list:
        """
        This function should measure the execution time of the training.py and ingestion.py scripts.
        It should return a Python list containing the execution time for each script in seconds.
        """
        timings = []
        
        # Measure time for training.py
        start_time = timeit.default_timer()
        trainer = ModelTrainer(output_model_path=self.dataset_csv_path, output_folder_path=self.dataset_csv_path)
        trainer.train()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)

        # Measure time for ingestion.py
        start_time = timeit.default_timer()
        ingestion = DataIngestion(input_folder_path=self.input_folder_path, output_folder_path=self.dataset_csv_path)
        ingestion.merge_multiple_dataframe()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)

        return timings
    

    def get_latest_version(
            self,
            package_name
    ) -> str:
        """
        Query PyPI for the latest version of a package.
        Args:
            package_name: The name of the package to check.
        Returns:
            The latest version of the package as a string, or "N/A" if not found.
        """
        url = f"https://pypi.org/pypi/{package_name}/json"
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return data["info"]["version"]
        except Exception:
            pass
        return "N/A"
    
    def parse_dep_string(
            self,
            dep_str
    ) -> tuple:
        """
        Parse a dependency string like 'flask (>=3.1.1)' into ('flask', '3.1.1').
        """
        match = re.match(r"^([^\s]+)\s*\((?:[<>=!~]*\s*)([\d\.]+)\)", dep_str)
        if match:
            name, version = match.groups()
            return name, version
        else:
            # If no version constraint, just return the name and None
            return dep_str.strip(), None
    

    def parse_pyproject_toml(
            self,
            toml_path="pyproject.toml"
    ) -> dict:
        """
        Parse pyproject.toml and return a dict of {package: version}.
        Args:
            toml_path: Path to the pyproject.toml file.
        Returns:
            A dictionary where keys are package names and values are their versions.
        """
        pyproject = toml.load(toml_path)
        packages = {}
        # Poetry dependencies are usually under [tool.poetry.dependencies]
        dependencies = pyproject.get("project", {}).get("dependencies", {})

        for pkg in dependencies:
            # Skip Python itself
            if pkg.lower() == "python":
                continue
            # If version is a dict (for extras), get the version string
            # 'flask (>=3.1.1)'
            package, version = self.parse_dep_string(pkg)
            packages[package] = version
        
        return packages
    

    def outdated_packages_list(
            self
    ) -> list:
        """
        This function should check for outdated packages in the poetry.lock file.
        It should return a Python list containing the package name, current version, and latest version for each outdated package.
        """
        lock_path = os.path.join(os.getcwd(), "pyproject.toml")
        packages = self.parse_pyproject_toml(lock_path)
        result = []
        for pkg, current_version in packages.items():
            latest_version = self.get_latest_version(pkg)
            result.append({
                "package": pkg,
                "current_version": current_version,
                "latest_version": latest_version
            })

        return result
