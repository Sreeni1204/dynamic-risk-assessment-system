# /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the main application for the dynamic risk assessment system.
It sets up the Flask app, defines endpoints for predictions, scoring, summary statistics,
and diagnostics, and handles the configuration and execution of the system.
"""

import argparse
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import json

from dynamic_risk_assessment_system.diagnostics.diagnostics import Diagnostics
from dynamic_risk_assessment_system.model_helpers.training import ModelTrainer
from dynamic_risk_assessment_system.model_helpers.scoring import ModelScorer
from dynamic_risk_assessment_system.data_ingestion.ingestion import DataIngestion
from dynamic_risk_assessment_system.model_helpers.deployment import ModelDeployment
from dynamic_risk_assessment_system.reporting.reporting import ModelReport


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

#######################First run for practice data
@app.route("/first_run_on_practice_data", methods=['GET','OPTIONS'])
def first_run():
    """ This function is used to run the first run on practice data.
    It will merge the multiple dataframes, train the model, score the model,
    deploy the model and generate a report.
    """
    try:
        # data ingestion
        data_ingestion = DataIngestion(
            input_folder_path=app.config['custom_config']['input_folder_path'],
            output_folder_path=app.config['custom_config']['output_folder_path']
        )
        data_ingestion.merge_multiple_dataframe()

        # model training
        model_trainer = ModelTrainer(
            output_model_path=app.config['custom_config']['output_folder_path'],
            output_folder_path=app.config['custom_config']['output_folder_path']
        )
        model_trainer.train()

        # model scoring
        model_scorer = ModelScorer(
            model_path=app.config['custom_config']['output_folder_path'],
            test_data_path=app.config['custom_config']['test_data_path']
        )
        model_scorer.load_model()
        model_scorer.load_test_data()
        _ = model_scorer.score_model()

        # model deployment
        model_deployment = ModelDeployment(
            output_folder_path=app.config['custom_config']['output_folder_path'],
            prod_deployment_path=app.config['custom_config']['prod_deployment_path']
        )
        model_deployment.deploy_model()

    except Exception as e:
        print("First run error:", e)
        return jsonify({"error": str(e)}), 500
    
    return jsonify({"message": "First run completed successfully"}), 200

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    """ This function should load the model from the production deployment path,
    and then use it to make predictions on the test data provided as a DataFrame.
    It should return a list of predictions.
    Args:
        test_data: A pandas DataFrame containing the test data for predictions.
    Returns:
        A list of predictions made by the model.
    """
    try:
        model_diagnostics = Diagnostics(
            dataset_csv_path=app.config['custom_config']['output_folder_path'],
            test_data_path=app.config['custom_config']['test_data_path'],
            prod_deployment_path=app.config['custom_config']['prod_deployment_path'],
            input_folder_path=app.config['custom_config']['input_folder_path']
        )

        #get the test data from the request
        test_data = request.get_json()
        if not test_data:
            return jsonify({"error": "No test data provided"}), 400
        test_df = pd.DataFrame(test_data)
        if test_df.empty:
            return jsonify({"error": "Test data is empty"}), 400
        #make predictions using the loaded model

        predictions = model_diagnostics.model_predictions(test_df)
        if not predictions:
            return jsonify({"error": "No predictions made"}), 500
        #convert predictions to a list
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        elif isinstance(predictions, pd.Series):
            predictions = predictions.values.tolist()
        elif isinstance(predictions, list):
            predictions = predictions
        else:
            return jsonify({"error": "Unexpected prediction format"}), 500

        model_report = ModelReport(
            dataset_csv_path=app.config['custom_config']['output_folder_path'],
            test_data_path=app.config['custom_config']['test_data_path'],
            prod_deployment_path=app.config['custom_config']['prod_deployment_path'],
            input_folder_path=app.config['custom_config']['input_folder_path']
        )
        model_report.generate_report()

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500
    
    return jsonify({"predictions": predictions}), 200


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring_stats():
    """ This function should load the model from the production deployment path,
    and then use it to score the test data provided as a DataFrame.
    It should return a single F1 score number.
    Args:
        test_data: A pandas DataFrame containing the test data for scoring.
    Returns:
        A single F1 score number representing the model's performance on the test data.
    """
    
    model_scorer = ModelScorer(
        model_path=app.config['custom_config']['prod_deployment_path'],
        test_data_path=app.config['custom_config']['test_data_path'],
    )
    model_scorer.load_model()
    model_scorer.load_test_data()
    f1_score = model_scorer.score_model()
    #check the score of the deployed model
    return jsonify({"f1_score": f1_score}), 200


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():
    """ This function should calculate summary statistics for each numerical column in the dataset stored in the directory specified
    by dataset_csv_path in config.json. It should output a Python list containing the mean,
    median, and standard deviation for each numerical column.
    """
    model_diagnostics = Diagnostics(
        dataset_csv_path=app.config['custom_config']['output_folder_path'],
        test_data_path=app.config['custom_config']['test_data_path'],
        prod_deployment_path=app.config['custom_config']['prod_deployment_path'],
        input_folder_path=app.config['custom_config']['input_folder_path']
    )

    summary_stats = model_diagnostics.dataframe_summary()
    if not summary_stats:
        return jsonify({"error": "No summary statistics calculated"}), 500
    #convert summary stats to a list
    if isinstance(summary_stats, pd.DataFrame):
        summary_stats = summary_stats.to_dict(orient='records')
    elif isinstance(summary_stats, list):
        summary_stats = summary_stats
    else:
        return jsonify({"error": "Unexpected summary statistics format"}), 500
    #return the summary statistics
    return jsonify({"summary_stats": summary_stats}), 200


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics_stats():
    """ This function should calculate the number and percentage of missing data for each column in the dataset stored
    in the directory specified by dataset_csv_path in config.json. It should output a Python list
    containing the column name, the number of missing values, and the percentage of missing values for each column.
    """
    model_diagnostics = Diagnostics(
        dataset_csv_path=app.config['custom_config']['output_folder_path'],
        test_data_path=app.config['custom_config']['test_data_path'],
        prod_deployment_path=app.config['custom_config']['prod_deployment_path'],
        input_folder_path=app.config['custom_config']['input_folder_path']
    )
    missing_data = model_diagnostics.missing_data()
    if not missing_data:
        return jsonify({"error": "No missing data found"}), 500
    #convert missing data to a list
    if isinstance(missing_data, pd.DataFrame):
        missing_data = missing_data.to_dict(orient='records')
    elif isinstance(missing_data, list):
        missing_data = missing_data
    else:
        return jsonify({"error": "Unexpected missing data format"}), 500
    
    execution_time = model_diagnostics.execution_time()
    if not execution_time:
        return jsonify({"error": "No execution time found"}), 500
    #convert execution time to a list
    if isinstance(execution_time, pd.DataFrame):
        execution_time = execution_time.to_dict(orient='records')
    elif isinstance(execution_time, list):
        execution_time = execution_time
    else:
        return jsonify({"error": "Unexpected execution time format"}), 500
    
    outdated_packages = model_diagnostics.outdated_packages_list()
    if not outdated_packages:
        return jsonify({"error": "No outdated packages found"}), 500
    #convert outdated packages to a list
    if isinstance(outdated_packages, pd.DataFrame):
        outdated_packages = outdated_packages.to_dict(orient='records')
    elif isinstance(outdated_packages, list):
        outdated_packages = outdated_packages
    else:
        return jsonify({"error": "Unexpected outdated packages format"}), 500
    
    # return missing data, execution time, and outdated packages as a dictionary
    diagnostics_info = {
        "missing_data": missing_data,
        "execution_time": execution_time,
        "outdated_packages": outdated_packages
    }
    return jsonify(diagnostics_info), 200


def main():
    parser = argparse.ArgumentParser(description='Run the dynamic risk assessment system.')
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host address for the Flask app'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for the Flask app'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='config.json',
        help='Path to the configuration file'
    )
    args = parser.parse_args()

    with open(args.config_path,'r') as f:
        config = json.load(f)

    app.config['custom_config'] = config

    app.run(
        host=args.host,
        port=args.port,
        debug=True  # Optional: enables auto-reload and debug logs
    )
