import os
import json
import requests

from dynamic_risk_assessment_system.model_helpers.training import ModelTrainer
from dynamic_risk_assessment_system.model_helpers.scoring import ModelScorer
from dynamic_risk_assessment_system.model_helpers.deployment import ModelDeployment
from dynamic_risk_assessment_system.diagnostics.diagnostics import Diagnostics
from dynamic_risk_assessment_system.reporting.reporting import ModelReport
from dynamic_risk_assessment_system.data_ingestion.ingestion import DataIngestion

##################Dynamic Risk Assessment System - Full Process Automation

##################Check and read new data
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

ingested_files_path = config['prod_deployment_path']
#first, read ingestedfiles.txt
ingested_files = os.path.join(ingested_files_path, 'ingestedfiles.txt')
if os.path.exists(ingested_files):
    with open(ingested_files, 'r') as file:
        ingested_files_list = file.read().splitlines()
else:
    ingested_files_list = []

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_data_path = config['input_folder_path']
source_files = os.listdir(source_data_path)
new_data_files = [file for file in source_files if file not in ingested_files_list]
if new_data_files:
    print(f"New data files found: {new_data_files}")
else:
    print("No new data files found.")
    new_data_files = None

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not new_data_files:
    print("No new data to process. Ending the process.")
    exit()
else:
    print("Proceeding with new data files.")
    # copy files in ingested_files_list to the source_data_path
    for file in ingested_files_list:
        base_file_path = os.path.join(config["base_data_path"], file)
        if os.path.exists(base_file_path):
            print(f"Copying {file} to source data path.")
            # Here you can implement the logic to copy the file if needed
            os.system(f'cp {base_file_path} {source_data_path}')
        else:
            print(f"File {file} does not exist in source data path.")

    ##################Data ingestion
    data_ingestion = DataIngestion(
        input_folder_path=config['input_folder_path'],
        output_folder_path=config['output_folder_path']
    )
    data_ingestion.merge_multiple_dataframe()

    ##################Model training
    model_trainer = ModelTrainer(
            output_model_path=config['output_folder_path'],
            output_folder_path=config['output_folder_path']
        )
    model_trainer.train()


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
model_score_file = os.path.join(config['prod_deployment_path'], 'latestscore.txt')
if os.path.exists(model_score_file):
    with open(model_score_file, 'r') as file:
        base_model_score = float(file.read().strip())
else:
    print("No base model score found. Assuming no previous model score.")
    base_model_score = 0.0

new_model_scorer = ModelScorer(
    model_path=config['output_folder_path'],
    test_data_path=config['test_data_path'],
)
new_model_scorer.load_model()
new_model_scorer.load_test_data()
new_score = new_model_scorer.score_model()

if new_score != base_model_score:
    print(f"Model drift detected: Base score {base_model_score}, New score {new_score}")
    model_deployment = ModelDeployment(
            output_folder_path=config['output_folder_path'],
            prod_deployment_path=config['prod_deployment_path']
        )
    model_deployment.deploy_model()
else:
    print(f"No model drift detected: Base score {base_model_score}, New score {new_score}")
    exit()



##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
# run apicalls.py to call the API endpoints
os.system('python apicalls.py')






