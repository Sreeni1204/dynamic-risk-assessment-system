# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deployment helper for the dynamic risk assessment system.
This module handles the deployment of the trained model and associated files
to a production environment.
"""

import pickle
import os


class ModelDeployment:
    """
    A class to handle the deployment of a trained model and its associated files.
    It copies the model, latest score, and ingested files to a specified production deployment path.
    """
    def __init__(
            self,
            output_folder_path,
            prod_deployment_path
    ) -> None:
        """
        Initializes the ModelDeployment class with paths for the model, latest score, ingested files,
        and the production deployment directory.
        Args:
            output_folder_path: Path where the model and related files are stored.
            prod_deployment_path: Path where the model and related files will be deployed.
        """
        self.model_path = os.path.join(output_folder_path, 'trainedmodel.pkl')
        self.latest_score_path = os.path.join(output_folder_path, 'latestscore.txt')
        self.ingest_files_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
        self.prod_deployment_path = prod_deployment_path

    def deploy_model(
            self
    ) -> None:
        """
        Deploys the model by copying the latest pickle file, latest score, and ingested files
        to the production deployment directory.
        Raises:
            FileNotFoundError: If any of the required files do not exist.
        """
        #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.latest_score_path):
            raise FileNotFoundError(f"Latest score file not found at {self.latest_score_path}")
        if not os.path.exists(self.ingest_files_path):
            raise FileNotFoundError(f"Ingest files file not found at {self.ingest_files_path}")
        
        # copy the model file
        model_dest_path = os.path.join(self.prod_deployment_path, 'trainedmodel.pkl')
        os.makedirs(os.path.dirname(model_dest_path), exist_ok=True)
        with open(self.model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(model_dest_path, 'wb') as model_dest_file:
            pickle.dump(model, model_dest_file)
        
        # copy the latest score file
        latest_score_dest_path = os.path.join(self.prod_deployment_path, 'latestscore.txt')
        with open(self.latest_score_path, 'r') as score_file:
            latest_score = score_file.read()
        with open(latest_score_dest_path, 'w') as score_dest_file:
            score_dest_file.write(latest_score)

        # copy the ingest files
        ingest_files_dest_path = os.path.join(self.prod_deployment_path, 'ingestedfiles.txt')
        with open(self.ingest_files_path, 'r') as ingest_file:
            ingest_files = ingest_file.read()
        with open(ingest_files_dest_path, 'w') as ingest_dest_file:
            ingest_dest_file.write(ingest_files)
