# /usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains the ModelScorer class which is responsible for scoring a machine learning model.
It loads a trained model, evaluates it on test data, and saves the score to a file.
"""

import pandas as pd
import pickle
import os
from sklearn import metrics


class ModelScorer:
    def __init__(self, model_path, test_data_path):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.model = None
        self.test_data = None

    def load_model(self):
        with open(os.path.join(self.model_path, "trainedmodel.pkl"), 'rb') as f:
            self.model = pickle.load(f)

    def load_test_data(self):
        self.test_data = pd.read_csv(os.path.join(self.test_data_path, "testdata.csv"))

    def score_model(self):
        if self.model is None or self.test_data is None:
            raise ValueError("Model or test data not loaded.")

        x_test = self.test_data.drop(['corporation', 'exited'], axis=1)
        y_test = self.test_data['exited']
        
        y_pred = self.model.predict(x_test)
        f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

        with open(os.path.join(self.model_path, 'latestscore.txt'), 'w') as f:
            f.write(str(f1_score))

        return f1_score
