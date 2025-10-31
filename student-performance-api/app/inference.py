from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch 
import pandas as pd
import numpy as np
import os

app = FastAPI()

# Load sklearn pipeline (includes scaler if you saved pipeline)
REG_MODEL_PATH = os.path.join("models", "best_regression_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
NN_WEIGHTS = os.path.join("models", "neural_regressor.pth")
reg_model = joblib.load(REG_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


##Define NN same arch as training

import torch.nn as nn
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

    
INPUT_DIM = 8
nn_model = NeuralNet(INPUT_DIM)
nn_model.load_state_dict(torch.load(NN_WEIGHTS, map_location='cpu'))
nn_model.eval()


class Features(BaseModel):
    gender: int
    parental_level_of_education:int
    lunch: int
    test_preparation_course: int
    race_ethnicity_group_B: int
    race_ethnicity_group_C: int
    race_ethnicity_group_D: int
    race_ethnicity_group_E: int

@app.post("/predict")

def predict(feat: Features):
    row = {
        "gender": feat.gender,
        "parental level of education": feat.parental_level_of_education,
        "lunch": feat.lunch,
        "test preparation course": feat.test_preparation_course,
        "race/ethnicity_group B": feat.race_ethnicity_group_B,
        "race/ethnicity_group C": feat.race_ethnicity_group_C,
        "race/ethnicity_group D": feat.race_ethnicity_group_D,
        "race/ethnicity_group E": feat.race_ethnicity_group_E
    }
    df = pd.DataFrame([row])

    reg_pred = reg_model.predict(df)[0]

    X_scaled = scaler.transform(df)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        nn_pred = nn_model(X_tensor).numpy().ravel()[0]

    return {
        "LinearRegressionPrediction": float(reg_pred),
        "NeuralNetworkPrediction": float(nn_pred)
    }