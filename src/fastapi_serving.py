import os
from typing import Any, Dict, List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from loguru import logger

from src.utils.general_util_functions import parse_cfg

app = FastAPI()

# Load the model at startup
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = os.getenv("MODEL_VERSION")
CONFIG_FILE_PATH = "./cfg/catalog.yaml"
config = parse_cfg(CONFIG_FILE_PATH)


class MLModelPredictor:
    def __init__(self):
        self.trained_model = None
        self.important_features = None

    def load_model(self):
        try:
            self.trained_model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_features_to_keep(self, config: dict):
        self.important_features = config["reduced_features"]

    def make_prediction(self, data: List[Dict[str, Any]]) -> List:
        data_df = pd.DataFrame(data)
        return self.trained_model.predict(data_df).tolist()


model = MLModelPredictor()


@app.router.on_startup.append
async def startup_event():
    try:
        model.load_model()
        model.load_features_to_keep(config)
        logger.info("Model and features loaded successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.post("/predict")
async def predict(data: List[Dict[str, Any]]) -> List:
    return model.make_prediction(data)


# uvicorn fastapi_serving:app --reload
