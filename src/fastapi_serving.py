import os
from typing import Any, Dict, List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from loguru import logger
from sklearn.metrics import accuracy_score

from src.main import process_data
from src.utils.general_util_functions import parse_cfg
from src.utils.synthetic_data_generator import generate_synthetic_data

app = FastAPI()

# Load the model at startup
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = os.getenv("MODEL_VERSION")
CONFIG_FILE_PATH = "./cfg/catalog.yaml"
config = parse_cfg(CONFIG_FILE_PATH)


class Model:
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


model = Model()


@app.router.on_startup.append
async def startup_event():
    try:
        model.load_model()
        model.load_features_to_keep(config)
        logger.info(model.important_features)
        synthetic_data = generate_synthetic_data(n_samples=30)
        processed_synthetic_data = process_data(config, synthetic_data)
        test_data = processed_synthetic_data[model.important_features]
        if set(model.important_features) != set(test_data.columns):
            raise ValueError(
                "Mismatch in the number of features between"
                "the model and the synthetic data"
            )
        else:
            logger.info("All good!")
        test_data_as_dict = test_data.to_dict(orient="records")
        logger.info(test_data_as_dict)
        predictions = model.make_prediction(test_data_as_dict)
        logger.info(predictions)
        logger.info(
            accuracy_score(processed_synthetic_data["Investment_Strategy"], predictions)
        )
    except Exception as e:
        logger.error(f"Error during startup: {e}")


@app.post("/predict")
async def predict(data: List[Dict[str, Any]]) -> List:
    return model.make_prediction(data)


# uvicorn fastapi_serving:app --reload
