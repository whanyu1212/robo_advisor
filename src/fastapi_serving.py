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

app = FastAPI()


def load_features_to_keep(config: dict) -> List[str]:
    """
    Load the list of features to keep from the config file.

    Args:
        config_file (str): path to the config file

    Returns:
        List[str]: a list of features to keep
    """

    return config["reduced_features"]


# Assume this is your model loading function
def load_model():
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    return model


@app.router.on_startup.append
async def startup_event():
    global trained_model
    global important_features
    try:
        trained_model = load_model()
        important_features = load_features_to_keep(config)
        logger.info(important_features)
        synthetic_data = generate_synthetic_data(n_samples=30)
        processed_synthetic_data = process_data(config, synthetic_data)
        test_data = processed_synthetic_data[important_features]
        if set(important_features) != set(test_data.columns):
            raise ValueError(
                "Mismatch in the number of features between"
                "the model and the synthetic data"
            )
        else:
            logger.info("All good!")
        test_data_as_dict = test_data.to_dict(orient="records")
        logger.info(test_data_as_dict)
        predictions = make_prediction(test_data_as_dict)
        logger.info(predictions)
        logger.info(
            accuracy_score(processed_synthetic_data["Investment_Strategy"], predictions)
        )
    except Exception as e:
        logger.error(f"Error during startup: {e}")


def make_prediction(data: List[Dict[str, Any]]) -> List:
    data_df = pd.DataFrame(data)
    return trained_model.predict(data_df).tolist()


@app.post("/predict")
async def predict(data: List[Dict[str, Any]]) -> List:
    return make_prediction(data)


# uvicorn fastapi_serving:app --reload
