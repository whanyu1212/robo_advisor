import os
from typing import Any, Dict, List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from loguru import logger
from mlflow.tracking import MlflowClient

from src.utils.general_util_functions import parse_cfg

app = FastAPI()
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = os.getenv("MODEL_VERSION")
CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH")
config = parse_cfg(CONFIG_FILE_PATH)


class MLModelPredictor:
    def __init__(self):
        # Initialize the trained model, important features, and MLflow client
        self.trained_model = None
        self.important_features = None
        self.client = MlflowClient()

    def get_model_details(self) -> tuple[str, str]:
        """
        Use the MLflow client to get the experiment ID and run ID.

        Returns:
            tuple[str, str]: a tuple of strings
            containing the experiment ID and run ID
        """

        # Get the model version details
        model_version_details = self.client.get_model_version(
            name=MODEL_NAME, version=MODEL_VERSION
        )

        # Extract the run ID from the model version details
        run_id = model_version_details.run_id

        # Get the run details using the run ID
        run = mlflow.get_run(run_id)

        # Extract the experiment ID from the run details
        experiment_id = run.info.experiment_id

        # Return the experiment ID and run ID
        return experiment_id, run_id

    def load_model(self, experiment_id: str, run_id: str):
        """
        Locate and load the trained model based on the experiment ID and
        run ID.

        Args:
            experiment_id (str): experiment ID of the best model
            run_id (str): run ID of the best model
        """
        try:
            self.trained_model = mlflow.pyfunc.load_model(
                model_uri=f"./mlruns/{experiment_id}/{run_id}/artifacts/lightgbm_model"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_features_to_keep(self, config: dict):
        """
        Based on the configuration file, extract the important features
        to keep from the trained model.

        Args:
            config (dict): configuration dictionary
        """
        self.important_features = config["reduced_features"]

    def make_prediction(self, data: List[Dict[str, Any]]) -> List:
        """
        Fastapi only works with JSON-compatible data. In order to make
        predictions, we need to convert the data to a pandas DataFrame
        first because ml model only accepts pandas DataFrame.

        Args:
            data (List[Dict[str, Any]]): data to make predictions on

        Returns:
            List: list of predictions
        """
        # Convert the data to a pandas DataFrame
        data_df = pd.DataFrame(data)

        # data_dict = data_df.to_dict(orient="records")

        return self.trained_model.predict(data_df).tolist()


model = MLModelPredictor()


# define a startup event handler
@app.router.on_startup.append
async def startup_event():
    """Create a startup event handler to load the model."""
    try:
        experiment_id, run_id = model.get_model_details()
        model.load_model(experiment_id, run_id)
        model.load_features_to_keep(config)
        logger.info("Model and features loaded successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


# define a route for a POST request to the predict end point
@app.post("/predict")
async def predict(data: List[Dict[str, Any]]) -> List:
    """
    Create a route for a POST request to the predict end point.

    Args:
        data (List[Dict[str, Any]]): data in the required format

    Returns:
        List: predictions stored in a list
    """
    return model.make_prediction(data)


# uvicorn src.fastapi_serving:app --reload
