from typing import Any, Dict, List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from sklearn.metrics import accuracy_score

from src.data_processing import DataProcessor
from src.utils.general_util_functions import parse_cfg
from src.utils.synthetic_data_generator import generate_synthetic_data

app = FastAPI()

# Load the model at startup
model_name = "Investment_strategy_model"
model_version = 7
CONFIG_FILE_PATH = "./cfg/catalog.yaml"
config = parse_cfg(CONFIG_FILE_PATH)

app = FastAPI()


def process_data(config: dict, synthetic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Initializes the DataProcessor class and processes the data.

    Args:
        config (dict): configuration for columns
        synthetic_data (pd.DataFrame): original synthetic data

    Returns:
        pd.DataFrame: processed synthetic data
    """
    processor = DataProcessor(
        synthetic_data,
        config["useless_features"],
        config["categorical_features"],
        config["numerical_features"],
    )
    df_processed = processor.remove_useless_columns()
    df_dummies = processor.encode_categorical_columns(df_processed)
    df_final = processor.combine_dummy_n_numeric(df_dummies, df_processed)
    return df_final


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
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    return model


@app.router.on_startup.append
async def startup_event():
    global model
    try:
        model = load_model()
        global features_to_keep
        features_to_keep = load_features_to_keep(config)
        print(features_to_keep)
        test_set = generate_synthetic_data(n_samples=30)
        test_set_encoded = process_data(config, test_set)
        test_data = test_set_encoded[features_to_keep]
        if set(features_to_keep) != set(test_data.columns):
            raise ValueError(
                "Mismatch in the number of features between"
                "the model and the synthetic data"
            )
        else:
            print("All good!")
        test_data_dict = test_data.to_dict(orient="records")
        print(test_data_dict)
        predictions = make_prediction(test_data_dict)
        print(predictions)
        print(accuracy_score(test_set["Investment_Strategy"], predictions))
    except Exception as e:
        print(f"Error during startup: {e}")


def make_prediction(data: List[Dict[str, Any]]) -> List:
    data_df = pd.DataFrame(data)
    return model.predict(data_df).tolist()


@app.post("/predict")
async def predict(data: List[Dict[str, Any]]) -> List:
    return make_prediction(data)
