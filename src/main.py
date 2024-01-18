import time

import pandas as pd
from colorama import Fore, Style, init
from loguru import logger

from src.data_processing import DataProcessor
from src.mlflow_tuning import Optuna_flow
from src.model_pipeline import ModelPipeline
from src.utils.data_schema import synthetic_data_schema
from src.utils.general_util_functions import parse_cfg, validate_config
from src.utils.synthetic_data_generator import generate_synthetic_data

init(autoreset=True)

logger.add("./logs/Workflow_logs.log")

# Global variables
CONFIG_FILE_PATH = "./cfg/catalog.yaml"
MODEL_NAME = "Investment_strategy_model"
MODEL_VERSION = 1


def log_time_taken(start_time: float):
    """
    Log the time taken for the entire pipeline to run.

    Args:
        start_time (float): the start timestamp in seconds
    """
    end_time = time.time()
    total_time = end_time - start_time
    time_unit = "seconds"

    if total_time > 60:
        total_time /= 60
        time_unit = "minutes"

        if total_time > 60:
            total_time /= 60
            time_unit = "hours"

    logger.info(
        f"Process finished. Time taken: {Fore.GREEN}{total_time} "
        f"{time_unit}{Style.RESET_ALL}. Exiting..."
    )


def get_csv_writer(df: pd.DataFrame, file_path: str) -> callable:
    """
    Lazy evaluation of csv writer.

    Args:
        df (pd.DataFrame): dataframe to be saved
        file_path (str): location to save the dataframe

    Returns:
        callable: write to csv function
    """

    def write_to_csv():
        df.to_csv(file_path, index=False)

    return write_to_csv


def generate_and_validate_data() -> pd.DataFrame:
    """
    Generate synthetic data and validate it against the schema.

    Returns:
        pd.DataFrame: validated synthetic data
    """
    synthetic_data = generate_synthetic_data()
    synthetic_data_schema.validate(synthetic_data)
    logger.success("Data schema validated.")
    return synthetic_data


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


def get_important_features(df_final: pd.DataFrame, n: int = 15) -> list[str]:
    """
    Fit the model without any feature selection and return the top n
    features.

    Args:
        df_final (pd.DataFrame): processed synthetic data
        n (int, optional): number of features to keep. Defaults to 15.

    Returns:
        list[str]: list of column names to keep
    """
    lgbm_pipeline = ModelPipeline(df_final, "Investment_Strategy")
    lgbm_pipeline.run(n)
    feature_importance = lgbm_pipeline.run(10)
    print(feature_importance.index)

    return feature_importance.index.tolist()


def trim_predictors(df_final: pd.DataFrame, reduced_features: list[str]) -> pd.DataFrame:
    """
    Trim the processed data to only keep the important features.

    Args:
        df_final (pd.DataFrame): processed synthetic data
        reduced_features (list[str]): list of column names to keep

    Returns:
        pd.DataFrame: trimmed processed synthetic data
    """
    reduced_features.append("Investment_Strategy")
    df_final = df_final[reduced_features]
    return df_final


def main():
    try:
        config = parse_cfg(CONFIG_FILE_PATH)
        validate_config(config)
        logger.info("Config validated.")
        pipeline_start_time = time.time()
        logger.info(f"{Fore.BLUE}Starting the ML Pipeline{Style.RESET_ALL}")

        csv_file_path = config["raw_filepath"]
        logger.info(f"{Fore.GREEN}Generating synthetic dataset for ML{Style.RESET_ALL}")

        synthetic_data = generate_and_validate_data()
        synthetic_data.to_csv(csv_file_path, index=False)

        df_final = process_data(config, synthetic_data)
        reduced_features = get_important_features(df_final)
        df_trimmed = trim_predictors(df_final, reduced_features)

        write_func_df_trimmed = get_csv_writer(
            df_trimmed, "./data/processed/df_trimmed.csv"
        )
        write_func_synthetic_data = get_csv_writer(
            synthetic_data, "./data/raw/synthetic_data.csv"
        )

        X, y = (
            df_trimmed.drop("Investment_Strategy", axis=1),
            df_trimmed["Investment_Strategy"],
        )

        Optuna_flow(MODEL_NAME, MODEL_VERSION, X, y, n_trials=10)

        # execute the lazy evaluation
        write_func_df_trimmed()
        write_func_synthetic_data()
        log_time_taken(pipeline_start_time)
    except Exception as e:
        logger.error(f"{Fore.RED}An error occurred: {e}")


if __name__ == "__main__":
    main()
