# import os
import time

from colorama import Fore, Style, init
from loguru import logger

from src.data_processing import DataProcessor
from src.mlflow_tuning import Optuna_flow
from src.model_pipeline import ModelPipeline
from src.utils.data_schema import synthetic_data_schema
from src.utils.general_util_functions import parse_cfg, validate_config

# import yaml
from src.utils.synthetic_data_generator import generate_synthetic_data

init(autoreset=True)

logger.add("./logs/Workflow_logs.log")

# Global variables
CONFIG_FILE_PATH = "./cfg/catalog.yaml"
GCP_AUTH_FILE_PATH = "./cfg/hy-learning-93fbebd91d63.json"
MODEL_NAME = "Investment_strategy_model"
MODEL_VERSION = 1


def log_time_taken(start_time):
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


def process_data(config, synthetic_data):
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


def get_important_features(df_final):
    lgbm_pipeline = ModelPipeline(df_final, "Investment_Strategy")
    lgbm_pipeline.run(15)
    feature_importance = lgbm_pipeline.run(10)
    print(feature_importance.index)

    return feature_importance.index.tolist()


def trim_predictors(df_final, reduced_features):
    reduced_features.append("Investment_Strategy")
    df_final = df_final[reduced_features]
    return df_final


def main():
    try:
        pipeline_start_time = time.time()
        logger.info(f"{Fore.BLUE}Starting the ML Pipeline{Style.RESET_ALL}")

        config = parse_cfg(CONFIG_FILE_PATH)
        validate_config(config)
        logger.info("Config validated.")

        # credential_path = config["gcp_auth_path"]
        # dataset_id = config["dataset_id"]
        # table_id = config["table_id"]
        csv_file_path = config["raw_filepath"]

        logger.info(f"{Fore.GREEN}Generating synthetic dataset for ML{Style.RESET_ALL}")
        synthetic_data = generate_synthetic_data()
        synthetic_data_schema.validate(synthetic_data)
        logger.success("Data schema validated.")
        synthetic_data.to_csv(csv_file_path, index=False)

        df_final = process_data(config, synthetic_data)

        reduced_features = get_important_features(df_final)

        df_trimmed = trim_predictors(df_final, reduced_features)
        print(df_trimmed.columns)

        X, y = (
            df_trimmed.drop("Investment_Strategy", axis=1),
            df_trimmed["Investment_Strategy"],
        )

        Optuna_flow(MODEL_NAME, MODEL_VERSION, X, y, n_trials=10)

        log_time_taken(pipeline_start_time)
    except Exception as e:
        logger.error(f"{Fore.RED}An error occurred: {e}")


if __name__ == "__main__":
    main()
