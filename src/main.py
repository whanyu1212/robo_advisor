# import os
import time

from colorama import Fore, Style, init
from loguru import logger

from src.data_processing import DataProcessor
from src.utils.data_schema import synthetic_data_schema
from src.utils.general_util_functions import parse_cfg, upload_csv_BQ, validate_config

# import yaml
from src.utils.synthetic_data_generator import generate_synthetic_data

init(autoreset=True)

logger.add("./logs/Workflow_logs.log")

# Global variables
CONFIG_FILE_PATH = "./cfg/catalog.yaml"
GCP_AUTH_FILE_PATH = "./cfg/hy-learning-93fbebd91d63.json"


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


if __name__ == "__main__":
    try:
        pipeline_start_time = time.time()
        logger.info(f"{Fore.BLUE}Starting the ML Pipeline{Style.RESET_ALL}")

        config = parse_cfg(CONFIG_FILE_PATH)
        validate_config(config)
        logger.info("Config validated.")

        credential_path = config["gcp_auth_path"]
        dataset_id = config["dataset_id"]
        table_id = config["table_id"]
        csv_file_path = config["raw_filepath"]

        logger.info(f"{Fore.GREEN}Generating synthetic dataset for ML{Style.RESET_ALL}")
        synthetic_data = generate_synthetic_data()
        synthetic_data_schema.validate(synthetic_data)
        logger.success("Data schema validated.")
        upload_csv_BQ(credential_path, dataset_id, table_id, csv_file_path)

        processor = DataProcessor(
            synthetic_data,
            config["useless_features"],
            config["categorical_features"],
            config["numerical_features"],
        )
        df_processed = (
            processor.remove_useless_columns()
            .encode_categorical_columns()
            .outlier_removal()
        )

        df_processed.to_csv("./data/processed/processed_data.csv", index=False)
        print(df_processed.head())

        log_time_taken(pipeline_start_time)
    except Exception as e:
        logger.error(f"{Fore.RED}An error occurred: {e}")
