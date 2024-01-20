import os
import time
from typing import IO, Dict, Union

import pandas as pd
import yaml
from colorama import Fore, Style
from dotenv import load_dotenv
from google.cloud import bigquery
from loguru import logger

load_dotenv()

# Global variables

JOB_STATE_DONE = os.getenv("JOB_STATE_DONE")
JOB_STATE_PENDING = os.getenv("JOB_STATE_PENDING")
JOB_STATE_RUNNING = os.getenv("JOB_STATE_RUNNING")


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


def parse_cfg(path) -> Dict:
    """
    Reading in the config.

    Returns:
        dict: with keys representing the parameters
    """
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            logger.info("Config file loaded successfully")
            return config
    except Exception as e:
        logger.exception(f"Error: Unable to load config file with error {str(e)}")
        exit(1)


def validate_config(config: Dict) -> None:
    """
    Check if the necessary config keys are present and have the correct
    type.

    Args:
        config (Dict): Config in dict format

    Raises:
        ValueError: If the key is not found
        TypeError: If the values are not of type str
        ValueError: if the values are empty
    """
    required_keys = ["gcp_auth_path", "dataset_id", "table_id", "raw_filepath"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

        value = config[key]
        if not isinstance(value, str):
            raise TypeError(
                "Expected string for config key {key},"
                f" but got {type(value).__name__}"
            )

        if not value:
            raise ValueError(f"Config key {key} must not be empty")


def log_important_features(features, config_file):
    # Load the existing config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Update the config with the new features
    config["reduced_features"] = features

    # Write the updated config back to the file
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)


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


def split_predictor_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into predictor and target.

    Args:
        df (pd.DataFrame): ML ready data

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: predictor and target, respectively
        2 pandas dataframes
    """
    X, y = (
        df.drop("Investment_Strategy", axis=1),
        df[["Investment_Strategy"]],
    )
    return X, y


def create_bigquery_client(credential_path: str) -> bigquery.Client:
    """
    Initialize the bigquery client.

    Args:
        credential_path (str): Path to the GCP service account key

    Returns:
        bigquery.Client: as the name suggests
    """
    return bigquery.Client.from_service_account_json(credential_path)


def create_job_config() -> bigquery.LoadJobConfig:
    """
    Iniatialize the job config for for uploading CSV.

    Returns:
        bigquery.LoadJobConfig: Config defining the job
    """
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.skip_leading_rows = 1
    job_config.autodetect = True
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    return job_config


def load_table_from_file(
    client: bigquery.Client,
    source_file: IO[bytes],
    table_ref: bigquery.TableReference,
    job_config: bigquery.LoadJobConfig,
) -> Union[bigquery.LoadJob, None]:
    try:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
        job.result()
        return job
    except Exception as e:
        logger.exception(
            f"Error: CSV export to Bigquery table failed with error {str(e)}"
        )
        return None


def handle_job_result(job: bigquery.LoadJob, csv_file_path: str) -> None:
    """
    Check the job state and log the result.

    Args:
        job (bigquery.LoadJob): Whether it is still running
        csv_file_path (str): path where the raw csv is stored
    """

    if job.state == JOB_STATE_DONE:
        logger.success(
            f"CSV file '{csv_file_path}' successfully "
            f"exported to Bigquery table, job ID: {job.job_id}"
        )
    elif job.state in [JOB_STATE_PENDING, JOB_STATE_RUNNING]:
        logger.info(
            f"Job {job.job_id} is still in progress," f"current state: {job.state}"
        )
    else:
        logger.error(f"Job {job.job_id} ended with state: {job.state}")


def open_source_file(csv_file_path: str) -> IO[bytes]:
    try:
        return open(csv_file_path, "rb")
    except (IOError, FileNotFoundError) as e:
        logger.error(f"Failed to open file {csv_file_path}: {str(e)}")
        raise


def upload_csv_BQ(
    credential_path: str, dataset_id: str, table_id: str, csv_file_path: str
) -> None:
    """
    Trigger the upload of the CSV to Bigquery.

    Args:
        credential_path (str): GCP_auth_file_path
        dataset_id (str): dataset created on BQ
        table_id (str): name given to the new table
        csv_file_path (str): path where the raw csv is stored
    """
    client = create_bigquery_client(credential_path)
    logger.info("BigQuery client initialized.")
    table_ref = client.dataset(dataset_id).table(table_id)
    job_config = create_job_config()

    source_file = open_source_file(csv_file_path)
    if source_file is not None:
        logger.info(f"Successfully opened file {csv_file_path}.")
        job = load_table_from_file(client, source_file, table_ref, job_config)
        if job is not None:
            handle_job_result(job, csv_file_path)
        else:
            logger.error("Error: Export" f"to Bigquery table failed for {csv_file_path}")
