from typing import Dict

import yaml
from google.cloud import bigquery
from loguru import logger


def parse_cfg(path) -> Dict:
    """
    Reading in the config.

    Returns:
        dict: with keys representing the parameters
    """
    try:
        with open(path, "r", encoding="utf-8") as yamlfile:
            cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    except Exception:
        logger.error("Error: Unable to parse the config file")
    return cfg


def upload_BQ(dataset_id: str, table_id: str, csv_file_path: str):
    """
    Upload COA flat file to BQ and use for joining later.

    Args:
        dataset_id (str): dataset id for this GCP Project
        table_id (str): the name we assign to the table
        csv_file_path (str): where this flat file is stored relative
        to this directory
    """
    client = bigquery.Client()
    table_ref = client.dataset(dataset_id).table(table_id)

    # Define the job configuration
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.skip_leading_rows = 1
    job_config.autodetect = True
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    try:
        with open(csv_file_path, "rb") as source_file:
            job = client.load_table_from_file(
                source_file, table_ref, job_config=job_config
            )
    except Exception:
        logger.error("Error: CSV export to Bigquery table failed")
        return

    job.result()

    if job.state == "DONE":
        logger.success(
            f"CSV file '{csv_file_path}' successfully exported \n"
            "to Bigquery table"
        )
    elif job.state in ["PENDING", "RUNNING"]:
        logger.info(f"Job is still in progress, current state: {job.state}")
    else:
        logger.error(f"Job ended with state: {job.state}")
