import os

import yaml
from colorama import Fore, init
from google.cloud import aiplatform, storage
from google.oauth2 import service_account
from loguru import logger

from src.utils.general_util_functions import parse_cfg

init(autoreset=True)

CONFIG_FILE_PATH = "./cfg/catalog.yaml"
META_FILE_PATH = "./mlruns/models/Investment_strategy_model/version-1/meta.yaml"


def extract_source_file_from_meta_file(meta_file_path: str) -> str:
    """
    Extract the source file path from the meta.yaml file.

    Args:
        meta_file_path (str): path to the meta.yaml file

    Returns:
        str: path to the source file
    """
    try:
        with open(meta_file_path, "r") as f:
            meta_file = yaml.safe_load(f)
        logger.info(f"{Fore.GREEN}Meta file loaded.")
    except FileNotFoundError:
        logger.info(
            f"{Fore.RED}File {meta_file_path} not found. Please check your code again."
        )

    logger.info(f"{Fore.GREEN}Extracting source file from meta.yaml file.")
    if meta_file["source"]:
        logger.info(f"{Fore.GREEN}Model source file found at {meta_file['source']}.")
        # return meta_file["source"].replace("file://", "") + "/model.pkl"
        return meta_file["source"].replace("file://", "")
    else:
        logger.info(
            f"{Fore.RED}Source file not found in meta.yaml file."
            f"{Fore.RED}Please check your mlflow run again."
        )


def upload_directory_to_gcs(bucket_name, source_directory, destination_blob_prefix):
    """Uploads a directory to the bucket."""
    storage_client = storage.Client.from_service_account_json(
        "/Users/hanyuwu/Study/robo_advisor/cfg/hy-learning-93fbebd91d63.json"
    )
    bucket = storage_client.bucket(bucket_name)

    for local_file in os.listdir(source_directory):
        local_file_path = os.path.join(source_directory, local_file)
        if os.path.isfile(local_file_path):
            blob_path = os.path.join(destination_blob_prefix, local_file)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{blob_path}")


def get_model_artifact_uri(gsutil_uri, destination_blob_name) -> str:
    return f"{gsutil_uri}/{destination_blob_name}"


def deploy_model(
    project: str,
    location: str,
    model_display_name: str,
    endpoint_display_name: str,
    model_artifact_uri: str,
    service_account_json: str,
    machine_type: str = "n1-standard-4",
):
    # Authenticate using the service account JSON key file
    credentials = service_account.Credentials.from_service_account_file(
        service_account_json
    )

    client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}

    # Initialize the Vertex AI Model Service client
    model_client = aiplatform.gapic.ModelServiceClient(
        client_options=client_options, credentials=credentials
    )

    # Initialize the Vertex AI Endpoint Service client
    endpoint_client = aiplatform.gapic.EndpointServiceClient(
        client_options=client_options, credentials=credentials
    )

    parent = f"projects/{project}/locations/{location}"

    # Upload the model
    model = {
        "display_name": model_display_name,
        "artifact_uri": model_artifact_uri,
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/"
            "prediction/sklearn-cpu.0-23:latest",
        },
    }
    operation = model_client.upload_model(parent=parent, model=model)
    print("Uploading model...")
    upload_model_response = operation.result()
    model_name = upload_model_response.model

    # Create an endpoint
    endpoint = {"display_name": endpoint_display_name}
    operation = endpoint_client.create_endpoint(parent=parent, endpoint=endpoint)
    print("Creating endpoint...")
    endpoint_response = operation.result()
    endpoint_name = endpoint_response.name

    # Deploy the model to the endpoint
    deployed_model = {
        "model": model_name,
        "display_name": model_display_name,
        "dedicated_resources": {
            "machine_spec": {"machine_type": machine_type},
            "min_replica_count": 1,
            "max_replica_count": 1,
        },
    }
    traffic_split = {"0": 100}
    operation = endpoint_client.deploy_model(
        endpoint=endpoint_name,
        deployed_model=deployed_model,
        traffic_split=traffic_split,
    )
    print("Deploying model to endpoint...")
    operation.result()

    print(f"Model deployed to endpoint: {endpoint_name}")


if __name__ == "__main__":
    cfg = parse_cfg(CONFIG_FILE_PATH)
    bucket_name = cfg["gcs_bucket"]
    source_file = extract_source_file_from_meta_file(META_FILE_PATH)
    destination_blob_name = cfg["destination_blob_name"]
    upload_directory_to_gcs(bucket_name, source_file, destination_blob_name)
    model_artifact_uri = get_model_artifact_uri(cfg["gsutil_uri"], destination_blob_name)

    deploy_model(
        cfg["gcp_project_id"],
        cfg["location"],
        cfg["model_display_name"],
        cfg["endpoint_display_name"],
        model_artifact_uri,
        cfg["gcp_auth_path"],
    )
