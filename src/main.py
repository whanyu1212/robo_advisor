# import os
import time

# import pandas as pd
# import yaml
from colorama import Fore, Style, init
from loguru import logger

init(autoreset=True)


# Global variables
CONFIG_FILE_PATH = "./cfg/catalog.yaml"
GCP_AUTH_FILE_PATH = "./cfg/hy-learning-93fbebd91d63.json"


def log_time_taken(start_time, task_name="Robo Advisor Orchestration"):
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(
        f"{Fore.BLUE}Total time taken for {task_name}: \n"
        f"{total_time} seconds{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    try:
        pipeline_start_time = time.time()
        logger.info(f"{Fore.BLUE}Starting the ML Pipeline{Style.RESET_ALL}")
        log_time_taken(pipeline_start_time)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
