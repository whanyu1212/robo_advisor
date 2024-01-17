# import os
import time

# import pandas as pd
# import yaml
from colorama import Fore, Style, init
from loguru import logger

init(autoreset=True)


# Global variables
CONFIG_FILE_PATH = "./config/default_config.yaml"
USER_INPUT_FILE_PATH = "./data/test_user_input.csv"


def log_time_taken(start_time, task_name="Robo Advisor Orchestration"):
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(
        f"{Fore.BLUE}Total time taken for {task_name}: \n"
        f"{total_time} seconds{Style.RESET_ALL}"
    )
