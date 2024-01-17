from typing import Dict

import yaml
from loguru import logger


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
        logger.exception(
            f"Error: Unable to load config file with error {str(e)}"
        )
        exit(1)
