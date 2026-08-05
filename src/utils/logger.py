from datetime import datetime
import logging
from pathlib import Path


def get_logger(name: str, log_dir: str, log_file: str):

    log_dir = Path("output/logs") / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / log_file

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_log_prefix(name: str) -> str:
    now = datetime.now()
    asctime = now.strftime("%Y-%m-%d %H:%M:%S") + f",{now.microsecond // 1000:03d}"
    return f"{asctime} | INFO | "


def print_configuration(config, logger):
    logger.info("===== CONFIG =====")
    for section, sub_dict in config.items():
        if isinstance(sub_dict, dict):
            for k, v in sub_dict.items():
                # Logs only the clean field name 'k' without the parent 'section.' prefix
                logger.info(f"{k:20}: {v}")
    logger.info("==================")
