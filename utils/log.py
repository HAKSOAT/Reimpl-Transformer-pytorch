import logging
import sys
from os import makedirs, path

BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))


def get_logger(run_name, log_path=None):
    log_dir = path.join(BASE_DIR, "logs")
    if not path.exists(log_dir):
        makedirs(log_dir)

    if log_path is None:
        log_path = path.join(log_dir, f"{run_name}.log")

    logger = logging.getLogger(run_name)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, "w", "utf-8")
        stream_handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter("[%(levelname)s] %(asctime)s > %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

    return logger
